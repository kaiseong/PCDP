import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
import pinocchio as pin
import os
from piper_sdk import *
import pcdp.common.mono_time as mono_time
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor

# 카메라-베이스 변환 행렬
camera_to_base = np.array([
    [ 0.0, -0.9063,  0.4226,  0.110],
    [-1.0,  0.0,    0.0,     0.0],
    [ 0.0, -0.4226, -0.9063,  0.510],
    [ 0.0,  0.0,    0.0,     1.0]
])

# 로봇 모델의 베이스-월드 좌표계 변환 행렬 (시각화 보정용)
robot_to_base = np.array([
    [1., 0., 0., 0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.0]
])

# 작업 공간 경계
workspace_bounds = np.array([
    [0.100, 0.800],    # X 범위 (미터)
    [-0.400, 0.400],   # Y 범위 (미터)
    [-0.100, 0.350]    # Z 범위 (미터)
])

def main():
    # --- 로봇 및 카메라 초기화 ---
    piper = C_PiperInterface_V2("can_slave")
    piper.ConnectPort()

    preprocess = PointCloudPreprocessor(camera_to_base,
                                        workspace_bounds,
                                        enable_sampling=False)
    pipeline = ob.Pipeline()
    cfg = ob.Config()
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
    cfg.enable_stream(depth_profile)
    cfg.enable_stream(color_profile)
    pipeline.enable_frame_sync()
    pipeline.start(cfg)
    align = ob.AlignFilter(align_to_stream = ob.OBStreamType.DEPTH_STREAM)
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)

    # --- Pinocchio 로봇 모델 로드 ---
    urdf_path = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/pcdp/dependencies"
    robot_wrapper = pin.RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
    robot_model = robot_wrapper.model
    visual_model = robot_wrapper.visual_model
    robot_data = robot_model.createData()
    visual_data = visual_model.createData()

    # --- Open3D 시각화 초기화 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='default', width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([1.0, 1.0, 1.0])
    
    pcd = o3d.geometry.PointCloud()

    # 로봇 메시 로드 및 원본 정점 저장
    robot_meshes = []
    original_vertices_list = []
    for geom_obj in visual_model.geometryObjects:
        mesh_path = geom_obj.meshPath
        if mesh_path:
            full_mesh_path = os.path.join(mesh_dir, mesh_path.split('dependencies/')[-1])
            mesh = o3d.io.read_triangle_mesh(full_mesh_path)
            mesh.compute_vertex_normals()
            # Pinocchio의 로컬 변환 적용
            mesh.transform(geom_obj.placement.homogeneous)
            robot_meshes.append(mesh)
            original_vertices_list.append(np.asarray(mesh.vertices))

    first_iter = True
    
    try:
        while True:
            # --- 데이터 취득 ---
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue
            
            # 로봇 관절 각도 (라디안)
            q_deg = np.asarray(piper.GetArmJointMsgs())
            q_rad = np.deg2rad(q_deg)

            # 포인트 클라우드 생성
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc = np.asarray(point_cloud)
            pc = pc[pc[:, 2] > 0.0]
            
            # 포인트 클라우드 전처리 (베이스 좌표계로 변환)
            processed_pc = preprocess(pc)
            
            # --- 로봇 모델 업데이트 (Forward Kinematics) ---
            pin.forwardKinematics(robot_model, robot_data, q_rad)
            pin.updateGeometryPlacements(robot_model, robot_data, visual_model, visual_data, q_rad)

            # --- 시각화 업데이트 ---
            # 포인트 클라우드 업데이트
            transformed_points = processed_pc[:, :3]
            transformed_colors = processed_pc[:, 3:6] / 255.0
            pcd.points = o3d.utility.Vector3dVector(transformed_points)
            pcd.colors = o3d.utility.Vector3dVector(transformed_colors)
            
            if first_iter:
                # 포인트 클라우드 추가
                vis.add_geometry(pcd)
                # 로봇 메시 추가
                for mesh in robot_meshes:
                    vis.add_geometry(mesh)

                # 월드/카메라 좌표계 및 작업 공간 시각화
                base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                vis.add_geometry(base_frame)
                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                camera_frame.transform(camera_to_base)
                vis.add_geometry(camera_frame)
                
                points = [
                    [workspace_bounds[0, 0], workspace_bounds[1, 0], workspace_bounds[2, 0]], [workspace_bounds[0, 1], workspace_bounds[1, 0], workspace_bounds[2, 0]],
                    [workspace_bounds[0, 0], workspace_bounds[1, 1], workspace_bounds[2, 0]], [workspace_bounds[0, 0], workspace_bounds[1, 0], workspace_bounds[2, 1]],
                    [workspace_bounds[0, 1], workspace_bounds[1, 1], workspace_bounds[2, 0]], [workspace_bounds[0, 1], workspace_bounds[1, 0], workspace_bounds[2, 1]],
                    [workspace_bounds[0, 0], workspace_bounds[1, 1], workspace_bounds[2, 1]], [workspace_bounds[0, 1], workspace_bounds[1, 1], workspace_bounds[2, 1]],
                ]
                lines = [[0, 1], [0, 2], [1, 4], [2, 4], [0, 3], [1, 5], [2, 6], [4, 7], [3, 5], [3, 6], [5, 7], [6, 7]]
                colors = [[1, 0, 0] for i in range(len(lines))]
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
                line_set.colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_set)
                
                # 뷰 컨트롤 설정
                ctr = vis.get_view_control()
                ctr.set_lookat([0.4, 0, 0.1])
                ctr.set_front([-0.5, -0.8, 0.2])
                ctr.set_up([0.2, 0.3, 1.0])
                ctr.set_zoom(0.7)
                first_iter = False
            else:
                vis.update_geometry(pcd)
            
            # 로봇 메시 자세 업데이트
            for i, mesh in enumerate(robot_meshes):
                # Pinocchio로부터 월드 좌표계 기준 변환 행렬 가져오기
                T_world_link = visual_data.oMg[i+1].homogeneous # oMg[0] is universe
                # 최종 변환 행렬 계산
                T_final = robot_to_base @ T_world_link
                # 원본 정점을 변환하여 메시 업데이트
                new_vertices = o3d.utility.Vector3dVector(
                    (T_final[:3, :3] @ original_vertices_list[i].T + T_final[:3, 3:4]).T
                )
                mesh.vertices = new_vertices
                mesh.compute_vertex_normals()
                vis.update_geometry(mesh)
            
            vis.poll_events()
            vis.update_renderer()

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        pipeline.stop()
        piper.DisconnectPort()

if __name__ == '__main__':
    main()