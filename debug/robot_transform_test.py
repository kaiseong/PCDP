import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
import pinocchio as pin
import os
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from piper_sdk import *

# ======= 환경에 맞게 수정 =======
URDF_PATH = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
MESH_DIRS = ["/home/moai/pcdp/dependencies"]
EEF_FRAME_NAME = "link6"
# ==============================

def get_q_from_joints_msg(jm):
    q_deg = np.array(jm, dtype=np.float64)
    return np.deg2rad(q_deg[:6])

# Transformation matrices and workspace bounds
camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
])
robot_to_base = np.array([
    [1.,         0.,         0.,          0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])
workspace_bounds = np.array([
    [0.100, 0.800],    # X range (meters)
    [-0.400, 0.400],   # Y range (meters)
    [-0.100, 0.350]    # Z range (meters)
])

def main():
    piper = C_PiperInterface_V2("can_slave")
    piper.ConnectPort()
    
    robot = pin.RobotWrapper.BuildFromURDF(URDF_PATH, MESH_DIRS)
    model, data = robot.model, robot.data
    fid = model.getFrameId(EEF_FRAME_NAME)
    assert fid != len(model.frames), f"Frame '{EEF_FRAME_NAME}' not found."
    
    preprocess = PointCloudPreprocessor(camera_to_base,
                                        workspace_bounds,
                                        enable_sampling=False)
    pipeline = ob.Pipeline()
    cfg = ob.Config()
    try:
        depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                        .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
        color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                        .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
        cfg.enable_stream(depth_profile)
        cfg.enable_stream(color_profile)
    except Exception as e:
        print(f"스트림 프로파일을 가져오는 데 실패했습니다: {e}")
        piper.DisconnectPort()
        return
        
    pipeline.enable_frame_sync()
    pipeline.start(cfg)
    align = ob.AlignFilter(align_to_stream = ob.OBStreamType.DEPTH_STREAM)
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)
    
    # Open3D 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='default', width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([1.0, 1.0, 1.0])
    
    pcd = o3d.geometry.PointCloud()
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    # ======== ✨ 원본 정점 저장 (핵심 수정) ✨ ========
    original_eef_vertices = np.asarray(eef_frame.vertices).copy()
    # ====================================================
    
    first_iter = True
    
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None: continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None: continue
            
            # 포인트 클라우드 처리
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc = np.asarray(point_cloud)
            pc = pc[np.isfinite(pc).all(axis=1)]
            if pc.size == 0: continue
            pc = pc[pc[:, 2] > 0.0]
            processed_pc = preprocess(pc)
            
            # 로봇 EEF 자세 처리 및 변환
            joints = piper.GetArmJointMsgs()
            q = get_q_from_joints_msg(joints)
            
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            M_fk = data.oMf[fid]
            T_eef_in_world = robot_to_base @ M_fk.homogeneous

            # 시각화 업데이트
            pcd.points = o3d.utility.Vector3dVector(processed_pc[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(processed_pc[:, 3:6] / 255.0)
            
            # ======== ✨ EEF 좌표계 업데이트 (핵심 수정) ✨ ========
            new_vertices = (T_eef_in_world[:3, :3] @ original_eef_vertices.T + T_eef_in_world[:3, 3:4]).T
            eef_frame.vertices = o3d.utility.Vector3dVector(new_vertices)
            eef_frame.compute_vertex_normals() # 조명을 위해 법선 벡터 재계산
            # =======================================================

            if first_iter:
                vis.add_geometry(pcd)
                vis.add_geometry(eef_frame)
                
                base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                vis.add_geometry(base_frame)
                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                camera_frame.transform(camera_to_base)
                vis.add_geometry(camera_frame)
                
                points = [ [x,y,z] for x in workspace_bounds[0] for y in workspace_bounds[1] for z in workspace_bounds[2] ]
                lines = [[0,1],[0,2],[1,3],[2,3], [4,5],[4,6],[5,7],[6,7], [0,4],[1,5],[2,6],[3,7]]
                line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines))
                line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])
                vis.add_geometry(line_set)
                
                ctr = vis.get_view_control()
                ctr.set_lookat([0.4, 0, 0.1])
                ctr.set_front([-0.5, -0.8, 0.2])
                ctr.set_up([0.2, 0.3, 1.0])
                ctr.set_zoom(0.7)
                first_iter = False
            else:
                vis.update_geometry(pcd)
                vis.update_geometry(eef_frame)
            
            vis.poll_events()
            vis.update_renderer()
            
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        pipeline.stop()
        piper.DisconnectPort()
        vis.destroy_window()

if __name__ == '__main__':
    main()