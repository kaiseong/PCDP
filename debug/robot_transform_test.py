import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from piper_sdk import C_PiperInterface_V2
from pcdp.common import RISE_transformation as rise_tf

# Transformation matrices and workspace bounds
camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
])
robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])

workspace_bounds = np.array([
    [-0.000, 0.740],    # X range (milli meters)
    [-0.400, 0.350],    # Y range (milli meters)
    [-0.100, 0.400]     # Z range (milli meters)
])

z_offset = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0.07], 
    [0, 0, 0, 1]])

def main():
    piper = C_PiperInterface_V2("can_slave")
    piper.ConnectPort()
    
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
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Live View', width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    pcd = o3d.geometry.PointCloud()
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame.transform(camera_to_base)
    last_eef_transform = np.identity(4)
    first_iter = True
    
    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None: 
                if not vis.poll_events(): break
                continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None: 
                if not vis.poll_events(): break
                continue
            
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc = np.asarray(point_cloud)
            processed_pc = preprocess(pc)
            
            pcd.points = o3d.utility.Vector3dVector(processed_pc[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(processed_pc[:, 3:6] / 255.0)
            
            eef_pose_raw = piper.GetArmEndPoseMsgs()
            eef_pos_m = np.array(eef_pose_raw[:3]) / 1000.0
            eef_rot_rad = np.deg2rad(eef_pose_raw[3:])
            
            eef_to_robot_base = rise_tf.rot_trans_mat(eef_pos_m, eef_rot_rad)
            current_eef_transform = robot_to_base @ eef_to_robot_base @ z_offset
            
            eef_frame.transform(current_eef_transform @ np.linalg.inv(last_eef_transform))
            last_eef_transform = current_eef_transform.copy()

            if first_iter:
                vis.add_geometry(pcd)
                vis.add_geometry(eef_frame)
                vis.add_geometry(base_frame)
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
            
            if not vis.poll_events():
                break
            vis.update_renderer()
            
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        pipeline.stop()
        piper.DisconnectPort()
        vis.destroy_window()

if __name__ == '__main__':
    main()