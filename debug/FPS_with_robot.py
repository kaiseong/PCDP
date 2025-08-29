import os 
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
import pcdp.common.mono_time as mono_time
from piper_sdk import *
import time
from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.RISE_transformation import xyz_rot_transform

URDF_PATH = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
MESH_DIRS = ["/home/moai/pcdp/dependencies"]
EEF_FRAME_NAME = "link6" 

def main():

    # piper setting
    piper = C_PiperInterface_V2("can_slave", start_sdk_joint_limit=True)
    piper.ConnectPort()
    piper.SetSDKJointLimitParam('j4', -1.7453, 1.7977)
    piper.SetSDKJointLimitParam('j5', -1.3265, 1.2741)
    piper.SetSDKJointLimitParam('j6', -2.0071, 2.2166)
    time.sleep(0.5)
    piper.EnableArm(7)
    time.sleep(1.0)


    # orbbec setting
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

    use_vis = True
    if use_vis:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.0, 0.0, 0.0])
        pcd = o3d.geometry.PointCloud()
    
    first_iter = True
    
    durations = np.array([])
    cnt = 0
    t_cnt=0
    try:
        while cnt<500:
            end_pose = piper.GetArmEndPoseMsgs()
            end_pose[:3] = end_pose[:3] * 0.001
            translation = end_pose[:3]
            rotation = end_pose[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
            T_k_matrix = self.robot_to_base @ eef_to_robot_base_k
            pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='XYZ'
            )




            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc=np.asarray(point_cloud) 
            pc = pc[pc[:, 2] > 0.0]

            if cnt>200:
                start=mono_time.now_ms()
                tim= mono_time.now_ms() - start
                if tim>50.0:
                    t_cnt+=1
                durations = np.append(durations, tim)

            if use_vis:
                vis_pc = np.asarray(point_cloud, dtype =np.float32)
                pcd.points = o3d.utility.Vector3dVector(vis_pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(vis_pc[:, 3:6] / 255.0)
                if first_iter:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
                    first_iter = False
                else:
                    vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
            cnt += 1

    except Exception as e:
        print(f"error: {e}")
    finally:
        print(f"duration\n\
            mean: {durations.mean()}\n \
            max: {durations.max()}\n \
            min: {durations.min()}")
        print(f"t_cnt: {t_cnt}")
        pipeline.stop()

if __name__ == '__main__':
    main()
