import os 
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor
import pcdp.common.mono_time as mono_time
from piper_sdk import *
import time
import pinocchio as pin
from pcdp.common import RISE_transformation as rise_tf



URDF_PATH = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
MESH_DIRS = ["/home/moai/pcdp/dependencies"]
EEF_FRAME_NAME = "link6" 

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

def main():

    pc_preprocess = PointCloudPreprocessor(enable_rgb_normalize=False,
                                           enable_sampling=False,
                                          enable_filter=True,
                                          enable_transform=True,
                                          )
    low_preprocess = LowDimPreprocessor()
    # piper setting
    piper = C_PiperInterface_V2("can_slave", start_sdk_joint_limit=True)
    piper.ConnectPort()
    piper.SetSDKJointLimitParam('j4', -1.7453, 1.7977)
    piper.SetSDKJointLimitParam('j5', -1.3265, 1.2741)
    piper.SetSDKJointLimitParam('j6', -2.0071, 2.2166)
    time.sleep(0.5)
    piper.EnableArm(7)
    time.sleep(1.0)

    # Pinocchio 로봇 로드
    robot = RobotWrapper.BuildFromURDF(URDF_PATH, MESH_DIRS)
    model, data = robot.model, robot.data
    fid = model.getFrameId(EEF_FRAME_NAME)

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
    
    # start sequence
    enable_flag = False
    timeout = 5
    start_time = mono_time.now_s()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = mono_time.now_s() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("상태 활성화:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,5000,0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("시간초과....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("프로그램 timeout으로 종료합니다")
        exit(0)

    use_vis = True
    if use_vis:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.4, 0.45, 0.4])
        pcd = o3d.geometry.PointCloud()
        eef_frame = None # Placeholder for the coordinate frame

    first_iter = True
    
    durations = np.array([])
    cnt = 0
    t_cnt=0
    try:
        while True:
            # Get both pose sources
            end_pose = np.array(piper.GetArmEndPoseMsgs(), dtype=np.float32)
            end_pose[:3] = end_pose[:3] * 0.001
            translation = end_pose[:3]
            rotation = end_pose[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
            T_k_matrix = self.robot_to_base @ eef_to_robot_base_k
            pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='ZYX'
            )

            

            # Transform to the world frame for visualization
            T_k_matrix = robot_to_base @ eef_to_robot_base_k

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
            pc = pc_preprocess.process(pc)

            if cnt>200:
                start=mono_time.now_ms()
                tim= mono_time.now_ms() - start
                if tim>50.0:
                    t_cnt+=1
                durations = np.append(durations, tim)

            if use_vis:
                pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(pc[:, 3:6] / 255.0)

                # Remove the old frame if it exists
                if eef_frame is not None:
                    vis.remove_geometry(eef_frame, reset_bounding_box=False)

                # Create a new frame, transform it, and add it
                eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                eef_frame.transform(T_k_matrix)
                
                if first_iter:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    vis.add_geometry(eef_frame, reset_bounding_box=False)
                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
                    first_iter = False
                else:
                    vis.add_geometry(eef_frame, reset_bounding_box=False)
                    vis.update_geometry(pcd)

                vis.poll_events()
                vis.update_renderer()
            cnt += 1

    except Exception as e:
        print(f"error: {e}")
    finally:
        if durations.size > 0:
            print(f"duration\n\
                mean: {durations.mean()}\n \
                max: {durations.max()}\n \
                min: {durations.min()}")
            print(f"t_cnt: {t_cnt}")
        pipeline.stop()

if __name__ == '__main__':
    main()
