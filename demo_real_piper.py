"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> 

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import open3d as o3d
import numpy as np
from termcolor import cprint
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env_piper import RealEnv
from diffusion_policy.real_world.joypad_shared_memory import JoypadSpacemouse
from diffusion_policy.common.precise_sleep import precise_wait
import diffusion_policy.common.mono_time as mono_time
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, vis_camera_idx, init_joints, frequency, command_latency):
    dt = 1/frequency
    
    # IK parameters
    urdf_path = "/home/moai/diffusion_policy/debug/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/diffusion_policy"
    ee_link_name = "link6"
    joints_to_lock_names = [] # No joints to lock in the no-gripper URDF

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            JoypadSpacemouse(shm_manager=shm_manager) as sm, \
            RealEnv(
                output_dir=output, 
                # IK params
                urdf_path=urdf_path,
                mesh_dir=mesh_dir,
                ee_link_name=ee_link_name,
                joints_to_lock_names=joints_to_lock_names,
                # recording resolution
                frequency=frequency,
                init_joints=init_joints,
                orbbec_mode="C2D",
                shm_manager=shm_manager
            ) as env:
            cv2.setNumThreads(1)
            
            
            # 간단히 "5초 이내"로 기다리면서 도달 여부 검사
            base_pose = [0.03751, 0.012182, 0.493991, 0.96503, 1.4663, 1.18428]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([base_pose], [plan_time])
            print("Moving to the base_pose, please wait...")
            start_block = time.time()
            while True:
                if time.time() - start_block > 5.0:
                    break  # 최대 5초만 기다림
                # 현재 로봇 위치
                state = env.get_robot_state()
                actual_pose = state['ArmEndPoseMsgs']  # length=6
                dist = np.linalg.norm(
                    np.array(actual_pose[:3]) - np.array(base_pose[:3]))
                if dist < 0.01:
                    # 어느정도 도달했다고 가정
                    break
                time.sleep(0.05)
            print("Base pose reached (or timed out).")

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = base_pose.copy()
            t_start = mono_time.now_s()
            iter_idx = 0
            stop = False
            is_recording = False
            # open3d visualization
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name='default', width=1280, height=720)
            opt = vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.array([0.0, 0.0, 0.0])
            pcd = o3d.geometry.PointCloud()
            down_smaple_scale = 100
            first = True
            pre_time = time.time_ns()
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()
                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(mono_time.now_s()+2*dt)
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # visualize
                # if is_recording:
                    # print("recoding!")
                vis_pc = obs["pointcloud"][-1].copy()
                episode_id = env.recorder.n_episodes
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                vis_pc = np.asarray(vis_pc)
                vis_pc = vis_pc[vis_pc[:, 2] > 0.0]  # remove points below ground
                vis_pc = vis_pc[:: int(down_smaple_scale)]  # downsample
                
                pcd.points = o3d.utility.Vector3dVector(vis_pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(vis_pc[:, 3:6] / 255.0)

                now = time.time_ns()
                if first:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
                    first = False
                else:
                    vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                pre_time = now


                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (env.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                
                if not sm.is_button_pressed(7):
                    # translation mode
                    drot_xyz[:] = 0
                else:
                    dpos[:] = 0
                if not sm.is_button_pressed(6):
                    # 2D translation mode
                    dpos[2] = 0

                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()

                # execute teleop command
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
