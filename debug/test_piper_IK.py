import numpy as np
from piper_sdk import *
from diffusion_policy.real_world.pinocchio_ik_controller import PinocchioIKController
import time
import scipy.spatial.transform as st
import pinocchio as pin
import diffusion_policy.common.mono_time as mono_time

def interpolate_pose_m_rad(start_pose_m_rad, target_pose_m_rad, num_steps):
    """
    시작과 목표 자세 사이를 [m, rad] 단위로 보간하여 경로를 생성합니다.

    Args:
        start_pose_m_rad (np.ndarray): 시작 자세 [x(m), y(m), z(m), r(rad), p(rad), y(rad)].
        target_pose_m_rad (np.ndarray): 목표 자세 [x(m), y(m), z(m), r(rad), p(rad), y(rad)].
        num_steps (int): 생성할 경로의 스텝 수.

    Returns:
        tuple: (보간된 위치 배열, 보간된 Rotation 객체 배열)
    """
    # 1. 위치 보간 (단위: m)
    start_pos = start_pose_m_rad[:3]
    target_pos = target_pose_m_rad[:3]
    interpolated_positions = np.linspace(start_pos, target_pos, num_steps)

    # 2. 회전 보간 (Slerp, 단위: rad)
    key_rotations = st.Rotation.from_rotvec([start_pose_m_rad[3:], target_pose_m_rad[3:]])
    key_times = [0, 1]
    slerp = st.Slerp(key_times, key_rotations)
    interp_times = np.linspace(0, 1, num_steps)
    interpolated_rotations = slerp(interp_times)

    return interpolated_positions, interpolated_rotations

def _deg_to_sdk_joint(rad_angles: np.ndarray) -> list[int]:
    """라디안 단위의 관절 각도를 Piper SDK 정수 형식으로 변환합니다."""
    return (np.rad2deg(rad_angles) * 1e3).astype(int).tolist()

# --- 메인 실행 코드 ---
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_master")
    piper.ConnectPort()
    try:
        ik_controller = PinocchioIKController(
            urdf_path="/home/moai/diffusion_policy/debug/piper_no_gripper_description.urdf",
            mesh_dir="/home/moai/diffusion_policy",
            ee_link_name="link6",
        )
        
        # 로봇 활성화 (기존 코드와 유사)
        piper.EnableArm(7)
        print("Waiting for arm to enable...")
        time.sleep(2) # 로봇이 활성화될 시간을 줍니다.
        
        # 초기 위치로 이동
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl([0, 0, 0, 0, 0, 0])
        print("Moving to initial joint state. Waiting for 2 seconds...")
        time.sleep(2)

        # 시작 자세 (q_init) [m, m, m, rad, rad, rad]
        # [549.52, 0.0, 203.386, 0.0, 85.0, 0.0] -> [m, rad]
        start_pose = np.array([0.054952, 0.0, 0.203386, 
                               np.deg2rad(0.0), np.deg2rad(85.0), np.deg2rad(0.0)])

        # 목표 자세 (target_pose) [m, m, m, rad, rad, rad]
        # [37.51, 12.182, 493.991, 0.0, 85.0, 0.0] -> [m, rad]
        target_pose = np.array([0.154952, 0.021820, 0.493991,
                                np.deg2rad(0.0), np.deg2rad(85.0), np.deg2rad(0.0)])

        # 보간 스텝 수
        num_interpolation_steps = 100
        
        # 포즈 보간 함수 호출
        trajectory_positions, trajectory_rotations = interpolate_pose_m_rad(
            start_pose, target_pose, num_interpolation_steps
        )
        
        print(f"--- Starting Trajectory Execution ({num_interpolation_steps} steps) ---")
        
        # 현재 관절 상태를 IK 초기 추정값으로 사용
        duration = np.array([])
        cnt =0
        for i in range(num_interpolation_steps):
            start_time = mono_time.now_s()
            current_joints_rad = np.deg2rad(np.asarray(piper.GetArmJointMsgs()))
            pos = trajectory_positions[i]
            rot = trajectory_rotations[i].as_matrix()
            target_se3 = pin.SE3(rot, pos)

            # IK 계산 (현재 관절 상태를 초기 추정값으로 사용)
            t1 = mono_time.now_ms()
            target_joints_rad = ik_controller.calculate_ik(target_se3, current_joints_rad)
            duration = np.append(duration, mono_time.now_ms() - t1)
            x,y,z = trajectory_positions[i]*1e6
            r =trajectory_rotations[i].as_euler('xyz', degrees=True)
            x=round(x)
            y=round(y)
            z=round(z)
            roll = round(r[0])*1000
            pitch = round(r[1])*1000
            yaw = round(r[2])*1000
            

            target = [x,y,z,roll, pitch, yaw]
            # print(target)
            piper.EndPoseCtrl(target)
            
            
            if target_joints_rad is not None:
                # 로봇에 명령 전송
                piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                # piper.JointCtrl(_deg_to_sdk_joint(target_joints_rad))
                
            else:
                # print(f"step: {i}, pose: {piper.GetArmEndPoseMsgs()}")
                cnt+=1
            elapsed = mono_time.now_s() - start_time
            sleep_time = max(0, 0.005 - elapsed)
            if sleep_time >0:
                time.sleep(sleep_time) 
        time.sleep(5)
        print(f"duration \n \
              mean: {duration.mean()} \n \
              max: {duration.max()}\n \
              min: {duration.min()}\n \
              failed: {cnt}")
    finally:
        print("Trajectory finished. Moving to zero position.")
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl([0, 0, 0, 0, 0, 0])
        time.sleep(6)
        piper.DisableArm(7)
        piper.DisconnectPort()
        print("Disconnected.")
