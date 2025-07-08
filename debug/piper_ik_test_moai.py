import numpy as np
from piper_sdk import *
import time
import scipy.spatial.transform as st
import diffusion_policy.common.mono_time as mono_time
# moai.py에서 컨트롤러를 가져옵니다.
from moai import PiperCartesianController

def interpolate_pose_m_rad(start_pose_m_rad, target_pose_m_rad, num_steps):
    """
    시작과 목표 자세 사이를 [m, rad] 단위로 보간하여 경로를 생성합니다.
    위치는 선형으로, 회전은 Slerp로 보간합니다.
    """
    # 위치 보간
    start_pos = start_pose_m_rad[:3]
    target_pos = target_pose_m_rad[:3]
    interpolated_positions = np.linspace(start_pos, target_pos, num_steps)

    # 회전 보간 (Scipy Slerp 사용)
    # 오일러 각도(회전 벡터)를 Rotation 객체로 변환
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
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    try:
        # moai.py의 컨트롤러를 생성합니다.
        ik_controller = PiperCartesianController()
        
        # 로봇 활성화
        piper.EnableArm(7)
        print("Waiting for arm to enable...")
        time.sleep(2)
        
        # 초기 위치로 이동
        piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper.JointCtrl([0, 0, 0, 0, 0, 0])
        print("Moving to initial joint state. Waiting for 5 seconds...")
        time.sleep(5)

        # 현재 로봇의 자세를 시작 자세로 사용
        current_joints_rad = np.deg2rad(np.asarray(piper.GetArmJointMsgs()))
        start_pos_m, start_rot_rad = ik_controller.calculate_forward_kinematics(current_joints_rad)
        start_pose = np.concatenate([start_pos_m, start_rot_rad])
        print(f"Start Pose: {np.round(start_pose, 3)}")
        print(f"get_start_pose: {np.round(piper.GetArmEndPoseMsgs(), 3)}")

        # 목표 자세 (target_pose) [m, m, m, rad, rad, rad]
        target_pose = np.array([0.054952, 0.0, 0.493991,
                                np.deg2rad(0.0), np.deg2rad(85.0), np.deg2rad(0.0)])
        print(f"Target Pose: {np.round(target_pose, 3)}")

        # 보간 스텝 및 제어 주기 설정
        num_interpolation_steps = 20
        control_period = 0.005 # 50Hz
        
        # 경로 보간
        trajectory_positions, trajectory_rotations = interpolate_pose_m_rad(
            start_pose, target_pose, num_interpolation_steps
        )
        
        print(f"--- Starting Trajectory Execution ({num_interpolation_steps} steps) ---")
        
        # 제어 루프 시작
        q_current = current_joints_rad
        duration = np.array([])
        cnt =0
        for i in range(num_interpolation_steps):
            start_time = mono_time.now_s()
            # 1. 현재 스텝의 목표 자세를 설정합니다.
            target_pos = trajectory_positions[i]
            # moai.py의 IK는 오일러 각도를 사용합니다.
            target_rot = trajectory_rotations[i].as_rotvec()

            # 2. moai 컨트롤러를 사용하여 IK를 계산합니다.
            # solve_inverse_kinematics(pos, rot, q_init) -> q, time, converged
            t1 = mono_time.now_ms()
            q_target, _, converged = ik_controller.solve_inverse_kinematics(
                target_position_m=target_pos.tolist(),
                target_rotation_rad=target_rot.tolist(),
                initial_joints_rad=q_current.tolist()
            )
            duration = np.append(duration, mono_time.now_ms() - t1)
            
            if converged:
                # 3. 로봇에 명령을 전송합니다.
                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                piper.JointCtrl(_deg_to_sdk_joint(q_target))
                # 다음 IK를 위해 현재 타겟을 초기 추정값으로 사용
                q_current = np.deg2rad(np.asarray(piper.GetArmJointMsgs()))
            else:
                # print(f"IK solution not found for step {i}. Skipping command.")
                # IK 실패 시, 실제 로봇의 현재 관절 각도를 다시 읽어옵니다.
                cnt+=1
                q_current = np.deg2rad(np.asarray(piper.GetArmJointMsgs()))

            # 5. 제어 주기를 맞춥니다.
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
        time.sleep(3)
        piper.DisableArm(7)
        piper.DisconnectPort()
        print("Disconnected.")