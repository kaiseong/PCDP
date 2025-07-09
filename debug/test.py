from piper_sdk import *
import numpy as np
from diffusion_policy.real_world.pinocchio_ik_controller import PinocchioIKController
import time
import scipy.spatial.transform as st
import pinocchio as pin
import diffusion_policy.common.mono_time as mono_time

def interpolate_pose(start_pose: np.ndarray, end_pose: np.ndarray, t: float) -> np.ndarray:
    """
    두 EndPose 사이를 보간합니다. 위치는 선형으로, 회전은 Slerp로 보간합니다.

    Args:
        start_pose (np.ndarray): 시작 Pose [x, y, z, rx, ry, rz] (회전은 라디안 단위의 회전 벡터)
        end_pose (np.ndarray): 종료 Pose [x, y, z, rx, ry, rz] (회전은 라디안 단위의 회전 벡터)
        t (float): 보간 계수. 0.0에서 1.0 사이의 값. 
                   0.0이면 start_pose를, 1.0이면 end_pose를 반환합니다.

    Returns:
        np.ndarray: 보간된 Pose [x, y, z, rx, ry, rz]
    """
    # t 값을 0.0과 1.0 사이로 제한
    t = np.clip(t, 0.0, 1.0)
    
    # 1. 위치(Position) 분리 및 선형 보간 (Lerp)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    interp_pos = start_pos * (1 - t) + end_pos * t

    # 2. 회전(Rotation) 분리 및 구면 선형 보간 (Slerp)
    # 회전 벡터로부터 Scipy의 Rotation 객체 생성
    key_rotations = st.Rotation.from_rotvec([start_pose[3:], end_pose[3:]])
    key_times = [0, 1] # 시작 시간 0, 종료 시간 1
    
    # Slerp 객체 생성
    slerp = st.Slerp(key_times, key_rotations)
    
    # t 시점의 회전 보간
    interp_rot = slerp(t)

    # 3. 보간된 위치와 회전을 다시 합치기
    # 보간된 회전을 다시 회전 벡터로 변환
    interp_rot_vec = interp_rot.as_rotvec()
    
    return np.concatenate((interp_pos, interp_rot_vec))


if __name__ == "__main__":
    
    piper_master = C_PiperInterface_V2("can_master")
    piper_slave = C_PiperInterface_V2("can_slave")
    piper_master.ConnectPort()
    piper_slave.ConnectPort()
    duration = np.array([])
    try:
        ik_controller = PinocchioIKController(
            urdf_path="/home/moai/diffusion_policy/debug/piper_no_gripper_description.urdf",
            mesh_dir="/home/moai/diffusion_policy",
            ee_link_name="link6",
        )

        piper_slave.EnableArm(7)
        print("Waiting for slave arm to enable...")
        time.sleep(2)
        
        piper_slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper_slave.JointCtrl([0, 0, 0, 0, 0, 0])
        print("Moving to initial joint state. Waiting for 2 seconds...")
        time.sleep(2)
        cnt=0
        tick=0
        
        t_du =None
        while True:
            start_time = mono_time.now_s()
            # [mm, mm, mm, deg, deg, deg]
            if tick % 20 == 0:
                EndPose = np.asarray(piper_master.GetArmEndPoseMsgs())
            
            position = EndPose[:3]/1e3
            rotation = np.deg2rad(EndPose[3:])
            rot_matrix = st.Rotation.from_euler('xyz', rotation).as_matrix()
            command_pose = pin.SE3(rot_matrix, position)

            current_joints_rad = np.deg2rad(np.asarray(piper_slave.GetArmJointMsgs()))
            target_joints_rad = ik_controller.calculate_ik(command_pose, current_joints_rad)
            if target_joints_rad is not None:
                piper_slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                cmd = (np.rad2deg(target_joints_rad)*1e3).astype(int).tolist()
                print(cmd)
                piper_slave.JointCtrl(cmd)
            else:
                cnt+=1
            elapsed = mono_time.now_s() - start_time
            sleep_time = max(0, 0.005 - elapsed)
            if sleep_time >0:
                time.sleep(sleep_time)
            now = mono_time.now_ms()
            if t_du is not None:
                duration = np.append(duration, now-t_du)
            t_du = now
            tick+=1
    finally:
        print(f"mean: {duration.mean()}\n \
                max: {duration.max()}\n \
                min: {duration.min()}")
        piper_slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper_slave.JointCtrl([0, 0, 0, 0, 0, 0])
        time.sleep(6)
        piper_slave.DisableArm(7)
        piper_slave.DisconnectPort()
        piper_master.DisconnectPort()
        print("Disconnected.")
    






        


