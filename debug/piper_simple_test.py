# fk_check.py
import time
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from piper_sdk import *
import pcdp.common.mono_time as mono_time

# ======= 환경에 맞게 수정 =======
URDF_PATH = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
MESH_DIRS = ["/home/moai/pcdp/dependencies"]
EEF_FRAME_NAME = "link6"      # FK로 뽑을 엔드프레임
# ==============================

def get_q_from_joints_msg(jm):
    """Piper 조인트(deg 리스트) -> rad numpy(6,)"""
    if isinstance(jm, (list, tuple, np.ndarray)):
        q_deg = np.array(jm, dtype=np.float64)
        return np.deg2rad(q_deg[:6])
    # 객체 형태일 경우 (필요시 수정)
    q_deg = np.array([
        jm.motor_1.joint_pos, jm.motor_2.joint_pos, jm.motor_3.joint_pos,
        jm.motor_4.joint_pos, jm.motor_5.joint_pos, jm.motor_6.joint_pos
    ], dtype=np.float64)
    return np.deg2rad(q_deg)

def main():
    # Piper 연결
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
    assert fid != len(model.frames), f"Frame '{EEF_FRAME_NAME}' not found."

    cnt = 0
    while cnt < 10000:
        joints = piper.GetArmJointMsgs()      # 예: [-3.434, 0.0, 0.0, -10.001, 18.181, 2.148] (deg)
        q = get_q_from_joints_msg(joints)     # (rad, 6)

        # FK 계산
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M_fk = data.oMf[fid]                  # SE3 (로봇베이스 기준)

        # (m, rad)
        t_m = M_fk.translation                # (3,)
        rpy_rad = pin.rpy.matrixToRpy(M_fk.rotation)  # (roll, pitch, yaw) in rad

        # (mm, deg)
        t_mm = t_m * 1000.0
        rpy_deg = np.rad2deg(rpy_rad)

        # 출력
        print(f"[FK on {EEF_FRAME_NAME}] "
              f"pos (m): {t_m[0]: .6f}, {t_m[1]: .6f}, {t_m[2]: .6f} | "
              f"rpy (rad): {rpy_rad[0]: .6f}, {rpy_rad[1]: .6f}, {rpy_rad[2]: .6f}")
        print(f"               "
              f"pos (mm): {t_mm[0]: .3f}, {t_mm[1]: .3f}, {t_mm[2]: .3f} | "
              f"rpy (deg): {rpy_deg[0]: .3f}, {rpy_deg[1]: .3f}, {rpy_deg[2]: .3f}")
        print(f"piper_endpose: {piper.GetArmEndPoseMsgs()}")
        cnt += 1
        time.sleep(0.0046)  # ~216Hz

    piper.DisconnectPort()

if __name__ == "__main__":
    main()
