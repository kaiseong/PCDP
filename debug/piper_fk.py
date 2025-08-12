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
EEF_FRAME_NAME = "link6"      # URDF의 EEF 프레임 이름
BASE_TO_ROBOT = np.eye(4)     # Piper EEF가 기준 'base'이고, RobotWrapper는 'robot base'일 때 관계가 다르면 수정
# ==============================

def euler_xyz_to_R(rx, ry, rz):
    """XYZ(intrinsic) 순서. 필요시 네 convention으로 바꿔도 됨."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=np.float64)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=np.float64)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=np.float64)
    return Rz @ Ry @ Rx

def piper_eef_to_SE3_base(eef_mm_deg):
    """eef: [x(mm), y(mm), z(mm), roll(deg), pitch(deg), yaw(deg)] → SE3(베이스 기준)"""
    x_mm, y_mm, z_mm, r_deg, p_deg, y_deg = eef_mm_deg[:6]
    t = np.array([x_mm, y_mm, z_mm], dtype=np.float64) / 1000.0
    R = euler_xyz_to_R(np.deg2rad(r_deg), np.deg2rad(p_deg), np.deg2rad(y_deg))
    return pin.SE3(R, t)

def baseSE3_to_robotSE3(M_base):
    """기준 베이스→로봇 베이스로 변환(필요시). 기본은 항등."""
    T_br = np.linalg.inv(BASE_TO_ROBOT)  # robot ← base
    return pin.SE3(T_br[:3,:3], T_br[:3,3]) * M_base

def pose_errors(M_meas: pin.SE3, M_fk: pin.SE3):
    """오차: 위치(m), 자세(rad)"""
    dM = M_meas.inverse() * M_fk
    trans_err = np.linalg.norm(dM.translation)           # ✅ 위치 오차
    rot_vec   = pin.log3(dM.rotation)                    # ✅ 3D 회전벡터 (numpy array)
    rot_err   = np.linalg.norm(rot_vec)   
    return trans_err, rot_err

def get_q_from_joints_msg(jm):
    """
    너가 말한 출력이 [-3.434, 0.0, 0.0, -10.001, 18.181, 2.148] (degree)이므로,
    리스트를 바로 numpy로 받아서 rad로 변환.
    """
    if isinstance(jm, (list, tuple, np.ndarray)):
        q_deg = np.array(jm, dtype=np.float64)
        assert q_deg.size >= 6
        q = np.deg2rad(q_deg[:6])
        return q
    # 객체 형태로 온다면 필드명을 여기에 맞춰서 꺼내자(예시)
    try:
        q_deg = np.array([
            jm.motor_1.joint_pos, jm.motor_2.joint_pos, jm.motor_3.joint_pos,
            jm.motor_4.joint_pos, jm.motor_5.joint_pos, jm.motor_6.joint_pos
        ], dtype=np.float64)
        return np.deg2rad(q_deg)
    except Exception as e:
        raise RuntimeError(f"조인트 메시지 파싱 실패: {e}")

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
    
    cnt=0
    pos_error =np.array([])
    ori_error =np.array([])
    # 현재 값 읽기
    while cnt<100:
        t1=mono_time.now_ms()
        joints = piper.GetArmJointMsgs()   # 예: [-3.434, 0.0, 0.0, -10.001, 18.181, 2.148]
        # print("Piper joints (deg):", joints)
        q = get_q_from_joints_msg(joints)  # → rad

        # Piper가 주는 EEF(있다면) 읽기
        # 만약 네 SDK에서 EEF 메시지가 없다면, 아래 3줄은 주석 처리하고 FK 결과만 출력해도 됨.
        eef_msg = piper.GetArmEndPoseMsgs()  # [mm, mm, mm, deg, deg, deg]
        M_base_eef_meas  = piper_eef_to_SE3_base(eef_msg)
        M_robot_eef_meas = baseSE3_to_robotSE3(M_base_eef_meas)

        # FK 계산 (로봇 베이스 기준)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        M_robot_eef_fk = data.oMf[fid]
        print(M_robot_eef_fk)

        # 결과 출력
        # print("\n=== FK 검증 ===")
        # print("q (rad):", q)
        # print("\nPiper EEF (robot base):")
        # print("t:", M_robot_eef_meas.translation)
        # print("R:\n", M_robot_eef_meas.rotation)

        # print("\nFK EEF (robot base):")
        # print("t:", M_robot_eef_fk.translation)
        # print("R:\n", M_robot_eef_fk.rotation)

        trans_err_m, rot_err_rad = pose_errors(M_robot_eef_meas, M_robot_eef_fk)
        # print("\nErrors:")
        pos_error = np.append(pos_error, trans_err_m*1000)
        ori_error = np.append(ori_error, np.rad2deg(rot_err_rad))
        cnt+=1
        time.sleep(0.0046)
    print(f"pos_error mean: {pos_error.mean()}mm")
    print(f"ori_error mean: {ori_error.mean()}deg")

    piper.DisconnectPort()

if __name__=="__main__":
    main()
