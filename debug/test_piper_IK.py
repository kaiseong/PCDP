import ikpy.chain
import numpy as np
import ikpy.utils.plot as plot_utils
import matplotlib.pyplot
from piper_sdk import *
import time
# from ikpy.utils.geometry import rpy_matrix
from pytracik.trac_ik import TracIK
import diffusion_policy.common.mono_time as mono_time

BASE_LINK = "base_link"
TIP_LINK = "link6"

try:
    from scipy.spatial.transform import Rotation, Slerp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

def enable_fun(piper:C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)
# --- 핵심 기능 함수 ---
def get_ik_solution(urdf_file_path, target_position, target_orientation_matrix, initial_q=None):
    """
    Piper 로봇의 IK를 계산하여 6개의 활성 관절 값을 반환합니다.

    Args:
        urdf_file_path (str): 로봇의 URDF 파일 경로.
        target_position (list or np.array): 목표 위치 [x, y, z] (미터).
        target_orientation_matrix (np.array): 3x3 목표 방향 회전 행렬.
        initial_q (list or np.array, optional): 계산을 시작할 초기 관절 각도(rad).
                                                지정하지 않으면 0에서 시작합니다.

    Returns:
        np.array: 성공 시, 6개 관절 각도(rad)가 담긴 Numpy 배열.
        None: IK 해를 찾지 못한 경우.
    """
    # 그리퍼를 제외한 6-DOF 암 관절만 활성화합니다.
    active_links_mask = [False, True, True, True, True, True, True, False, False]
    
    try:
        # URDF로부터 로봇 체인 생성
        piper_chain = ikpy.chain.Chain.from_urdf_file(
            urdf_file_path,
            active_links_mask=active_links_mask
        )
    except FileNotFoundError:
        print(f"오류: '{urdf_file_path}' 파일을 찾을 수 없습니다.")
        return None
    except Exception as e:
        print(f"URDF 파일을 파싱하는 중 오류 발생: {e}")
        print("URDF 파일의 XML 문법이 올바른지 확인해주세요.")
        return None

    # 초기 관절 각도 설정
    if initial_q is None:
        # ikpy는 전체 링크 수(9개)에 맞는 벡터를 기대합니다.
        initial_q_full = np.zeros(len(piper_chain.links))
    else:
        # 사용자가 6개 관절 값만 제공하면 전체 벡터로 확장
        initial_q_full = np.zeros(len(piper_chain.links))
        initial_q_full[active_links_mask] = initial_q

    try:
        # 역기구학 계산
        ik_solution_rad_full = piper_chain.inverse_kinematics(
            target_position=target_position,
            target_orientation=target_orientation_matrix,
            orientation_mode='all',
            initial_position=initial_q_full
        )
        
        # 활성화된 6개 관절의 값만 추출하여 반환
        active_joint_values = ik_solution_rad_full[active_links_mask]
        return active_joint_values

    except Exception as e:
        # ikpy가 해를 찾지 못했을 때 발생하는 예외 처리
        print(f"IK 해를 찾지 못했습니다: {e}")
        return None

def _interp_orientations(rpy_start, rpy_end, n):
    """n 개 구간의 회전행렬 배열 반환"""
    if _HAS_SCIPY:
        # R (rad) → quaternions → SLERP
        key_rots = Rotation.from_euler("xyz", [rpy_start, rpy_end])
        slerp    = Slerp([0, 1], key_rots)
        return slerp(np.linspace(0, 1, n)).as_matrix()
    else:
        # RPY 각도를 선형보간(간단)
        rpy_path = np.vstack([np.linspace(s, e, n) for s, e in zip(rpy_start, rpy_end)]).T
        return np.array([rpy_matrix(*r) for r in rpy_path])  # ikpy.utils.geometry.rpy_matrix
# -------------------------------------------------
def linear_cartesian_move(piper, urdf_file,
                          pos_start, pos_end,
                          rpy_start, rpy_end,
                          n_points=100, pause=0.05,
                          q_initial=None, speed_pct=10):
    """
    위치·자세(roll pitch yaw [rad]) 모두 선형/SLERP 보간하며 이동
    """
    # ── 1. 궤적 생성 ─────────────────────
    pos_path = np.stack([np.linspace(s, e, n_points)
                         for s, e in zip(pos_start, pos_end)], axis=1)
    ori_path = _interp_orientations(rpy_start, rpy_end, n_points)

    # ── 2. 로봇 MOVE J 속도·모드 세팅(한 번) ──
    piper.MotionCtrl_2(ctrl_mode=0x01, move_mode=0x01,
                       move_spd_rate_ctrl=speed_pct, is_mit_mode=0x00)
    times = np.array([])
    # ── 3. 루프 전송 ─────────────────────
    for idx, (p_xyz, R) in enumerate(zip(pos_path, ori_path)):
        t_start = mono_time.now_ms()
        ik_rad = get_ik_solution(urdf_file, p_xyz, R, initial_q=q_initial)
        if ik_rad is None:
            print(f"[WARN] idx {idx}: IK 실패, 스킵")
            continue
        cmd_mdeg = (np.degrees(ik_rad) * 1000).astype(int)   # 0.001°
        piper.JointCtrl(cmd_mdeg)
        q_initial = ik_rad      # 다음 점의 초기값으로
        times = np.append(times, mono_time.now_ms()-t_start)
        time.sleep(pause)
    print(f"mean: {times.mean()}\
            max: {times.max()}\
            min: {times.min()}")


# --- 사용 예시 ---
if __name__ == '__main__':
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    factor = 57295.7795
    
    # 1. URDF 파일 경로 설정
    URDF_FILE = "/home/moai/diffusion_policy/debug/piper_description.urdf" 

    

    # 2. 목표 자세 설정
    target_pos = [0.3, 0.0, 0.6]
    roll_deg = 0
    pitch_deg = 90
    yaw_deg = -90

    # 도(degree)를 라디안(radian)으로 변환
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)

    # rpy_matrix 함수를 사용하여 3x3 회전 행렬 생성
    target_ori = rpy_matrix(roll_rad, pitch_rad, yaw_rad)

    # 위치(m)
    start_pos = [0.3, 0.1, 0.3]
    end_pos   = [0.2, 0.0, 0.6]

    # 시작/끝 RPY (°) → rad
    start_rpy_deg = [  60,  90, 0]
    end_rpy_deg   = [ 0,  90, -90]    # 예: roll 0→30°, pitch·yaw 고정
    rpy_s = np.radians(start_rpy_deg)
    rpy_e = np.radians(end_rpy_deg)

    print("직선+자세 보간 궤적 실행…")
    linear_cartesian_move(piper,
                          URDF_FILE,
                          start_pos, end_pos,
                          rpy_s, rpy_e,
                          n_points=120,      # 총 6 초(≈ 20 Hz)
                          pause=0.01,
                          speed_pct=50)      # MOVE J 속도 15 %
    print("궤적 완료")
