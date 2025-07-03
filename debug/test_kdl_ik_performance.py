import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import time
import statistics
import os

def main():
    # 1. URDF 로드
    urdf_path ="/home/nscl/diffusion_policy/debug/piper_no_gripper_description.urdf"
    mesh_dir = "/home/nscl/diffusion_policy"

    try:
        robot = RobotWrapper.BuildFromURDF(urdf_path,[mesh_dir])
        model = robot.model
        data = robot.data
        print("Pinocchio RobotWrapper 로드 성공!")
    except Exception as e:
        print(f"Pinocchio RobotWrapper 로드 실패: {e}")
        return

    # 2. IK를 위한 설정
    ee_link_name = "link6"
    if not model.existFrame(ee_link_name):
        print(f"EEF 링크 '{ee_link_name}'를 찾을 수 없습니다.")
        return
    EE_FRAME_ID = model.getFrameId(ee_link_name)

    target_pos = np.array([0.3, 0.1, 0.2])
    target_rot = np.identity(3)
    TARGET_POSE = pin.SE3(target_rot, target_pos)

    # 3. 벤치마킹 실행

    # [최종 수정!] 6-DOF 모델을 위한 좋은 초기 추정치를 명시적으로 설정합니다.
    if model.nq == 6:
        q_init = np.array([0.0, 0.5, -1.0, 0.0, 1.0, 0.0])
    else:
        print(f"경고: 로봇 관절 수가 {model.nq}입니다. q_init을 직접 조정해야 합니다.")
        q_init = pin.neutral(model)

    durations = []
    print(f"로봇 관절 수: {model.nq}")
    print(f"사용된 초기 추정치 q_init: {q_init.flatten()}")
    print("\n벤치마킹 시작 (100회 반복)...")

    # --- Pinocchio 표준 IK 알고리즘 ---
    DAMPING = 1e-4
    MAX_ITERATIONS = 100
    TOLERANCE = 1e-4

    for n_trial in range(100):
        start_time = time.perf_counter()
        q = q_init.copy()
        success = False

        verbose = (n_trial == 0)
        if verbose: print("\n--- 첫 번째 IK 시도 과정 ---")

        for i in range(MAX_ITERATIONS):
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)

            error_vec =pin.log6(TARGET_POSE.actInv(data.oMf[EE_FRAME_ID])).vector
            error_norm = np.linalg.norm(error_vec)

            # 자코비안 계산 및 상태 확인
            J = pin.computeFrameJacobian(model, data, q,EE_FRAME_ID, pin.ReferenceFrame.LOCAL)
            cond_J = np.linalg.cond(J)
            if verbose: print(f"  반복 {i:2d}: 오차 크기 = {error_norm:.6f}, 자코비안 조건수 = {cond_J:.2f}")

            if error_norm < TOLERANCE:
                success = True
                break

            delta_q = np.linalg.solve(J.T @ J + DAMPING *np.eye(J.shape[1]), J.T @ error_vec)
            q = pin.integrate(model, q, -delta_q)

        end_time = time.perf_counter()
        if success and n_trial >= 20:
            durations.append((end_time - start_time) * 1000)
            q_init = q
        elif verbose:
            print("--- 첫 번째 IK 시도 실패 ---")

    # 4. 결과 출력
    if not durations:
        print("\nIK 해를 한 번도 찾지 못했습니다.")
        return

    print("\n--- Pinocchio IK Performance (ms) ---")
    print(f"  성공률: {len(durations)}/100")
    print(f"  평균: {statistics.mean(durations):.6f}")
    print(f"  최대: {max(durations):.6f}")
    print(f"  최소: {min(durations):.6f}")

if __name__ == '__main__':
    main()

