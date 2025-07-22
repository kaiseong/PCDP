# tracik_ik_controller.py

import numpy as np
import pinocchio as pin  # Pinocchio의 SE3 타입을 입력으로 받기 위해 필요
# 사용자님께서 설치에 성공하신 tracikpy 라이브러리를 사용합니다.
from tracikpy import TracIKSolver
from typing import Union

class TracIKController:
    """
    mjd3/tracikpy 라이브러리를 사용하여 역기구학을 계산하는 최종 컨트롤러.
    PinocchioIKController와 호환되는 인터페이스를 가집니다.
    """
    def __init__(self,
                 urdf_path: str,
                 ee_link_name: str,
                 base_link_name: str = "base_link",
                 solve_type: str = "Speed",
                 **kwargs): # 호환성을 위해 나머지 인자들을 받음
        """
        TracIKSolver를 초기화합니다.

        Args:
            urdf_path (str): 로봇의 URDF 파일 경로.
            ee_link_name (str): 제어할 End-Effector 링크의 이름.
            base_link_name (str): 로봇의 베이스 링크 이름.
            solve_type (str): TRAC-IK의 풀이 타입.
        """
        # TracIKSolver 클래스를 인스턴스화합니다.
        # 이 클래스는 __init__에서 파일 경로를 직접 받아 처리합니다.
        self.ik_solver = TracIKSolver(
            urdf_file=urdf_path,
            base_link=base_link_name,
            tip_link=ee_link_name,
            timeout=0.005,  # 200Hz에 맞춰 타임아웃 설정
            epsilon=1e-5,
            solve_type=solve_type,
        )
        
        print("--- tracikpy 기반의 최종 IK 컨트롤러 초기화 성공 ---")
        print(f"EE Link: {ee_link_name}, Solve Type: {solve_type}")
        print(f"관절 이름: {self.ik_solver.joint_names}")

    def calculate_ik(self,
                     target_pose: pin.SE3,
                     q_init: np.ndarray) -> Union[np.ndarray, None]:
        """
        주어진 목표 포즈에 대한 IK를 계산합니다.
        PinocchioIKController와 동일한 입력 형식을 받습니다.

        Args:
            target_pose (pin.SE3): Pinocchio의 SE3 객체로 표현된 목표 Pose.
            q_init (np.ndarray): IK 계산을 위한 초기 관절 각도 추정치 (seed_state).

        Returns:
            np.ndarray or None: 성공 시 목표 관절 각도(rad), 실패 시 None.
        """
        if isinstance(target_pose, pin.SE3):
            target_matrix = target_pose.homogeneous      # ← 핵심 수정
        else:
            target_matrix = np.asarray(target_pose)
        # -------------------------------------------------------------------

        result_joints = self.ik_solver.ik(
            ee_pose=target_matrix,
            qinit=q_init,
        )
        return result_joints