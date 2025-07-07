
import pinocchio as pin
import numpy as np
import os
from typing import Union

class PinocchioIKController:
    """
    Pinocchio를 사용하여 역기구학(Inverse Kinematics)을 계산하는 컨트롤러.
    """
    def __init__(self, 
                 urdf_path: str, 
                 mesh_dir: str,
                 ee_link_name: str,
                 joints_to_lock_names: list = []):
        """
        IK 컨트롤러를 초기화합니다.

        :param urdf_path: 로봇의 URDF 파일 경로.
        :param mesh_dir: 메쉬 파일이 포함된 디렉토리 경로.
        :param ee_link_name: 제어할 End-Effector 링크의 이름.
        :param joints_to_lock_names: 모델에서 제외할 'fixed' 또는 'gripper' 관절 이름 리스트.
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")

        self.ee_link_name = ee_link_name
        self._initialize_model(urdf_path, mesh_dir, joints_to_lock_names)

        # IK 계산을 위한 파라미터
        self.DAMPING = 1e-2
        self.MAX_ITERATIONS = 100
        self.TOLERANCE = 1e-4

    def _initialize_model(self, urdf_path, mesh_dir, joints_to_lock_names):
        """URDF에서 모델을 로드하고 축소 모델을 생성합니다."""
        try:
            # RobotWrapper를 사용하여 전체 모델 로드
            full_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
            full_model = full_robot.model
            
            # 지정된 관절을 잠궈 축소 모델 생성
            joints_to_lock_ids = [full_model.getJointId(name) for name in joints_to_lock_names if full_model.existJoint(name)]
            
            self.model = pin.buildReducedModel(
                full_model, 
                joints_to_lock_ids, 
                pin.neutral(full_model)
            )
            self.data = self.model.createData()
            
            print(f"Pinocchio IK 컨트롤러 초기화 성공. 축소 모델 관절 수: {self.model.nq}")
            # 관절 한계 출력 (디버깅용)
            print(f"  - Lower joint limits: {self.model.lowerPositionLimit}")
            print(f"  - Upper joint limits: {self.model.upperPositionLimit}")

        except Exception as e:
            print(f"Pinocchio 모델 초기화 실패: {e}")
            raise

        if not self.model.existFrame(self.ee_link_name):
            raise ValueError(f"EEF 링크 '{self.ee_link_name}'를 모델에서 찾을 수 없습니다.")
        self.ee_frame_id = self.model.getFrameId(self.ee_link_name)
        self.n_joints = self.model.nq




    def calculate_ik(self, 
                     target_pose: pin.SE3, 
                     q_init: np.ndarray) -> Union[np.ndarray, None]:
        """
        주어진 목표 포즈에 대한 IK를 계산합니다.

        :param target_pose: 목표 EEF 포즈 (pin.SE3 객체).
        :param q_init: IK 계산을 위한 초기 관절 각도 추정치.
        :return: 성공 시 목표 관절 각도(np.ndarray), 실패 시 None.
        """
        q = q_init.copy()
        
        for i in range(self.MAX_ITERATIONS):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pose = self.data.oMf[self.ee_frame_id]
            # Note: error is computed in the frame of the target pose
            error_vec = pin.log6(current_pose.inverse() * target_pose).vector
            
            if np.linalg.norm(error_vec) < self.TOLERANCE:
                # Solution found, now check joint limits
                is_within_limits = np.all(q >= self.model.lowerPositionLimit) and np.all(q <= self.model.upperPositionLimit)
                if is_within_limits:
                    return q  # Success!
                else:
                    # Converged to a solution that violates joint limits
                    return None

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            # Damped Least Squares
            # A*dq = b --> (J.T*J + D*I) dq = J.T * error
            A = J.T @ J + self.DAMPING * np.eye(self.model.nv)
            b = J.T @ error_vec
            delta_q = np.linalg.solve(A, b)
            
            q = pin.integrate(self.model, q, delta_q)
            
        return None # Convergence failed
