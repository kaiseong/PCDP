
import pinocchio as pin
import numpy as np
import os
from typing import Union

class PinocchioIKController:
    
    def __init__(self, 
                 urdf_path: str, 
                 mesh_dir: str,
                 ee_link_name: str,
                 joints_to_lock_names: list = []):
        
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF 파일을 찾을 수 없습니다: {urdf_path}")

        self.ee_link_name = ee_link_name
        self._initialize_model(urdf_path, mesh_dir, joints_to_lock_names)

        # IK 계산을 위한 파라미터
        self.DAMPING = 1e-4
        self.MAX_ITERATIONS = 100
        self.TOLERANCE = 1e-4

    def _initialize_model(self, urdf_path, mesh_dir, joints_to_lock_names):
        try:
            full_robot = pin.RobotWrapper.BuildFromURDF(urdf_path, [mesh_dir])
            full_model = full_robot.model
            
            joints_to_lock_ids = [full_model.getJointId(name) for name in joints_to_lock_names if full_model.existJoint(name)]
            
            self.model = pin.buildReducedModel(
                full_model, 
                joints_to_lock_ids, 
                pin.neutral(full_model)
            )
            self.data = self.model.createData()
            

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
        
        q = q_init.copy()
        
        for i in range(self.MAX_ITERATIONS):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pose = self.data.oMf[self.ee_frame_id]
            error_vec = pin.log6(current_pose.inverse() * target_pose).vector
            
            if np.linalg.norm(error_vec) < self.TOLERANCE:
                # Solution found, now check joint limits
                is_within_limits = np.all(q >= self.model.lowerPositionLimit) and np.all(q <= self.model.upperPositionLimit)
                if is_within_limits:
                    return q  # Success!
                else:
                    return None

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            A = J.T @ J + self.DAMPING * np.eye(self.model.nv)
            b = J.T @ error_vec
            delta_q = np.linalg.solve(A, b)
            
            q = pin.integrate(self.model, q, delta_q)
            
        return None # Convergence failed
