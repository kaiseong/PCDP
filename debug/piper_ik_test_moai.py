import numpy as np
import time
from typing import Optional, Tuple, List
from piper_sdk import C_PiperInterface_V2

class PiperFastIK:
    """Fast inverse kinematics solver for Piper robot using Levenberg-Marquardt"""
    
    def __init__(self):
        # DH Parameters [a, alpha, d, theta_offset] in meters
        self.dh_params = [
            [0,       0,       0.123,   0],
            [0,       np.pi/2, 0,       -np.pi/2],
            [0.28503, 0,       0,       0],
            [0.022,   np.pi/2, 0.25075, 0],
            [0,       -np.pi/2,0,       0],
            [0,       np.pi/2, 0.091,   0]
        ]
        
        # Joint limits [min, max] in radians
        self.joint_limits = [
            [-2.618, 2.618],
            [0, 3.14],
            [-2.697, 0],
            [-1.832, 1.832],
            [-1.22, 1.22],
            [-3.14, 3.14]
        ]
        
        self.n_joints = 6
    
    def dh_transform(self, a: float, alpha: float, d: float, theta: float) -> np.ndarray:
        """Compute DH transformation matrix"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d],
            [0,   0,      0,     1]
        ])
    
    def forward_kinematics(self, joint_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics"""
        T = np.eye(4)
        for i in range(self.n_joints):
            a, alpha, d, theta_offset = self.dh_params[i]
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(a, alpha, d, theta)
        
        position = T[:3, 3]
        rotation = T[:3, :3]
        return position, rotation
    
    def rotation_to_euler(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to Euler angles"""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        
        if sy < 1e-6:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        else:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        
        return np.array([x, y, z])
    
    def compute_jacobian(self, q: np.ndarray, delta: float = 0.001) -> np.ndarray:
        """Numerical Jacobian using finite differences"""
        J = np.zeros((6, self.n_joints))
        
        # Current pose
        pos0, rot0 = self.forward_kinematics(q)
        euler0 = self.rotation_to_euler(rot0)
        
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] = min(q[i] + delta, self.joint_limits[i][1])
            
            pos_plus, rot_plus = self.forward_kinematics(q_plus)
            euler_plus = self.rotation_to_euler(rot_plus)
            
            if q_plus[i] != q[i]:
                J[:3, i] = (pos_plus - pos0) / (q_plus[i] - q[i])
                J[3:, i] = (euler_plus - euler0) / (q_plus[i] - q[i])
        
        return J
    
    def wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def solve_ik(self, target_pose: List[float], 
                 initial_joints: Optional[List[float]] = None) -> Tuple[Optional[np.ndarray], float]:
        """
        Solve inverse kinematics using Levenberg-Marquardt method
        
        Args:
            target_pose: [x, y, z, rx, ry, rz] target pose in meters and radians
            initial_joints: Initial joint angles for faster convergence (optional)
            
        Returns:
            (joint_angles, computation_time_ms) or (None, computation_time_ms) if failed
        """
        start_time = time.perf_counter()
        
        # Parse target
        target_pos = np.array(target_pose[:3])
        target_euler = np.array(target_pose[3:])
        
        # Initial guess
        if initial_joints is not None:
            q = np.array(initial_joints)
        else:
            q = np.array([0, 1.57, -1.0, 0, 0, 0])  # Default initial guess
        
        # Levenberg-Marquardt parameters
        damping = 0.01
        max_iter = 10
        
        for _ in range(max_iter):
            # Current pose
            pos, rot = self.forward_kinematics(q)
            euler = self.rotation_to_euler(rot)
            
            # Error
            pos_error = target_pos - pos
            euler_error = np.array([self.wrap_angle(target_euler[i] - euler[i]) for i in range(3)])
            error = np.concatenate([pos_error, euler_error * 0.5])
            
            if np.linalg.norm(error) < 0.005:   # 5mm(threshold)
                end_time = time.perf_counter()
                computation_time_ms = (end_time - start_time) * 1000
                return q, computation_time_ms
            
            # Jacobian and update
            J = self.compute_jacobian(q)
            JtJ = J.T @ J
            Jte = J.T @ error
            
            try:
                H = JtJ + damping * np.eye(self.n_joints)
                dq = np.linalg.solve(H, Jte)
                
                # Update with limits
                q_new = q + 0.5 * dq
                for i in range(self.n_joints):
                    q_new[i] = np.clip(q_new[i], 
                                      self.joint_limits[i][0], 
                                      self.joint_limits[i][1])
                
                # Check improvement
                pos_new, rot_new = self.forward_kinematics(q_new)
                euler_new = self.rotation_to_euler(rot_new)
                error_new = np.concatenate([target_pos - pos_new, 
                                          np.array([self.wrap_angle(target_euler[i] - euler_new[i]) 
                                                   for i in range(3)]) * 0.5])
                
                if np.linalg.norm(error_new) < np.linalg.norm(error):
                    q = q_new
                    damping *= 0.5
                    damping = max(damping, 1e-6)
                else:
                    damping *= 2.0
                    damping = min(damping, 1e-1)
            except:
                break
        
        end_time = time.perf_counter()
        computation_time_ms = (end_time - start_time) * 1000
        
        # Return best attempt even if not perfect
        return q, computation_time_ms

def connect_robot():
    """Connect and initialize robot"""
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    print("Robot connected successfully")
    return piper

def get_current_joints(piper):
    """Read current joint angles (millidegrees -> radians)"""
    joint_msgs = piper.GetArmJointMsgs()
    # Extract joint values from joint_state (millidegrees)
    joints_millideg = np.asarray(joint_msgs) * 1000
    # Convert millidegrees to radians
    joints_rad = joints_millideg / 57295.7795
    return joints_millideg, joints_rad

def move_to_joints_ctrl(piper, joint_angles_rad):
    """Move robot to calculated joint angles"""
    # Convert radians to millidegrees (integer)
    factor = 57295.7795
    joints_millideg = [int(round(angle * factor)) for angle in joint_angles_rad]
    
    print(f"Target joints (millideg): {joints_millideg}")
    
    # Set motion control
    piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
    
    # Control joints (6 axes)
    piper.JointCtrl(joints_millideg[0], joints_millideg[1], joints_millideg[2],
                    joints_millideg[3], joints_millideg[4], joints_millideg[5])
    
    return joints_millideg

def move_to_endpose():
    """Move to target end-effector pose"""
    target_endpose = [-0.3468, -0.1866, -0.0328, 2.8362, 0.3947, -2.1769]

    # Create IK solver
    ik_solver = PiperFastIK()
    
    # Connect robot
    piper = connect_robot()
    
    # Read current joints
    current_millideg, current_rad = get_current_joints(piper)
    print(f"Current joints (millideg): {current_millideg}")
    
    # Calculate IK
    print(f"Target endpose: {target_endpose}")
    result_joints_rad, computation_time = ik_solver.solve_ik(target_endpose, initial_joints=current_rad)
    
    if result_joints_rad is not None:
        result_millideg = [int(round(angle * 57295.7795)) for angle in result_joints_rad]
        print(f"Calculated joints (millideg): {result_millideg}")
        print(f"Computation time: {computation_time:.1f} ms")
        
        move_to_joints_ctrl(piper, result_joints_rad)
        print("Movement completed")
    else:
        print("IK failed")

if __name__ == "__main__":
    # Example: [x, y, z, rx, ry, rz] in meters and radians
    move_to_endpose()