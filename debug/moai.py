#!/usr/bin/env python3
import numpy as np
import time
from typing import Optional, Tuple, List
from piper_sdk import C_PiperInterface_V2
from piper_sdk.kinematics.piper_fk import C_PiperForwardKinematics

class PiperCartesianController:
    """IK solver for PiPER robot cartesian control"""
    
    def __init__(self):
        # Official FK solver (with offset matching URDF)
        self.fk_solver = C_PiperForwardKinematics(dh_is_offset=0x01)
        
        # Joint limits [min, max] in radians
        self.joint_limits = [
            [-2.618, 2.618],  # Joint 1: Â±150Â°
            [0, 3.14],        # Joint 2: 0~180Â°
            [-2.697, 0],      # Joint 3: -154Â°~0Â°
            [-1.832, 1.832],  # Joint 4: Â±105Â°
            [-1.22, 1.22],    # Joint 5: Â±70Â°
            [-3.14, 3.14]     # Joint 6: Â±180Â°
        ]
        
        self.n_joints = 6
    
    def calculate_forward_kinematics(self, joint_angles_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate forward kinematics using official FK"""
        # FK calculation (radians input)
        fk_result = self.fk_solver.CalFK(joint_angles_rad.tolist())
        
        # Extract end-effector pose
        end_effector = fk_result[-1]  # [x, y, z, rx, ry, rz]
        
        # Position: mm to m conversion
        position_m = np.array([
            end_effector[0] / 1000.0,  # x
            end_effector[1] / 1000.0,  # y
            end_effector[2] / 1000.0   # z
        ])
        
        # Rotation: degrees to radians conversion
        rotation_rad = np.array([
            end_effector[3] * np.pi / 180.0,  # rx
            end_effector[4] * np.pi / 180.0,  # ry
            end_effector[5] * np.pi / 180.0   # rz
        ])
        
        return position_m, rotation_rad
    
    def compute_numerical_jacobian_fast(self, q: np.ndarray, pos0: np.ndarray, euler0: np.ndarray, delta: float = 0.005) -> np.ndarray:
        """Fast numerical Jacobian with larger delta and caching"""
        J = np.zeros((6, self.n_joints))
        pose0 = np.concatenate([pos0, euler0])
        
        for i in range(self.n_joints):
            q_plus = q.copy()
            q_plus[i] = min(q[i] + delta, self.joint_limits[i][1])
            
            if q_plus[i] != q[i]:
                pos_plus, euler_plus = self.calculate_forward_kinematics(q_plus)
                pose_plus = np.concatenate([pos_plus, euler_plus])
                J[:, i] = (pose_plus - pose0) / (q_plus[i] - q[i])
        
        return J
    
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def solve_inverse_kinematics(self, 
                                target_position_m: List[float], 
                                target_rotation_rad: List[float],
                                initial_joints_rad: Optional[List[float]] = None,
                                position_tolerance_m: float = 2e-3,
                                max_iterations: int = 30) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Fast IK solver using Broyden's method (quasi-Newton)
        """
        start_time = time.perf_counter()
        
        # Target setup
        target_pos = np.array(target_position_m)
        target_euler = np.array(target_rotation_rad)
        
        # Initial guess
        if initial_joints_rad is not None:
            q = np.array(initial_joints_rad)
        else:
            q = np.array([0, 1.57, -1.0, 0, 0, 0])
        
        # First FK call
        pos, euler = self.calculate_forward_kinematics(q)
        
        # Initial Jacobian (compute only once)
        J = self.compute_numerical_jacobian_fast(q, pos, euler)
        
        # Broyden's method parameters
        damping = 5e-2 # ori: 5e-2
        
        
        for iteration in range(max_iterations):
            # Error calculation
            pos_error = target_pos - pos
            euler_error = np.array([
                self.normalize_angle(target_euler[i] - euler[i]) 
                for i in range(3)
            ])
            error = np.concatenate([pos_error, euler_error * 0.3])  # Less weight on rotation
            position_error_norm = np.linalg.norm(pos_error)
            
            if position_error_norm < position_tolerance_m:
                computation_time_ms = (time.perf_counter() - start_time) * 1000
                return q, computation_time_ms, True
            
            # Solve for step
            try:
                # Fast solve using current Jacobian estimate
                dq = np.linalg.lstsq(J + damping * np.eye(6), error, rcond=None)[0]
            except:
                # Fallback to simple gradient
                dq = 0.1 * J.T @ error
            
            # Line search for optimal step
            alpha = 1.0
            q_new = q + alpha * dq
            
            # Apply joint limits
            for i in range(self.n_joints):
                q_new[i] = np.clip(q_new[i], 
                                  self.joint_limits[i][0] + 0.01,
                                  self.joint_limits[i][1] - 0.01)
            
            # New FK
            pos_new, euler_new = self.calculate_forward_kinematics(q_new)
            
            # Broyden update for Jacobian (avoid recomputing)
            if iteration < max_iterations - 1:  # Don't update on last iteration
                dq_actual = q_new - q
                pose_old = np.concatenate([pos, euler])
                pose_new = np.concatenate([pos_new, euler_new])
                dy = pose_new - pose_old - J @ dq_actual
                
                if np.linalg.norm(dq_actual) > 1e-6:
                    # Broyden rank-1 update
                    J += np.outer(dy, dq_actual) / (np.linalg.norm(dq_actual)**2)
            
            q = q_new
            pos = pos_new
            euler = euler_new
        
        computation_time_ms = (time.perf_counter() - start_time) * 1000
        return q, computation_time_ms, False

def main():
    """Main test function"""
    # Create controller
    controller = PiperCartesianController()
    robot = PiperRobotInterface()
    
    # Connect robot
    if not robot.connect():
        return
    
    # Read current state
    current_millideg, current_rad = robot.get_current_joint_positions()
    
    # MODIFY THIS: Set your target position here
    target_endpose = [
        0.05,    # X: 50mm
        -0.02,   # Y: -20mm  
        0.35,    # Z: 350mm
        3.14159, # RX: 180 degrees in radians
        1.3439,  # RY: 77 degrees in radians
        -3.0543  # RZ: -175 degrees in radians
    ]
    
    print(f"Target: X={target_endpose[0]*1000:.0f}mm, Y={target_endpose[1]*1000:.0f}mm, Z={target_endpose[2]*1000:.0f}mm")
    
    # Calculate IK
    result_joints, computation_time, converged = controller.solve_inverse_kinematics(
        target_endpose[:3],  # position
        target_endpose[3:],  # rotation
        current_rad,
        position_tolerance_m=0.002
    )
    
    if converged:
        print(f"IK time: {computation_time:.1f}ms")
        
        # Execute movement
        robot.move_to_joint_positions(result_joints)
        
        # Check actual position after 3 seconds
        time.sleep(3)
        new_millideg, new_rad = robot.get_current_joint_positions()
        print(f"Final joints (deg): {[round(a/1000, 1) for a in new_millideg]}")
        
        new_pos, new_euler = controller.calculate_forward_kinematics(np.array(new_rad))
        print(f"Actual: X={new_pos[0]*1000:.0f}mm, Y={new_pos[1]*1000:.0f}mm, Z={new_pos[2]*1000:.0f}mm")
    else:
        print("IK FAILED")

if __name__ == "__main__":
    main()