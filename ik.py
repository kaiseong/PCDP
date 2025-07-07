import numpy as np
import math
import time
from scipy.optimize import minimize
import warnings
from piper_sdk import *
import diffusion_policy.common.mono_time as mono_time

warnings.filterwarnings('ignore')

class PiperIK:
    def __init__(self):
        # DH Parameters [a, alpha, d, theta_offset]
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
    
    def dh_transform(self, a, alpha, d, theta):
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct,    -st*ca,  st*sa,   a*ct],
            [st,     ct*ca, -ct*sa,   a*st],
            [0,      sa,     ca,      d],
            [0,      0,      0,       1]
        ])
    
    def forward_kinematics(self, joint_angles):
        T = np.eye(4)
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params):
            theta = joint_angles[i] + theta_offset
            T = T @ self.dh_transform(a, alpha, d, theta)
        
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        return position, rotation_matrix
    
    def rotation_matrix_to_euler(self, R):
        sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
        
        if sy < 1e-6:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        else:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        
        return np.array([x, y, z])
    
    def objective_function(self, joint_angles, target_position, target_euler):
        try:
            current_pos, current_rot = self.forward_kinematics(joint_angles)
            current_euler = self.rotation_matrix_to_euler(current_rot)
            
            pos_error = np.linalg.norm(current_pos - target_position)
            euler_error = np.linalg.norm(current_euler - target_euler)
            
            return pos_error + 0.1 * euler_error
        except:
            return 1e6
    
    def solve_ik_original(self, target_pose, max_attempts=5):
        """Original method (slower but very accurate)"""
        target_pos = np.array(target_pose[:3])
        target_euler = np.array(target_pose[3:])
        
        best_solution = None
        best_error = float('inf')
        
        # Fixed initial guesses for consistent results
        initial_guesses = [
            [0, 1.57, -1.0, 0, 0, 0],      # Standard position
            [0, 1.0, -1.5, 0, 0, 0],       # Alternative 1
            [0.5, 1.2, -1.2, 0, 0, 0],     # Alternative 2
            [-0.5, 1.8, -0.8, 0, 0, 0],    # Alternative 3
            [0, 2.0, -2.0, 0, 0, 0],       # Alternative 4
        ]
        
        for i in range(max_attempts):
            if i < len(initial_guesses):
                init_angles = initial_guesses[i]
            else:
                # Fallback to random if needed
                init_angles = []
                for min_val, max_val in self.joint_limits:
                    angle = np.random.uniform(min_val + 0.1, max_val - 0.1)
                    init_angles.append(angle)
            
            try:
                result = minimize(
                    self.objective_function,
                    init_angles,
                    args=(target_pos, target_euler),
                    method='SLSQP',
                    bounds=self.joint_limits,
                    options={'ftol': 1e-6, 'maxiter': 1000}
                )
                
                if result.success and result.fun < best_error:
                    # Check joint limits
                    if all(self.joint_limits[i][0] <= result.x[i] <= self.joint_limits[i][1] 
                           for i in range(6)):
                        best_solution = result.x
                        best_error = result.fun
                        
                        if best_error < 0.005:
                            break
            except:
                continue
        
        return best_solution if best_solution is not None else None
    
    def solve_ik_fast_newton(self, target_pose, max_iter=3):
        """Fast Newton-Raphson method (faster, good accuracy)"""
        target_pos = np.array(target_pose[:3])
        target_euler = np.array(target_pose[3:])
        
        # Good initial guess
        q = np.array([0, 1.57, -1.0, 0, 0, 0])
        
        for _ in range(max_iter):
            # Forward kinematics
            pos, rot = self.forward_kinematics(q)
            euler = self.rotation_matrix_to_euler(rot)
            
            # Error
            pos_error = target_pos - pos
            euler_error = target_euler - euler
            error = np.concatenate([pos_error, euler_error])
            
            if np.linalg.norm(error) < 0.001:
                break
            
            # Simple Jacobian approximation
            J = self._compute_jacobian_approx(q)
            
            # Newton step
            try:
                dq = 0.1 * np.linalg.pinv(J) @ error
                q += dq
                
                # Clamp to joint limits
                for i, (min_val, max_val) in enumerate(self.joint_limits):
                    q[i] = np.clip(q[i], min_val, max_val)
            except:
                break
        
        return q
    
    def _compute_jacobian_approx(self, q, delta=0.001):
        """Fast approximate Jacobian computation"""
        J = np.zeros((6, 6))
        
        pos0, rot0 = self.forward_kinematics(q)
        euler0 = self.rotation_matrix_to_euler(rot0)
        endpose0 = np.concatenate([pos0, euler0])
        
        for i in range(6):
            q_plus = q.copy()
            q_plus[i] += delta
            
            pos_plus, rot_plus = self.forward_kinematics(q_plus)
            euler_plus = self.rotation_matrix_to_euler(rot_plus)
            endpose_plus = np.concatenate([pos_plus, euler_plus])
            
            J[:, i] = (endpose_plus - endpose0) / delta
        
        return J

# Global solver instance
_ik_solver = None

def solve_ik(endpose, method='fast'):
    """
    6DOF IK solver with performance timing
    
    Args:
        endpose: [x, y, z, rx, ry, rz] in meters and radians
        method: 'fast' for Fast Newton (recommended), 'original' for scipy.optimize
        
    Returns:
        tuple: (joint_angles, computation_time_ms) 
               joint_angles: [Î¸1, Î¸2, Î¸3, Î¸4, Î¸5, Î¸6] in radians, or None if failed
               computation_time_ms: float, computation time in milliseconds
    """
    start_time = time.perf_counter()
    
    global _ik_solver
    if _ik_solver is None:
        _ik_solver = PiperIK()
    
    if len(endpose) != 6:
        end_time = time.perf_counter()
        computation_time_ms = (end_time - start_time) * 1000
        raise ValueError("endpose must be [x, y, z, rx, ry, rz]")
    
    # Set fixed seed for consistent results
    np.random.seed(42)
    
    if method == 'fast':
        result = _ik_solver.solve_ik_fast_newton(endpose)
    elif method == 'original':
        result = _ik_solver.solve_ik_original(endpose)
    else:
        raise ValueError("method must be 'fast' or 'original'")
    
    np.random.seed(None)  # Reset seed
    
    end_time = time.perf_counter()
    computation_time_ms = (end_time - start_time) * 1000
    
    return result, computation_time_ms

def solve_ik_batch(endposes, method='fast'):
    """
    Batch IK solver with timing
    
    Args:
        endposes: list of [x, y, z, rx, ry, rz]
        method: 'fast' or 'original'
        
    Returns:
        tuple: (results, total_time_ms, average_time_ms)
               results: list of joint_angles or None for failed cases
               total_time_ms: total computation time in milliseconds
               average_time_ms: average time per IK solution
    """
    start_time = time.perf_counter()
    results = []
    individual_times = []
    
    for endpose in endposes:
        joint_angles, comp_time = solve_ik(endpose, method)
        results.append(joint_angles)
        individual_times.append(comp_time)
    
    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    average_time_ms = np.mean(individual_times) if individual_times else 0
    
    return results, total_time_ms, average_time_ms



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

# Test function
if __name__ == "__main__":
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    factor = 57295.7795

    start_endpose = [0.042, -0.433, -0.024, 3.142, 0.500, -0.571]
    end_endpose = [-0.012829, -0.028897, 0.472968, -1.0345, 1.4931, -1.1319]
    
    

    while True:
        piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
        joint_angles, computation_time = solve_ik(end_endpose, method='original')
        print(f"consume time: {computation_time}")
        joints = np.asarray(joint_angles, np.float32)
        joints = (joints * 1000).astype(np.int32)
        print(f"joints: {joints}")
        print(f"encoder_joints: {piper.GetArmJointMsgs()}")
        print(f"encoder_end_pose: {piper.GetArmEndPoseMsgs()}")
        piper.JointCtrl(joints)
        time.sleep(0.005)

    if joint_angles is not None:
        print(f"Target pose: {test_endpose}")
        print(f"Target angle: {np.round(joint_angles, 4).tolist()}")
        print(f"Time: {computation_time:.3f} ms")
    else:
        print(f"Target pose: {test_endpose}")
        print(f"Target angle: Failed")
        print(f"Time: {computation_time:.3f} ms")