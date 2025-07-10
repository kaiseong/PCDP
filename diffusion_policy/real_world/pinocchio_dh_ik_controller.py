
import pinocchio as pin
import numpy as np

import pinocchio as pin
import numpy as np

# (PinocchioDhIkController 클래스 정의는 그대로 둡니다)

def build_model_from_dh():
    """
    Builds a Pinocchio model from Denavit-Hartenberg parameters.
    This version implements the Standard DH convention correctly.
    The transform from frame i-1 to i is: T_i = Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
    """
    model = pin.Model()

    # DH parameters from piper_fk.py (dh_is_offset = 0x01), units converted to meters
    a = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
    alpha = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
    d = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0
    theta_offset = np.array([0.0, -np.pi * 172.22 / 180, -102.78 / 180 * np.pi, 0.0, 0.0, 0.0])

    # Initialize the current transformation as identity (representing the world frame)
    current_transform = pin.SE3.Identity()

    for i in range(6):
        # The placement of the joint i's frame is defined relative to the previous frame (i-1).
        # We first apply the fixed transformations from the previous link (Trans_x and Rot_x).
        # This defines the location of joint i in the frame of joint i-1.
        
        # 1. Transformation from previous link's frame (i-1)
        # For i=0, this is identity. For i>0, this is Trans_x(a_{i-1}) * Rot_x(alpha_{i-1})
        if i > 0:
            trans_x_a = pin.SE3(np.eye(3), np.array([a[i-1], 0, 0]))
            rot_x_alpha = pin.SE3(pin.rpy.rpyToMatrix(alpha[i-1], 0, 0), np.zeros(3))
            current_transform = current_transform * trans_x_a * rot_x_alpha

        # 2. Define the joint's motion axis.
        # This is a rotation around Z, with a fixed offset along Z (d_i) and a theta offset.
        # Joint placement is defined by Trans_z(d_i) * Rot_z(theta_offset_i)
        trans_z_d = pin.SE3(np.eye(3), np.array([0, 0, d[i]]))
        rot_z_theta = pin.SE3(pin.rpy.rpyToMatrix(0, 0, theta_offset[i]), np.zeros(3))
        joint_placement = current_transform * rot_z_theta * trans_z_d
        
        # Add the joint to the model. The joint model itself provides the Rot_z(q_i) motion.
        joint_id = model.addJoint(0, pin.JointModelRZ(), joint_placement, f'joint_{i+1}')

    # Add an end-effector frame. It is placed relative to the last JOINT FRAME (not the last link frame).
    # The last link's fixed transform must be applied.
    trans_x_a_last = pin.SE3(np.eye(3), np.array([a[5], 0, 0]))
    rot_x_alpha_last = pin.SE3(pin.rpy.rpyToMatrix(alpha[5], 0, 0), np.zeros(3))
    eef_placement = joint_placement * trans_x_a_last * rot_x_alpha_last

    model.addFrame(pin.Frame("end_effector", 0, eef_placement, pin.FrameType.OP_FRAME))
    
    # Rebuild data structures after model modifications
    model.frames[-1].previousFrame = model.njoints # Link to the last joint
    pin.crba(model, model.createData(), np.zeros(model.nv)) # Recompute CRBA to update model internals
    pin.forwardKinematics(model,model.createData(),np.zeros(model.nv)) # Update frame placements

    return model

# (파일의 나머지 부분은 그대로 둡니다)

class PinocchioDhIkController:
    """
    An Inverse Kinematics controller using a model built from DH parameters.
    """
    def __init__(self, ee_link_name="end_effector"):
        """
        Initializes the IK controller.
        """
        self.model = build_model_from_dh()
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId(ee_link_name)
        self.n_joints = self.model.nq

        # IK calculation parameters
        self.DAMPING = 1e-4
        self.MAX_ITERATIONS = 100
        self.TOLERANCE = 1e-4

        print(f"Pinocchio DH IK controller initialized. Model has {self.model.nq} joints.")

    def calculate_ik(self, 
                     target_pose: pin.SE3, 
                     q_init: np.ndarray) -> np.ndarray | None:
        """
        Calculates the inverse kinematics for a given target pose.

        :param target_pose: The desired EEF pose (as a pin.SE3 object).
        :param q_init: The initial guess for the joint configuration.
        :return: The calculated joint angles (q) or None if no solution is found.
        """
        q = q_init.copy()
        
        for i in range(self.MAX_ITERATIONS):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            
            current_pose = self.data.oMf[self.ee_frame_id]
            error_vec = pin.log6(current_pose.inverse() * target_pose).vector
            
            if np.linalg.norm(error_vec) < self.TOLERANCE:
                return q  # Success!

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            
            # Damped Least Squares
            A = J.T @ J + self.DAMPING * np.eye(self.model.nv)
            b = J.T @ error_vec
            delta_q = np.linalg.solve(A, b)
            
            q = pin.integrate(self.model, q, delta_q)
            
        return None # Convergence failed
