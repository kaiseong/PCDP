
import pinocchio as pin
import numpy as np

def build_model_from_dh():
    """
    Builds a Pinocchio model from Denavit-Hartenberg parameters.
    The DH parameters are taken from the C_PiperForwardKinematics class.
    Note: Pinocchio uses meters, so we convert mm to m.
    """
    model = pin.Model()

    # DH parameters from piper_fk.py (dh_is_offset = 0x00)
    # Link lengths (a) in meters
    a = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
    # Link twists (alpha) in radians
    alpha = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
    # Link offsets (d) in meters
    d = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0
    # Joint angle offsets (theta) in radians
    theta_offset = np.array([0.0, -np.pi * 174.22 / 180, -100.78 / 180 * np.pi, 0.0, 0.0, 0.0])

    parent_joint_id = 0
    # The transformation from the base to the first joint
    # This is often implicitly handled in URDFs but needs to be explicit here.
    # Based on the DH table, the first transformation is along Z.
    base_to_j1 = pin.SE3(np.eye(3), np.array([0, 0, d[0]]))

    for i in range(6):
        # The transformation for a link i is: Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
        # In Pinocchio, we define the joint and then the body attached to it.
        # The joint applies Rot_z(theta_i), and the body inertia has the fixed transformation.
        
        # Placement of the joint in the parent frame
        M = pin.SE3.Identity()
        if i == 0:
            # The first joint is placed relative to the world origin
            M.translation = np.array([0.0, 0.0, d[0]])
        else:
            # Subsequent joints are placed relative to the previous joint's frame
            # after its fixed transformation.
            M = pin.SE3(pin.rpy.rpyToMatrix(alpha[i-1], 0, theta_offset[i-1]), np.array([a[i-1], 0, d[i-1]]))

        joint_id = model.addJoint(parent_joint_id, pin.JointModelRZ(), M, f"joint_{i+1}")
        parent_joint_id = joint_id

    # Add an end-effector frame for IK
    # The final transformation from the last joint to the EEF
    eef_placement = pin.SE3(pin.rpy.rpyToMatrix(alpha[5], 0, theta_offset[5]), np.array([a[5], 0, d[5]]))
    model.addFrame(pin.Frame("end_effector", parent_joint_id, 0, eef_placement, pin.FrameType.OP_FRAME))

    return model

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
