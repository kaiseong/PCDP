
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
    for i in range(6):
        # Pinocchio's convention for revolute joints is around the Z-axis.
        # The DH convention involves a sequence of transformations.
        # We can represent a DH link as a fixed joint in Pinocchio.
        # The transformation for a link i is: Rot_z(theta_i) * Trans_z(d_i) * Trans_x(a_i) * Rot_x(alpha_i)
        
        # 1. Transformation for the joint angle (theta_i) + offset
        joint_placement = pin.SE3(np.eye(3), np.array([0, 0, 0]))
        joint_id = model.addJoint(parent_joint_id, pin.JointModelRZ(), joint_placement, f"joint_{i+1}")

        # 2. Transformation for the rest of the DH parameters (d, a, alpha)
        # This forms the rigid body (link) attached to the joint.
        link_placement = pin.SE3.Identity()
        link_placement.translation = np.array([a[i], 0, d[i]])
        link_placement.rotation = pin.rpy.rpyToMatrix(alpha[i], 0, 0)

        # We need to adjust the placement to account for the theta offset
        # by pre-multiplying by a Z-rotation.
        theta_rotation = pin.SE3(pin.rpy.rpyToMatrix(0, 0, theta_offset[i]), np.zeros(3))
        
        # The final placement of the *next* joint's frame relative to the current one
        frame_placement = theta_rotation * link_placement

        model.appendBodyToJoint(joint_id, pin.Inertia.Zero(), frame_placement)
        parent_joint_id = joint_id

    # Add an end-effector frame for IK
    ee_frame_placement = pin.SE3.Identity() # Assuming EEF is at the origin of the last link frame
    model.addFrame(pin.Frame("end_effector", parent_joint_id, 0, ee_frame_placement, pin.FrameType.OP_FRAME))

    return model

class DhIkController:
    """
    An Inverse Kinematics controller using a model built from DH parameters.
    """
    def __init__(self, eps=1e-4, max_iters=1000, damping=1e-12):
        """
        Initializes the IK controller.
        :param eps: Accuracy threshold for the IK solution.
        :param max_iters: Maximum number of iterations for the IK solver.
        :param damping: Damping factor for the IK solver.
        """
        self.model = build_model_from_dh()
        self.data = self.model.createData()
        self.ee_frame_id = self.model.getFrameId("end_effector")
        self.eps = eps
        self.max_iters = max_iters
        self.damping = damping

    def calculate_ik(self, target_pose: pin.SE3, q0: np.ndarray = None) -> np.ndarray | None:
        """
        Calculates the inverse kinematics for a given target pose.
        :param target_pose: The desired end-effector pose (as a pin.SE3 object).
        :param q0: The initial guess for the joint configuration. If None, uses random values.
        :return: The calculated joint angles (q) or None if no solution is found.
        """
        if q0 is None:
            q0 = pin.randomConfiguration(self.model)
        
        q = q0.copy()
        
        for i in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            
            # Current pose of the end-effector
            current_pose = self.data.oMf[self.ee_frame_id]
            
            # Calculate the error between the current and target poses
            err = pin.log(target_pose.inverse() * current_pose).vector
            
            if np.linalg.norm(err) < self.eps:
                # Solution found
                return q

            # Compute the Jacobian of the end-effector frame
            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)
            
            # Solve for the change in joint angles using damped least squares
            # (J.T * J + damping * I) * dq = J.T * err
            A = J.T.dot(J) + self.damping * np.eye(self.model.nv)
            b = J.T.dot(err)
            delta_q = np.linalg.solve(A, b)
            
            # Update the joint configuration
            q = pin.integrate(self.model, q, -delta_q)

        # No solution found within the iteration limit
        return None

