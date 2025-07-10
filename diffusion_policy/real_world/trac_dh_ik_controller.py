

import numpy as np
import pinocchio as pin
from tracikpy import TracIKSolver
from typing import Union

def generate_urdf_from_dh() -> str:
    """
    Generates a URDF XML string dynamically from Denavit-Hartenberg parameters.
    """
    # DH parameters from piper_fk.py (dh_is_offset = 0x00), converted to meters
    a = np.array([0.0, 0.0, 285.03, -21.98, 0.0, 0.0]) / 1000.0
    alpha = np.array([0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2])
    d = np.array([123.0, 0.0, 0.0, 250.75, 0.0, 91.0]) / 1000.0
    theta_offset = np.array([0.0, -np.pi * 174.22 / 180, -100.78 / 180 * np.pi, 0.0, 0.0, 0.0])

    urdf = '<?xml version="1.0"?>\n'
    urdf += '<robot name="piper_dh">\n'
    
    # Base Link
    urdf += '  <link name="base_link">\n'
    urdf += '    <visual><geometry><box size="0.1 0.1 0.1"/></visual>\n'
    urdf += '  </link>\n'

    last_link = "base_link"
    for i in range(6):
        link_name = f"link_{i+1}"
        joint_name = f"joint_{i+1}"

        # The transformation from the previous link's frame to the current joint's frame
        # is defined by the DH parameters of the *previous* link (i-1).
        # For URDF, the joint origin is defined relative to the parent link's frame.
        # URDF transform: Rz(theta) -> Tz(d) -> Tx(a) -> Rx(alpha)
        
        # We need to create a chain of transformations.
        # The origin of joint `i` is placed relative to link `i-1`'s frame.
        # The transformation is composed of parameters from the previous step.
        if i == 0:
            # First joint is relative to the base link
            # Transformation: Rz(theta_offset_0) * Tz(d_0) * Tx(a_0) * Rx(alpha_0)
            x, y, z = 0, 0, d[0]
            roll, pitch, yaw = 0, 0, theta_offset[0]
        else:
            # Subsequent joints
            # Transformation: Rz(theta_offset_{i-1}) * Tz(d_{i-1}) * Tx(a_{i-1}) * Rx(alpha_{i-1})
            # This is a bit tricky because URDF combines transformations.
            # A standard approach is to represent the DH params as a series of links and joints.
            # Let's use a simpler representation that TRAC-IK can parse.
            # The origin of joint i is at the end of link i-1.
            # The transformation from joint i-1 to joint i is defined by DH params of link i-1.
            
            # Create a rotation matrix from the previous alpha and theta_offset
            prev_alpha = alpha[i-1]
            prev_theta = theta_offset[i-1]
            rot_mat = np.array([
                [np.cos(prev_theta), -np.sin(prev_theta), 0],
                [np.sin(prev_theta)*np.cos(prev_alpha), np.cos(prev_theta)*np.cos(prev_alpha), -np.sin(prev_alpha)],
                [np.sin(prev_theta)*np.sin(prev_alpha), np.cos(prev_theta)*np.sin(prev_alpha), np.cos(prev_alpha)]
            ])
            rpy = pin.rpy.matrixToRpy(rot_mat)
            roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

            # Create the translation vector
            x, y, z = a[i-1], -np.sin(alpha[i-1])*d[i], np.cos(alpha[i-1])*d[i]

        # Joint definition
        urdf += f'  <joint name="{joint_name}" type="revolute">\n'
        urdf += f'    <parent link="{last_link}"/>\n'
        urdf += f'    <child link="{link_name}"/>\n'
        # The origin needs to combine the transformations.
        # Let's define the transform from frame i-1 to i
        urdf += f'    <origin xyz="{a[i-1]} {0} {d[i-1]}" rpy="{alpha[i-1]} {0} {theta_offset[i-1]}"/>\n'
        urdf += '    <axis xyz="0 0 1"/>\n'
        urdf += '    <limit lower="-3.14" upper="3.14" effort="100" velocity="1.0"/>\n'
        urdf += '  </joint>\n'

        # Link definition
        urdf += f'  <link name="{link_name}">\n'
        urdf += '    <visual><geometry><box size="0.05 0.05 0.05"/></visual>\n'
        urdf += '  </link>\n'
        last_link = link_name

    # Add End Effector Link
    urdf += '  <joint name="end_effector_joint" type="fixed">\n'
    urdf += f'   <parent link="{last_link}"/>\n'
    urdf += '   <child link="end_effector"/>\n'
    urdf += f'   <origin xyz="{a[5]} 0 {d[5]}" rpy="{alpha[5]} 0 {theta_offset[5]}"/>\n'
    urdf += '  </joint>\n'
    urdf += '  <link name="end_effector"/>\n'

    urdf += '</robot>\n'
    return urdf

class TracDhIkController:
    """
    A TRAC-IK controller using a model built dynamically from DH parameters.
    """
    def __init__(self,
                 ee_link_name: str = "end_effector",
                 base_link_name: str = "base_link",
                 solve_type: str = "Speed",
                 **kwargs):
        """
        Initializes the TracIKSolver with a URDF string generated from DH params.
        """
        urdf_string = generate_urdf_from_dh()
        
        self.ik_solver = TracIKSolver(
            urdf_string=urdf_string, # Use the generated string
            base_link=base_link_name,
            tip_link=ee_link_name,
            timeout=0.005,
            epsilon=1e-5,
            solve_type=solve_type,
        )
        
        print("--- TRAC-IK DH-based controller initialized successfully ---")
        print(f"EE Link: {ee_link_name}, Solve Type: {solve_type}")
        print(f"Joint Names: {self.ik_solver.joint_names}")

    def calculate_ik(self,
                     target_pose: pin.SE3,
                     q_init: np.ndarray) -> Union[np.ndarray, None]:
        """
        Calculates the inverse kinematics for a given target pose.
        """
        if isinstance(target_pose, pin.SE3):
            target_matrix = target_pose.homogeneous
        else:
            target_matrix = np.asarray(target_pose)

        result_joints = self.ik_solver.ik(
            ee_pose=target_matrix,
            qinit=q_init,
        )
        return result_joints

