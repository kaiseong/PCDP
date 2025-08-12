import numpy as np

def homogeneous_transform(tx, ty, tz, ang_y_deg, ang_z_deg):
    """
    Compute the 4×4 homogeneous transformation matrix that maps points from
    the camera frame C to the base frame B.

    Parameters:
    - tx, ty, tz : translation of the camera origin in the base frame (meters or units)
    - ang_y_deg   : intrinsic rotation angle about the camera's local Y-axis (degrees)
    - ang_z_deg   : intrinsic rotation angle about the camera's local Z-axis (degrees)

    Returns:
    - H (4×4 numpy.ndarray): homogeneous transform such that
        [p_B; 1] = H @ [p_C; 1]
    """
    # Convert degrees to radians
    ry = np.deg2rad(ang_y_deg)
    rz = np.deg2rad(ang_z_deg)

    # Rotation about local Y-axis
    R_y = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [          0, 1,          0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Rotation about local Z-axis
    R_z = np.array([
        [ np.cos(rz), -np.sin(rz), 0],
        [ np.sin(rz),  np.cos(rz), 0],
        [          0,           0, 1]
    ])

    # Combined intrinsic rotation: first Y, then Z
    R = R_y @ R_z

    # Assemble homogeneous transform
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = [tx, ty, tz]

    return H

# Example usage
if __name__ == "__main__":
    tx, ty, tz = 0.04, -0.29,-0.03     # translation
    ang_y, ang_z = 1, 0       # rotations
    H = homogeneous_transform(tx, ty, tz, ang_y, ang_z)
    print("Homogeneous transform H_C_to_B:\n", H)
