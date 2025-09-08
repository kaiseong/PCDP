import numpy as np

def homogeneous_transform(tx, ty, tz, ang_x_deg, ang_y_deg, ang_z_deg):
    """
    Compute the 4x4 homogeneous transformation matrix that maps points from
    the camera frame C to the base frame B, based on ZYX Euler angles.

    Parameters:
    - tx, ty, tz : translation of the camera origin in the base frame (meters or units)
    - ang_x_deg   : intrinsic rotation angle about the camera's local X-axis (degrees)
    - ang_y_deg   : intrinsic rotation angle about the camera's local Y-axis (degrees)
    - ang_z_deg   : intrinsic rotation angle about the camera's local Z-axis (degrees)

    Returns:
    - H (4x4 numpy.ndarray): homogeneous transform such that
        [p_B; 1] = H @ [p_C; 1]
    """
    # Convert degrees to radians
    rx = np.deg2rad(ang_x_deg)
    ry = np.deg2rad(ang_y_deg)
    rz = np.deg2rad(ang_z_deg)

    # Rotation about local X-axis
    R_x = np.array([
        [1,           0,            0],
        [0,  np.cos(rx), -np.sin(rx)],
        [0,  np.sin(rx),  np.cos(rx)]
    ])

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

    # Combined intrinsic rotation: first X, then Y, then Z (equivalent to extrinsic ZYX)
    R = R_z @ R_y @ R_x

    # Assemble homogeneous transform
    H = np.eye(4)
    H[:3, :3] = R
    H[:3,  3] = [tx, ty, tz]

    return H

# Example usage
if __name__ == "__main__":
    # Set the translation components
    tx, ty, tz = 0.0, 0.0, 0.0

    # Set the rotation angles for Z, Y, X axes in degrees
    ang_z, ang_y, ang_x = 90, 00, 33

    # Compute the transformation matrix
    H = homogeneous_transform(tx, ty, tz, ang_x, ang_y, ang_z)
    
    # Print the resulting matrix, formatted to 3 decimal places for clarity
    with np.printoptions(precision=3, suppress=True):
        print("Homogeneous transform H_C_to_B (ZYX order):\n", H)