import os
import numpy as np

from pcdp.common.RISE_transformation import xyz_rot_to_mat, mat_to_xyz_rot

camera_to_origin = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
    ])

workspace_bounds = np.array([
    [-0.000, 0.740],    # X range (m)
    [-0.400, 0.350],    # Y range (m)
    [-0.100, 0.400]     # Z range (m)
])


robot_to_origin = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])

z_offset = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0.07], 
    [0, 0, 0, 1]])

class Projector:
    def __init__(self, calib_path):
        pass
        
    def project_tcp_to_camera_coord(self, tcp, cam, rotation_rep = "", rotation_rep_convention = None):
        pass

    def project_tcp_to_base_coord(self, tcp, cam, rotation_rep = "", rotation_rep_convention = None):
        
        
