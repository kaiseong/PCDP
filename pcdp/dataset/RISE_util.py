import numpy as np

# tcp normalization and gripper width normalization
ACTION_TRANS_MAX = np.array([0.408983, 0.026530, 0.341521]) 
ACTION_TRANS_MIN = np.array([0.089008, -0.335518, 0.058755]) 
OBS_TRANS_MAX = np.array([0.408523, 0.026380, 0.339402]) 
OBS_TRANS_MIN = np.array([0.088132, -0.335480, 0.058757]) 
OBS_GRIP_MAX = np.array([94.1])
OBS_GRIP_MIN = np.array([0.0])


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

