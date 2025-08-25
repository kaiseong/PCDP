import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.1234, 0.1234, 0.1234])
IMG_STD = np.array([0.2620, 0.2710, 0.2709])

# tcp normalization and gripper width normalization
ACTION_TRANS_MIN = np.array([0.109933, -0.265188, 0.07253]) 
ACTION_TRANS_MAX = np.array([0.387143, -0.018333, 0.265922]) 
OBS_TRANS_MIN = np.array([0.109933, -0.265188, 0.07253]) 
OBS_TRANS_MAX = np.array([0.387143, -0.018333, 0.265922]) 
OBS_GRIP_MIN = np.array([95.0])
OBS_GRIP_MAX = np.array([65.0])


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

