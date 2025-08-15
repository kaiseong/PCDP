import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.1234, 0.1234, 0.1234])
IMG_STD = np.array([0.2620, 0.2710, 0.2709])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([0.109933, -0.265188, 0.07253]), np.array([0.387143, -0.018333, 0.265922]) 


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

