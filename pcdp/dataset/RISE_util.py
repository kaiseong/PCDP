import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 1.0]) 


# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

