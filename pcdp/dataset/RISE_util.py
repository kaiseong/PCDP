import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.0217, 0.0217, 0.0217])
IMG_STD = np.array([0.1474, 0.1474, 0.1474])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.018614, -0.53738, 0.10401]), np.array([0.486704, 0.4232, 0.448772]) 

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']

