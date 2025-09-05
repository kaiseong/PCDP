
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pcdp.common.RISE_transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform
import numpy as np
import time
from pcdp.dataset.RISE_util import *
from pcdp.common import RISE_transformation as rise_tf
import torch
TRANS_MIN = np.array([-1.0, -0.5, -0.3])
TRANS_MAX = np.array([1.0, 0.5, 0.3])

camera_to_base = np.array([
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


robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])


# 1단계 시연 수집된 action 데이터
pose_eulers = np.array([
    [0.189517, 0.070388, 0.241700, -3.071465, 0.863798, -2.837434, 1],
    [0.4081271, 0.0243388, 0.163988, 3.1056912, 1.209059, -2.2664297, 0],
    ])

# 2단계 학습시 캐시 데이터셋으로 변환시 base 좌표계로 TF되는 action 데이터
TF_action_7d =[]
for pose_euler in pose_eulers:
    pose_6d = pose_euler[:6]
    gripper = pose_euler[6]
    translation = pose_6d[:3]
    rotation = pose_6d[3:6]
    eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
    T_k_matrix = robot_to_base @ eef_to_robot_base_k
    transformed_pose_6d = rise_tf.mat_to_xyz_rot(
        T_k_matrix,
        rotation_rep='euler_angles',
        rotation_rep_convention='ZYX'
    )
    new_action_7d = np.concatenate([transformed_pose_6d, [gripper]])
    TF_action_7d.append(new_action_7d)

# (2, 7)
TF_action_7d = np.array(TF_action_7d, dtype=np.float32)
print(f"캐시 데이터셋(base 좌표계로 TF) 변환 후의 값: {TF_action_7d}")


# 3단계 roation을 rotation 6d로 변환
TF_pose_6d=TF_action_7d[:, :6]
gripper = TF_action_7d[:, 6:]

pose_9d = xyz_rot_transform(TF_pose_6d, from_rep="euler_angles", to_rep="rotation_6d", from_convention="ZYX")
actions_10d = np.concatenate([pose_9d, gripper], axis=-1)
print(f"rotation 6d로 변환 후의 값: {actions_10d}")

# 4단계 normalize
actions_10d[:, :3] = (actions_10d[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
actions_10d[:, -1] = actions_10d[:, -1] * 2 - 1
print(f"정규화 적용한 값: {actions_10d}")

print("=============================== 학습 진행.... ==============================")

# model에서 나온 데이터도 rotation 6d에 normalize된 상태일 것

# 5단계 unormalize
# NumPy 배열과 PyTorch 텐서를 직접 연산할 수 없으므로, 타입을 통일합니다.
# 연산을 위해 actions_7d를 PyTorch 텐서로 변환합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actions_10d_tensor = torch.from_numpy(actions_10d).to(device).float()

trans_min_exp = torch.from_numpy(TRANS_MIN).to(device).float()
trans_max_exp = torch.from_numpy(TRANS_MAX).to(device).float()

# PyTorch 텐서로 un-normalize 연산 수행
actions_10d_tensor[..., :3] = (actions_10d_tensor[..., :3] + 1) / 2.0 * (trans_max_exp - trans_min_exp) + trans_min_exp
actions_10d_tensor[..., -1] = (actions_10d_tensor[..., -1] + 1) / 2.0 
print(f"역 정규화 적용한 값: {actions_10d_tensor}")

# 이후 코드에서 NumPy 배열이 필요할 수 있으므로 다시 변환해줍니다.
pred = actions_10d_tensor.cpu().numpy()

# 6단계 다시 rotation 6d를 euler rot로 바꾸기
pos   = pred[:, :3]
rot6d = pred[:, 3:9]
grip  = pred[:, 9:]

xyz_rot6d = np.concatenate([pos, rot6d], axis=-1) 
xyz_eulers = xyz_rot_transform(
        xyz_rot6d,
        from_rep="rotation_6d",
        to_rep="euler_angles",
        to_convention="ZYX"
    )
print(f"rotation_6d 해제 값: {xyz_eulers}")

# 7단계 다시 robot base로 TF

base_to_robot_matrix = np.linalg.inv(robot_to_base)

original_poses = []

for xyz_euler in xyz_eulers:
    translation = xyz_euler[:3]
    rotation = xyz_euler[3:6]
    transformed_matrix = rise_tf.rot_trans_mat(translation, rotation)
    original_matrix = base_to_robot_matrix @ transformed_matrix

    original_pose_6d = rise_tf.mat_to_xyz_rot(
        original_matrix,
        rotation_rep='euler_angles',
        rotation_rep_convention='ZYX'
    )
    original_poses.append(original_pose_6d)

action_sequence_7d = np.concatenate([original_poses, grip], axis=-1)
print(f"다시 로봇 좌표계로 TF 값: {action_sequence_7d}")

print(f"원본 - 복원 값 [0]: {pose_eulers[0] - action_sequence_7d[0]}")
print(f"원본 - 복원 값 [0]: {pose_eulers[1] - action_sequence_7d[1]}")