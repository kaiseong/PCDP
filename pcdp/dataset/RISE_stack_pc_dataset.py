# real_stack_pc_dataset.py
from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import json
import hashlib
import copy
import open3d as o3d
import MinkowskiEngine as ME
import torchvision.transforms as T
import collections.abc as container_abcs
from PIL import Image

from pcdp.common.pytorch_util import dict_apply
from pcdp.dataset.base_dataset import BasePointCloudDataset
from pcdp.model.common.normalizer import LinearNormalizer
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.real_world.real_data_pc_conversion import _get_replay_buffer, PointCloudPreprocessor
from pcdp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from pcdp.model.common.normalizer import SingleFieldLinearNormalizer

# Assuming these util/transformation files are created by the user
from pcdp.dataset.RISE_util import *
from pcdp.common.RISE_transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

class RealStackPointCloudDataset(BasePointCloudDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=20,
            pad_before=0,
            pad_after=0,
            n_obs_steps=1,
            n_latency_steps=0,
            use_cache=True,
            seed=42,
            voxel_size=0.005,
            val_ratio=0.0,
            max_train_episodes=None,
            enable_preprocessing=True,
            preprocessor_config=None,
            # RISE specific params
            aug=False,
            aug_trans_min=None,
            aug_trans_max=None,
            aug_rot_min=None,
            aug_rot_max=None,
            aug_jitter=False,
            aug_jitter_params=None,
            aug_jitter_prob=0.2,
            split='train'
        ):

        # =========== 1. ADDED: Data Loading Logic ===========
        # This section was missing and is now added to actually load data.
        super().__init__() # Call parent __init__ but don't pass args

        preprocessor = None
        if enable_preprocessing:
            preprocessor = PointCloudPreprocessor(**(preprocessor_config or {}))

        replay_buffer = None
        if use_cache:
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    print('Cache does not exist. Creating!')
                    replay_buffer = _get_replay_buffer(
                        dataset_path=dataset_path,
                        shape_meta=shape_meta,
                        store=zarr.MemoryStore(),
                        preprocessor=preprocessor
                    )
                    print('Saving cache to disk.')
                    with zarr.ZipStore(cache_zarr_path) as zip_store:
                        replay_buffer.save_to_store(store=zip_store)
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                preprocessor=preprocessor
            )
        
        # Parse shape meta to identify keys
        pointcloud_keys = [k for k, v in shape_meta.obs.items() if v.type == 'pointcloud']
        lowdim_keys = [k for k, v in shape_meta.obs.items() if v.type == 'low_dim']
        
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        if max_train_episodes is not None:
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
        
        episode_mask = train_mask if split =='train' else val_mask

        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=episode_mask)
        
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.pointcloud_keys = pointcloud_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.horizon = horizon
        # =======================================================

        # Add RISE-specific augmentation and processing parameters
        self.aug = aug
        self.aug_trans_min = np.array(aug_trans_min if aug_trans_min is not None else [-0.2, -0.2, -0.2])
        self.aug_trans_max = np.array(aug_trans_max if aug_trans_max is not None else [0.2, 0.2, 0.2])
        self.aug_rot_min = np.array(aug_rot_min if aug_rot_min is not None else [-30, -30, -30])
        self.aug_rot_max = np.array(aug_rot_max if aug_rot_max is not None else [30, 30, 30])
        self.aug_jitter = aug_jitter
        self.aug_jitter_params = np.array(aug_jitter_params if aug_jitter_params is not None else [0.4, 0.4, 0.2, 0.1])
        self.aug_jitter_prob = aug_jitter_prob
        self.voxel_size = voxel_size
        self.split = split
        self.n_latency_steps = n_latency_steps
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler =SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask
        )
        val_set.val_mask = self.val_mask
        val_set.split = 'val'
        return val_set

    def _augmentation(self, clouds, tcps_euler):
        # To correctly augment, we should convert euler to a rotation matrix/quaternion,
        # apply transformations, and then convert back if needed.
        # For simplicity, let's assume the augmentation function handles this.
        # We will convert to quaternion here before passing to augmentation.
        tcps_quat = xyz_rot_transform(tcps_euler, from_rep="euler_angles", to_rep="quaternion", from_convention="XYZ")

        translation_offsets = np.random.rand(3) * (self.aug_trans_max - self.aug_trans_min) + self.aug_trans_min
        rotation_angles = np.random.rand(3) * (self.aug_rot_max - self.aug_rot_min) + self.aug_rot_min
        rotation_angles = rotation_angles / 180 * np.pi
        aug_mat = rot_trans_mat(translation_offsets, rotation_angles)
        
        center = clouds[-1][..., :3].mean(axis=0)

        for i in range(len(clouds)):
            clouds[i][..., :3] -= center
            clouds[i] = apply_mat_to_pcd(clouds[i], aug_mat)
            clouds[i][..., :3] += center

        tcps_quat[..., :3] -= center
        tcps_quat = apply_mat_to_pose(tcps_quat, aug_mat, rotation_rep="quaternion")
        tcps_quat[..., :3] += center

        # Convert back to euler angles for consistency if other parts of the code expect it
        tcps_aug_euler = xyz_rot_transform(tcps_quat, from_rep="quaternion", to_rep="euler_angles", to_convention="XYZ")
        return clouds, tcps_aug_euler

    def _normalize_tcp(self, tcp_list):
        # Assuming tcp_list is (N, 9) for 6D rot or (N, 7) for euler+gripper
        # This needs to be robust to the input shape.
        tcp_list[:, :3] = (tcp_list[:, :3] - TRANS_MIN) / (TRANS_MAX - TRANS_MIN) * 2 - 1
        # Normalize gripper which is the last element
        # The user's dataset uses binary gripper values (0 for closed, 1 for open).
        # Map 0 to -1 and 1 to 1.
        tcp_list[:, -1] = tcp_list[:, -1] * 2 - 1
        return tcp_list

    def __len__(self):
        return len(self.sampler)
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = {key: data[key][T_slice] for key in self.pointcloud_keys + self.lowdim_keys}
        
        clouds = [obs_dict['pointcloud'][i].astype(np.float32) for i in range(self.n_obs_steps)]
        
        # Action is (T, 7) with (x,y,z, r,p,y, gripper)
        actions_euler = data['action'].astype(np.float32)

        # Color jitter
        if self.split == 'train' and self.aug_jitter:
            jitter_transform = T.ColorJitter(
                brightness=self.aug_jitter_params[0],
                contrast=self.aug_jitter_params[1],
                saturation=self.aug_jitter_params[2],
                hue=self.aug_jitter_params[3]
            )
            jitter = T.RandomApply([jitter_transform], p=self.aug_jitter_prob)
            
            for i in range(len(clouds)):
                cloud_colors = (clouds[i][:, 3:] * 255).astype(np.uint8)
                # To apply torchvision transforms, data needs to be in (H, W, C) or (C, H, W).
                # A point cloud is an unordered set. A simple reshape might not be meaningful.
                # As a simple workaround, we treat the points as a 1D image (1, N, 3).
                pil_img = Image.fromarray(cloud_colors.reshape(1, -1, 3))
                jittered_img = jitter(pil_img)
                jittered_colors = np.array(jittered_img).reshape(-1, 3) / 255.0
                clouds[i][:, 3:] = jittered_colors.astype(np.float32)
        
        # Normalize colors
        for i in range(len(clouds)):
            clouds[i][:, 3:] = clouds[i][:, 3:] / 255.0
        for i in range(len(clouds)):
            clouds[i][:, 3:] = (clouds[i][:, 3:] - IMG_MEAN) / IMG_STD

        # Point cloud augmentation
        if self.split == 'train' and self.aug:
            clouds, actions_euler = self._augmentation(clouds, actions_euler)

        # =========== 2. MODIFIED: Action Representation Conversion ===========
        # Convert action from Euler angles to 6D representation for the policy
        pose_euler = actions_euler[:, :6]
        gripper_action = actions_euler[:, 6:]

        pose_6d = xyz_rot_transform(pose_euler, from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")
        
        actions_7d = np.concatenate([pose_6d, gripper_action], axis=-1)
        # Normalize actions
        actions_normalized = self._normalize_tcp(actions_7d.copy())

        # Voxelize for MinkowskiEngine
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
            input_coords_list.append(coords)
            input_feats_list.append(cloud.astype(np.float32))

        return {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'action': torch.from_numpy(actions_7d).float(),
            'action_normalized': torch.from_numpy(actions_normalized).float(),
            'action_euler': torch.from_numpy(actions_euler).float()
        }

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        # This needs to be implemented according to the data format
        # For now, returning an identity normalizer
        normalizer = LinearNormalizer()
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
        for key in self.pointcloud_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

# RISE-specific collate function
def collate_fn(batch):
    if not isinstance(batch, list):
        return batch
    
    elem = batch[0]
    if isinstance(elem, container_abcs.Mapping):
        ret_dict = {}
        for key in elem:
            if key in ['action', 'action_normalized', 'action_euler']:
                ret_dict[key] = torch.stack([d[key] for d in batch], 0)
            elif key in ['input_coords_list', 'input_feats_list']:
                flat_list = [item for sublist in [d[key] for d in batch] for item in sublist]
                if key == 'input_coords_list':
                    coords_list = flat_list
                else:
                    feats_list = flat_list
            else:
                ret_dict[key] = [d[key] for d in batch]

        coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
        return ret_dict

    raise TypeError(f"Batch must contain dicts, but found {type(elem)}")