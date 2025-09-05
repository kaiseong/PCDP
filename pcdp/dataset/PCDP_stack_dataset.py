# pcdp_stack_dataset.py
from typing import Dict, List
import torch
import numpy as np
import zarr
import os
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import json
import hashlib
import copy
import MinkowskiEngine as ME
import collections.abc as container_abcs

from pcdp.common.pytorch_util import dict_apply
from pcdp.dataset.base_dataset import BasePointCloudDataset
from pcdp.model.common.normalizer import LinearNormalizer
from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.real_world.real_data_pc_conversion import _get_replay_buffer, PointCloudPreprocessor, LowDimPreprocessor
from pcdp.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from pcdp.model.common.normalizer import SingleFieldLinearNormalizer

from tqdm import tqdm
from pcdp.common.RISE_transformation import xyz_rot_transform

class PCDP_RealStackPointCloudDataset(BasePointCloudDataset):
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
            enable_pc_preprocessing=True,
            pc_preprocessor_config=None,
            enable_low_dim_preprocessing=True,
            low_dim_preprocessor_config=None,
        ):

        super().__init__() 

        assert os.path.isdir(dataset_path), f"Dataset path does not exist: {dataset_path}"

        pc_preprocessor = None
        if enable_pc_preprocessing:
            pc_preprocessor = PointCloudPreprocessor(**(pc_preprocessor_config or {}))
        
        low_dim_preprocessor = None
        if enable_low_dim_preprocessing:
            low_dim_preprocessor = LowDimPreprocessor(**(low_dim_preprocessor_config or {}))
        
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
                        pc_preprocessor=pc_preprocessor,
                        lowdim_preprocessor=low_dim_preprocessor
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
                pc_preprocessor=pc_preprocessor,
                lowdim_preprocessor=low_dim_preprocessor
            )
        
        pointcloud_keys = [k for k, v in shape_meta.obs.items() if v.type == 'pointcloud']
        lowdim_keys = [k for k, v in shape_meta.obs.items() if v.type == 'low_dim']
        
        key_first_k = dict()
        if n_obs_steps is not None:
            for key in pointcloud_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

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
        
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.pointcloud_keys = pointcloud_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.val_mask = val_mask
        self.horizon = horizon
        self.voxel_size = voxel_size
        self.n_latency_steps = n_latency_steps
        self.normalizer = None

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
        val_set.normalizer = self.normalizer
        return val_set

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer

    def get_normalizer(self, device='cpu') -> LinearNormalizer:
        normalizer = LinearNormalizer()
        
        all_actions = torch.from_numpy(self.replay_buffer['action'][:]).to(device)
        all_robot_obs = torch.from_numpy(self.replay_buffer['robot_obs'][:]).to(device)

        action_trans_data = all_actions[:, :3]
        obs_trans_data = all_robot_obs[:, :3]
        obs_grip_data = all_robot_obs[:, 6:7]

        normalizer['action_translation'] = SingleFieldLinearNormalizer.create_fit(action_trans_data)
        normalizer['obs_translation'] = SingleFieldLinearNormalizer.create_fit(obs_trans_data)
        normalizer['obs_gripper'] = SingleFieldLinearNormalizer.create_fit(obs_grip_data)

        all_colors = []
        for i in tqdm(range(self.replay_buffer.n_episodes), desc="Calculating PointCloud Color Stats"):
            data = self.replay_buffer.get_episode(i)
            points = data['pointcloud']
            for pc in points:
                if len(pc) > 0:
                    all_colors.append(pc[:, 3:6])
        
        if all_colors:
            all_colors = np.concatenate(all_colors, axis=0) 
            all_colors = torch.from_numpy(all_colors).to(device)
            color_normalizer = SingleFieldLinearNormalizer.create_fit(all_colors, mode='gaussian')
            normalizer['pointcloud_color'] = color_normalizer
        
        return normalizer

    def __len__(self):
        return len(self.sampler)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = {key: data[key][T_slice] for key in self.pointcloud_keys + self.lowdim_keys}
        
        clouds = [obs_dict['pointcloud'][i].astype(np.float32) for i in range(self.n_obs_steps)]
        robot_obs_euler = np.array([obs_dict['robot_obs'][i] for i in range(self.n_obs_steps)], dtype=np.float32)
        
        actions_euler = data['action'].astype(np.float32)

        # Convert obs from Euler angles to 6D representation
        obs_pose_euler = robot_obs_euler[:, :6]
        obs_gripper = robot_obs_euler[:, 6:]
        obs_9d = xyz_rot_transform(obs_pose_euler, from_rep='euler_angles', to_rep='rotation_6d', from_convention='ZYX')
        robot_obs_10d = np.concatenate([obs_9d, obs_gripper], axis=-1)

        # Convert action from Euler angles to 6D representation
        pose_euler = actions_euler[:, :6]
        gripper_action = actions_euler[:, 6:]
        pose_9d = xyz_rot_transform(pose_euler, from_rep="euler_angles", to_rep="rotation_6d", from_convention="ZYX")
        actions_10d = np.concatenate([pose_9d, gripper_action], axis=-1)

        # Voxelize for MinkowskiEngine
        input_coords_list = []
        input_feats_list = []
        for cloud in clouds:
            coords = np.ascontiguousarray(cloud[:, :3] / self.voxel_size, dtype=np.int32)
            # Return unnormalized RGB in features
            input_feats_list.append(cloud.astype(np.float32))
            input_coords_list.append(coords)

        return {
            'input_coords_list': input_coords_list,
            'input_feats_list': input_feats_list,
            'robot_obs': torch.from_numpy(robot_obs_10d).float(),
            'action': torch.from_numpy(actions_10d).float()
        }

def collate_fn(batch):
    if not isinstance(batch, list):
        return batch
    
    elem = batch[0]
    if not isinstance(elem, container_abcs.Mapping):
        raise TypeError(f"Batch must contain dicts, but found {type(elem)}")
    
    ret_dict = {}
    coords_list = list()
    feats_list = list()

    for key in elem:
        if key in ['input_coords_list', 'input_feats_list']:
            flat_list = [item for d in batch for item in d[key]]
            if key == 'input_coords_list':
                coords_list.extend(flat_list)
            else:
                feats_list.extend(flat_list)
        else:
            ret_dict[key] = torch.stack([d[key] for d in batch], dim =0)
    if coords_list and feats_list:
        coords_batch, feats_batch = ME.utils.sparse_collate(coords=coords_list, feats=feats_list)
        ret_dict['input_coords_list'] = coords_batch
        ret_dict['input_feats_list'] = feats_batch
    return ret_dict