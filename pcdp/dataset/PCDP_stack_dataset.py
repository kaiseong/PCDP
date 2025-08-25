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

# Assuming these util/transformation files are created by the user
from pcdp.dataset.RISE_util import *
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
        
        # Parse shape meta to identify keys
        pointcloud_keys = [k for k, v in shape_meta.obs.items() if v.type == 'pointcloud']
        lowdim_keys = [k for k, v in shape_meta.obs.items() if v.type == 'low_dim']
        
        print(f"  - Pointcloud keys: {pointcloud_keys}")
        print(f"  - Low-dim keys: {lowdim_keys}")

        self.check_replay_buffer_keys(replay_buffer, pointcloud_keys, lowdim_keys)
        

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from pointcloud and lowdim data
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
    
    def check_replay_buffer_keys(self, replay_buffer: ReplayBuffer, pointcloud_keys: List[str], lowdim_keys: List[str]):
        """
        Validate that replay buffer contains all expected keys from shape_meta.
        
        Args:
            replay_buffer: The loaded replay buffer
            pointcloud_keys: Expected pointcloud keys
            lowdim_keys: Expected lowdim keys
        """
        
        buffer_keys = set(replay_buffer.keys())
        
        # Check pointcloud keys
        for key in pointcloud_keys:
            if key not in buffer_keys:
                raise ValueError(f"Expected pointcloud key '{key}' not found in replay buffer. "
                               f"Available keys: {list(buffer_keys)}")
        
        # Check lowdim keys
        for key in lowdim_keys:
            if key not in buffer_keys:
                raise ValueError(f"Expected lowdim key '{key}' not found in replay buffer. "
                               f"Available keys: {list(buffer_keys)}")
        
        # Check action key
        if 'action' not in buffer_keys:
            raise ValueError("Expected 'action' key not found in replay buffer. "
                           f"Available keys: {list(buffer_keys)}")
        
        print(f"Validation passed: All expected keys found in replay buffer")

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
        return val_set


    def _normalize_(self, tcp_list, trans_max, trans_min, grip_max=None, grip_min=None):
        # Assuming tcp_list is (N, 9) for 6D rot or (N, 7) for euler+gripper
        # This needs to be robust to the input shape.
        tcp_list[:, :3] = (tcp_list[:, :3] - trans_min) / (trans_max - trans_min) * 2 - 1
        # Normalize gripper which is the last element
        # The user's dataset uses binary gripper values (0 for closed, 1 for open).
        # Map 0 to -1 and 1 to 1.
        if grip_max is not None and grip_min is not None:
            tcp_list[:, -1] = (tcp_list[:, -1] - grip_min ) / (grip_max - grip_min) *2 -1
        else:
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
        robot_obs = np.array([obs_dict['robot_obs'][i] for i in range(self.n_obs_steps)], dtype=np.float32)
        

        # Action is (T, 7) with (x,y,z, r,p,y, gripper)
        actions_euler = data['action'].astype(np.float32)

        # =========== 2. MODIFIED: Action Representation Conversion ===========
        # Convert action from Euler angles to 6D representation for the policy
        pose_euler = actions_euler[:, :6]
        gripper_action = actions_euler[:, 6:]

        pose_9d = xyz_rot_transform(pose_euler, from_rep="euler_angles", to_rep="rotation_6d", from_convention="XYZ")
        
        actions_10d = np.concatenate([pose_9d, gripper_action], axis=-1)

        obs_pose_euler = robot_obs[:, :6]
        obs_gripper = robot_obs[:, 6:]
        obs_9d = xyz_rot_transform(obs_pose_euler, from_rep='euler_angles', to_rep='rotation_6d', from_convention='XYZ')
        
        obs_10d = np.concatenate([obs_9d, obs_gripper], axis=-1)
        robot_obs = obs_10d


        # Normalize actions
        actions_normalized = self._normalize_(actions_10d.copy(), ACTION_TRANS_MAX, ACTION_TRANS_MIN)
        # Normalize obs
        robot_obs = self._normalize_(robot_obs, OBS_TRANS_MAX, OBS_TRANS_MIN, OBS_GRIP_MAX, OBS_GRIP_MIN)

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
            'robot_obs': torch.from_numpy(robot_obs).float(),
            'action': torch.from_numpy(actions_10d).float(),
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
