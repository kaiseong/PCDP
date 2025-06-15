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
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BasePointCloudDataset 
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from diffusion_policy.real_world.pointcloud_data_conversion import real_pointcloud_data_to_replay_buffer

class RealStackPointCloudDataset(BasePointCloudDataset):  
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
            apply_preprocessing=False,
            use_cuda=False
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_pointcloud_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            apply_preprocessing=apply_preprocessing,
                            use_cuda=use_cuda
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.exists(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_pointcloud_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support 7D action (6DOF + gripper)
            assert actions.shape[1] <= 7
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        pointcloud_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'pointcloud':
                pointcloud_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from pointclouds
            for key in pointcloud_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
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
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action (7D: 6DOF + gripper)
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # pointcloud 
        for key in self.pointcloud_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.pointcloud_keys:
            # T,N,6 pointcloud data
            # XYZ is float32, RGB is 0-1 Normalization
            pc_data = data[key][T_slice].astype(np.float32)
            # RGB Normalization (0-255 → 0-1)
            pc_data[..., 3:6] = pc_data[..., 3:6] / 255.0
            obs_dict[key] = pc_data
            # save ram
            del data[key]
            
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_pointcloud_replay_buffer(dataset_path, shape_meta, store, apply_preprocessing, use_cuda):
    # parse shape meta
    pointcloud_keys = list()
    lowdim_keys = list()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    n_points = None
    
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        
        if type == 'pointcloud':
            pointcloud_keys.append(key)
            # pointcloud is (N_points, 6) shape
            assert len(shape) == 2 and shape[1] == 6, f"PointCloud shape must be (N, 6), got {shape}"
            n_points = shape[0]
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,),(7,)]  # adding 7D support
    
    action_shape = tuple(shape_meta['action']['shape'])
    assert action_shape in [(2,),(6,),(7,)]  # adding 7D support

    # load data
    with threadpool_limits(1):
        replay_buffer = real_pointcloud_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            lowdim_keys=lowdim_keys + ['action'],
            pointcloud_keys=pointcloud_keys,
            n_points=n_points,
            apply_preprocessing=apply_preprocessing,
            use_cuda=use_cuda
        )

    # transform lowdim dimensions - 7D action
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    elif action_shape == (6,):
        # 6D action space, 6DOF pose
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1,2,3,4,5])
    # 7D (6DOF + gripper)
    
    for key, shape in lowdim_shapes.items():
        if 'pose' in key:
            if shape == (2,):
                # only take X and Y
                zarr_arr = replay_buffer[key]
                zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
            elif shape == (6,):
                # 6DOF pose
                zarr_arr = replay_buffer[key]
                zarr_resize_index_last_dim(zarr_arr, idxs=[0,1,2,3,4,5])

    return replay_buffer
