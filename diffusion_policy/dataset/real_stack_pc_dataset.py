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
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BasePointCloudDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.real_world.real_data_pc_conversion import _get_replay_buffer, PointCloudPreprocessor
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

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
            # Preprocessing parameters (configurable via YAML)
            enable_preprocessing=True,
            preprocessor_config=None
        ):
        """
        Pointcloud-based dataset for Piper robot demonstrations.
        
        Args:
            shape_meta: Dictionary defining the shape and type of observations and actions
            dataset_path: Path to the recorder_data directory containing episode folders
            horizon: Number of action steps to predict
            pad_before: Number of steps to pad before the sequence
            pad_after: Number of steps to pad after the sequence
            n_obs_steps: Number of observation steps to use (if None, use all)
            n_latency_steps: Number of latency steps to account for
            use_cache: Whether to use caching for faster loading
            seed: Random seed for reproducibility
            val_ratio: Ratio of episodes to use for validation
            max_train_episodes: Maximum number of training episodes to use
            enable_preprocessing: Whether to enable pointcloud preprocessing
            preprocessor_config: Configuration for pointcloud preprocessing
        """
        assert os.path.isdir(dataset_path), f"Dataset path does not exist: {dataset_path}"
        
        # Setup preprocessor if enabled
        preprocessor = None
        if enable_preprocessing:
            preprocessor = self._create_preprocessor(preprocessor_config or {})
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta for cache validation
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exist
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore(),
                            preprocessor=preprocessor
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
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore(),
                preprocessor=preprocessor
            )
        
        # Parse shape meta to identify pointcloud and lowdim keys dynamically
        pointcloud_keys = list()
        lowdim_keys = list()
        
        obs_shape_meta = shape_meta.get('obs', {})
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get('type', 'low_dim')
            if obs_type == 'pointcloud':
                pointcloud_keys.append(key)
            elif obs_type == 'low_dim':
                lowdim_keys.append(key)
                
        print(f"Parsed observation keys from shape_meta:")
        print(f"  - Pointcloud keys: {pointcloud_keys}")
        print(f"  - Low-dim keys: {lowdim_keys}")

        # Validate that replay buffer contains expected keys
        self._validate_replay_buffer_keys(replay_buffer, pointcloud_keys, lowdim_keys)
        
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

    
    def _validate_replay_buffer_keys(self, replay_buffer: ReplayBuffer, pointcloud_keys: List[str], lowdim_keys: List[str]):
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

    def _create_preprocessor(self, config: dict):
        """
        Create pointcloud preprocessor from configuration.
        
        Args:
            config: Preprocessor configuration dictionary
            
        Returns:
            PointCloudPreprocessor instance
        """
        
        
        return PointCloudPreprocessor(
            extrinsics_matrix=config.get('extrinsics_matrix'),
            workspace_bounds=config.get('workspace_bounds'),
            target_num_points=config.get('target_num_points', 1024),
            enable_transform=config.get('enable_transform', True),
            enable_cropping=config.get('enable_cropping', True),
            enable_sampling=config.get('enable_sampling', True),
            use_cuda=config.get('use_cuda', True),
            verbose=config.get('verbose', False)
        )

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

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # lowdim obs - use standard normalization
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # pointcloud - use identity normalizer (no normalization)
        # 실제 거리를 정규화하면, 거리 정보를 온전히 이해 못할 수도 있음, => 강인성에서 멀어짐
        for key in self.pointcloud_keys:
            # Get stats but don't actually normalize pointcloud data
            # stats = array_to_stats(self.replay_buffer[key][:])
            # normalizer[key] = get_identity_normalizer_from_stat(stats)
            normalizer[key] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        
        # process pointcloud data
        for key in self.pointcloud_keys:
            # Keep pointcloud as T,N,6 (time, points, XYZRGB)
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
            
        # process lowdim data    
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data