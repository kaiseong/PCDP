# real_data_pc_conversion.py
from typing import Sequence, Tuple, Dict, Optional, Union, List
import os
import pathlib
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm

import torch
try:
    import pytorch3d.ops as torch3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. FPS will use fallback method.")
from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.replay_buffer import ReplayBuffer, get_optimal_chunks

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

class PointCloudPreprocessor:
    """Pointcloud preprocessing class with coordinate transformation, cropping, and sampling."""
    def __init__(self, 
                extrinsics_matrix=None,
                workspace_bounds=None,
                rgb_mean=None,
                rgb_std=None,
                target_num_points=1024,
                enable_transform=True,
                enable_cropping=True,
                enable_sampling=True,
                enable_normalize=True,
                use_cuda=True,
                verbose=False):
        """
        Initialize pointcloud preprocessor.
        
        Args:
            extrinsics_matrix: 4x4 transformation matrix from camera to robot coordinates
            workspace_bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            target_num_points: Target number of points after sampling
            enable_transform: Whether to apply coordinate transformation
            enable_cropping: Whether to crop to workspace
            enable_sampling: Whether to apply farthest point sampling
            use_cuda: Whether to use GPU acceleration
            verbose: Whether to print debug information
        """
        # Default extrinsics matrix (camera to robot transform)
        if extrinsics_matrix is None:
            self.extrinsics_matrix = np.array([
                [  0.007131,  -0.91491,    0.403594,  0.05116],
                [ -0.994138,   0.003833,   0.02656,  -0.00918],
                [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
                [  0.,         0. ,        0. ,        1.      ]
            ])
        else:
            self.extrinsics_matrix = np.array(extrinsics_matrix)
            
        # Default workspace bounds
        if workspace_bounds is None:
            self.workspace_bounds = [
                [-0.000, 0.740],    # X range (m)
                [-0.400, 0.350],    # Y range (m)
                [-0.100, 0.400]     # z range
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
        if rgb_mean is None:
            self.rgb_mean = np.array([0.1234, 0.1234, 0.1234])
        else:
            self.rgb_mean=rgb_mean

        if rgb_std is None:
            self.rgb_std = np.array([0.2620, 0.2710, 0.2709])
        else:
            self.rgb_std=rgb_std
            
        self.target_num_points = target_num_points
        self.enable_transform = enable_transform
        self.enable_cropping = enable_cropping
        self.enable_sampling = enable_sampling
        self.enable_normalize = enable_normalize
        self.use_cuda = use_cuda and torch.cuda.is_available() and PYTORCH3D_AVAILABLE
        self.verbose = verbose
        
        if self.verbose:
            print(f"PointCloudPreprocessor initialized:")
            print(f"  - Transform: {self.enable_transform}")
            print(f"  - Cropping: {self.enable_cropping}")
            print(f"  - Sampling: {self.enable_sampling} (target: {self.target_num_points})")
            print(f"  - CUDA: {self.use_cuda}")
            
    def __call__(self, points):
        """Process pointcloud through the full pipeline."""
        return self.process(points)
        
    def process(self, points):
        """
        Process pointcloud through transformation, cropping, and sampling.
        
        Args:
            points: numpy array of shape (N, 6) containing XYZRGB data
            
        Returns:
            processed_points: numpy array of processed pointcloud
        """
        import pcdp.common.mono_time as mono_time
        if len(points) == 0:
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        # Ensure points is float32
        points = points.astype(np.float32)

        if self.enable_normalize:
            points = self._apply_normalize(points)
        # Coordinate transformation
        if self.enable_transform:
            points = self._apply_transform(points)
        # Workspace cropping
        if self.enable_cropping:
            points = self._crop_workspace(points)
        # Point FPS sampling
        if self.enable_sampling:
            points = self._sample_points(points)
        return points

    def _apply_normalize(self, points):
        points[:, 3:6] = points[:, 3:6] / 255.0
        points[:, 3:6] = (points[:, 3:6] - self.rgb_mean) / self.rgb_std

        if self.verbose and len(points) > 0:
            print(f"mean: {points[:, 3:6].mean()}")

        return points
    def _apply_transform(self, points):
        """Apply extrinsics transformation and scaling."""
        # Scale from mm to m (Orbbec specific)
        point_xyz = points[:, :3] * 0.001
        
        # Apply extrinsics transformation
        point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
        point_transformed = np.dot(point_homogeneous, self.extrinsics_matrix.T)
        
        # Update XYZ coordinates
        points[:, :3] = point_transformed[:, :3]
        
        if self.verbose and len(points) > 0:
            print(f"After transform: {len(points)} points, "
                  f"XYZ range: [{points[:, :3].min(axis=0)} - {points[:, :3].max(axis=0)}]")
        
        return points
        
    def _crop_workspace(self, points):
        """Crop points to workspace bounds."""
        if len(points) == 0:
            return points
            
        mask = (
            (points[:, 0] >= self.workspace_bounds[0][0]) & 
            (points[:, 0] <= self.workspace_bounds[0][1]) &
            (points[:, 1] >= self.workspace_bounds[1][0]) & 
            (points[:, 1] <= self.workspace_bounds[1][1]) &
            (points[:, 2] >= self.workspace_bounds[2][0]) & 
            (points[:, 2] <= self.workspace_bounds[2][1])
        )
        
        cropped_points = points[mask]
        
        if self.verbose:
            print(f"After cropping: {len(cropped_points)}/{len(points)} points remain")
            
        return cropped_points
        
    def _sample_points(self, points):
        """Apply farthest point sampling to reduce number of points."""
        if len(points) == 0:
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        if len(points) <= self.target_num_points:
            # Pad with zeros if not enough points
            padded_points = np.zeros((self.target_num_points, 6), dtype=np.float32)
            padded_points[:len(points)] = points
            return padded_points
            
        try:
            # Use farthest point sampling
            points_xyz = points[:, :3]
            sampled_xyz, sample_indices = self._farthest_point_sampling(
                points_xyz, self.target_num_points)
                
            # Reconstruct full points with RGB
            if self.use_cuda:
                sample_indices = sample_indices.cpu().numpy().flatten()
            else:
                sample_indices = sample_indices.numpy().flatten()
                
            sampled_points = points[sample_indices]
            
            if self.verbose:
                print(f"After sampling: {len(sampled_points)} points")
                
            return sampled_points
            
        except Exception as e:
            if self.verbose:
                print(f"FPS failed: {e}, using random sampling")
            # Fallback to random sampling
            indices = np.random.choice(len(points), self.target_num_points, replace=False)
            return points[indices]
            
    def _farthest_point_sampling(self, points, num_points):
        """Apply farthest point sampling using pytorch3d."""
        if not PYTORCH3D_AVAILABLE:
            raise ImportError("pytorch3d not available")
            
        points_tensor = torch.from_numpy(points)
        
        if self.use_cuda:
            points_tensor = points_tensor.cuda()
            
        # pytorch3d expects batch dimension
        points_batch = points_tensor.unsqueeze(0)
        
        # Apply FPS
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points_batch, K=[num_points])
            
        # Remove batch dimension
        sampled_points = sampled_points.squeeze(0)
        indices = indices.squeeze(0)
        
        if self.use_cuda:
            sampled_points = sampled_points.cpu()
            
        return sampled_points.numpy(), indices


def create_default_preprocessor(target_num_points=1024, use_cuda=True, verbose=False):
    """Create a preprocessor with default settings."""
    return PointCloudPreprocessor(
        target_num_points=target_num_points,
        use_cuda=use_cuda,
        verbose=verbose
    )


def downsample_obs_data(obs_data, downsample_factor=3):
    """
    Downsample observation data by taking every Nth sample.
    
    Args:
        obs_data: Dictionary of observation arrays
        downsample_factor: Factor to downsample by (e.g., 3 for 30Hz->10Hz)
        
    Returns:
        downsampled_data: Dictionary with downsampled arrays
    """
    downsampled_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            downsampled_data[key] = value[::downsample_factor].copy()
        else:
            downsampled_data[key] = value
    return downsampled_data


def align_obs_action_data(obs_data, action_data, obs_timestamps, action_timestamps):
    """
    Align observation and action data based on timestamps.
    For each obs timestamp, find the first action timestamp that comes after it.
    
    Args:
        obs_data: Dictionary of observation arrays
        action_data: Dictionary of action arrays  
        obs_timestamps: Array of observation timestamps
        action_timestamps: Array of action timestamps
        
    Returns:
        aligned_obs_data: Dictionary of aligned observation data
        aligned_action_data: Dictionary of aligned action data
        valid_indices: Indices where alignment was successful
    """
    valid_indices = []
    aligned_action_indices = []
    
    for i, obs_ts in enumerate(obs_timestamps):
        # Find first action timestamp >= obs timestamp
        future_actions = action_timestamps >= obs_ts
        if np.any(future_actions):
            action_idx = np.where(future_actions)[0][0]
            valid_indices.append(i)
            aligned_action_indices.append(action_idx)
    
    if len(valid_indices) == 0:
        print("Warning: No valid obs-action alignments found!")
        return {}, {}, []
    
    # Filter obs data to only valid indices
    aligned_obs_data = {}
    for key, value in obs_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_obs_data[key] = value[valid_indices]
        else:
            aligned_obs_data[key] = value
            
    # Filter action data to aligned indices
    aligned_action_data = {}
    for key, value in action_data.items():
        if isinstance(value, np.ndarray) and len(value.shape) > 0:
            aligned_action_data[key] = value[aligned_action_indices]
        else:
            aligned_action_data[key] = value
    # print(f"Aligned {len(valid_indices)} obs-action pairs from {len(obs_timestamps)} obs and {len(action_timestamps)} actions")
    return aligned_obs_data, aligned_action_data, valid_indices



def process_single_episode(episode_path, preprocessor=None, downsample_factor=3):
    """
    Process a single episode: load, downsample, align, and optionally preprocess.
    
    Args:
        episode_path: Path to episode directory
        preprocessor: Optional PointCloudPreprocessor instance
        downsample_factor: Factor for downsampling obs data
        
    Returns:
        episode_data: Dictionary containing processed episode data
    """

    episode_path = pathlib.Path(episode_path)
    
    obs_zarr_path = episode_path / 'obs_replay_buffer.zarr'
    action_zarr_path = episode_path / 'action_replay_buffer.zarr'
    
    if not obs_zarr_path.exists() or not action_zarr_path.exists():
        raise FileNotFoundError(f"Missing zarr files in {episode_path}")
    
    obs_replay_buffer = ReplayBuffer.create_from_path(str(obs_zarr_path), mode='r')
    action_replay_buffer = ReplayBuffer.create_from_path(str(action_zarr_path), mode='r')

    obs_data ={}
    for key in obs_replay_buffer.keys():
        obs_data[key] = obs_replay_buffer[key][:]
    
    action_data ={}
    for key in action_replay_buffer.keys():
        action_data[key] = action_replay_buffer[key][:]

    # Downsample obs data from 30Hz to 10Hz
    downsampled_obs = downsample_obs_data(obs_data, downsample_factor)
    downsampled_obs_timestamps = downsampled_obs['align_timestamp']
    action_timestamps = action_data['timestamp']
    
    # Align obs and action data based on timestamps
    aligned_obs, aligned_action, valid_indices = align_obs_action_data(
        downsampled_obs, action_data, 
        downsampled_obs_timestamps, action_timestamps)
    
    if len(valid_indices) == 0:
        return None
        
    # Apply pointcloud preprocessing if provided
    if preprocessor is not None and 'pointcloud' in aligned_obs:
        processed_pointclouds = []
        for pc in aligned_obs['pointcloud']:
            processed_pc = preprocessor.process(pc)
            processed_pointclouds.append(processed_pc)
        aligned_obs['pointcloud'] = np.array(processed_pointclouds, dtype=object)
    
    if preprocessor is not None:
        original_actions = aligned_action['action']
        processed_actions = []
        
        for action_7d in original_actions:
            pose_6d = action_7d[:6]
            gripper = action_7d[6]

            translation = pose_6d[:3]
            rotation = pose_6d[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)

            T_k_matrix = robot_to_base @ eef_to_robot_base_k
            transformed_pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='XYZ'
            )

            new_action_7d = np.concatenate([transformed_pose_6d, [gripper]])
            processed_actions.append(new_action_7d)
        
        aligned_action['action'] = np.array(processed_actions, dtype=np.float32)
    
    # Combine obs and action data
    episode_data = {}
    episode_data.update(aligned_obs)
    episode_data.update(aligned_action)
    
    return episode_data


def parse_shape_meta(shape_meta: dict) -> Tuple[List[str], List[str], dict, dict]:
    """
    Parse shape_meta to extract pointcloud keys, lowdim keys, and their configurations.
    
    Args:
        shape_meta: Shape metadata dictionary from config
        
    Returns:
        pointcloud_keys: List of pointcloud observation keys
        lowdim_keys: List of lowdim observation keys  
        pointcloud_configs: Configuration for each pointcloud key
        lowdim_configs: Configuration for each lowdim key
    """
    pointcloud_keys = []
    lowdim_keys = []
    pointcloud_configs = {}
    lowdim_configs = {}
    
    # Parse obs shape meta
    obs_shape_meta = shape_meta.get('obs', {})
    for key, attr in obs_shape_meta.items():
        obs_type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape', []))
        
        if obs_type == 'pointcloud':
            pointcloud_keys.append(key)
            pointcloud_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
        elif obs_type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_configs[key] = {
                'shape': shape,
                'type': obs_type
            }
    
    return pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs


def validate_episode_data_with_shape_meta(episode_data: dict, shape_meta: dict) -> bool:
    """
    Validate that episode data matches the expected shape_meta.
    
    Args:
        episode_data: Processed episode data
        shape_meta: Expected shape metadata
        
    Returns:
        bool: True if validation passes
    """
    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    # Validate pointcloud data
    for key in pointcloud_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = pointcloud_configs[key]['shape']
            if len(data.shape) >= 2:
                # Check that last dimensions match expected shape
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected pointcloud key '{key}' not found in episode data")
            return False
    
    # Validate lowdim data
    for key in lowdim_keys:
        if key in episode_data:
            data = episode_data[key]
            expected_shape = lowdim_configs[key]['shape']
            if len(expected_shape)==1:
                if expected_shape[0] == 1 and len(data.shape) == 1:
                    continue

            if len(data.shape) >= 1:
                # Check that last dimensions match expected shape
                if data.shape[-len(expected_shape):] != expected_shape:
                    print(f"Warning: {key} shape mismatch. Expected: {expected_shape}, Got: {data.shape}")
                    return False
        else:
            print(f"Warning: Expected lowdim key '{key}' not found in episode data")
            return False
    
    # Validate action data
    action_shape_meta = shape_meta.get('action', {})
    if 'action' in episode_data and 'shape' in action_shape_meta:
        expected_action_shape = tuple(action_shape_meta['shape'])
        actual_action_shape = episode_data['action'].shape[-len(expected_action_shape):]
        if actual_action_shape != expected_action_shape:
            print(f"Warning: Action shape mismatch. Expected: {expected_action_shape}, Got: {actual_action_shape}")
            return False
    
    return True


def _get_replay_buffer(
        dataset_path: str,
        shape_meta: dict,
        store: Optional[zarr.ABSStore] = None,
        preprocessor: Optional[PointCloudPreprocessor] = None,
        downsample_factor: int = 3,
        max_episodes: Optional[int] = None,
        n_workers: int = 1
) -> ReplayBuffer:
    """
    Convert Piper demonstration data to ReplayBuffer format.
    
    Args:
        dataset_path: Path to recorder_data directory containing episode folders
        shape_meta: Dictionary defining observation and action shapes/types
        store: Zarr store for output (if None, uses MemoryStore)
        preprocessor: Optional pointcloud preprocessor
        downsample_factor: Factor for downsampling obs data (30Hz -> 10Hz = 3)
        max_episodes: Maximum number of episodes to process
        n_workers: Number of worker processes (currently unused)
        
    Returns:
        replay_buffer: ReplayBuffer containing processed data
    """
    if store is None:
        store = zarr.MemoryStore()
        
    dataset_path = pathlib.Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Parse shape_meta to understand data structure
    pointcloud_keys, lowdim_keys, pointcloud_configs, lowdim_configs = parse_shape_meta(shape_meta)
    
    print(f"Parsed shape_meta:")
    print(f"  - Pointcloud keys: {pointcloud_keys}")
    print(f"  - Lowdim keys: {lowdim_keys}")
    print(f"  - Action shape: {shape_meta.get('action', {}).get('shape', 'undefined')}")
    
    # Find all episode directories
    episode_dirs = []
    for item in sorted(dataset_path.iterdir()):
        if item.is_dir() and item.name.startswith('episode_'):
            episode_dirs.append(item)
            
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]
        
    print(f"Found {len(episode_dirs)} episodes to process")
    
    if len(episode_dirs) == 0:
        raise ValueError("No episode directories found")
    
    # Create ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr(storage=store)
    
    # Process episodes
    with tqdm(total=len(episode_dirs), desc="Processing episodes", mininterval=1.0) as pbar:
        for episode_dir in episode_dirs:
            try:
                episode_data = process_single_episode(
                    episode_dir, preprocessor, downsample_factor)
                
                if episode_data is not None:
                    # Validate episode data against shape_meta
                    if validate_episode_data_with_shape_meta(episode_data, shape_meta):
                        # Add episode to replay buffer
                        replay_buffer.add_episode(episode_data,
                            object_codecs={'pointcloud': numcodecs.Pickle()})
                        pbar.set_postfix(
                            episodes=replay_buffer.n_episodes,
                            steps=replay_buffer.n_steps
                        )
                    else:
                        print(f"Skipping episode {episode_dir.name} due to shape validation failure")
                else:
                    print(f"Skipping empty episode: {episode_dir.name}")
                    
            except Exception as e:
                print(f"Error processing {episode_dir.name}: {e}")
                continue
                
            pbar.update(1)
    
    print(f"Successfully processed {replay_buffer.n_episodes} episodes "
        f"with {replay_buffer.n_steps} total steps")
    
    return replay_buffer