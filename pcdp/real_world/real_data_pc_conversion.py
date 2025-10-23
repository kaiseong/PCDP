# real_data_pc_conversion.py
from typing import Sequence, Tuple, Dict, Optional, Union, List
import os
import pathlib
import numpy as np
import zarr
import numcodecs
from tqdm import tqdm
import open3d as o3d
import torch
import pcdp.common.mono_time as mono_time
try:
    import pytorch3d.ops as torch3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. FPS will use fallback method.")
from pcdp.common import RISE_transformation as rise_tf
from pcdp.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from pcdp.common.RISE_transformation import xyz_rot_transform

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

class PointCloudPreprocessor:
    """Pointcloud preprocessing with optional temporal memory for fused cache export.
    When enable_temporal=True and export_mode='fused', each call to `process(points)`
    updates a stateful voxel map (per episode) with exponential decay and returns
    an Nx7 array: [x,y,z,r,g,b,c], where `c` is temporal confidence (recency).
    """
    def __init__(self, 
                enable_sampling=False,
                target_num_points=1024,
                enable_transform=True,
                extrinsics_matrix=None,
                enable_cropping=True,
                workspace_bounds=None,
                enable_filter=False,
                nb_points=10,
                sor_std=1.7,
                use_cuda=True,
                verbose=False,
                enable_temporal=False,
                export_mode='off',
                temporal_voxel_size=0.005,
                temporal_decay=0.90,
                temporal_c_min=0.20,
                temporal_max_points=None,
                ):

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
                [-0.000, 0.715],    # X range (m)
                [-0.400, 0.350],    # Y range (m)
                [-0.100, 0.400]     # z range
            ]
        else:
            self.workspace_bounds = workspace_bounds
        
            
        self.target_num_points = target_num_points
        self.nb_points = nb_points
        self.sor_std = sor_std
        self.enable_transform = enable_transform
        self.enable_cropping = enable_cropping
        self.enable_sampling = enable_sampling
        self.enable_filter = enable_filter
        self.use_cuda = use_cuda and torch.cuda.is_available() and PYTORCH3D_AVAILABLE
        self.verbose = verbose
        
        # temporal memory
        self.enable_temporal = enable_temporal
        self.export_mode = export_mode
        self._temporal_voxel_size = float(temporal_voxel_size)
        self._temporal_decay = float(temporal_decay)
        self._temporal_c_min = float(temporal_c_min)
        self._temporal_max_points = temporal_max_points
        
        self._frame_idx = 0
        self._mem = {}

        if self.verbose:
            print(f"PointCloudPreprocessor initialized:")
            print(f"  - Transform: {self.enable_transform}")
            print(f"  - Cropping: {self.enable_cropping}")
            print(f"  - Sampling: {self.enable_sampling} (target: {self.target_num_points})")
            print(f"  - CUDA: {self.use_cuda}")
            
    def __call__(self, points):
        """Process pointcloud through the full pipeline."""
        return self.process(points)
    
    def reset_temporal(self):
        """Call at the start of each episode."""
        self._mem.clear()
        self._frame_idx = 0

    def _key_from_xyz(self, xyz: np.ndarray):
        return tuple(np.floor(xyz / self._temporal_voxel_size).astype(np.int32).tolist())

    def _decay_and_prune(self, now_step: int):
        if not self._mem: 
            return
        drop = []
        for k, (xyz, rgb, c, s) in self._mem.items():
            dt = max(0, now_step - s)
            c_new = self._temporal_decay ** dt
            if c_new < self._temporal_c_min:
                drop.append(k)
            else:
                self._mem[k] = (xyz, rgb, c_new, s)
        for k in drop:
            self._mem.pop(k, None)
    
    def _export_array_from_mem(self) -> np.ndarray:
        if not self._mem:
            return np.zeros((0,7), dtype=np.float32)
        arr = np.asarray([[*xyz, *rgb, c] for (xyz, rgb, c, _) in self._mem.values()], dtype=np.float32)
        # optional global budget
        if self._temporal_max_points is not None and len(arr) > self._temporal_max_points:
            idx = np.random.choice(len(arr), self._temporal_max_points, replace=False)
            arr = arr[idx]
        return arr
        
    def process(self, points):
        """
        Process pointcloud through transformation, cropping, and sampling.
        
        Args:
            points: numpy array of shape (N, 6) containing XYZRGB data
            
        Returns:
            processed_points: numpy array of processed pointcloud
        """
        import pcdp.common.mono_time as mono_time
        if points is None or len(points) == 0:
            # temporal 켜져 있으면 (0,7), 아니면 기존처럼 (target_num_points, 6)
            if self.enable_temporal and self.export_mode == 'fused':
                self._frame_idx += 1
                return np.zeros((0,7), dtype=np.float32)
            return np.zeros((self.target_num_points, 6), dtype=np.float32)
            
        # Ensure points is float32
        points = points.astype(np.float32)

        # Coordinate transformation
        if self.enable_transform:
            points = self._apply_transform(points)
        # Workspace cropping
        if self.enable_cropping:
            points = self._crop_workspace(points)
        if self.enable_filter:
            points = self._apply_filter(points)
        if not self.enable_temporal or self.export_mode=="off":
            # Point FPS sampling
            if self.enable_sampling:
                points = self._sample_points(points)
            return points
        
        t0 = mono_time.now_ms()
        now_step = self._frame_idx
        self._decay_and_prune(now_step)

        for i in range(len(points)):
            xyz = points[i, :3]
            rgb = points[i, 3:6]
            k = self._key_from_xyz(xyz)
            self._mem[k] = (xyz.copy(), rgb.copy(), 1.0, now_step)
        
        out = self._export_array_from_mem()
        
        self._frame_idx += 1
        print(f"cost time: {mono_time.now_ms()-t0}")
        return out


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

    def _apply_filter(self, points):
        if len(points) == 0:
            raise ValueError("points empty")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=self.nb_points, std_ratio=self.sor_std)
        return points[ind]
        
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

class LowDimPreprocessor:
    def __init__(self,
                 robot_to_base=None
                ):
        if robot_to_base is None:
            self.robot_to_base=np.array([
                [1., 0., 0., -0.04],
                [0., 1., 0., -0.29],
                [0., 0., 1., -0.03],
                [0., 0., 0.,  1.0]
            ])
        else:
            self.robot_to_base = np.array(robot_to_base, dtype=np.float32)
        
    
    def TF_process(self, robot_7ds):
        assert robot_7ds.shape[-1] == 7, f"robot_7ds data shape shoud be (..., 7), but got {robot_7ds.shape}"
        processed_robot7d = []
        for robot_7d in robot_7ds:
            pose_6d = robot_7d[:6]
            gripper = robot_7d[6]
            
            translation = pose_6d[:3]
            rotation = pose_6d[3:6]
            eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)
            T_k_matrix = self.robot_to_base @ eef_to_robot_base_k
            transformed_pose_6d = rise_tf.mat_to_xyz_rot(
                T_k_matrix,
                rotation_rep='euler_angles',
                rotation_rep_convention='ZYX'
            )
            new_robot_7d = np.concatenate([transformed_pose_6d, [gripper]])
            processed_robot7d.append(new_robot_7d)
        
        return np.array(processed_robot7d, dtype=np.float32)




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



def process_single_episode(episode_path, pc_preprocessor=None, lowdim_preprocessor=None, downsample_factor=3):
    """
    Process a single episode: load, downsample, align, and optionally preprocess.
    
    Args:
        episode_path: Path to episode directory
        pc_preprocessor: Optional PointCloudPreprocessor instance
        lowdim_preprocessor: Optional LowDimPreprocessor instance
        downsample_factor: Factor for downsampling obs data
        
    Returns:
        episode_data: Dictionary containing processed episode data
    """

    episode_path = pathlib.Path(episode_path)

    if pc_preprocessor is not None and hasattr(pc_preprocessor, "reset_temporal"):
        pc_preprocessor.reset_temporal()
    
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
    if pc_preprocessor is not None and 'pointcloud' in aligned_obs:
        processed_pointclouds = []
        for pc in aligned_obs['pointcloud']:
            processed_pc = pc_preprocessor.process(pc)
            processed_pointclouds.append(processed_pc)
        aligned_obs['pointcloud'] = np.array(processed_pointclouds, dtype=object)
    
    
    # Create robot_obs by concatenating pose and gripper width
    robot_eef_pose = aligned_obs['robot_eef_pose']
    robot_gripper_width = aligned_obs['robot_gripper'][:, :1] # Keep it as a column vector
    aligned_obs['robot_obs'] = np.concatenate([robot_eef_pose, robot_gripper_width], axis=1) 
    

    # TF to based on origin frames
    if lowdim_preprocessor is not None:
        aligned_obs['robot_obs'] = lowdim_preprocessor.TF_process(aligned_obs['robot_obs'])
        aligned_action['action'] = lowdim_preprocessor.TF_process(aligned_action['action'])
    
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
        pc_preprocessor: Optional[PointCloudPreprocessor] = None,
        lowdim_preprocessor: Optional[LowDimPreprocessor] = None,
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
        pc_preprocessor: Optional pointcloud preprocessor
        lowdim_preprocessor: Optional lowdim preprocessor
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
                    episode_dir, pc_preprocessor, lowdim_preprocessor, downsample_factor)
                
                if episode_data is not None:
                    # Validate episode data against shape_meta
                    if validate_episode_data_with_shape_meta(episode_data, shape_meta):
                        # Ensure all data are numpy arrays before adding to buffer
                        for key in episode_data.keys():
                            if isinstance(episode_data[key], list):
                                episode_data[key] = np.asarray(episode_data[key])
                        
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