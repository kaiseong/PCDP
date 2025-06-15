import os
import time
import numpy as np
import threading
from collections import deque
import zarr
import numcodecs
import logging
from typing import Optional, Union
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs


class PointCloudRecorder:
    def __init__(self, compression_level=3):
        self.compression_level = compression_level
        
        self.recording = False
        self.ready = False
        self.lock = threading.Lock()
        
        # save parameter
        self.output_path = None
        self.start_time = None
        self.next_global_idx = 0
        self.frame_count = 0
        
        # Zarr parameter
        self.zarr_store = None
        self.zarr_root = None
        self.points_dataset = None
        self.timestamps_dataset = None
        
        self.logger = logging.getLogger(__name__)

    def start(self, file_path: str, start_time: Optional[float] = None):
        with self.lock:
            if self.recording:
                self.logger.warning("Already recording")
                return
                
            self.output_path = file_path
            self.start_time = start_time if start_time is not None else time.time()
            self.next_global_idx = 0
            self.frame_count = 0
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            self._init_zarr_store()
            
            self.recording = True
            self.ready = True
            
            self.logger.info(f"PointCloudRecorder started: {file_path}")

    def _init_zarr_store(self):
        self.zarr_store = zarr.DirectoryStore(self.output_path)
        self.zarr_root = zarr.open(store=self.zarr_store, mode='w')
        
        compressor = numcodecs.Blosc(
            cname='zstd', 
            clevel=self.compression_level, 
            shuffle=numcodecs.Blosc.SHUFFLE
        )
        
        self.points_dataset = self.zarr_root.create_dataset(
            'points',
            shape=(0, 6),  # (N_points, 6)
            chunks=(30000, 6),
            dtype=np.float32,
            compressor=compressor,
            fill_value=0.0
        )
        
        self.timestamps_dataset = self.zarr_root.create_dataset(
            'timestamps',
            shape=(0,),
            chunks=(1000,),
            dtype=np.float64,
            compressor=compressor
        )
        
        self.frame_indices_dataset = self.zarr_root.create_dataset(
            'frame_indices',
            shape=(0,),
            chunks=(1000,),
            dtype=np.int32,
            compressor=compressor
        )

    def write_frame(self, pointcloud: np.ndarray, frame_time: Optional[float] = None):
        if not self.is_ready():
            return
            
        if frame_time is None:
            frame_time = time.time()
            
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 6:
            self.logger.error(f"Invalid pointcloud shape: {pointcloud.shape}")
            return
            
        # valid points filtering (NaN, Inf remove)
        valid_mask = np.isfinite(pointcloud).all(axis=1)
        valid_pointcloud = pointcloud[valid_mask].astype(np.float32)
        
        
        with self.lock:
            self._write_pointcloud_to_zarr(valid_pointcloud, frame_time)
            self.frame_count += 1

    def _write_pointcloud_to_zarr(self, pointcloud: np.ndarray, timestamp: float):
        n_points = len(pointcloud)
        
        old_points_len = self.points_dataset.shape[0]
        old_frames_len = self.timestamps_dataset.shape[0]
        
        self.points_dataset.resize((old_points_len+n_points, 6))
        self.timestamps_dataset.resize((old_frames_len+1,))
        self.frame_indices_dataset.resize((old_frames_len+1,))
        
        # save data
        self.points_dataset[old_points_len:old_points_len + n_points] = pointcloud
        self.timestamps_dataset[old_frames_len] = timestamp
        self.frame_indices_dataset[old_frames_len] = n_points

    def stop(self):
        with self.lock:
            if not self.recording:
                return
                
            self.recording = False
            self.ready = False
            
            if self.zarr_store is not None:
                self.zarr_store.close()
                self.zarr_store = None
                
            self.logger.info(f"PointCloudRecorder stopped. Total frames: {self.frame_count}")

    def is_ready(self) -> bool:
        return self.ready and self.recording

    @classmethod
    def create_default(cls, fps=30, **kwargs):
        return cls(fps=fps, **kwargs)

