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
    def __init__(self, fps=30, compression_level=3):
        """
        VideoRecorder와 동일한 인터페이스로 설계된 PointCloudRecorder
        """
        self.fps = fps
        self.dt = 1.0 / fps
        self.compression_level = compression_level
        
        # VideoRecorder와 동일한 상태 관리
        self.recording = False
        self.ready = False
        self.lock = threading.Lock()
        
        # 저장 관련
        self.output_path = None
        self.start_time = None
        self.next_global_idx = 0
        self.frame_count = 0
        
        # Zarr 관련
        self.zarr_store = None
        self.zarr_root = None
        self.points_dataset = None
        self.timestamps_dataset = None
        
        self.logger = logging.getLogger(__name__)

    def start(self, file_path: str, start_time: Optional[float] = None):
        """VideoRecorder.start()와 동일한 인터페이스"""
        with self.lock:
            if self.recording:
                self.logger.warning("Already recording")
                return
                
            self.output_path = file_path
            self.start_time = start_time if start_time is not None else time.time()
            self.next_global_idx = 0
            self.frame_count = 0
            
            # 출력 디렉토리 생성
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Zarr 스토어 초기화
            self._init_zarr_store()
            
            self.recording = True
            self.ready = True
            
            self.logger.info(f"PointCloudRecorder started: {file_path}")

    def _init_zarr_store(self):
        """Zarr 저장소 초기화"""
        self.zarr_store = zarr.DirectoryStore(self.output_path)
        self.zarr_root = zarr.open(store=self.zarr_store, mode='w')
        
        # 압축 설정
        compressor = numcodecs.Blosc(
            cname='zstd', 
            clevel=self.compression_level, 
            shuffle=numcodecs.Blosc.SHUFFLE
        )
        
        # 데이터셋 생성 
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
        
        # 프레임 인덱스 (각 프레임이 몇 개의 포인트를 가지는지)
        self.frame_indices_dataset = self.zarr_root.create_dataset(
            'frame_indices',
            shape=(0,),
            chunks=(1000,),
            dtype=np.int32,
            compressor=compressor
        )

    def write_frame(self, pointcloud: np.ndarray, frame_time: Optional[float] = None):
        """
        VideoRecorder.write_frame()과 동일한 인터페이스
        기존 get_accumulate_timestamp_idxs 함수 사용
        """
        if not self.is_ready():
            return
            
        if frame_time is None:
            frame_time = time.time()
            
        # 포인트클라우드 검증 및 전처리
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 6:
            self.logger.error(f"Invalid pointcloud shape: {pointcloud.shape}")
            return
            
        # 유효한 포인트만 필터링 (NaN, Inf 제거)
        valid_mask = np.isfinite(pointcloud).all(axis=1)
        valid_pointcloud = pointcloud[valid_mask].astype(np.float32)
        
        
        with self.lock:
            self._write_pointcloud_to_zarr(valid_pointcloud, frame_time)
            self.frame_count += 1

    def _write_pointcloud_to_zarr(self, pointcloud: np.ndarray, timestamp: float):
        """실제 Zarr에 포인트클라우드 저장"""
        n_points = len(pointcloud)
        
        # 데이터셋 크기 확장
        old_points_len = self.points_dataset.shape[0]
        old_frames_len = self.timestamps_dataset.shape[0]
        
        self.points_dataset.resize((old_points_len+n_points, 6))
        self.timestamps_dataset.resize((old_frames_len+1,))
        self.frame_indices_dataset.resize((old_frames_len+1,))
        
        # 데이터 저장
        self.points_dataset[old_points_len:old_points_len + n_points] = pointcloud
        self.timestamps_dataset[old_frames_len] = timestamp
        self.frame_indices_dataset[old_frames_len] = n_points

    def stop(self):
        """VideoRecorder.stop()과 동일한 인터페이스"""
        with self.lock:
            if not self.recording:
                return
                
            self.recording = False
            self.ready = False
            
            # Zarr 저장소 정리
            if self.zarr_store is not None:
                self.zarr_store.close()
                self.zarr_store = None
                
            self.logger.info(f"PointCloudRecorder stopped. Total frames: {self.frame_count}")

    def is_ready(self) -> bool:
        """VideoRecorder.is_ready()와 동일한 인터페이스"""
        return self.ready and self.recording

    @classmethod
    def create_default(cls, fps=30, **kwargs):
        """VideoRecorder.create_h264()와 유사한 팩토리 메서드"""
        return cls(fps=fps, **kwargs)

# 포인트클라우드 읽기 함수 (데이터셋 변환용)
def read_pointcloud_sequence(pointcloud_path: str, dt: float, start_time: float = 0.0, 
                           downsample_factor: int = 1, max_points: int = None):
    """
    기존 get_accumulate_timestamp_idxs를 사용한 포인트클라우드 시퀀스 읽기
    """
    try:
        zarr_store = zarr.DirectoryStore(pointcloud_path)
        root = zarr.open(store=zarr_store, mode='r')
        
        points = root['points']
        timestamps = root['timestamps']
        frame_indices = root['frame_indices']
        
        next_global_idx = 0
        current_point_idx = 0
        
        # 모든 타임스탬프를 한 번에 처리
        all_timestamps = timestamps[:]
        local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
            timestamps=all_timestamps.tolist(),
            start_time=start_time,
            dt=dt,
            next_global_idx=0
        )
        
        # 선택된 프레임들만 yield
        for local_idx in local_idxs:
            frame_idx = local_idx
            n_points = frame_indices[frame_idx]
            
            # 해당 프레임까지의 포인트 인덱스 계산
            start_point_idx = sum(frame_indices[:frame_idx])
            end_point_idx = start_point_idx + n_points
            
            # 해당 프레임의 포인트클라우드 추출
            frame_points = points[start_point_idx:end_point_idx]
            
            # 다운샘플링
            if downsample_factor > 1:
                frame_points = frame_points[::downsample_factor]
            
            # 최대 포인트 수 제한
            if max_points is not None and len(frame_points) > max_points:
                indices = np.random.choice(len(frame_points), max_points, replace=False)
                frame_points = frame_points[indices]
            
            yield frame_points
            
    except Exception as e:
        logging.error(f"Error reading pointcloud sequence: {e}")
        raise
    finally:
        if 'zarr_store' in locals():
            zarr_store.close()
