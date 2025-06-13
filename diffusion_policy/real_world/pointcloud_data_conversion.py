from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
import pytorch3d.ops as torch3d_ops
import torch

def read_pointcloud_sequence(
        pointcloud_path: str, 
        dt: float, 
        start_time: float = 0.0,
        downsample_factor: int = 1, 
        n_points: int = None,
    ) -> np.ndarray:
    """
    30Hz pointcloud zarr에서 10Hz로 프레임 추출
    real_data_conversion.py의 read_video()와 동일한 패턴
    """
    try:
        zarr_store = zarr.DirectoryStore(pointcloud_path)
        root = zarr.open(store=zarr_store, mode='r')
        
        points = root['points']
        timestamps = root['timestamps']
        frame_indices = root['frame_indices']
        
        # 모든 타임스탬프를 10Hz 기준으로 처리
        current_point_idx = 0
        next_global_idx = 0
        
        for frame_idx in range(len(timestamps)):
            frame_time = timestamps[frame_idx]
            
            # get_accumulate_timestamp_idxs와 동일한 로직
            local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            
            if len(global_idxs) > 0:
                n_points = frame_indices[frame_idx]
                
                # 해당 프레임의 포인트클라우드 추출
                start_point_idx = current_point_idx
                end_point_idx = current_point_idx + n_points
                frame_points = points[start_point_idx:end_point_idx]
                
                # 다운샘플링
                if downsample_factor > 1:
                    frame_points = frame_points[::downsample_factor]
                
                # 최대 포인트 수 제한
                if N_points is not None and len(frame_points) > N_points:
                    assert len(frame_points) >= N_points
                
                # 반복 저장 (드롭 프레임 보상)
                for _ in range(len(global_idxs)):
                    yield frame_points
            
            current_point_idx += frame_indices[frame_idx]
            
    except Exception as e:
        print(f"Error reading pointcloud sequence: {e}")
        raise
    finally:
        if 'zarr_store' in locals():
            zarr_store.close()

def real_pointcloud_data_to_replay_buffer(
        dataset_path: str,
        out_store: Optional[zarr.ABSStore] = None,
        lowdim_keys: Optional[Sequence[str]] = None,
        pointcloud_keys: Optional[Sequence[str]] = None,
        lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
        pointcloud_compressor: Optional[numcodecs.abc.Codec] = None,
        n_points: int = 368640,
        downsample_factor=1,
        n_decoding_threads: int = multiprocessing.cpu_count(),
        n_encoding_threads: int = multiprocessing.cpu_count(),
        max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
        verify_read: bool = True,
        apply_preprocessing: bool = False,
        use_cuda: bool = False,
) -> ReplayBuffer:
    """
    real_data_conversion.py와 동일한 패턴으로 pointcloud 데이터 변환
    """
    
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if pointcloud_compressor is None:
        # 포인트클라우드용 압축기 (이미지보다 낮은 압축률)
        pointcloud_compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    # 입력 검증
    input_path = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input_path.joinpath('replay_buffer.zarr')
    in_pointcloud_dir = input_path.joinpath('orbbec_points.zarr')
    
    assert in_zarr_path.is_dir(), f"replay_buffer.zarr not found: {in_zarr_path}"
    assert in_pointcloud_dir.is_dir(), f"pointcloud dir not found: {in_pointcloud_dir}"
    
    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r')

    # lowdim 데이터를 단일 청크로 저장
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape
        compressor_map[key] = lowdim_compressor

    print('Loading lowdim data')
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=lowdim_keys,
        chunks=chunks_map,
        compressors=compressor_map
    )
    
    # 포인트클라우드 처리
    def put_pointcloud(zarr_arr, zarr_idx, pointcloud):
        try:
            zarr_arr[zarr_idx] = pointcloud
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            print(f"Error putting pointcloud at {zarr_idx}: {e}")
            return False

    n_steps = in_replay_buffer.n_steps
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    episode_lengths = in_replay_buffer.episode_lengths
    timestamps = in_replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]  # 10Hz 간격

    # 포인트클라우드 키 추정
    if pointcloud_keys is None:
        pointcloud_keys = ['pointcloud']  # 기본값

    with tqdm(total=n_steps * len(pointcloud_keys), desc="Loading pointcloud data", mininterval=1.0) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor:
            futures = set()
            
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_pointcloud_dir = in_pointcloud_dir.joinpath(str(episode_idx))
                episode_start = episode_starts[episode_idx]
                
                if not episode_pointcloud_dir.exists():
                    print(f"Warning: pointcloud data not found for episode {episode_idx}")
                    continue

                for pointcloud_key in pointcloud_keys:
                    # 포인트클라우드 배열 할당
                    if pointcloud_key not in out_replay_buffer:
                        _ = out_replay_buffer.data.require_dataset(
                            name=pointcloud_key,
                            shape=(n_steps, n_points, 6),  # XYZRGB
                            chunks=(1, n_points, 6),
                            compressor=pointcloud_compressor,
                            dtype=np.float32
                        )
                    arr = out_replay_buffer[pointcloud_key]

                    # 포인트클라우드 시퀀스 읽기
                    episode_start_time = timestamps[episode_start]
                    for step_idx, pointcloud in enumerate(read_pointcloud_sequence(
                            pointcloud_path=str(episode_pointcloud_dir),
                            dt=dt,
                            start_time=episode_start_time,
                            downsample_factor=downsample_factor,
                            n_points=n_points,
                            n_decoding_threads=n_decoding_threads
                    )):
                        if len(futures) >= max_inflight_tasks:
                            # 진행 중인 작업 수 제한
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode pointcloud!')
                            pbar.update(len(completed))
                        
                        global_idx = episode_start + step_idx
                        
                        if apply_preprocessing:
                            pointcloud = preprocess_point_cloud(pointcloud, use_cuda = use_cuda)
                        
                        futures.add(executor.submit(put_pointcloud, arr, global_idx, pointcloud))

                        if step_idx == (episode_length - 1):
                            break
            
            # 남은 작업 완료
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode pointcloud!')
            pbar.update(len(completed))
    
    return out_replay_buffer

# pointcloud_data_conversion.py 수정
def preprocess_point_cloud(points, use_cuda=True):
    """3D Diffusion Policy와 동일한 전처리"""
    num_points = 1024
    
    extrinsics_matrix = np.array([[ 0.5213259,  -0.84716441,  0.10262438,  0.04268034],
                                  [ 0.25161211,  0.26751035,  0.93012341,  0.15598059],
                                  [-0.81542053, -0.45907589,  0.3526169,   0.47807532],
                                  [ 0.,          0.,          0.,          1.        ]])

    WORK_SPACE = [
        [0.65, 1.1],   # 작업 공간에 맞게 조정
        [0.45, 0.66], 
        [-0.7, 0]
    ]

    # scale 조정 (Orbbec 카메라에 맞게)
    point_xyz = points[..., :3] * 0.001  # mm to m 변환
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
    # crop
    points = points[np.where((points[..., 0] > WORK_SPACE[0][0]) & 
                            (points[..., 0] < WORK_SPACE[0][1]) &
                            (points[..., 1] > WORK_SPACE[1][0]) & 
                            (points[..., 1] < WORK_SPACE[1][1]) &
                            (points[..., 2] > WORK_SPACE[2][0]) & 
                            (points[..., 2] < WORK_SPACE[2][1]))]

    # FPS 샘플링
    if len(points) > num_points:
        points_xyz = points[..., :3]
        points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
        sample_indices = sample_indices.cpu()
        points_rgb = points[sample_indices, 3:][0]
        points = np.hstack((points_xyz, points_rgb))
    
    return points

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    """FPS 샘플링 함수"""
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(
            points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()
    return sampled_points, indices

