#!/usr/bin/env python3
"""
PointCloud 데이터 전용 뷰어
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import zarr
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import argparse

def analyze_pointcloud_data(dataset_path: str):
    """포인트클라우드 데이터만 분석"""
    print(f"Loading dataset: {dataset_path}")
    
    if dataset_path.endswith('.zip'):
        with zarr.ZipStore(dataset_path, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore()
            )
    else:
        replay_buffer = ReplayBuffer.create_from_path(dataset_path, mode='r')
    
    print(f"Episodes: {replay_buffer.n_episodes}")
    print(f"Total steps: {replay_buffer.n_steps}")
    print(f"All keys: {list(replay_buffer.keys())}")
    
    # 포인트클라우드 키 찾기
    pointcloud_keys = []
    for key in replay_buffer.keys():
        data = replay_buffer[key]
        if len(data.shape) == 3 and data.shape[2] == 6:  # (N_steps, N_points, 6)
            pointcloud_keys.append(key)
    
    if not pointcloud_keys:
        print("❌ No pointcloud data found!")
        return
    
    print(f"\n✅ PointCloud keys found: {pointcloud_keys}")
    
    for pc_key in pointcloud_keys:
        print(f"\n" + "="*50)
        print(f"ANALYZING POINTCLOUD: {pc_key}")
        print("="*50)
        
        pc_data = replay_buffer[pc_key]
        print(f"Shape: {pc_data.shape}")
        print(f"Data type: {pc_data.dtype}")
        
        # 압축 정보
        if hasattr(pc_data, 'compressor') and pc_data.compressor:
            print(f"Compressor: {pc_data.compressor}")
        if hasattr(pc_data, 'chunks'):
            print(f"Chunks: {pc_data.chunks}")
        
        # 첫 번째 프레임 분석
        first_frame = pc_data[0]
        print(f"\nFirst frame shape: {first_frame.shape}")
        
        # 유효한 포인트 찾기 (NaN이 아닌 포인트)
        valid_mask = np.isfinite(first_frame[:, :3]).all(axis=1)
        valid_points = first_frame[valid_mask]
        
        print(f"Valid points in first frame: {len(valid_points)}/{len(first_frame)}")
        
        if len(valid_points) > 0:
            # XYZ 통계
            xyz_data = valid_points[:, :3]
            print(f"\nXYZ Statistics:")
            print(f"  X range: [{xyz_data[:, 0].min():.3f}, {xyz_data[:, 0].max():.3f}]")
            print(f"  Y range: [{xyz_data[:, 1].min():.3f}, {xyz_data[:, 1].max():.3f}]")
            print(f"  Z range: [{xyz_data[:, 2].min():.3f}, {xyz_data[:, 2].max():.3f}]")
            print(f"  Center: [{xyz_data.mean(axis=0)[0]:.3f}, {xyz_data.mean(axis=0)[1]:.3f}, {xyz_data.mean(axis=0)[2]:.3f}]")
            
            # RGB 통계
            rgb_data = valid_points[:, 3:6]
            print(f"\nRGB Statistics:")
            print(f"  R range: [{rgb_data[:, 0].min():.1f}, {rgb_data[:, 0].max():.1f}]")
            print(f"  G range: [{rgb_data[:, 1].min():.1f}, {rgb_data[:, 1].max():.1f}]")
            print(f"  B range: [{rgb_data[:, 2].min():.1f}, {rgb_data[:, 2].max():.1f}]")
            
            # 첫 5개 포인트 출력
            print(f"\nFirst 5 points (XYZ + RGB):")
            for i in range(min(5, len(valid_points))):
                point = valid_points[i]
                print(f"  Point {i}: XYZ=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) "
                      f"RGB=({point[3]:.0f}, {point[4]:.0f}, {point[5]:.0f})")
        
        # 여러 프레임의 포인트 수 통계
        print(f"\nPoints per frame statistics (first 10 frames):")
        for frame_idx in range(min(10, pc_data.shape[0])):
            frame_data = pc_data[frame_idx]
            valid_count = np.isfinite(frame_data[:, :3]).all(axis=1).sum()
            print(f"  Frame {frame_idx}: {valid_count} valid points")
        
        # 전체 프레임 통계 (샘플링)
        sample_size = min(100, pc_data.shape[0])
        sample_indices = np.linspace(0, pc_data.shape[0]-1, sample_size, dtype=int)
        points_per_frame = []
        
        for idx in sample_indices:
            frame_data = pc_data[idx]
            valid_count = np.isfinite(frame_data[:, :3]).all(axis=1).sum()
            points_per_frame.append(valid_count)
        
        print(f"\nOverall statistics (from {sample_size} frames):")
        print(f"  Mean points per frame: {np.mean(points_per_frame):.0f}")
        print(f"  Min points per frame: {np.min(points_per_frame)}")
        print(f"  Max points per frame: {np.max(points_per_frame)}")
        print(f"  Std points per frame: {np.std(points_per_frame):.0f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze PointCloud data only')
    parser.add_argument('dataset', help='Dataset path (.zarr or .zarr.zip)')
    args = parser.parse_args()
    
    analyze_pointcloud_data(args.dataset)
