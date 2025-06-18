#!/usr/bin/env python3
"""
PointCloud 데이터셋 변환 및 테스트 스크립트

Usage:
python test_pointcloud_dataset.py --config-name=real_stack_pointcloud
"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import numpy as np
import torch
from diffusion_policy.dataset.real_stack_pc_dataset import RealStackPointCloudDataset
from diffusion_policy.common.pytorch_util import dict_apply
import time
from termcolor import cprint

# OmegaConf resolver 등록
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')), config_name="real_stack_pc_workspace")
def main(cfg: DictConfig) -> None:
    """메인 테스트 함수"""
    
    print("=" * 60)
    cprint("POINTCLOUD DATASET TEST", "green", attrs=["bold"])
    print("=" * 60)
    
    # 설정 정보 출력
    print(f"Task: {cfg.task.name}")
    print(f"Dataset path: {cfg.task.dataset_path}")
    print(f"Horizon: {cfg.horizon}")
    print(f"N obs steps: {cfg.n_obs_steps}")
    print(f"Use cache: {cfg.task.dataset.use_cache}")
    
    # 데이터셋 경로 확인
    dataset_path = pathlib.Path(cfg.task.dataset_path)
    if not dataset_path.exists():
        cprint(f"Error: Dataset path does not exist: {dataset_path}", "red", attrs=["bold"])
        return
    
    required_files = [
        dataset_path / "replay_buffer.zarr",
        dataset_path / "orbbec_points.zarr"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            cprint(f"Error: Required file missing: {file_path}", "red", attrs=["bold"])
            return
    
    cprint("✓ All required files found", "green")
    
    # 데이터셋 생성
    print("\n" + "=" * 60)
    cprint("CREATING DATASET", "blue", attrs=["bold"])
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Hydra를 통한 데이터셋 인스턴스화
        dataset: RealStackPointCloudDataset = hydra.utils.instantiate(cfg.task.dataset)
        
        creation_time = time.time() - start_time
        cprint(f"✓ Dataset created successfully in {creation_time:.2f} seconds", "green")
        
    except Exception as e:
        cprint(f"Error creating dataset: {e}", "red")
        raise
    
    # 데이터셋 정보 출력
    print("\n" + "=" * 60)
    cprint("DATASET INFORMATION", "blue", attrs=["bold"])
    print("=" * 60)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Episodes: {dataset.replay_buffer.n_episodes}")
    print(f"Total steps: {dataset.replay_buffer.n_steps}")
    print(f"Episode lengths: {dataset.replay_buffer.episode_lengths}")
    
    # 데이터 키 정보
    print(f"\nData keys: {list(dataset.replay_buffer.keys())}")
    print(f"PointCloud keys: {dataset.pointcloud_keys}")
    print(f"LowDim keys: {dataset.lowdim_keys}")
    
    # 데이터 형태 정보
    for key in dataset.replay_buffer.keys():
        data = dataset.replay_buffer[key]
        print(f"  {key}: {data.shape} ({data.dtype})")
    
    """
    # 샘플 데이터 테스트
    print("\n" + "=" * 60)
    cprint("SAMPLE DATA TEST", "blue", attrs=["bold"])
    print("=" * 60)
    
    # 첫 번째 샘플 로드
    print("Loading first sample...")
    sample_start = time.time()
    
    try:
        sample = dataset[0]
        sample_time = time.time() - sample_start
        
        cprint(f"✓ Sample loaded in {sample_time:.3f} seconds", "green")
        
        # 샘플 구조 분석
        print(f"\nSample keys: {sample.keys()}")
        print(f"Obs keys: {sample['obs'].keys()}")
        
        for key, value in sample['obs'].items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
                if key in dataset.pointcloud_keys:
                    # 포인트클라우드 통계
                    pc_data = value.numpy()
                    xyz_data = pc_data[..., :3]
                    rgb_data = pc_data[..., 3:6]
                    
                    print(f"    XYZ range: [{xyz_data.min():.3f}, {xyz_data.max():.3f}]")
                    print(f"    RGB range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")
                    
                    # 유효한 포인트 수 계산
                    valid_points = np.isfinite(xyz_data).all(axis=-1).sum(axis=-1)
                    print(f"    Valid points per frame: {valid_points}")
        
        print(f"Action shape: {sample['action'].shape} ({sample['action'].dtype})")
        print(f"Action range: [{sample['action'].min():.3f}, {sample['action'].max():.3f}]")
        
    except Exception as e:
        cprint(f"Error loading sample: {e}", "red")
        raise
    
    # 정규화기 테스트
    print("\n" + "=" * 60)
    cprint("NORMALIZER TEST", "blue", attrs=["bold"])
    print("=" * 60)
    
    try:
        normalizer = dataset.get_normalizer()
        cprint("✓ Normalizer created successfully", "green")
        
        print(f"Normalizer keys: {list(normalizer.keys())}")
        
        # 정규화 테스트
        normalized_sample = normalizer.normalize(sample)
        print("✓ Sample normalization test passed")
        
        # 역정규화 테스트
        denormalized_sample = normalizer.unnormalize(normalized_sample)
        print("✓ Sample denormalization test passed")
        
    except Exception as e:
        cprint(f"Error with normalizer: {e}", "red")
        raise
    
    # 배치 로딩 테스트
    print("\n" + "=" * 60)
    cprint("BATCH LOADING TEST", "blue", attrs=["bold"])
    print("=" * 60)
    
    try:
        from torch.utils.data import DataLoader
        
        # 작은 배치로 테스트
        dataloader = DataLoader(
            dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=0  # 테스트용으로 0 사용
        )
        
        batch_start = time.time()
        batch = next(iter(dataloader))
        batch_time = time.time() - batch_start
        
        cprint(f"✓ Batch loaded in {batch_time:.3f} seconds", "green")
        
        print(f"Batch obs keys: {batch['obs'].keys()}")
        for key, value in batch['obs'].items():
            print(f"  {key}: {value.shape}")
        print(f"Batch action shape: {batch['action'].shape}")
        
    except Exception as e:
        cprint(f"Error with batch loading: {e}", "red")
        raise
    
    # 성능 벤치마크
    print("\n" + "=" * 60)
    cprint("PERFORMANCE BENCHMARK", "blue", attrs=["bold"])
    print("=" * 60)
    
    try:
        # 여러 샘플 로딩 시간 측정
        n_samples = min(10, len(dataset))
        times = []
        
        for i in range(n_samples):
            start = time.time()
            _ = dataset[i]
            times.append(time.time() - start)
        
        print(f"Sample loading times (n={n_samples}):")
        print(f"  Mean: {np.mean(times):.3f}s")
        print(f"  Std: {np.std(times):.3f}s")
        print(f"  Min: {np.min(times):.3f}s")
        print(f"  Max: {np.max(times):.3f}s")
        
    except Exception as e:
        cprint(f"Error in benchmark: {e}", "red")
        raise
    
    # 검증 데이터셋 테스트
    print("\n" + "=" * 60)
    cprint("VALIDATION DATASET TEST", "blue", attrs=["bold"])
    print("=" * 60)
    
    try:
        val_dataset = dataset.get_validation_dataset()
        print(f"Validation dataset length: {len(val_dataset)}")
        
        if len(val_dataset) > 0:
            val_sample = val_dataset[0]
            print("✓ Validation sample loaded successfully")
        else:
            print("! No validation data (val_ratio=0.0)")
        
    except Exception as e:
        cprint(f"Error with validation dataset: {e}", "red")
        raise
    
    # 완료 메시지
    print("\n" + "=" * 60)
    cprint("TEST COMPLETED SUCCESSFULLY!", "green", attrs=["bold"])
    print("=" * 60)
    
    total_time = time.time() - start_time
    print(f"Total test time: {total_time:.2f} seconds")
    
    # 요약 정보
    print(f"\nDataset Summary:")
    print(f"  Episodes: {dataset.replay_buffer.n_episodes}")
    print(f"  Total steps: {dataset.replay_buffer.n_steps}")
    print(f"  Sample loading time: {np.mean(times):.3f}s")
    print(f"  PointCloud keys: {dataset.pointcloud_keys}")
    print(f"  Action dimensions: {sample['action'].shape[-1]}")
    """
    # 요약 정보
    print(f"\nDataset Summary:")
    print(f"  Episodes: {dataset.replay_buffer.n_episodes}")
    print(f"  Total steps: {dataset.replay_buffer.n_steps}")
    # print(f"  Sample loading time: {np.mean(times):.3f}s")
    print(f"  PointCloud keys: {dataset.pointcloud_keys}")

if __name__ == "__main__":
    main()
