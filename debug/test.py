## -*- coding: utf-8 -*-
"""
# test.py
# PCDP 데이터셋 로드 및 정보 출력 스크립트

"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import open3d as o3d
from diffusion_policy.common.replay_buffer import ReplayBuffer



# 시각화를 위한 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_pcdp_data(data_dir):
    """
    PCDP 데이터를 로드하는 함수
    
    Args:
        data_dir (str): recorder_data 디렉토리 경로
    
    Returns:
        obs_buffer, action_buffer: ReplayBuffer 객체들
    """
    data_path = Path(data_dir)
    
    # Observation 데이터 로드
    obs_path = data_path / 'obs_replay_buffer.zarr'
    action_path = data_path / 'action_replay_buffer.zarr'
    
    if not obs_path.exists():
        raise FileNotFoundError(f"Observation 데이터를 찾을 수 없습니다: {obs_path}")
    if not action_path.exists():
        raise FileNotFoundError(f"Action 데이터를 찾을 수 없습니다: {action_path}")
    
    # ReplayBuffer로 로드
    obs_buffer = ReplayBuffer.copy_from_path(str(obs_path), backend='numpy')
    action_buffer = ReplayBuffer.copy_from_path(str(action_path), backend='numpy')
    
    return obs_buffer, action_buffer

def print_dataset_info(obs_buffer, action_buffer):
    """데이터셋 기본 정보 출력"""
    print("=" * 50)
    print("PCDP 데이터셋 정보")
    print("=" * 50)
    
    # Episode 정보
    print(f"총 Episode 수: {obs_buffer.n_episodes}")
    print(f"총 Observation 스텝: {obs_buffer.n_steps}")
    print(f"총 Action 스텝: {action_buffer.n_steps}")
    
    # Episode 길이 분포
    obs_lengths = obs_buffer.episode_lengths
    action_lengths = action_buffer.episode_lengths
    
    print(f"\nObservation Episode 길이:")
    print(f"  평균: {np.mean(obs_lengths):.2f}")
    print(f"  최소: {np.min(obs_lengths)}")
    print(f"  최대: {np.max(obs_lengths)}")
    
    print(f"\nAction Episode 길이:")
    print(f"  평균: {np.mean(action_lengths):.2f}")
    print(f"  최소: {np.min(action_lengths)}")
    print(f"  최대: {np.max(action_lengths)}")
    
    # 데이터 키 정보
    print(f"\nObservation 키: {list(obs_buffer.keys())}")
    print(f"Action 키: {list(action_buffer.keys())}")
    
    # 데이터 형태 정보
    print(f"\n데이터 형태:")
    for key in obs_buffer.keys():
        print(f"  {key}: {obs_buffer[key].shape} ({obs_buffer[key].dtype})")
    
    for key in action_buffer.keys():
        print(f"  {key}: {action_buffer[key].shape} ({action_buffer[key].dtype})")

def analyze_episode(obs_buffer, action_buffer, episode_idx=0):
    """특정 episode 분석"""
    if episode_idx >= obs_buffer.n_episodes:
        print(f"Episode {episode_idx}는 존재하지 않습니다. (최대: {obs_buffer.n_episodes-1})")
        return
    
    # Episode 데이터 추출
    obs_episode = obs_buffer.get_episode(episode_idx)
    action_episode = action_buffer.get_episode(episode_idx)
    
    print(f"Episode {episode_idx} 분석:")
    print(f"  Observation 길이: {len(obs_episode['pointcloud'])}")
    print(f"  Action 길이: {len(action_episode['action'])}")
    
    # 타임스탬프 분석
    obs_timestamps = obs_episode['align_timestamp']
    action_timestamps = action_episode['timestamp']
    
    print(f"  시간 범위:")
    print(f"    Obs: {obs_timestamps[0]:.3f} ~ {obs_timestamps[-1]:.3f} ({obs_timestamps[-1] - obs_timestamps[0]:.3f}s)")
    print(f"    Action: {action_timestamps[0]:.3f} ~ {action_timestamps[-1]:.3f} ({action_timestamps[-1] - action_timestamps[0]:.3f}s)")
    
    # 포인트클라우드 정보
    pointcloud = obs_episode['pointcloud'][0]  # 첫 번째 프레임
    valid_points = pointcloud[pointcloud[:, 2] > 0]  # Z > 0인 유효한 점들
    print(f"  포인트클라우드 (첫 프레임):")
    print(f"    전체 점 수: {len(pointcloud)}")
    print(f"    유효 점 수: {len(valid_points)}")
    print(f"    XYZ 범위: X[{valid_points[:,0].min():.3f}, {valid_points[:,0].max():.3f}], "
          f"Y[{valid_points[:,1].min():.3f}, {valid_points[:,1].max():.3f}], "
          f"Z[{valid_points[:,2].min():.3f}, {valid_points[:,2].max():.3f}]")
    
    # 로봇 상태 분석
    robot_pose = obs_episode['robot_eef_pose']
    print(f"  로봇 말단 자세 범위:")
    print(f"    위치 (XYZ): [{robot_pose[:,:3].min():.3f}, {robot_pose[:,:3].max():.3f}]")
    print(f"    회전 (RPY): [{robot_pose[:,3:].min():.3f}, {robot_pose[:,3:].max():.3f}]")
    
    return obs_episode, action_episode

def validate_timestamp_sync(obs_buffer, action_buffer, episode_idx=0):
    """타임스탬프 동기화 품질 검증"""
    obs_episode = obs_buffer.get_episode(episode_idx)
    action_episode = action_buffer.get_episode(episode_idx)
    
    obs_ts = obs_episode['align_timestamp']
    robot_ts = obs_episode['robot_timestamp']
    action_ts = action_episode['timestamp']
    
    # Observation-Robot 동기화 오차
    sync_errors = np.abs(obs_ts - robot_ts)
    
    print(f"Episode {episode_idx} 동기화 검증:")
    print(f"  Obs-Robot 동기화 오차:")
    print(f"    평균: {np.mean(sync_errors)*1000:.2f}ms")
    print(f"    최대: {np.max(sync_errors)*1000:.2f}ms")
    print(f"    표준편차: {np.std(sync_errors)*1000:.2f}ms")
    
    # 타임스탬프 간격 분석
    obs_intervals = np.diff(obs_ts)
    action_intervals = np.diff(action_ts)
    
    print(f"  데이터 수집 주기:")
    print(f"    Obs 평균 간격: {np.mean(obs_intervals)*1000:.2f}ms ({1/np.mean(obs_intervals):.1f}Hz)")
    print(f"    Action 평균 간격: {np.mean(action_intervals)*1000:.2f}ms ({1/np.mean(action_intervals):.1f}Hz)")
    
    # 동기화 품질 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(sync_errors * 1000)
    plt.title('Obs-Robot 동기화 오차')
    plt.xlabel('Frame')
    plt.ylabel('오차 (ms)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(sync_errors * 1000, bins=30, alpha=0.7)
    plt.title('동기화 오차 분포')
    plt.xlabel('오차 (ms)')
    plt.ylabel('빈도')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return sync_errors

def generate_quality_report(obs_buffer, action_buffer, output_file=None):
    """전체 데이터셋 품질 리포트 생성"""
    report = []
    report.append("PCDP 데이터셋 품질 리포트")
    report.append("=" * 50)
    
    # 기본 통계
    report.append(f"총 Episode 수: {obs_buffer.n_episodes}")
    report.append(f"총 데이터 포인트: Obs({obs_buffer.n_steps}), Action({action_buffer.n_steps})")
    
    # Episode별 분석
    all_sync_errors = []
    all_pos_errors = []
    
    for ep_idx in range(obs_buffer.n_episodes):
        obs_ep = obs_buffer.get_episode(ep_idx)
        action_ep = action_buffer.get_episode(ep_idx)
        
        # 동기화 오차
        sync_error = np.abs(obs_ep['align_timestamp'] - obs_ep['robot_timestamp'])
        all_sync_errors.extend(sync_error)
        
        # 위치 추종 오차
        pos_error = np.linalg.norm(
            obs_ep['robot_eef_pose'][:, :3] - obs_ep['robot_eef_target'][:, :3], axis=1)
        all_pos_errors.extend(pos_error)
    
    all_sync_errors = np.array(all_sync_errors)
    all_pos_errors = np.array(all_pos_errors)
    
    # 품질 지표
    report.append(f"\n품질 지표:")
    report.append(f"  타임스탬프 동기화:")
    report.append(f"    평균 오차: {np.mean(all_sync_errors)*1000:.2f}ms")
    report.append(f"    최대 오차: {np.max(all_sync_errors)*1000:.2f}ms")
    report.append(f"    99% 백분위수: {np.percentile(all_sync_errors, 99)*1000:.2f}ms")
    
    report.append(f"  로봇 제어 정확도:")
    report.append(f"    평균 위치 오차: {np.mean(all_pos_errors)*1000:.2f}mm")
    report.append(f"    최대 위치 오차: {np.max(all_pos_errors)*1000:.2f}mm")
    report.append(f"    99% 백분위수: {np.percentile(all_pos_errors, 99)*1000:.2f}mm")
    
    # 포인트클라우드 품질
    sample_pc = obs_buffer.get_episode(0)['pointcloud'][0]
    valid_points = sample_pc[sample_pc[:, 2] > 0]
    report.append(f"  포인트클라우드:")
    report.append(f"    전체 점 수: {len(sample_pc)}")
    report.append(f"    유효 점 수: {len(valid_points)} ({len(valid_points)/len(sample_pc)*100:.1f}%)")
    
    # 리포트 출력 및 저장
    full_report = "\n".join(report)
    print(full_report)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"\n리포트가 {output_file}에 저장되었습니다.")
    
    return full_report



# 데이터 로드 예시
data_dir = "/home/nscl/diffusion_policy/bb/recorder_data"  # 실제 경로로 변경
obs_buffer, action_buffer = load_pcdp_data(data_dir)

# 정보 출력
print_dataset_info(obs_buffer, action_buffer)

# 동기화 검증
sync_errors = validate_timestamp_sync(obs_buffer, action_buffer, episode_idx=0)


# Episode 0 분석
obs_ep, action_ep = analyze_episode(obs_buffer, action_buffer, episode_idx=0)

# 품질 리포트 생성
report = generate_quality_report(obs_buffer, action_buffer, "pcdp_quality_report.txt")