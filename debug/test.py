# episode_analyzer.py
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
import time

import csv
def save_timestamp_duration_to_csv(timestamps, filename):
    """타임스탬프 배열을 CSV 파일로 저장"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'timestamp'])
        for i, ts in enumerate(timestamps):
            writer.writerow([i, ts])


class EpisodeAnalyzer:
    """에피소드별 저장된 PCDP 데이터 분석 클래스"""
    
    def __init__(self, recorder_data_dir):
        self.recorder_data_dir = Path(recorder_data_dir)
        self.episodes = self._discover_episodes()
        
    def _discover_episodes(self):
        """저장된 에피소드 목록 발견"""
        episodes = []
        if self.recorder_data_dir.exists():
            for item in sorted(self.recorder_data_dir.iterdir()):
                if item.is_dir() and item.name.startswith('episode_'):
                    episodes.append(item.name)
        return episodes
    
    def load_episode(self, episode_name):
        """특정 에피소드 로드"""
        episode_dir = self.recorder_data_dir / episode_name
        
        obs_path = episode_dir / 'obs_replay_buffer.zarr'
        action_path = episode_dir / 'action_replay_buffer.zarr'
        
        if not obs_path.exists() or not action_path.exists():
            raise FileNotFoundError(f"Episode data not found: {episode_dir}")
        
        obs_buffer = ReplayBuffer.copy_from_path(str(obs_path), backend='numpy')
        action_buffer = ReplayBuffer.copy_from_path(str(action_path), backend='numpy')
        
        return obs_buffer, action_buffer
    
    def get_episode_summary(self):
        """전체 에피소드 요약 정보"""
        summary = {
            'total_episodes': len(self.episodes),
            'episode_list': self.episodes,
            'episode_details': []
        }
        
        for episode_name in self.episodes:
            try:
                obs_buffer, action_buffer = self.load_episode(episode_name)
                detail = {
                    'name': episode_name,
                    'obs_steps': obs_buffer.n_steps,
                    'action_steps': action_buffer.n_steps,
                    'obs_episodes': obs_buffer.n_episodes,
                    'action_episodes': action_buffer.n_episodes,
                    'status': 'OK' if obs_buffer.n_episodes == 1 and action_buffer.n_episodes == 1 else 'WARNING'
                }
                summary['episode_details'].append(detail)
            except Exception as e:
                summary['episode_details'].append({
                    'name': episode_name,
                    'status': f'ERROR: {str(e)}'
                })
        
        return summary

def analyze_episode_quality(obs_buffer, action_buffer, episode_name):
    """에피소드 데이터 품질 분석"""
    print(f"\n=== {episode_name} 품질 분석 ===")
    
    # 기본 정보
    print(f"Observation 스텝: {obs_buffer.n_steps}")
    print(f"Action 스텝: {action_buffer.n_steps}")
    print(f"Observation 에피소드 수: {obs_buffer.n_episodes}")
    print(f"Action 에피소드 수: {action_buffer.n_episodes}")
    
    # 에피소드 분할 문제 확인
    if obs_buffer.n_episodes != 1:
        print(f"⚠️  WARNING: Observation이 {obs_buffer.n_episodes}개 에피소드로 분할됨")
    if action_buffer.n_episodes != 1:
        print(f"⚠️  WARNING: Action이 {action_buffer.n_episodes}개 에피소드로 분할됨")
    
    # 데이터 키 확인
    print(f"\nObservation 키: {list(obs_buffer.keys())}")
    print(f"Action 키: {list(action_buffer.keys())}")
    
    # 첫 번째 에피소드 데이터 분석
    if obs_buffer.n_episodes > 0:
        obs_episode = obs_buffer.get_episode(0)
        action_episode = action_buffer.get_episode(0)
        
        # 타임스탬프 분석
        obs_align_timestamp = obs_episode['align_timestamp']
        obs_robot_timestamp = obs_episode['robot_timestamp']
        action_timestamps = action_episode['timestamp']

        obs_capture_timestamp = obs_episode['capture_timestamp']

        pre_time =obs_capture_timestamp[0]
        cnt = 0
        for now in obs_capture_timestamp:
            if now-pre_time > 35.0:
                cnt +=1
            pre_time=now
        print(f"cnt: {cnt}")


        with open("obs_dataset.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['index', 'align_timestamp', 'robot_timestamp', 'capture_time'])
            for i in range(len(obs_align_timestamp)):
                writer.writerow([i, obs_align_timestamp[i], obs_robot_timestamp[i], obs_capture_timestamp[i]])
        
        action_csv='action_dataset.csv'
        save_timestamp_duration_to_csv(action_timestamps, action_csv)

# 사용 예시
analyzer = EpisodeAnalyzer("/home/nscl/diffusion_policy/bb/recorder_data")
summary = analyzer.get_episode_summary()
print(f"총 에피소드 수: {summary['total_episodes']}")
for detail in summary['episode_details']:
    print(f"{detail['name']}: {detail.get('obs_steps', 'N/A')} obs steps, "
          f"{detail.get('action_steps', 'N/A')} action steps - {detail['status']}")

# 사용 예시
obs_buffer, action_buffer = analyzer.load_episode('episode_0004')
analyze_episode_quality(obs_buffer, action_buffer, 'episode_0004')
