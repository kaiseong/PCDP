"""
모델 추론으로 저장된 obs zarr데이터들
csv로 뽑아내고 영상 재생하는 디버깅용 코드
"""

import sys
import os
import argparse

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import zarr
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import open3d as o3d
from pcdp.common.replay_buffer import ReplayBuffer
import time
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.common import RISE_transformation as rise_tf
import csv


robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])


camera_to_base = np.array([
      [  0.007131,  -0.91491,    0.403594,  0.05116],
      [ -0.994138,   0.003833,   0.02656,  -0.00918],
      [ -0.025717,  -0.403641,  -0.914552, 0.50821],
      [ 0.,         0. ,        0. ,        1.    ]
            ])

workspace_bounds = np.array([
    [0.000, 0.715],    # X range (milli meters)
    [-0.400, 0.350],    # Y range (milli meters)
    [-0.100, 0.400]     # Z range (milli meters)
])



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


def point_cloud_visualize(obs_episode):
    """
    에피소드의 포인트 클라우드 시퀀스를 동영상처럼 재생하여 시각화합니다.
    """

    preprocess = PointCloudPreprocessor(extrinsics_matrix=camera_to_base,
                                        workspace_bounds=workspace_bounds,
                                        enable_sampling=False,
                                        enable_rgb_normalize=False,
                                        enable_filter=True)
    pts_seq = obs_episode['pointcloud']
    if len(pts_seq) == 0:
        print("시각화할 포인트 클라우드 데이터가 없습니다.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Sequence", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0.5, 0.55, 0.5])
    
    pcd = o3d.geometry.PointCloud()
    is_first_frame = True
    
    print("포인트 클라우드 시퀀스를 재생합니다. (창이 활성화된 상태에서 'Q'를 누르면 종료됩니다)")
    
    for i, pts in enumerate(pts_seq):
        pc=preprocess(pts)
        xyz = pc[:, :3].astype(np.float64)
        rgb = pc[:, 3:6].astype(np.float64) / 255.0
        
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        
        if is_first_frame:
            vis.add_geometry(pcd)
            ctr = vis.get_view_control()
            ctr.set_lookat(pcd.get_axis_aligned_bounding_box().get_center())
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.4)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
        
        if not vis.poll_events():
            break
        vis.update_renderer()
        
        time.sleep(1/30) 
            
    print("시각화 종료.")
    vis.destroy_window()

def analyze_episode_quality(obs_buffer, action_buffer, episode_name):
    """에피소드 데이터 품질 분석"""
    print(f"\n=== {episode_name} 품질 분석 ===")
    
    print(f"Observation 스텝: {obs_buffer.n_steps}")
    print(f"Action 스텝: {action_buffer.n_steps}")
    
    obs_episode = obs_buffer.get_episode(0)
    action_episode = action_buffer.get_episode(0)
    
    obs_align_timestamp = obs_episode['align_timestamp']
    action = action_episode['action']
    
    output_filename_prefix = f"{episode_name}_output"

    with open(f"{output_filename_prefix}_obs_dataset.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'align_timestamp'])
        for i, ts in enumerate(obs_align_timestamp):
            writer.writerow([i, ts])

    with open(f"{output_filename_prefix}_action.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper'])
        for i, act in enumerate(action):
            writer.writerow([i] + act.tolist())
            
    point_cloud_visualize(obs_episode)

if __name__ == "__main__":
    analyzer = EpisodeAnalyzer("/home/nscl/diffusion_policy/data/simple_stack_3_output_final/recorder_data")
    obs_buffer, action_buffer = analyzer.load_episode('episode_0004')
    analyze_episode_quality(obs_buffer, action_buffer, 'episode_0004')