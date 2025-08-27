"""
    30hz로 저장된 obs zarr데이터들
    csv로 뽑아내고 영상 재생하는 디버깅용 코드
"""

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
    [0.000, 0.740],    # X range (milli meters)
    [-0.400, 0.350],    # Y range (milli meters)
    [-0.100, 0.400]     # Z range (milli meters)
])


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
    
def point_cloud_visualize(obs_episode):
    """
    에피소드의 포인트 클라우드 시퀀스를 동영상처럼 재생하여 시각화합니다.
    실시간 렌더링 모범 사례를 적용하여 수정되었습니다[1].
    """

    preprocess = PointCloudPreprocessor(camera_to_base,
                                        workspace_bounds,
                                        enable_sampling=False,
                                        enable_rgb_normalize=False)
    pts_seq = obs_episode['pointcloud']
    if len(pts_seq) == 0:
        print("시각화할 포인트 클라우드 데이터가 없습니다.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Sequence", width=1280, height=720)
    
    opt = vis.get_render_option()
    opt.point_size = 1.0
    opt.background_color = np.array([0.5, 0.6, 0.5])
    
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
            # 카메라 뷰 초기 설정
            ctr = vis.get_view_control()
            ctr.set_lookat(pcd.get_axis_aligned_bounding_box().get_center())
            ctr.set_front([0.0, 0.0, -1.0])
            ctr.set_up([0.0, -1.0, 0.0])
            ctr.set_zoom(0.4)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
        
        # 렌더링 업데이트 및 이벤트 처리
        if not vis.poll_events():
            break # 창이 닫혔을 경우 루프 종료
        vis.update_renderer()
        
        # 재생 속도 조절 (예: 30fps)
        time.sleep(1/30) 
            
    print("시각화 종료.")
    vis.destroy_window()

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
        action = action_episode['action']
        
        print(f'pre_action: {action[0]}')
        # process_actions = []
        # for action_7d in action:
        #     pose_6d = action_7d[:6]
        #     gripper = action_7d[6]

        #     translation = pose_6d[:3]
        #     rotation = pose_6d[3:6]
        #     eef_to_robot_base_k = rise_tf.rot_trans_mat(translation, rotation)

        #     T_k_matrix = robot_to_base @ eef_to_robot_base_k
        #     transformed_pose_6d = rise_tf.mat_to_xyz_rot(
        #         T_k_matrix,
        #         rotation_rep='euler_angles',
        #         rotation_rep_convention='XYZ'
        #     )

        #     new_action_7d = np.concatenate([transformed_pose_6d, [gripper]])
        #     process_actions.append(new_action_7d)
        
        # action = np.array(process_actions, dtype=np.float32)
        print(f'after_action: {action[0]}')
        obs_capture_timestamp = obs_episode['capture_timestamp']

        pre_time =obs_capture_timestamp[0]
        cnt = 0
        for now in obs_capture_timestamp:
            if now-pre_time > 35.0:
                cnt +=1
            pre_time=now
        
        print(f"cnt: {cnt}")
        
        # with open(f"{episode_name}_obs_dataset.csv", 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['index', 'align_timestamp', 'robot_timestamp', 'capture_time'])
        #     for i in range(len(obs_align_timestamp)):
        #         writer.writerow([i, obs_align_timestamp[i], obs_robot_timestamp[i], obs_capture_timestamp[i]])
        # with open(f"{episode_name}_action.csv", 'w', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow(['index', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        #     for i in range(len(action)):
        #         writer.writerow([i, action[i][0], action[i][1], action[i][2], action[i][3], action[i][4], action[i][5]])
        # action_csv=f'{episode_name}_action_timestamp_dataset.csv'
        # save_timestamp_duration_to_csv(action_timestamps, action_csv)
        point_cloud_visualize(obs_episode)
            

# 사용 예시
analyzer = EpisodeAnalyzer("/home/moai/pcdp/data/test1_output/recorder_data")
# summary = analyzer.get_episode_summary()
# print(f"총 에피소드 수: {summary['total_episodes']}")
# for detail in summary['episode_details']:
#     print(f"{detail['name']}: {detail.get('obs_steps', 'N/A')} obs steps, "
#           f"{detail.get('action_steps', 'N/A')} action steps - {detail['status']}")

# 사용 예시
obs_buffer, action_buffer = analyzer.load_episode('episode_0003')
analyze_episode_quality(obs_buffer, action_buffer, 'episode_0003')