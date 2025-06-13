
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

#!/usr/bin/env python3
"""
PointCloud 데이터셋 분석 및 시각화 도구

Usage:
python pointcloud_dataset_analyzer.py --dataset /path/to/training_dataset.zarr

python pointcloud_dataset_analyzer.py \
    --dataset /path/to/training_dataset.zarr \
    --episode 2 \
    --frame_step 5

python pointcloud_dataset_analyzer.py \
    --dataset /path/to/training_dataset.zarr \
    --export_samples ./sample_pointclouds \
    --n_samples 10

python pointcloud_dataset_analyzer.py \
    --dataset /path/to/training_dataset.zarr \
    --no_plot --no_vis

"""
import os
import argparse
import pathlib
import numpy as np
import zarr
import matplotlib.pyplot as plt
import open3d as o3d
from typing import Optional, Dict, List
import time
from diffusion_policy.common.replay_buffer import ReplayBuffer

class PointCloudDatasetAnalyzer:
    def __init__(self, dataset_path: str):
        self.dataset_path = pathlib.Path(dataset_path)
        self.replay_buffer = ReplayBuffer.create_from_path(str(self.dataset_path), mode='r')
        
    def print_dataset_info(self):
        """데이터셋 기본 정보 출력"""
        print("=" * 60)
        print("POINTCLOUD DATASET ANALYSIS")
        print("=" * 60)
        print(f"Dataset path: {self.dataset_path}")
        print(f"Total episodes: {self.replay_buffer.n_episodes}")
        print(f"Total steps: {self.replay_buffer.n_steps}")
        print(f"Backend: {self.replay_buffer.backend}")
        
        if self.replay_buffer.backend == 'zarr':
            print(f"Chunk size: {self.replay_buffer.chunk_size}")
        
        print("\nDataset keys:")
        for key in self.replay_buffer.keys():
            data = self.replay_buffer[key]
            print(f"  {key}: {data.shape} ({data.dtype})")
            if hasattr(data, 'compressor') and data.compressor:
                print(f"    Compressor: {data.compressor}")
        
        print(f"\nEpisode lengths: {self.replay_buffer.episode_lengths}")
        
    def analyze_pointcloud_statistics(self):
        """포인트클라우드 통계 분석"""
        print("\n" + "=" * 60)
        print("POINTCLOUD STATISTICS")
        print("=" * 60)
        
        if 'pointcloud' not in self.replay_buffer:
            print("No pointcloud data found!")
            return
            
        pointcloud_data = self.replay_buffer['pointcloud']
        print(f"Pointcloud shape: {pointcloud_data.shape}")
        
        # 샘플 데이터 로드 (메모리 절약을 위해 일부만)
        sample_size = min(100, pointcloud_data.shape[0])
        sample_indices = np.linspace(0, pointcloud_data.shape[0]-1, sample_size, dtype=int)
        sample_data = pointcloud_data[sample_indices]
        
        # XYZ 통계
        xyz_data = sample_data[:, :, :3].reshape(-1, 3)
        valid_mask = np.isfinite(xyz_data).all(axis=1)
        xyz_valid = xyz_data[valid_mask]
        
        print(f"\nXYZ Statistics (from {len(xyz_valid):,} valid points):")
        print(f"  X range: [{xyz_valid[:, 0].min():.3f}, {xyz_valid[:, 0].max():.3f}]")
        print(f"  Y range: [{xyz_valid[:, 1].min():.3f}, {xyz_valid[:, 1].max():.3f}]")
        print(f"  Z range: [{xyz_valid[:, 2].min():.3f}, {xyz_valid[:, 2].max():.3f}]")
        print(f"  Center: [{xyz_valid.mean(axis=0)[0]:.3f}, {xyz_valid.mean(axis=0)[1]:.3f}, {xyz_valid.mean(axis=0)[2]:.3f}]")
        
        # RGB 통계 (6채널인 경우)
        if sample_data.shape[2] >= 6:
            rgb_data = sample_data[:, :, 3:6].reshape(-1, 3)
            rgb_valid = rgb_data[valid_mask]
            print(f"\nRGB Statistics:")
            print(f"  R range: [{rgb_valid[:, 0].min():.1f}, {rgb_valid[:, 0].max():.1f}]")
            print(f"  G range: [{rgb_valid[:, 1].min():.1f}, {rgb_valid[:, 1].max():.1f}]")
            print(f"  B range: [{rgb_valid[:, 2].min():.1f}, {rgb_valid[:, 2].max():.1f}]")
        
        # 포인트 수 통계
        points_per_frame = []
        for i in range(min(sample_size, pointcloud_data.shape[0])):
            frame_data = sample_data[i]
            valid_points = np.isfinite(frame_data[:, :3]).all(axis=1).sum()
            points_per_frame.append(valid_points)
        
        print(f"\nPoints per frame statistics:")
        print(f"  Mean: {np.mean(points_per_frame):.0f}")
        print(f"  Min: {np.min(points_per_frame)}")
        print(f"  Max: {np.max(points_per_frame)}")
        print(f"  Std: {np.std(points_per_frame):.0f}")
        
    def analyze_robot_data(self):
        """로봇 데이터 분석"""
        print("\n" + "=" * 60)
        print("ROBOT DATA ANALYSIS")
        print("=" * 60)
        
        # 타임스탬프 분석
        if 'timestamp' in self.replay_buffer:
            timestamps = self.replay_buffer['timestamp'][:]
            dt_values = np.diff(timestamps)
            print(f"Timestamp analysis:")
            print(f"  Total duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
            print(f"  Mean dt: {dt_values.mean():.4f} seconds ({1/dt_values.mean():.1f} Hz)")
            print(f"  Dt std: {dt_values.std():.4f} seconds")
            print(f"  Min dt: {dt_values.min():.4f} seconds")
            print(f"  Max dt: {dt_values.max():.4f} seconds")
        
        # 액션 분석
        if 'action' in self.replay_buffer:
            actions = self.replay_buffer['action'][:]
            print(f"\nAction analysis:")
            print(f"  Shape: {actions.shape}")
            print(f"  Range: [{actions.min():.3f}, {actions.max():.3f}]")
            print(f"  Mean: {actions.mean(axis=0)}")
            print(f"  Std: {actions.std(axis=0)}")
        
        # 로봇 상태 분석
        robot_keys = ['robot_eef_pose', 'robot_joint', 'robot_gripper']
        for key in robot_keys:
            if key in self.replay_buffer:
                data = self.replay_buffer[key][:]
                print(f"\n{key} analysis:")
                print(f"  Shape: {data.shape}")
                print(f"  Range: [{data.min():.3f}, {data.max():.3f}]")
                print(f"  Mean: {data.mean(axis=0)}")
    
    def visualize_episode(self, episode_idx: int = 0, frame_step: int = 10):
        """특정 에피소드의 포인트클라우드 시각화"""
        print(f"\n" + "=" * 60)
        print(f"VISUALIZING EPISODE {episode_idx}")
        print("=" * 60)
        
        if episode_idx >= self.replay_buffer.n_episodes:
            print(f"Episode {episode_idx} does not exist!")
            return
            
        episode_data = self.replay_buffer.get_episode(episode_idx)
        
        if 'pointcloud' not in episode_data:
            print("No pointcloud data in episode!")
            return
            
        pointclouds = episode_data['pointcloud']
        print(f"Episode {episode_idx} has {len(pointclouds)} frames")
        
        # Open3D 시각화
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f'Episode {episode_idx} PointCloud', width=1280, height=720)
        
        # 렌더링 옵션 설정
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.1, 0.1, 0.1])
        
        pcd = o3d.geometry.PointCloud()
        
        print(f"Press 'Q' to quit, Space to pause/resume")
        print(f"Showing every {frame_step} frames...")
        
        frame_indices = range(0, len(pointclouds), frame_step)
        
        for i, frame_idx in enumerate(frame_indices):
            frame_data = pointclouds[frame_idx]
            
            # 유효한 포인트만 필터링
            valid_mask = np.isfinite(frame_data[:, :3]).all(axis=1)
            valid_points = frame_data[valid_mask]
            
            if len(valid_points) == 0:
                continue
                
            # 포인트클라우드 업데이트
            pcd.points = o3d.utility.Vector3dVector(valid_points[:, :3])
            
            if valid_points.shape[1] >= 6:  # RGB 정보가 있는 경우
                colors = valid_points[:, 3:6] / 255.0
                colors = np.clip(colors, 0, 1)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            if i == 0:
                vis.add_geometry(pcd, reset_bounding_box=True)
                
                # 카메라 설정
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                ctr.set_lookat(bbox.get_center())
                ctr.set_front([0.0, 0.0, -1.0])
                ctr.set_up([0.0, -1.0, 0.0])
                ctr.set_zoom(0.5)
            else:
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()
            
            print(f"\rFrame {frame_idx}/{len(pointclouds)-1} ({len(valid_points)} points)", end='', flush=True)
            time.sleep(0.1)  # 시각화 속도 조절
            
        print(f"\nVisualization completed!")
        vis.destroy_window()
    
    def plot_statistics(self):
        """데이터셋 통계 플롯"""
        print(f"\n" + "=" * 60)
        print("PLOTTING STATISTICS")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PointCloud Dataset Statistics', fontsize=16)
        
        # 에피소드 길이 분포
        episode_lengths = self.replay_buffer.episode_lengths
        axes[0, 0].bar(range(len(episode_lengths)), episode_lengths)
        axes[0, 0].set_title('Episode Lengths')
        axes[0, 0].set_xlabel('Episode Index')
        axes[0, 0].set_ylabel('Number of Steps')
        
        # 타임스탬프 간격 분포
        if 'timestamp' in self.replay_buffer:
            timestamps = self.replay_buffer['timestamp'][:1000]  # 처음 1000개만
            dt_values = np.diff(timestamps)
            axes[0, 1].hist(dt_values, bins=50, alpha=0.7)
            axes[0, 1].set_title('Timestamp Intervals Distribution')
            axes[0, 1].set_xlabel('dt (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(dt_values.mean(), color='red', linestyle='--', 
                              label=f'Mean: {dt_values.mean():.4f}s')
            axes[0, 1].legend()
        
        # 액션 분포
        if 'action' in self.replay_buffer:
            actions = self.replay_buffer['action'][:1000]  # 처음 1000개만
            for i in range(min(actions.shape[1], 3)):
                axes[1, 0].hist(actions[:, i], bins=30, alpha=0.5, label=f'Action {i}')
            axes[1, 0].set_title('Action Distribution')
            axes[1, 0].set_xlabel('Action Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # 포인트 수 분포
        if 'pointcloud' in self.replay_buffer:
            sample_size = min(100, self.replay_buffer['pointcloud'].shape[0])
            sample_indices = np.linspace(0, self.replay_buffer['pointcloud'].shape[0]-1, 
                                       sample_size, dtype=int)
            points_per_frame = []
            
            for idx in sample_indices:
                frame_data = self.replay_buffer['pointcloud'][idx]
                valid_points = np.isfinite(frame_data[:, :3]).all(axis=1).sum()
                points_per_frame.append(valid_points)
            
            axes[1, 1].hist(points_per_frame, bins=20, alpha=0.7)
            axes[1, 1].set_title('Points per Frame Distribution')
            axes[1, 1].set_xlabel('Number of Points')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def export_sample_data(self, output_dir: str, n_samples: int = 5):
        """샘플 데이터 내보내기"""
        output_path = pathlib.Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n" + "=" * 60)
        print(f"EXPORTING SAMPLE DATA to {output_path}")
        print("=" * 60)
        
        # 랜덤 샘플 선택
        total_steps = self.replay_buffer.n_steps
        sample_indices = np.random.choice(total_steps, min(n_samples, total_steps), replace=False)
        
        for i, idx in enumerate(sample_indices):
            if 'pointcloud' in self.replay_buffer:
                pointcloud = self.replay_buffer['pointcloud'][idx]
                
                # 유효한 포인트만 필터링
                valid_mask = np.isfinite(pointcloud[:, :3]).all(axis=1)
                valid_pointcloud = pointcloud[valid_mask]
                
                if len(valid_pointcloud) > 0:
                    # PLY 파일로 저장
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(valid_pointcloud[:, :3])
                    
                    if valid_pointcloud.shape[1] >= 6:
                        colors = valid_pointcloud[:, 3:6] / 255.0
                        colors = np.clip(colors, 0, 1)
                        pcd.colors = o3d.utility.Vector3dVector(colors)
                    
                    ply_path = output_path / f"sample_{i:03d}_step_{idx:06d}.ply"
                    o3d.io.write_point_cloud(str(ply_path), pcd)
                    print(f"Saved: {ply_path}")
        
        print(f"Exported {len(sample_indices)} sample point clouds")

def main():
    parser = argparse.ArgumentParser(description='Analyze PointCloud Dataset')
    parser.add_argument('--dataset', '-d', required=True, 
                       help='Path to the zarr dataset')
    parser.add_argument('--episode', '-e', type=int, default=0,
                       help='Episode index to visualize')
    parser.add_argument('--frame_step', type=int, default=10,
                       help='Frame step for visualization')
    parser.add_argument('--export_samples', type=str, default=None,
                       help='Directory to export sample point clouds')
    parser.add_argument('--n_samples', type=int, default=5,
                       help='Number of samples to export')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting statistics')
    parser.add_argument('--no_vis', action='store_true',
                       help='Skip 3D visualization')
    
    args = parser.parse_args()
    
    # 데이터셋 분석기 생성
    analyzer = PointCloudDatasetAnalyzer(args.dataset)
    
    # 기본 정보 출력
    analyzer.print_dataset_info()
    
    # 포인트클라우드 통계 분석
    analyzer.analyze_pointcloud_statistics()
    
    # 로봇 데이터 분석
    analyzer.analyze_robot_data()
    
    # 통계 플롯
    if not args.no_plot:
        analyzer.plot_statistics()
    
    # 3D 시각화
    if not args.no_vis:
        analyzer.visualize_episode(args.episode, args.frame_step)
    
    # 샘플 데이터 내보내기
    if args.export_samples:
        analyzer.export_sample_data(args.export_samples, args.n_samples)

if __name__ == '__main__':
    main()

