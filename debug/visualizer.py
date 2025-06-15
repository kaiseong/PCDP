"""
    용도: zarr로 저장된 pointcloud 시각화 재생용 코드
    사용법: 원하는 episode_idx와 fps 값 넣어서 실행
"""


import zarr
import numpy as np
import open3d as o3d
import time
import os

def visualize_episode_pointcloud(output_dir, episode_idx, fps=30):
    """각 프레임의 포인트 수가 고정된 경우의 시각화"""
    
    pointcloud_path = f"{output_dir}/orbbec_points.zarr/{episode_idx}"
    
    if not os.path.exists(pointcloud_path):
        print(f"에피소드 {episode_idx} 경로가 존재하지 않습니다: {pointcloud_path}")
        return
    
    zarr_store = zarr.DirectoryStore(pointcloud_path)
    root = zarr.open(store=zarr_store, mode='r')
    
    points = root['points']  # 플랫 배열: (total_points, 6)
    timestamps = root['timestamps']  # (n_frames,)
    frame_indices = root['frame_indices']  # 각 프레임의 포인트 수 (모두 368640)
    
    print(f"에피소드 {episode_idx} 정보:")
    print(f"- 총 프레임 수: {len(timestamps)}")
    print(f"- 전체 포인트 수: {points.shape[0]}")
    print(f"- 프레임당 포인트 수: {frame_indices[0]}")
    print(f"- 시작 타임스탬프: {timestamps[0]:.3f}")
    print(f"- 종료 타임스탬프: {timestamps[-1]:.3f}")
    
    # Open3D 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Episode {episode_idx} PointCloud Playback", 
        width=1280, 
        height=720
    )
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.0, 0.0, 0.0])
    
    pcd = o3d.geometry.PointCloud()
    
    print("\n재생 제어:")
    print("- Q: 종료")
    print("- 마우스: 시점 조정")
    
    frame_idx = 0
    points_per_frame = frame_indices[0]  # 368640 (C2D 모드)
    
    try:
        while frame_idx < len(timestamps):
            start_time = time.time()
            
            # 각 프레임의 포인트 인덱스 범위 계산
            start_point_idx = frame_idx * points_per_frame
            end_point_idx = start_point_idx + points_per_frame
            
            # 현재 프레임의 포인트 추출
            frame_points = points[start_point_idx:end_point_idx]
            
            if len(frame_points) == 0:
                print(f"프레임 {frame_idx}: 포인트가 없습니다.")
                frame_idx += 1
                continue
            
            # 유효한 포인트만 필터링 (Z > 0인 포인트만)
            valid_mask = frame_points[:, 2] > 0  # Z 좌표가 0보다 큰 포인트만
            if not np.any(valid_mask):
                print(f"프레임 {frame_idx}: 유효한 포인트가 없습니다.")
                frame_idx += 1
                continue
            
            frame_points = frame_points[valid_mask]
            
            # XYZ와 RGB 분리
            xyz = frame_points[:, :3].astype(np.float64)
            rgb = frame_points[:, 3:6].astype(np.float64)
            
            # RGB 정규화 (0-255 범위를 0-1로 변환)
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            
            # NaN/Inf 제거
            valid_xyz = np.isfinite(xyz).all(axis=1)
            valid_rgb = np.isfinite(rgb).all(axis=1)
            valid_points = valid_xyz & valid_rgb
            
            if not np.any(valid_points):
                print(f"프레임 {frame_idx}: 유효한 포인트가 없습니다 (NaN/Inf 제거 후).")
                frame_idx += 1
                continue
            
            xyz = xyz[valid_points]
            rgb = rgb[valid_points]
            
            # Open3D 업데이트
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            
            if frame_idx == 0:
                vis.add_geometry(pcd, reset_bounding_box=True)
                
                # 카메라 시점 설정
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                if not bbox.is_empty():
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
            else:
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()
            
            # 프레임 정보 출력
            print(f"\rFrame {frame_idx+1}/{len(timestamps)} | "
                  f"Timestamp: {timestamps[frame_idx]:.3f}s | "
                  f"Valid Points: {len(xyz)} / {points_per_frame} | "
                  f"Range: {start_point_idx}-{end_point_idx}", end='')
            
            frame_idx += 1
            
            # 프레임 레이트 조절
            elapsed = time.time() - start_time
            sleep_time = max(0, 1/fps - elapsed)
            time.sleep(sleep_time)
        
        print(f"\n에피소드 {episode_idx} 재생 완료!")
        
    except KeyboardInterrupt:
        print("\n재생이 중단되었습니다.")
    finally:
        vis.destroy_window()

# 사용 예시
visualize_episode_pointcloud("aa", episode_idx=2, fps=30)
