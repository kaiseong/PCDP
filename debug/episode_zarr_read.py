"""
    저장된 episode의 pointcloud 읽어오는 코드
"""
import numpy as np, open3d as o3d
from pyorbbecsdk import *
import time
from point_recorder import PointCloudRecorder
import zarr
import os
import math
import csv


def save_timestamp_duration_to_csv(timestamps, filename):
    """타임스탬프 배열을 CSV 파일로 저장"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'timestamp'])
        for i, ts in enumerate(timestamps):
            writer.writerow([i, ts])

episode=5
zarr_path = f"./{episode}"
if os.path.exists(zarr_path):
    zarr_store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store=zarr_store, mode='r')
    print("성공적으로 열림")
else:
    print(f"파일이 존재하지 않습니다: {zarr_path}")

# 데이터 확인
print(f"name: {zarr_path}")
print("저장된 데이터셋:", list(root.keys()))
print("포인트 데이터 형태:", root['points'].shape)
print("타임스탬프 개수:", root['timestamps'].shape)
print("프레임 인덱스:", root['frame_indices'].shape)

# 모든 포인트클라우드 데이터 로드
all_points = root['points'][:]  # (N_total_points, 6) XYZRGB
all_timestamps = root['timestamps'][:] # (N_frames,)
all_frame_indices = root['frame_indices'][:]  # (N_frames,)


print(f"총 포인트 수: {len(all_points)}")
print(f"총 프레임 수: {len(all_timestamps)}")
print(f"0:100: {(all_timestamps[100:200]-all_timestamps[0])*1000}")
print(f"first: {all_timestamps[0]}\n"
     f"last: {all_timestamps[-1]}")

anormaly_cnt=0
pre_time=0
idxs=[]
for idx, now in enumerate(all_timestamps):
    if abs(now-pre_time)*1000<20 or abs(now-pre_time)*1000 > 40:
        anormaly_cnt+=1
        idxs.append(idx)
    pre_time=now
print(f"idx: {idxs}")
print(f"anormaly_cnt: {anormaly_cnt}")

csv_filename = f"epsiode_{episode}.csv"
save_timestamp_duration_to_csv(all_timestamps, csv_filename)

def get_frame_pointcloud(root, frame_idx):
    """특정 프레임의 포인트클라우드 추출"""
    # 해당 프레임까지의 포인트 인덱스 계산
    start_idx = sum(root['frame_indices'][:frame_idx]) if frame_idx > 0 else 0
    end_idx = start_idx + root['frame_indices'][frame_idx]
    
    # 해당 프레임의 포인트클라우드 반환
    frame_points = root['points'][start_idx:end_idx]
    frame_timestamp = root['timestamps'][frame_idx]
    
    return frame_points, frame_timestamp
