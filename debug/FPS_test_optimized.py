

import os 
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
import torch
import time

try:
    import pytorch3d.ops as torch3d_ops
    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False
    print("Warning: pytorch3d not available. FPS will use fallback method.")

import pcdp.common.mono_time as mono_time

# --- PointCloudPreprocessor 로직을 여기에 통합 ---
# 이 함수들은 원래 PointCloudPreprocessor 클래스에 있던 메소드들입니다.
# FPS 테스트를 위해 필요한 부분만 가져와 최적화합니다.

def apply_transform_and_crop(points_np, extrinsics_matrix, workspace_bounds):
    """ CPU에서 수행되는 변환 및 필터링 """
    # 1. mm to m 스케일링 및 동차 좌표 변환
    point_xyz = points_np[:, :3] * 0.001
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_transformed = np.dot(point_homogeneous, extrinsics_matrix.T)
    points_np[:, :3] = point_transformed[:, :3]

    # 2. 작업 공간 필터링
    mask = (
        (points_np[:, 0] >= workspace_bounds[0][0]) & (points_np[:, 0] <= workspace_bounds[0][1]) &
        (points_np[:, 1] >= workspace_bounds[1][0]) & (points_np[:, 1] <= workspace_bounds[1][1]) &
        (points_np[:, 2] >= workspace_bounds[2][0]) & (points_np[:, 2] <= workspace_bounds[2][1])
    )
    return points_np[mask]

def farthest_point_sampling_gpu(points_gpu, target_num_points):
    """ GPU에서 수행되는 Farthest Point Sampling """
    if not PYTORCH3D_AVAILABLE or points_gpu.shape[0] == 0:
        return torch.zeros((target_num_points, 6), device=points_gpu.device)

    if points_gpu.shape[0] <= target_num_points:
        padded_points = torch.zeros((target_num_points, 6), device=points_gpu.device)
        padded_points[:points_gpu.shape[0]] = points_gpu
        return padded_points

    # pytorch3d는 XYZ 좌표만 사용합니다.
    points_xyz_gpu = points_gpu[:, :3].unsqueeze(0) # 배치 차원 추가
    _, indices = torch3d_ops.sample_farthest_points(points_xyz_gpu, K=target_num_points)
    
    # 전체 포인트 클라우드에서 샘플링된 인덱스를 사용하여 반환
    return points_gpu[indices.squeeze(0)]

def main():
    # --- 설정 ---
    workspace = [[-10, 10], [-10, 10], [-10, 10]]
    target_num_points = 1024 *2
    use_cuda = True and torch.cuda.is_available() and PYTORCH3D_AVAILABLE
    device = torch.device("cuda" if use_cuda else "cpu")

    # PointCloudPreprocessor의 기본 extrinsics matrix
    extrinsics_matrix = np.array([
        [ 0.5213259,  -0.84716441,  0.10262438,  0.04268034],
        [ 0.25161211,  0.26751035,  0.93012341,  0.15598059],
        [-0.81542053, -0.45907589,  0.3526169,   0.47807532],
        [ 0.,          0.,          0.,          1.        ]
    ])

    # --- 카메라 초기화 ---
    pipeline = ob.Pipeline()
    cfg = ob.Config()
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(640, 576, ob.OBFormat.Y16, 30)
    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                    .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)
    cfg.enable_stream(depth_profile)
    cfg.enable_stream(color_profile)
    pipeline.enable_frame_sync()
    pipeline.start(cfg)

    align = ob.AlignFilter(align_to_stream = ob.OBStreamType.DEPTH_STREAM)
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)

    # --- 최적화를 위한 버퍼 사전 할당 ---
    # Orbbec SDK의 최대 포인트 수 (640*576)를 기반으로 버퍼 생성
    max_points = 640 * 576
    # 1. 고정 메모리(Pinned Memory) 버퍼
    pinned_buffer = torch.empty((max_points, 6), dtype=torch.float32).pin_memory()
    # 2. GPU 텐서 버퍼
    gpu_buffer = torch.empty((max_points, 6), dtype=torch.float32, device=device)

    durations = np.array([])
    cnt = 0
    try:
        while cnt < 500:
            frames = pipeline.wait_for_frames(1)
            if frames is None: continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None: continue

            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            
            # SDK -> NumPy 배열로 복사 (불가피한 단계)
            pc_np = np.asarray(point_cloud, dtype=np.float32)
            pc_np = pc_np[pc_np[:, 2] > 0.0]
            num_points = pc_np.shape[0]
            if num_points == 0: continue

            if cnt > 200:
                torch.cuda.synchronize() # 이전 루프의 GPU 작업 완료 대기
                start = mono_time.now_ms()

                # --- 최적화된 파이프라인 ---
                # 1. NumPy -> Pinned Memory 복사
                pinned_buffer[:num_points].copy_(torch.from_numpy(pc_np))

                # 2. Pinned Memory -> GPU 비동기 복사 시작
                gpu_buffer[:num_points].copy_(pinned_buffer[:num_points], non_blocking=True)
                
                # 3. GPU가 데이터를 받는 동안 CPU는 변환/필터링 작업 수행
                #    (여기서는 간단함을 위해 동기적으로 처리. 실제로는 별도 스레드에서 수행하면 더 좋음)
                processed_pc_np = apply_transform_and_crop(pc_np, extrinsics_matrix, workspace)
                num_processed_points = processed_pc_np.shape[0]
                if num_processed_points == 0: continue

                # 4. CPU 처리 결과를 GPU로 복사 (이미 대부분의 포인트가 필터링되어 양이 적음)
                gpu_buffer[:num_processed_points].copy_(torch.from_numpy(processed_pc_np))

                # 5. FPS를 수행하기 전, 데이터 전송이 완료되도록 동기화
                torch.cuda.synchronize()
                
                # 6. GPU에서 Farthest Point Sampling 수행
                final_pc_gpu = farthest_point_sampling_gpu(gpu_buffer[:num_processed_points], target_num_points)

                # 7. 최종 결과가 나올 때까지 동기화하고 시간 측정
                torch.cuda.synchronize()
                durations = np.append(durations, mono_time.now_ms() - start)

            cnt += 1

    except Exception as e:
        print(f"error: {e}")
    finally:
        if len(durations) > 0:
            print(f"duration\n\
                mean: {durations.mean()}\n \
                max: {durations.max()}\n \
                min: {durations.min()}")
        pipeline.stop()

if __name__ == '__main__':
    main()
