import sys
import os
import time

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import pcdp.common.mono_time as mono_time

def rs_points_to_array(points: rs.points,
                       color_frame: rs.video_frame,
                       min_z: float = 0.07,
                       max_z: float | None = 0.60,
                       bilinear: bool = True) -> np.ndarray:
    """points + color -> (N,6) float32 [x,y,z,r,g,b], RGB in [0,1]."""
    # 1) zero-copy view
    xyz = np.frombuffer(points.get_vertices(), dtype=np.float32).reshape(-1, 3)
    uv  = np.frombuffer(points.get_texture_coordinates(), dtype=np.float32).reshape(-1, 2)

    color = np.asanyarray(color_frame.get_data())  # HxWx3 BGR uint8
    H, W, _ = color.shape

    # 2) 유효 마스크
    mask = np.isfinite(xyz).all(axis=1)
    mask &= (xyz[:, 2] > min_z)
    if max_z is not None:
        mask &= (xyz[:, 2] < max_z)
    mask &= (uv[:, 0] >= 0) & (uv[:, 0] < 1) & (uv[:, 1] >= 0) & (uv[:, 1] < 1)
    if not np.any(mask):
        return np.empty((0, 6), dtype=np.float32)

    xyz = xyz[mask]
    uv  = uv[mask]

    # 3) 텍스처 샘플링
    if not bilinear:
        u = (uv[:, 0] * (W - 1)).astype(np.int32)
        v = (uv[:, 1] * (H - 1)).astype(np.int32)
        bgr = color[v, u].astype(np.float32)
    else:
        u  = uv[:, 0] * (W - 1)
        v  = uv[:, 1] * (H - 1)
        u0 = np.floor(u).astype(np.int32); v0 = np.floor(v).astype(np.int32)
        u1 = np.clip(u0 + 1, 0, W - 1);    v1 = np.clip(v0 + 1, 0, H - 1)
        du = (u - u0)[..., None];          dv = (v - v0)[..., None]

        c00 = color[v0, u0].astype(np.float32)
        c10 = color[v0, u1].astype(np.float32)
        c01 = color[v1, u0].astype(np.float32)
        c11 = color[v1, u1].astype(np.float32)
        bgr = (c00 * (1 - du) * (1 - dv) +
               c10 * (    du) * (1 - dv) +
               c01 * (1 - du) * (    dv) +
               c11 * (    du) * (    dv))

    rgb = bgr[..., ::-1] / 255.0  # BGR->RGB, [0,1]
    return np.concatenate([xyz.astype(np.float32), rgb.astype(np.float32)], axis=1)

def main():
    # 사용할 포인트 개수 고정
    NUM_POINTS = 8192*2

    # 파이프라인 생성 및 설정
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)

    serials = list()
    for d in rs.context().devices:
        if d.get_info(rs.camera_info.name).lower() != 'platform camera':
            serial = d.get_info(rs.camera_info.serial_number)
            product_line = d.get_info(rs.camera_info.product_line)
            if product_line == 'D400':
                # only works with D400 series
                serials.append(serial)
    serials = sorted(serials)
    print(serials)

    # 스트림 시작
    profile = pipeline.start(config)
    
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 4.0)  # High Accuracy
    except Exception:
        print("not support")
        pass

    # 워밍업 프레임
    for i in range(30):
        pipeline.wait_for_frames()

    use_vis = False  # 시각화 활성화
    if use_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window("D405 Colored PointCloud", 1280, 720)
        opt = vis.get_render_option(); opt.point_size = 1.0
        pcd = o3d.geometry.PointCloud(); added = False

    make_pc_ms = np.array([])
    downsample_ms = np.array([]) # 다운샘플링 시간 측정용
    loop_ms = np.array([])
    t_prev = mono_time.now_ms()
    
    pc = rs.pointcloud()

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # 포인트클라우드 객체 생성 및 색상 맵핑
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            t0 = mono_time.now_ms()
            # rs_points_to_array 함수를 사용하여 포인트 클라우드 생성
            pc_array = rs_points_to_array(points, color_frame,
                                     min_z=0.07, max_z=0.40, bilinear=True)
            make_pc_ms = np.append(make_pc_ms, mono_time.now_ms()-t0)

            if pc_array.shape[0] == 0:
                continue

            # ===== 다운샘플링/패딩 로직 시작 =====
            t_downsample_start = mono_time.now_ms()
            

            num_valid_points = pc_array.shape[0]

            if num_valid_points > NUM_POINTS:
                # 다운샘플링
                indices = np.random.choice(num_valid_points, NUM_POINTS, replace=False)
                final_pc = pc_array[indices]
            elif num_valid_points < NUM_POINTS:
                # 패딩
                padding = np.zeros((NUM_POINTS - num_valid_points, 6), dtype=np.float32)
                final_pc = np.vstack([pc_array, padding])
            else:
                # 크기가 정확히 맞을 경우
                final_pc = pc_array

            downsample_ms = np.append(downsample_ms, mono_time.now_ms() - t_downsample_start)
            # ===== 다운샘플링/패딩 로직 끝 =====

            if use_vis:
                pcd.points = o3d.utility.Vector3dVector(final_pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(final_pc[:, 3:])

                if not added:
                    vis.add_geometry(pcd, reset_bounding_box=True); added = True
                else:
                    vis.update_geometry(pcd)
                
                vis.poll_events()
                vis.update_renderer()

            loop_ms = np.append(loop_ms, mono_time.now_ms()-t_prev)
            t_prev = mono_time.now_ms()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        pipeline.stop()
        if use_vis:
            vis.destroy_window()
        
        if make_pc_ms.size > 0:
            print(f"""
--- Point Cloud Generation ---
Mean: {make_pc_ms.mean():.2f} ms
Max:  {make_pc_ms.max():.2f} ms
Min:  {make_pc_ms.min():.2f} ms""")
        if downsample_ms.size > 0:
            print(f"""--- Downsampling/Padding ---
Mean: {downsample_ms.mean():.2f} ms
Max:  {downsample_ms.max():.2f} ms
Min:  {downsample_ms.min():.2f} ms""")
        if loop_ms.size > 0:
            print(f"""--- Total Loop ---
Mean: {loop_ms.mean():.2f} ms
Max:  {loop_ms.max():.2f} ms
Min:  {loop_ms.min():.2f} ms""")


if __name__ == "__main__":
    main()