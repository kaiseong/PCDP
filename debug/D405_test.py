#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


import pyrealsense2 as rs
import numpy as np
import open3d as o3d
import time
import pcdp.common.mono_time as mono_time
import cv2


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
    pipe = rs.pipeline()
    cfg = rs.config()
    # D405 추천 저해상도(속도) 예: 424x240@30
    cfg.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
    cfg.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8,60)
    profile = pipe.start(cfg)


    # 옵션(정확도 프리셋)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.visual_preset):
            depth_sensor.set_option(rs.option.visual_preset, 5.0)  # High Accuracy
    except Exception:
        print("not support")
        pass

    pc = rs.pointcloud()

    # Post-Processing 필터 초기화
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 2)

    threshold = rs.threshold_filter()
    threshold.set_option(rs.option.min_distance, 0.05)  # 10cm
    threshold.set_option(rs.option.max_distance, 0.5)  # 50cm

    depth_to_disparity = rs.disparity_transform(True)
    disparity_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    spatial.set_option(rs.option.holes_fill, 3)

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
    temporal.set_option(rs.option.filter_smooth_delta, 20)
    # temporal.set_option(rs.option.persistence_control, 3)

    # 워밍업
    for _ in range(15):
        pipe.wait_for_frames()

    use_vis = True  # 필요시 True
    if use_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window("D405 Colored PointCloud", 1280, 720)
        opt = vis.get_render_option(); opt.point_size = 1.0
        opt.background_color = np.array([0.4, 0.44, 0.4])
        pcd = o3d.geometry.PointCloud(); added = False

    make_pc_ms = np.array([])
    loop_ms = np.array([])
    t_prev = mono_time.now_ms()
    


    try:
        for _ in range(50000):
            frames = pipe.wait_for_frames()

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth or not color:
                continue

            # Post-Processing 필터 적용
            # Decimation은 해상도를 바꾸므로 color와 매칭이 깨질 수 있어 주석 처리합니다.
            # 사용하려면 rs.align 처리가 추가로 필요합니다.
            # depth = decimation.process(depth)
            
            depth = threshold.process(depth)
            depth = depth_to_disparity.process(depth)
            depth = spatial.process(depth)
            depth = temporal.process(depth)
            depth = disparity_to_depth.process(depth)

            print(f"depth: {depth.get_timestamp()}, color: {color.get_timestamp()}")
            t0 = mono_time.now_ms()
            # UV 계산 + 포인트클라우드
            pc.map_to(color)
            points = pc.calculate(depth)

            arr = rs_points_to_array(points, color,
                                     min_z=0.04, max_z=0.50, bilinear=True)
            print(f"arr.shape: {arr.shape}")
            make_pc_ms = np.append(make_pc_ms, mono_time.now_ms()-t0)

            if arr.shape[0] == 0:
                loop_ms.append((time.perf_counter() - t_prev) * 1000.0)
                t_prev = time.perf_counter()
                continue

            if use_vis:
                pcd.points = o3d.utility.Vector3dVector(arr[:, :3].astype(np.float64))
                pcd.colors = o3d.utility.Vector3dVector(arr[:, 3:].astype(np.float64))
                if not added:
                    vis.add_geometry(pcd, reset_bounding_box=True); added = True
                else:
                    vis.update_geometry(pcd)
                vis.poll_events(); vis.update_renderer()

            loop_ms = np.append(loop_ms, mono_time.now_ms()-t_prev)
            t_prev = mono_time.now_ms()

    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if use_vis:
            vis.destroy_window()
        if make_pc_ms.size > 0:
            make_pc_ms=make_pc_ms[1:]
            print(f"""make_pc_ms: mean: {make_pc_ms.mean()}
                      max: {make_pc_ms.max()}
                      min: {make_pc_ms.min()}""")
            print(f"max_index: {np.argmax(make_pc_ms)}, min_index: {np.argmin(make_pc_ms)}")
        if loop_ms.size > 0:
            loop_ms=loop_ms[1:]
            print(f"""loop_ms: mean: {loop_ms.mean()}
                      max: {loop_ms.max()}
                      min: {loop_ms.min()}""")
            print(f"max_index: {np.argmax(loop_ms)}, min_index: {np.argmin(loop_ms)}")
if __name__ == "__main__":
    main()
