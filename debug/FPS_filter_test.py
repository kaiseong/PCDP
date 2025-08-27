import os
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
import pcdp.common.mono_time as mono_time


# ======== 실시간 튜닝 파라미터 ========
NEAR_CLIP = 0.30       # [m] 0.3m 미만 컷 (센서 노이즈/근거리 왜곡 제거)
FAR_CLIP  = 3.00       # [m] 3.0m 초과 컷 (원거리 희박 노이즈 제거)
VOXEL_SIZE_PRE = 0.0   # [m] 0이면 비활성. 0.005~0.01 권장(실시간 허용 시)
MAX_INPUT_POINTS = 200_000  # 프리프로세서에 넘기기 전 상한(초과 시 랜덤 샘플)
ALIGN_TO_DEPTH = True  # RGB텍스처가 반드시 필요 없으면 False로 두어 정렬 비용 절약
TRY_HW_NOISE_REMOVE = True  # 지원 기기에서 하드웨어 노이즈 제거 on 시도
# ====================================


def fast_workspace_mask(pc_xyz, bounds):
    """워크스페이스 AABB 마스크 (NumPy 배열용, 매우 빠름)"""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds
    m = (
        (pc_xyz[:, 0] >= xmin) & (pc_xyz[:, 0] <= xmax) &
        (pc_xyz[:, 1] >= ymin) & (pc_xyz[:, 1] <= ymax) &
        (pc_xyz[:, 2] >= zmin) & (pc_xyz[:, 2] <= zmax)
    )
    return m


def voxel_downsample_numpy_o3d(pc_np, voxel):
    """
    Open3D를 이용해 컬러 포함 포인트클라우드 한 번에 다운샘플.
    - pc_np: (N,6) [x,y,z,r,g,b], r/g/b in [0..255]
    - 반환: (M,6)
    """
    if voxel <= 0:
        return pc_np

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])
    # Open3D 색상은 0~1 float
    colors = pc_np[:, 3:6] / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel)
    if len(pcd_ds.points) == 0:
        return pc_np

    out_xyz = np.asarray(pcd_ds.points, dtype=np.float32)
    out_rgb = (np.asarray(pcd_ds.colors, dtype=np.float32) * 255.0).clip(0, 255)
    out = np.concatenate([out_xyz, out_rgb], axis=1).astype(pc_np.dtype, copy=False)
    return out


def main():
    # 워크스페이스(AABB). 프리프로세서에서도 적용하지만,
    # 프리프로세서 호출 전에 한 번 더 얇게 걸러 연산량을 줄입니다.

    pipeline = ob.Pipeline()
    cfg = ob.Config()

    # NOTE: depth 해상도는 연산량에 직접 영향. 320x288은 이미 가벼운 편.
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR) \
        .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)

    # 컬러가 꼭 필요하지 않다면 아래 프로파일/정렬/텍스처링 과정을 생략해도 좋습니다.
    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR) \
        .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)

    cfg.enable_stream(depth_profile)
    cfg.enable_stream(color_profile)

    # 프레임 동기화
    pipeline.enable_frame_sync()
    pipeline.start(cfg)

    # (선택) 하드웨어 노이즈 제거 시도: 미지원이면 조용히 패스
    if TRY_HW_NOISE_REMOVE:
        try:
            dev = pipeline.get_device()
            # 기기/SDK 버전에 따라 프로퍼티 ID가 다를 수 있어 try/except 처리
            dev.set_bool_property(ob.OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL, True)
        except Exception:
            pass

    # 정렬/포인트클라우드 변환기
    align = ob.AlignFilter(
        align_to_stream=ob.OBStreamType.DEPTH_STREAM if ALIGN_TO_DEPTH else ob.OBStreamType.COLOR_STREAM
    )
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)  # x,y,z,r,g,b
    # 깊이 스케일은 매 프레임 갱신(디바이스/모드에 따라 변동 가능성)
    voxel_size = VOXEL_SIZE_PRE  # Open3D 다운샘플(선택)

    use_vis = False
    if use_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.0, 0.0, 0.0])
        pcd = o3d.geometry.PointCloud()
        first_iter = True
    else:
        first_iter = False

    durations = []
    cnt, t_cnt = 0, 0

    try:
        while cnt < 500:
            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue

            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue

            # (선택) 정렬: 컬러 텍스처가 필요 없다면 생략해도 됨
            frame = align.process(frames) if ALIGN_TO_DEPTH else frames

            # 포인트클라우드 변환
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc = np.asarray(point_cloud)  # (N,6): x,y,z,r,g,b
            if pc.size == 0:
                continue

            # 1) 깊이 범위 컷(초저비용)
            #    Z<=0 제거 + NEAR/FAR 범위 마스킹
            z = pc[:, 2]
            m_z = (z > 0.0) & (z >= NEAR_CLIP) & (z <= FAR_CLIP)
            if not np.any(m_z):
                cnt += 1
                continue
            pc = pc[m_z]

            # 2) 워크스페이스 AABB 컷(초저비용)
            m_ws = fast_workspace_mask(pc[:, :3], workspace)
            if not np.any(m_ws):
                cnt += 1
                continue
            pc = pc[m_ws]

            # 3) (선택) 매우 가벼운 다운샘플: voxel 한 번
            if voxel_size > 0.0 and pc.shape[0] > target_num_points * 2:
                pc = voxel_downsample_numpy_o3d(pc, voxel_size)

            # 4) (선택) 입력 상한: 지나치게 많으면 랜덤 샘플로 load 억제
            if pc.shape[0] > MAX_INPUT_POINTS:
                idx = np.random.choice(pc.shape[0], size=MAX_INPUT_POINTS, replace=False)
                pc = pc[idx]

            # ===== 성능 측정: 프리프로세서 전후 시간 =====
            if cnt > 200:
                start = mono_time.now_ms()
                process_pc = preprocessor.process(pc)  # (target_num_points, ?)
                tim = mono_time.now_ms() - start
                if tim > 50.0:
                    t_cnt += 1
                durations.append(tim)
            else:
                # 워밍업: GPU/캐시 안정화
                _ = preprocessor.process(pc)

            # (선택) 시각화
            if use_vis:
                vis_pc = pc.astype(np.float32, copy=False)
                pcd.points = o3d.utility.Vector3dVector(vis_pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector((vis_pc[:, 3:6] / 255.0).clip(0, 1))
                if first_iter:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
                    first_iter = False
                else:
                    vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

            cnt += 1

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if len(durations) > 0:
            durations = np.asarray(durations, dtype=np.float32)
            print(
                "duration(ms)\n"
                f"  mean: {durations.mean():.3f}\n"
                f"  max : {durations.max():.3f}\n"
                f"  min : {durations.min():.3f}"
            )
        else:
            print("duration: no samples (warmup only or early exit)")

        print(f"t_cnt(>50ms): {t_cnt}")
        pipeline.stop()


if __name__ == '__main__':
    main()
