import os
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
import pcdp.common.mono_time as mono_time


def main():

    pipeline = ob.Pipeline()
    cfg = ob.Config()
    dev = pipeline.get_device()

    
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR) \
        .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)

    color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR) \
        .get_video_stream_profile(1280, 720, ob.OBFormat.RGB, 30)

    cfg.enable_stream(depth_profile)
    cfg.enable_stream(color_profile)

    pipeline.enable_frame_sync()
    pipeline.start(cfg)

    try:
        dev = pipeline.get_device()
        dev.set_bool_property(ob.OBPropertyID.OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL, True)
    except Exception:
        pass
        print(f"no HW Filter")

    align = ob.AlignFilter(align_to_stream=ob.OBStreamType.DEPTH_STREAM)
    pc_filter = ob.PointCloudFilter()
    cam_param = pipeline.get_camera_param()
    pc_filter.set_camera_param(cam_param)
    pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)  # x,y,z,r,g,b

    use_vis = False
    if use_vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.5, 0.6, 0.5])
        pcd = o3d.geometry.PointCloud()

    first_iter = True

    durations = []
    cnt, t_cnt = 0, 0
    start =0
    pc_preprocess = PointCloudPreprocessor(enable_rgb_normalize=False,
                                           enable_sampling=False)
    try:
        while cnt < 250:
            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue

            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue

            frame = align.process(frames) 

            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc = np.asarray(point_cloud)  # (N,6): x,y,z,r,g,b
            pc = pc_preprocess(pc)

            if cnt > 10:
                if  pc.size > 0:
                    pcd_tmp = o3d.geometry.PointCloud()
                    pcd_tmp.points = o3d.utility.Vector3dVector(pc[:, :3])  # XYZ만 넘김

                    # 반경 ROR: 반경 내 이웃 수가 ROR_MIN_NB_POINTS 미만인 점 제거
                    pcd_f, ind = pcd_tmp.remove_radius_outlier(
                        nb_points=12, # 최소 이웃 수
                        radius=0.01 # meter 단위 반경
                    )
                    print(f"pre_pc: {pc.shape}")
                    pc = pc[ind]  # 색상까지 동기 보존 (x,y,z,r,g,b)
                    print(f"after_pc: {pc.shape}")
                tim = mono_time.now_ms() - start
                start = mono_time.now_ms()
                if tim > 50.0:
                    t_cnt += 1
                durations.append(tim)

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
            durations=durations[1:]
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
