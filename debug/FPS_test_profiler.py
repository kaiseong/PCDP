import os 
import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
import pcdp.common.mono_time as mono_time

def main():
    workspace = [[-10, 10], [-10, 10], [-10, 10]]
    target_num_points = 1024 

    preprocessor = PointCloudPreprocessor(
        workspace_bounds=workspace,
        target_num_points=target_num_points,
        use_cuda=True,
        verbose=False
    )

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
    voxel_size = 0.005

    use_vis = False
    if use_vis:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([0.0, 0.0, 0.0])
        pcd = o3d.geometry.PointCloud()
    
    first_iter = True
    
    durations = np.array([])
    cnt = 0
    try:
        while cnt<500:
            frames = pipeline.wait_for_frames(1)
            if frames is None:
                continue
            
            depth, color = frames.get_depth_frame(), frames.get_color_frame()
            if depth is None or color is None:
                continue
            frame = align.process(frames)
            pc_filter.set_position_data_scaled(depth.get_depth_scale())
            point_cloud = pc_filter.calculate(pc_filter.process(frame))
            pc=np.asarray(point_cloud) 
            pc = pc[pc[:, 2] > 0.0]
            mean_z = pc[:,2].mean()
            print(mean_z)

            if cnt>200:
                start=mono_time.now_ms()
                process_pc = preprocessor.process(pc)
                durations = np.append(durations, mono_time.now_ms() - start)
            # print(f"preprocess_pc shape: {process_pc.shape}")


            # vis
            if use_vis:
                vis_pc = np.asarray(point_cloud, dtype =np.float32)
                pcd.points = o3d.utility.Vector3dVector(vis_pc[:, :3])
                pcd.colors = o3d.utility.Vector3dVector(vis_pc[:, 3:6] / 255.0)
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
        print(f"error: {e}")
    finally:
        print(f"duration\n\
            mean: {durations.mean()}\n \
            max: {durations.max()}\n \
            min: {durations.min()}")
        pipeline.stop()

if __name__ == '__main__':
    main()
