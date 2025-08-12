import numpy as np
import pyorbbecsdk as ob
import open3d as o3d
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
import pcdp.common.mono_time as mono_time

camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
    ])

workspace_bounds = np.array([
    [-0.800, 0.800],    # X range (milli meters)
    [-0.800, 0.800],    # Y range (milli meters)
    [-0.800, 0.350]     # Z range (milli meters)
])



def main():
    preprocess = PointCloudPreprocessor(camera_to_base,
                                        workspace_bounds,
                                        enable_sampling=False)
    pipeline = ob.Pipeline()
    cfg = ob.Config()
    depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                    .get_video_stream_profile(320, 288, ob.OBFormat.Y16, 30)
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

    use_vis = True
    if use_vis:
        # open3d visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='default', width=1280, height=720)
        opt = vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.array([1.0, 1.0, 1.0])
        pcd = o3d.geometry.PointCloud()
    
    first_iter = True
    duration = np.array([])

    cnt=0
    
    try:
        while True:
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
            
            # Preprocess the point cloud
            processed_pc = preprocess(pc)
            if not first_iter and cnt > 200:
                duration = np.append(duration, mono_time.now_ms()- t)
            t = mono_time.now_ms()
            
            cnt += 1

            if use_vis:
                # Separate points and colors for visualization
                transformed_points = processed_pc[:, :3]
                transformed_colors = processed_pc[:, 3:6] / 255.0 # Normalize colors to [0, 1]

                pcd.points = o3d.utility.Vector3dVector(transformed_points)
                pcd.colors = o3d.utility.Vector3dVector(transformed_colors)
                if first_iter:
                    vis.add_geometry(pcd, reset_bounding_box=True)
                    # Add base coordinate system at the origin
                    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.1, origin=[0, 0, 0])
                    vis.add_geometry(base_frame)

                    # Add camera coordinate system, transformed to its pose relative to the base
                    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=0.05, origin=[0, 0, 0])  # Smaller size to distinguish
                    camera_frame.transform(camera_to_base)
                    vis.add_geometry(camera_frame)
                    
                    # Draw workspace bounding box
                    points = [
                        [workspace_bounds[0, 0], workspace_bounds[1, 0], workspace_bounds[2, 0]],
                        [workspace_bounds[0, 1], workspace_bounds[1, 0], workspace_bounds[2, 0]],
                        [workspace_bounds[0, 0], workspace_bounds[1, 1], workspace_bounds[2, 0]],
                        [workspace_bounds[0, 0], workspace_bounds[1, 0], workspace_bounds[2, 1]],
                        [workspace_bounds[0, 1], workspace_bounds[1, 1], workspace_bounds[2, 0]],
                        [workspace_bounds[0, 1], workspace_bounds[1, 0], workspace_bounds[2, 1]],
                        [workspace_bounds[0, 0], workspace_bounds[1, 1], workspace_bounds[2, 1]],
                        [workspace_bounds[0, 1], workspace_bounds[1, 1], workspace_bounds[2, 1]],
                    ]
                    lines = [
                        [0, 1], [0, 2], [1, 4], [2, 4],
                        [0, 3], [1, 5], [2, 6], [4, 7],
                        [3, 5], [3, 6], [5, 7], [6, 7]
                    ]
                    colors = [[1, 0, 0] for i in range(len(lines))]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(points)
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    line_set.colors = o3d.utility.Vector3dVector(colors)
                    vis.add_geometry(line_set)

                    ctr = vis.get_view_control()
                    bbox = pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([-1.0, 0.0, 0.0])
                    ctr.set_up([0.0, 0.0, 1.0])
                    ctr.set_zoom(0.4)
                    first_iter = False
                
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()

    except Exception as e:
        print(f"error: {e}")
    finally:
        pipeline.stop()
        print(f"mean: {duration.mean()}, max: {duration.max()}, min: {duration.min()}")

if __name__ == '__main__':
    main()