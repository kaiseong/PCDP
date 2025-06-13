if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import click
import pathlib
import zarr
import cv2
import threadpoolctl
from diffusion_policy.real_world.pointcloud_data_conversion import real_pointcloud_data_to_replay_buffer

@click.command()
@click.option('--input', '-i',  required=True)
@click.option('--output', '-o', default=None)
@click.option('--max_points', type=int, default=368640)
@click.option('--downsample_factor', default=1, type=int)
def main(input, output, max_points, downsample_factor):
    input = pathlib.Path(os.path.expanduser(input))
    print(input)
    in_zarr_path = input.joinpath('replay_buffer.zarr')
    in_points_dir = input.joinpath('orbbec_points.zarr')
    assert in_zarr_path.is_dir()
    assert in_points_dir.is_dir()
    if output is None:
        output = input.joinpath('640x480' + '.zarr.zip')
    else:
        output = pathlib.Path(os.path.expanduser(output))

    if output.exists():
        click.confirm('Output path already exists! Overrite?', abort=True)

    out_store = zarr.DirectoryStore(output)
    with threadpoolctl.threadpool_limits(1):
        replay_buffer = real_pointcloud_data_to_replay_buffer(
            dataset_path=str(input),
            out_store=out_store,
            lowdim_keys=['timestamp', 'action', 'robot_eef_pose', 'robot_joint', 'robot_gripper'],
            pointcloud_keys=['pointcloud'],
            max_points=max_points,
            downsample_factor=downsample_factor
        )
    
    print(f"Conversion completed! {replay_buffer.n_episodes} episodes, {replay_buffer.n_steps} steps")
    print('Saving to disk')

if __name__ == '__main__':
    main()
