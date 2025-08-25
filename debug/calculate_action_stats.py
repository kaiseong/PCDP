import sys
import os
import pathlib
from zipfile import ZipFile
import numpy as np
import click
import zarr
from tqdm import tqdm

@click.command()
@click.option('-i', '--dataset-dir', required=True, help="Directory containing the zarr.zip dataset file.")
def main(dataset_dir):
    # Find the .zarr.zip file in the directory
    dataset_dir = pathlib.Path(dataset_dir)
    dataset_paths = list(dataset_dir.rglob('*.zarr.zip'))
    if len(dataset_paths) == 0:
        print(f"No .zarr.zip file found in {dataset_dir}")
        return
    if len(dataset_paths) > 1:
        print(f"Multiple .zarr.zip files found, please specify a more precise directory.")
        return
    
    dataset_path = dataset_paths[0]
    print(f"Processing: {dataset_path}")

    # Load the dataset
    with ZipFile(dataset_path, 'r') as zf:
        with zf.open('data/episode_ends.json', 'r') as f:
            episode_ends = np.array(eval(f.read()))
        
        root = zarr.open(zf)
        data_group = root['data']
        
        all_trans = []
        all_rots = []
        all_gripper_widths = []

        for episode_idx in tqdm(range(len(episode_ends)), desc="Calculating Stats"):
            start_idx = 0
            if episode_idx > 0:
                start_idx = episode_ends[episode_idx-1]
            end_idx = episode_ends[episode_idx]
            
            # tcp is (x,y,z,roll,pitch,yaw)
            tcp_data = data_group['tcp'][start_idx:end_idx]
            # gripper_width is a single value
            gripper_width_data = data_group['gripper_width'][start_idx:end_idx]

            all_trans.extend(tcp_data[:, :3])
            all_rots.extend(tcp_data[:, 3:])
            all_gripper_widths.extend(gripper_width_data)

    # Convert to numpy arrays
    all_trans = np.array(all_trans)
    all_rots = np.array(all_rots)
    all_gripper_widths = np.array(all_gripper_widths)

    # Calculate stats
    trans_stats = {
        'min': np.min(all_trans, axis=0),
        'max': np.max(all_trans, axis=0)
    }
    rots_stats = {
        'min': np.min(all_rots, axis=0),
        'max': np.max(all_rots, axis=0)
    }
    gripper_stats = {
        'min': np.min(all_gripper_widths),
        'max': np.max(all_gripper_widths)
    }

    # Print the results in a copy-paste friendly format
    print("\n" + "="*20)
    print("Stats calculated successfully!")
    print("Copy these values into your RISE_util.py file.")
    print("="*20 + "\n")
    
    print("TRANS_MIN = [{:e}, {:e}, {:e}]".format(*trans_stats['min']))
    print("TRANS_MAX = [{:e}, {:e}, {:e}]".format(*trans_stats['max']))
    print("\n")
    print("ROTS_MIN = [{:e}, {:e}, {:e}]".format(*rots_stats['min']))
    print("ROTS_MAX = [{:e}, {:e}, {:e}]".format(*rots_stats['max']))
    print("\n")
    print("GRIPPER_WIDTH_MIN = {:e}".format(gripper_stats['min']))
    print("GRIPPER_WIDTH_MAX = {:e}".format(gripper_stats['max']))
    print("\n")


if __name__ == '__main__':
    main()