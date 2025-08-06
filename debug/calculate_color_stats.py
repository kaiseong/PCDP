
import numpy as np
import zarr
import os
from tqdm import tqdm
import argparse

# This script calculates the mean and standard deviation of RGB colors
# from all point clouds in a Zarr dataset.

def calculate_stats(dataset_path):
    """
    Calculates the mean and std of the pointcloud colors in the dataset.

    Args:
        dataset_path (str): Path to the directory containing the Zarr archives.

    Example:
        python calculate_color_status.py --dataset_path ./data/stack_pc_dataset

    Returns:
        tuple: (mean, std) as numpy arrays.
    """
    all_colors = []
    
    # Find all zarr files in the dataset path
    zarr_files = [f for f in os.listdir(dataset_path) if f.endswith('.zarr.zip')]
    if not zarr_files:
        raise FileNotFoundError(f"No .zarr.zip files found in {dataset_path}")

    print(f"Found {len(zarr_files)} Zarr archives. Processing...")

    for zarr_file in tqdm(zarr_files, desc="Processing Zarr files"):
        try:
            archive_path = os.path.join(dataset_path, zarr_file)
            with zarr.ZipStore(archive_path, mode='r') as store:
                replay_buffer = zarr.group(store=store)
                
                # Check if 'pointcloud' data exists
                if 'obs.pointcloud' not in replay_buffer:
                    print(f"Warning: 'obs.pointcloud' not found in {zarr_file}. Skipping.")
                    continue

                pointclouds = replay_buffer['obs.pointcloud']
                num_episodes = replay_buffer.attrs['n_episodes']
                episode_ends = replay_buffer['meta.episode_ends'][:]

                start_idx = 0
                for i in range(num_episodes):
                    end_idx = episode_ends[i]
                    # Extract colors (assuming shape is [time, num_points, 6] and colors are the last 3 dims)
                    # The data is stored flattened, so we need to consider the shape attribute
                    # For simplicity, we'll just grab all color data and concatenate.
                    # This assumes point cloud colors are stored in the last 3 channels.
                    colors = pointclouds[start_idx:end_idx][..., 3:]
                    
                    # Reshape to a long list of [R, G, B] values
                    # The shape of colors will be (time, num_points, 3)
                    # We want to treat every point's color as an independent sample.
                    all_colors.append(colors.reshape(-1, 3))
                    
                    start_idx = end_idx

        except Exception as e:
            print(f"Error processing file {zarr_file}: {e}")
            continue

    if not all_colors:
        print("No color data could be extracted. Aborting.")
        return None, None

    # Concatenate all color values into a single large numpy array
    all_colors_np = np.concatenate(all_colors, axis=0)
    
    # The color values are stored as uint8 (0-255), so we convert to float (0-1)
    all_colors_np = all_colors_np.astype(np.float32) / 255.0

    # Calculate mean and std per channel (R, G, B)
    mean = np.mean(all_colors_np, axis=0)
    std = np.std(all_colors_np, axis=0)

    return mean, std

def main():
    parser = argparse.ArgumentParser(description="Calculate RGB statistics for a point cloud dataset.")
    parser.add_argument('--dataset_path', type=str, required=True, 
                        help='Path to the directory containing your Zarr dataset files (e.g., ./data/your_dataset).')
    
    args = parser.parse_args()

    mean, std = calculate_stats(args.dataset_path)

    if mean is not None and std is not None:
        print("\n" + "="*50)
        print("Calculation Complete!")
        print("="*50)
        print("Copy the following lines into pcdp/dataset/RISE_util.py:")
        print("\n--------------------------------------------------")
        print(f"IMG_MEAN = np.array([{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}])")
        print(f"IMG_STD = np.array([{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}])")
        print("--------------------------------------------------\n")

if __name__ == "__main__":
    main()
