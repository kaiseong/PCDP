
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
    printed_once = False  # Flag to print debug info only once

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
                
                # Check for the required data groups and keys
                if 'data' not in replay_buffer or 'meta' not in replay_buffer:
                    print(f"Warning: 'data' or 'meta' group not found in {zarr_file}. Skipping.")
                    continue
                
                data_group = replay_buffer['data']
                meta_group = replay_buffer['meta']

                if 'pointcloud' not in data_group or 'episode_ends' not in meta_group:
                    print(f"Warning: 'pointcloud' or 'meta.episode_ends' not found in {zarr_file}. Skipping.")
                    continue

                pointclouds = data_group['pointcloud']
                # Assuming n_episodes is an attribute of the root group
                num_episodes = replay_buffer.attrs.get('n_episodes', len(meta_group['episode_ends']))
                episode_ends = meta_group['episode_ends'][:]

                start_idx = 0
                for i in range(num_episodes):
                    end_idx = episode_ends[i]
                    # episode_data is an object array, where each element is a single frame's point cloud array.
                    episode_data = pointclouds[start_idx:end_idx]

                    # Iterate through each frame's point cloud in the episode
                    for point_cloud_array in episode_data:
                        # Ensure the array is 2D and has enough columns for color
                        if isinstance(point_cloud_array, np.ndarray) and point_cloud_array.ndim == 2 and point_cloud_array.shape[1] >= 6:
                            # --- DEBUG: Print first 5 color values ---
                            if not printed_once:
                                print("\n--- DEBUG: First 5 Color Values from the first valid point cloud ---")
                                # The data is float32, not uint8 yet. The script converts later.
                                print(point_cloud_array[:40, 3:6])
                                print("---------------------------------------------------------------------\
")
                                printed_once = True
                            # --- END DEBUG ---

                            # Extract colors (assuming xyzrgb...)
                            colors = point_cloud_array[:, 3:6]
                            all_colors.append(colors)
                    
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
