import numpy as np
import zarr
import os
from tqdm import tqdm
import argparse

def calculate_lowdim_stats(dataset_path):
    """
    Calculates normalization stats for low-dimensional action and observation spaces
    from all Zarr archives in a directory.

    Args:
        dataset_path (str): Path to the directory containing the Zarr archives.

    Example:
        python debug/calculate_lowdim_stats.py --dataset_path ./data/your_dataset_directory

    Returns:
        dict: A dictionary containing the calculated min and max values.
    """
    all_actions = []
    all_robot_obs = []

    zarr_files = [f for f in os.listdir(dataset_path) if f.endswith('.zarr.zip')]
    if not zarr_files:
        print(f"Error: No .zarr.zip files found in '{dataset_path}'")
        return None

    print(f"Found {len(zarr_files)} Zarr archives. Processing...")

    for zarr_file in tqdm(zarr_files, desc="Processing Zarr files"):
        try:
            archive_path = os.path.join(dataset_path, zarr_file)
            with zarr.ZipStore(archive_path, mode='r') as store:
                replay_buffer = zarr.group(store=store)
                
                if 'data' not in replay_buffer:
                    print(f"Warning: 'data' group not found in {zarr_file}. Skipping.")
                    continue
                
                data_group = replay_buffer['data']

                if 'action' in data_group:
                    all_actions.append(data_group['action'][:])
                else:
                    print(f"Warning: 'action' array not found in {zarr_file}.")

                if 'robot_obs' in data_group:
                    all_robot_obs.append(data_group['robot_obs'][:])
                else:
                    print(f"Warning: 'robot_obs' array not found in {zarr_file}.")

        except Exception as e:
            print(f"Error processing file {zarr_file}: {e}")
            continue

    if not all_actions and not all_robot_obs:
        print("No 'action' or 'robot_obs' data could be extracted. Aborting.")
        return None

    stats = dict()

    # --- Action Stats ---
    if all_actions:
        # Shape of each element in all_actions: (n_steps, 7)
        # Concatenate all episodes and steps
        concatenated_actions = np.concatenate(all_actions, axis=0)
        print(f"Total action steps processed: {concatenated_actions.shape[0]}")
        
        # action: [x, y, z, roll, pitch, yaw, gripper_on_off]
        action_translation = concatenated_actions[:, :3]
        stats['ACTION_TRANS_MAX'] = np.max(action_translation, axis=0)
        stats['ACTION_TRANS_MIN'] = np.min(action_translation, axis=0)
    
    # --- Robot Observation Stats ---
    if all_robot_obs:
        # Shape of each element in all_robot_obs: (n_steps, 7)
        concatenated_obs = np.concatenate(all_robot_obs, axis=0)
        print(f"Total robot_obs steps processed: {concatenated_obs.shape[0]}")

        # robot_obs: [x, y, z, roll, pitch, yaw, gripper_width]
        obs_translation = concatenated_obs[:, :3]
        gripper_width = concatenated_obs[:, 6]

        stats['OBS_TRANS_MAX'] = np.max(obs_translation, axis=0)
        stats['OBS_TRANS_MIN'] = np.min(obs_translation, axis=0)
        stats['GRIP_MAX'] = np.max(gripper_width)
        stats['GRIP_MIN'] = np.min(gripper_width)

    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Calculate low-dimensional statistics from all Zarr files in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--dataset_path', type=str, required=True,
                        help="Path to the directory containing your Zarr dataset files.\n"
                             "Example: python debug/calculate_lowdim_stats.py --dataset_path ./data/real_stack/recorder_data")

    args = parser.parse_args()

    stats = calculate_lowdim_stats(args.dataset_path)

    if stats:
        print("\n" + "="*80)
        print("      Normalization Parameter Calculation Complete!")
        print("="*80)
        print("Copy the following lines into your configuration file (e.g., pcdp/config/task/your_task.yaml):\n")
        
        if 'ACTION_TRANS_MAX' in stats:
            max_vals = stats['ACTION_TRANS_MAX']
            min_vals = stats['ACTION_TRANS_MIN']
            print("  # Action space normalization")
            print(f"  action_trans_max: [{max_vals[0]:.6f}, {max_vals[1]:.6f}, {max_vals[2]:.6f}]")
            print(f"  action_trans_min: [{min_vals[0]:.6f}, {min_vals[1]:.6f}, {min_vals[2]:.6f}]")
            print("-" * 40)

        if 'OBS_TRANS_MAX' in stats:
            max_vals = stats['OBS_TRANS_MAX']
            min_vals = stats['OBS_TRANS_MIN']
            print("  # Observation space normalization")
            print(f"  obs_trans_max: [{max_vals[0]:.6f}, {max_vals[1]:.6f}, {max_vals[2]:.6f}]")
            print(f"  obs_trans_min: [{min_vals[0]:.6f}, {min_vals[1]:.6f}, {min_vals[2]:.6f}]")
            print(f"  grip_max: {stats['GRIP_MAX']:.6f}")
            print(f"  grip_min: {stats['GRIP_MIN']:.6f}")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    main()
