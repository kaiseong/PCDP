import argparse
import zarr
import numpy as np

def main():
    """
    Calculates and prints the minimum and maximum values for the 'action' key
    in a Zarr dataset. These values correspond to TRANS_MIN and TRANS_MAX
    used for normalization in the training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Calculate TRANS_MIN and TRANS_MAX for a dataset's action data."
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help="Path to the Zarr dataset file (e.g., 'data/your_dataset.zarr')."
    )
    parser.add_argument(
        '--action-key',
        type=str,
        default='action',
        help="The key for the action data in the Zarr dataset. Defaults to 'action'."
    )
    args = parser.parse_args()

    print(f"Loading dataset from: {args.dataset_path}")

    try:
        dataset = zarr.open(args.dataset_path, 'r')
    except Exception as e:
        print(f"Error: Failed to open Zarr dataset at '{args.dataset_path}'.")
        print(f"Details: {e}")
        return

    # Navigate to the correct data group
    data_group = dataset
    if 'data' in dataset and isinstance(dataset['data'], zarr.Group):
        data_group = dataset['data']

    # Check if the action key exists in the determined group
    if args.action_key not in data_group:
        print(f"Error: Action key '{args.action_key}' not found in the dataset.")
        print(f"Available keys in the root group: {list(dataset.keys())}")
        if 'data' in dataset and isinstance(dataset['data'], zarr.Group):
            print(f"Available keys in the 'data' group: {list(dataset['data'].keys())}")
        return

    # Load action data into memory
    action_data = data_group[args.action_key][:]
    print(f"Successfully loaded action data with shape: {action_data.shape}")

    # Calculate min and max across the first axis (the sequence axis)
    trans_min = np.min(action_data, axis=0)
    trans_max = np.max(action_data, axis=0)

    # Print the results in a copy-paste friendly format
    print("\n" + "="*50)
    print("Action Statistics")
    print("-"*50)
    print(f"TRANS_MIN: {trans_min.tolist()}")
    print(f"TRANS_MAX: {trans_max.tolist()}")
    print("="*50)
    print("\nCopy and paste these lists into the 'normalize_min' and 'normalize_max' fields")
    print("in your YAML configuration file (e.g., train_diffusion_RISE_workspace.yaml).")


if __name__ == "__main__":
    main()
