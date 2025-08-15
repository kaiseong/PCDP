
"""
model_test.py

Usage:
python debug/model_test.py checkpoint_path=/path/to/your/checkpoint.ckpt

Description:
This script loads a trained diffusion policy model and a dataset.
It then performs the following steps for each sample in the dataset:
1. Takes an observation (point cloud).
2. Uses the model to predict the action sequence.
3. Compares the predicted action sequence with the ground-truth action sequence.
4. Visualizes both the ground-truth and predicted action trajectories in 3D using Open3D.
"""

import sys
import os
import numpy as np
import torch
import open3d as o3d
import hydra
from omegaconf import OmegaConf, MISSING
import MinkowskiEngine as ME
import torch.nn.functional as F

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pcdp.dataset.RISE_stack_pc_dataset import collate_fn
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.common import RISE_transformation as rise_tf

# --- Constants ---
# Normalization constants from the dataset, converted to tensors
TRANS_MIN = torch.from_numpy(np.array([0.109933, -0.265188, 0.07253])).float()
TRANS_MAX = torch.from_numpy(np.array([0.387143, -0.018333, 0.265922])).float()

# Visualization constants
EEF_FRAME_SIZE = 0.03
POINT_SIZE = 2.0
HORIZON = 16

# --- Normalization Functions ---
def unnormalize_action(normalized_action):
    """
    Un-normalizes an action from [-1, 1] range to the original scale.
    """
    unnormalized_action = normalized_action.clone()
    
    # Un-normalize translation (first 3 dimensions)
    unnormalized_action[..., :3] = (normalized_action[..., :3] + 1) / 2 * (TRANS_MAX - TRANS_MIN) + TRANS_MIN
    
    # Un-normalize gripper (last dimension)
    unnormalized_action[..., -1] = (normalized_action[..., -1] + 1) / 2
    
    return unnormalized_action

@hydra.main(version_base=None, config_path="../pcdp/config", config_name="train_diffusion_RISE_workspace")
def main(cfg: OmegaConf):
    # --- Visualization Control ---
    vis_gt = False
    vis_pred = True
    # ---------------------------

    # Allow adding new keys to config
    OmegaConf.set_struct(cfg, False)

    # --- 1. Load Checkpoint Path from command line ---
    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        print("Error: Please provide the checkpoint path using the override 'checkpoint_path=/path/to/your.ckpt'")
        sys.exit(1)
    
    checkpoint_path = cfg.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path not found at {checkpoint_path}")
        sys.exit(1)

    # --- 2. Setup: Device, Model, Dataset ---
    device = torch.device(cfg.training.device)

    # Move normalization constants to device
    global TRANS_MIN, TRANS_MAX
    TRANS_MIN = TRANS_MIN.to(device)
    TRANS_MAX = TRANS_MAX.to(device)

    # Load model from config
    model: RISEPolicy = hydra.utils.instantiate(cfg.policy).to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # --- Smartly load model state_dict ---
    model_state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dicts' in checkpoint and isinstance(checkpoint['state_dicts'], dict):
            if 'model' in checkpoint['state_dicts']:
                model_state_dict = checkpoint['state_dicts']['model']
                print("Loaded state_dict from checkpoint['state_dicts']['model']")
            elif 'policy' in checkpoint['state_dicts']:
                model_state_dict = checkpoint['state_dicts']['policy']
                print("Loaded state_dict from checkpoint['state_dicts']['policy']")
            else:
                print(f"Error: Could not find 'model' or 'policy' key inside 'state_dicts'.")
                print(f"Available keys in 'state_dicts': {list(checkpoint['state_dicts'].keys())}")
                sys.exit(1)
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
            print("Loaded state_dict from 'model' key.")
        elif 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
            print("Loaded state_dict from 'state_dict' key.")
        else:
            print(f"Error: Checkpoint dictionary does not contain a known model key.")
            print(f"Available keys: {list(checkpoint.keys())}")
            sys.exit(1)
    else:
        # If the checkpoint is not a dict, assume it's the state_dict itself
        model_state_dict = checkpoint
        print("Checkpoint is not a dictionary, loading it directly as state_dict.")

    if model_state_dict:
        model.load_state_dict(model_state_dict)
        print("Model loaded successfully.")
    else:
        print("Error: Could not extract model state_dict from checkpoint.")
        sys.exit(1)

    model.eval()

    # Load dataset from config
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **cfg.dataloader)
    batch_iter = iter(dataloader)
    
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- 3. Open3D Visualization Setup ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Model Test: Prediction vs Ground Truth", 1280, 720)
    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = np.asarray([0.1, 0.1, 0.1])

    pcd = o3d.geometry.PointCloud()
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    geometries = {
        'pcd': pcd,
        'base_frame': base_frame,
        'gt_frames': [],
        'pred_frames': []
    }
    
    first_load = True
    global_sample_counter = 0

    def load_next_sample(vis):
        nonlocal batch_iter, first_load, global_sample_counter

        try:
            batch = next(batch_iter)
            sample_idx = 0 # Visualize the first sample of the batch

            # --- 4. Get Data ---
            # Observation
            coords = batch['input_coords_list']
            feats = batch['input_feats_list']
            point_indices = (coords[:, 0] == sample_idx)
            
            obs_pcd_np = feats[point_indices].numpy()
            obs_coords = coords[point_indices]
            
            # Ground truth action
            gt_action_euler = batch['action_euler'][sample_idx].to(device) # T, 7

            # --- 5. Model Prediction ---
            with torch.no_grad():
                # Prepare observation for model
                obs_coords_batch = obs_coords.to(device)
                obs_feats_batch = torch.from_numpy(obs_pcd_np).to(device)
                obs_tensor = ME.SparseTensor(features=obs_feats_batch, coordinates=obs_coords_batch, device=device)
                
                # Predict action by calling the forward method without the 'actions' parameter
                pred_action_normalized = model.forward(obs_tensor, batch_size=1) # T, 10 (6d_rot + gripper)
                
                # --- 6. Post-processing and Comparison ---
                # Un-normalize predicted action
                pred_action_10d = unnormalize_action(pred_action_normalized)
                
                # Convert predicted 6D rot to euler for comparison
                pred_action_euler = rise_tf.xyz_rot_transform(
                    pred_action_10d[..., :9].cpu().numpy(),
                    from_rep='rotation_6d',
                    to_rep='euler_angles',
                    to_convention='XYZ'
                )
                pred_action_euler = np.concatenate([pred_action_euler, pred_action_10d[..., 9:].cpu().numpy()], axis=-1)
                pred_action_euler = torch.from_numpy(pred_action_euler).to(device)
                pred_action_euler = pred_action_euler.squeeze(0)

                # Calculate MSE loss for position
                pos_loss = F.mse_loss(pred_action_euler[:, :3], gt_action_euler[:, :3])
                print(f"\n--- Sample {global_sample_counter} ---")
                print(f"Positional MSE Loss: {pos_loss.item():.6f}")

            # --- 7. Visualization ---
            # Clear previous geometries
            if not first_load:
                vis.remove_geometry(geometries['pcd'], reset_bounding_box=False)
                for frame in geometries['gt_frames'] + geometries['pred_frames']:
                    vis.remove_geometry(frame, reset_bounding_box=False)
            geometries['gt_frames'].clear()
            geometries['pred_frames'].clear()

            # Update point cloud
            xyz = obs_pcd_np[:, :3]
            rgb = obs_pcd_np[:, 3:6]
            geometries['pcd'].points = o3d.utility.Vector3dVector(xyz)
            geometries['pcd'].colors = o3d.utility.Vector3dVector(rgb)
            
            # Create frames for GT and Pred trajectories
            for t in range(min(HORIZON, len(gt_action_euler))):
                if vis_gt:
                    # Ground Truth (Green -> Red)
                    color_gt = [t / HORIZON, 1 - (t / HORIZON), 0]
                    gt_pose_matrix = rise_tf.rot_trans_mat(gt_action_euler[t, :3].cpu(), gt_action_euler[t, 3:6].cpu())
                    gt_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=EEF_FRAME_SIZE)
                    gt_frame.transform(gt_pose_matrix)
                    gt_frame.paint_uniform_color(color_gt)
                    geometries['gt_frames'].append(gt_frame)

                if vis_pred:
                    # Prediction (Blue -> Yellow)
                    color_pred = [0, t / HORIZON, 1 - (t / HORIZON)]
                    pred_pose_matrix = rise_tf.rot_trans_mat(pred_action_euler[t, :3].cpu(), pred_action_euler[t, 3:6].cpu())
                    
                    # pred_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=EEF_FRAME_SIZE)
                    # pred_frame.transform(pred_pose_matrix)
                    # pred_frame.paint_uniform_color(color_pred)
                    # geometries['pred_frames'].append(pred_frame)

                    # # green→red: (r,g,b) = (w, 1-w, 0)
                    pred_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    pred_frame.compute_vertex_normals()
                    pred_frame.paint_uniform_color(color_pred)

                    center_k = pred_pose_matrix[:3, 3]  # (x,y,z)
                    pred_frame.translate(center_k)
                    geometries['pred_frames'].append(pred_frame)

            # Add geometries to visualizer
            if first_load:
                vis.add_geometry(geometries['pcd'], reset_bounding_box=True)
                vis.add_geometry(geometries['base_frame'], reset_bounding_box=False)
            else:
                vis.add_geometry(geometries['pcd'], reset_bounding_box=False)

            if vis_gt:
                for frame in geometries['gt_frames']:
                    vis.add_geometry(frame, reset_bounding_box=False)
            
            if vis_pred:
                for frame in geometries['pred_frames']:
                    vis.add_geometry(frame, reset_bounding_box=False)

            if first_load:
                # Set initial camera view
                ctr = vis.get_view_control()
                ctr.set_lookat([0.2, 0, 0.1])
                ctr.set_front([-0.2, -0.8, -0.5])
                ctr.set_up([0, 0, 1])
                ctr.set_zoom(0.7)
                first_load = False
            
            vis.update_renderer()

            global_sample_counter += 1

        except StopIteration:
            print("All samples processed. Press 'Q' to exit.")

    # --- 8. Run Visualizer ---
    vis.register_key_callback(ord("N"), load_next_sample)
    vis.register_key_callback(ord("Q"), lambda v: v.close())
    
    print("\n" + "="*50)
    print("Controls: [N] for next sample, [Q] to quit.")
    print("Legend: GT Trajectory (Green->Red), Predicted Trajectory (Blue->Yellow)")
    print("="*50 + "\n")

    load_next_sample(vis) # Load the first sample
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
