"""
model_test_RISE.py

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

from pcdp.dataset.RISE_stack_dataset import collate_fn
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.common import RISE_transformation as rise_tf
from pcdp.dataset.RISE_util import *

robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])

# Visualization constants
EEF_FRAME_SIZE = 0.03
POINT_SIZE = 2.0
HORIZON = 20

@hydra.main(version_base=None, config_path="../pcdp/config", config_name="train_diffusion_RISE_workspace")
def main(cfg: OmegaConf):
    # Register the 'eval' resolver
    OmegaConf.register_new_resolver("eval", eval)
    
    # --- Visualization Control ---
    vis_gt = False
    vis_pred = True
    # ---------------------------

    # Allow adding new keys to config
    OmegaConf.set_struct(cfg, False)

    # Set default visualization mode if not provided from command line
    if 'vis_mode' not in cfg:
        cfg.vis_mode = 'coord'

    # --- 1. Load Checkpoint Path ---
    # The project_root is defined in the global scope.
    # We use it to create an absolute path to the checkpoint to avoid issues with Hydra's CWD management.
    relative_path = "data/outputs/2025.10.31/RISE_Filter_ON/checkpoints/latest.ckpt"
    checkpoint_path = os.path.join(project_root, relative_path)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path not found at {checkpoint_path}")
        sys.exit(1)

    # --- 2. Setup: Device, Model, Dataset ---
    device = torch.device(cfg.training.device)

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
            else:
                print(f"Error: Could not find 'model' key inside 'state_dicts'.")
                print(f"Available keys: {list(checkpoint['state_dicts'].keys())}")
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
    
    # Get and set normalizer
    print("Computing normalizer from dataset...")
    # normalizer = dataset.get_normalizer(device=device)
    # model.set_normalizer(normalizer)
    print("Using normalizer loaded from checkpoint (do not override).")
    print("Normalizer set on model.")

    dataloader = torch.utils.data.DataLoader(dataset, collate_fn=collate_fn, **cfg.dataloader)
    batch_iter = iter(dataloader)
    
    print(f"Dataset loaded with {len(dataset)} samples.")

    # --- 3. Open3D Visualization Setup ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Model Test All Samples: Prediction vs Ground Truth", 1280, 720)
    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = np.asarray([0.9, 0.9, 0.9])

    pcd = o3d.geometry.PointCloud()
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    geometries = {
        'pcd': pcd,
        'base_frame': base_frame,
        'gt_frames': [],
        'pred_frames': []
    }
    
    # State variables for iterating through all samples
    first_load = True
    global_batch_counter = 0
    current_batch = None
    current_sample_idx = -1

    def load_next_sample(vis):
        nonlocal batch_iter, first_load, global_batch_counter, current_batch, current_sample_idx

        # Determine batch size from config
        batch_size = cfg.dataloader.get('batch_size', 1)

        # Check if we need to load a new batch or advance the sample index
        if current_batch is None or current_sample_idx >= batch_size - 1:
            try:
                current_batch = next(batch_iter)
                current_sample_idx = 0
                global_batch_counter += 1
            except StopIteration:
                print("All samples and batches processed. Press 'Q' to exit.")
                return
        else:
            current_sample_idx += 1
        
        # Check if the new batch is smaller than the configured batch_size (last batch)
        actual_batch_size = current_batch['action'].shape[0]
        if current_sample_idx >= actual_batch_size:
            try:
                current_batch = next(batch_iter)
                current_sample_idx = 0
                global_batch_counter += 1
                actual_batch_size = current_batch['action'].shape[0]
            except StopIteration:
                print("All samples and batches processed. Press 'Q' to exit.")
                return

        sample_idx = current_sample_idx
        batch = current_batch

        # --- 4. Get Data for the current sample_idx ---
        # Observation
        coords = batch['input_coords_list']
        feats = batch['input_feats_list']
        
        # Filter point cloud for the current sample
        point_indices = (coords[:, 0] == sample_idx)
        obs_pcd_np = feats[point_indices].numpy()
        obs_coords = coords[point_indices]

        # Get robot obs and ground truth for the current sample

        gt_action_10d = batch['action'][sample_idx].to(device) # T, 10
        
        # Convert ground truth 6D rot to euler for visualization
        gt_action_euler = rise_tf.xyz_rot_transform(
            gt_action_10d[..., :9].cpu().numpy(),
            from_rep='rotation_6d',
            to_rep='euler_angles',
            to_convention='ZYX'
        )
        gt_action_euler = np.concatenate([gt_action_euler, gt_action_10d[..., 9:].cpu().numpy()], axis=-1)
        gt_action_euler = torch.from_numpy(gt_action_euler).to(device)

        # --- 5. Model Prediction ---
        with torch.no_grad():
            # Prepare observation for model
            model_obs_coords = obs_coords.clone()
            model_obs_coords[:, 0] = 0 # Reset batch index for the single-sample batch
            obs_coords_batch = model_obs_coords.to(device)
            obs_feats_batch = torch.from_numpy(obs_pcd_np).to(device)
            obs_tensor = ME.SparseTensor(features=obs_feats_batch, coordinates=obs_coords_batch, device=device)
            
            # Predict action. The model now returns un-normalized actions directly.
            pred_action_10d = model.forward(obs_tensor, batch_size=1)
            pred_action_10d = pred_action_10d.squeeze(0) # Remove batch dimension
            
            # --- 6. Post-processing and Comparison ---
            # Convert predicted 6D rot to euler for comparison and visualization
            pred_action_euler = rise_tf.xyz_rot_transform(
                pred_action_10d[..., :9].cpu().numpy(),
                from_rep='rotation_6d',
                to_rep='euler_angles',
                to_convention='ZYX'
            )
            pred_action_euler = np.concatenate([pred_action_euler, pred_action_10d[..., 9:].cpu().numpy()], axis=-1)
            pred_action_euler = torch.from_numpy(pred_action_euler).to(device)

            # Calculate MSE loss for the whole action (pos, rot, gripper)
            pos_loss = F.mse_loss(pred_action_euler, gt_action_euler)
            print(f"\n--- Batch {global_batch_counter}, Sample {sample_idx + 1}/{actual_batch_size} ---")

            # DEBUG PRINT: Compare GT and Pred coordinates for the first timestep
            gt_pos_t0 = gt_action_euler[0, :3].cpu().numpy()
            pred_pos_t0 = pred_action_euler[0, :3].cpu().numpy()
            offset_vec = gt_pos_t0 - pred_pos_t0
            print(f"DEBUG: First Timestep Position (x,y,z)")
            print(f"  - Ground Truth: {gt_pos_t0}")
            print(f"  - Prediction:   {pred_pos_t0}")
            print(f"  - Offset (GT-Pred): {offset_vec}")

            print(f"Total Action MSE Loss: {pos_loss.item():.6f}")

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
        # Un-normalize color for visualization
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
                # Common calculations for prediction
                pred_pose_matrix = rise_tf.rot_trans_mat(pred_action_euler[t, :3].cpu(), pred_action_euler[t, 3:6].cpu())
                gripper_val = pred_action_euler[t, -1].item()

                # --- COORD + GRIPPER SPHERE MODE ---
                if cfg.vis_mode == 'coord':
                    # 1. Create the coordinate frame (default RGB colors, no fade)
                    pred_frame_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=EEF_FRAME_SIZE)
                    pred_frame_coord.transform(pred_pose_matrix)
                    geometries['pred_frames'].append(pred_frame_coord)
                    
                    # 2. Create the gripper status sphere
                    if gripper_val > 60:
                        base_color = np.array([1.0, 0.0, 0.0]) # Red
                    else:
                        base_color = np.array([0.0, 0.0, 1.0]) # Blue
                    
                    fade_factor = 1.0 - (t / HORIZON) * 0.9
                    final_color = base_color * fade_factor
                    
                    gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                    gripper_sphere.compute_vertex_normals()
                    gripper_sphere.paint_uniform_color(final_color)
                    center_k = pred_pose_matrix[:3, 3]
                    gripper_sphere.translate(center_k)
                    geometries['pred_frames'].append(gripper_sphere)

                # --- SPHERE-ONLY MODE ---
                elif cfg.vis_mode == 'sphere':
                    if gripper_val > 0.8:
                        base_color = np.array([1.0, 0.0, 0.0]) # Red
                    else:
                        base_color = np.array([0.0, 0.0, 1.0]) # Blue
                    
                    fade_factor = 1.0 - (t / HORIZON) * 0.9
                    final_color = base_color * fade_factor
                    
                    pred_frame = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    pred_frame.compute_vertex_normals()
                    pred_frame.paint_uniform_color(final_color)
                    
                    center_k = pred_pose_matrix[:3, 3]
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

    # --- 8. Run Visualizer ---
    vis.register_key_callback(ord("N"), load_next_sample)
    vis.register_key_callback(ord("Q"), lambda v: v.close())
    
    print("\n" + "="*50)
    print("Controls: [N] for next sample (advances through batch), [Q] to quit.")
    print("Legend: GT Trajectory (Green->Red), Predicted Trajectory (Blue->Yellow)")
    print("="*50 + "\n")

    load_next_sample(vis) # Load the first sample
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()