import sys
import os
import time
from pathlib import Path
import numpy as np
import torch
import open3d as o3d
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from pcdp.common.replay_buffer import ReplayBuffer
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.policy.diffusion_SPEC_policy_mono import SPECPolicyMono
from pcdp.common import RISE_transformation as rise_tf
from pcdp.real_world.real_data_pc_conversion import LowDimPreprocessor, PointCloudPreprocessor
from pytorch3d.transforms import rotation_6d_to_matrix

import MinkowskiEngine as ME

# ==================== USER CONFIGURATION ====================
# 여기에 경로와 에피소드 인덱스를 직접 입력하세요.
# -------------------------------------------------------------
# 모델 체크포인트 파일의 절대 경로
CHECKPOINT_PATH = "/home/nscl/diffusion_policy/data/outputs/SPEC_SOR_120/checkpoints/latest.ckpt" 
# 데이터셋 디렉토리의 절대 경로
DATASET_PATH = "/home/nscl/diffusion_policy/data/please_please/recorder_data"
# 시각화할 에피소드 번호
EPISODE_IDX = 163
# 'N' 키를 눌렀을 때 건너뛸 스텝 수
STEP_SIZE = 3
# 시각화 모드 ('sphere' 또는 'frame')
VIS_MODE = 'sphere'
# 사용할 모델 타입 ('RISE' 또는 'SPEC_mono')
MODEL_TYPE = 'SPEC_mono'
# ============================================================


# ==================== Visualization Parameters ====================
VIS_MODE_SPHERE = 'sphere'
VIS_MODE_FRAME = 'frame'

SPHERE_RADIUS = 0.005
COLOR_MAP_SPHERE = plt.get_cmap('viridis')

FRAME_SIZE = 0.025
FRAME_ALPHA_MIN = 0.2
FRAME_ALPHA_MAX = 1.0
VOXEL_SIZE = 0.005

EEF_TO_GRIP_TRANSLATION = np.array([0.0, 0.0, 0.12], dtype=np.float64)

# ==================== State Management ====================
class VisController:
    def __init__(self, total_steps, step_size=1):
        self.total_steps = total_steps
        self.current_step = 0
        self.step_size = step_size
        self.vis_changed = True
        self.denoising_step = 0
        self.denoising_total_steps = 0
        self.run_denoising_step = False
        self.start_new_prediction = True

    def next_scene(self, vis):
        if self.current_step < self.total_steps - 1:
            self.current_step = min(self.current_step + self.step_size, self.total_steps - 1)
            self.vis_changed = True
            self.start_new_prediction = True
            print(f"Scene: {self.current_step}/{self.total_steps - 1}")
        return False

    def next_denoising_step(self, vis):
        if self.denoising_step < self.denoising_total_steps:
            self.run_denoising_step = True
            print(f"Denoising Step: {self.denoising_step + 1}/{self.denoising_total_steps}")
        else:
            print("Denoising complete for this scene. Press 'N' for next scene.")
        return False

# ==================== Helper Functions ====================
def load_model(checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    config_path = Path(checkpoint_path).parent.parent / '.hydra' / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Hydra config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    
    if MODEL_TYPE == 'RISE':
        policy_class = RISEPolicy
    elif MODEL_TYPE == 'SPEC_mono':
        policy_class = SPECPolicyMono
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}")
    
    cfg.policy._target_ = f"{policy_class.__module__}.{policy_class.__name__}"
    model = hydra.utils.instantiate(cfg.policy).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_state = state_dict['state_dicts']['model'] if 'state_dicts' in state_dict else state_dict

    model.load_state_dict(model_state)
    model.eval()
    
    if 'state_dicts' in state_dict and 'normalizer' in state_dict['state_dicts']:
        normalizer_state = state_dict['state_dicts']['normalizer']
        model.normalizer.load_state_dict(normalizer_state)

    print(f"Model '{MODEL_TYPE}' and normalizer loaded successfully.")
    return model, cfg

def load_episode_data(dataset_path, episode_idx):
    episode_name = f"episode_{episode_idx:04d}"
    episode_dir = Path(dataset_path) / episode_name
    obs_path = episode_dir / 'obs_replay_buffer.zarr'
    if not obs_path.exists():
        raise FileNotFoundError(f"Episode data not found: {obs_path}")
    
    replay_buffer = ReplayBuffer.copy_from_path(str(obs_path), backend='numpy')
    episode_data = replay_buffer.get_episode(0)
    print(f"Loaded episode {episode_name} with {len(episode_data['pointcloud'])} steps.")
    return episode_data

def create_frame_geometry(transform, alpha):
    points = np.array([[0, 0, 0], [FRAME_SIZE, 0, 0], [0, FRAME_SIZE, 0], [0, 0, FRAME_SIZE]])
    lines = [[0, 1], [0, 2], [0, 3]]
    base_colors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    grey = np.array([0.8, 0.8, 0.8])
    colors = [color * alpha + grey * (1 - alpha) for color in base_colors]
    
    frame = o3d.geometry.LineSet()
    frame.points = o3d.utility.Vector3dVector(points)
    frame.lines = o3d.utility.Vector2iVector(lines)
    frame.colors = o3d.utility.Vector3dVector(colors)
    frame.transform(transform)
    return frame

def create_action_geometries(trajectory, vis_mode):
    geometries = []
    num_steps = trajectory.shape[0]
    
    for i in range(num_steps):
        pose_10d = trajectory[i]

        # EEF → GRIP 변환 적용
        grip_pos, grip_rot = eef_to_grip_from_action_pose(pose_10d)

        if vis_mode == VIS_MODE_SPHERE:
            t = i / max(1, num_steps - 1)
            color = COLOR_MAP_SPHERE(1.0 - t)[:3] 
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS)
            sphere.paint_uniform_color(color)
            sphere.translate(grip_pos)  # ← EEF 위치가 아니라 GRIP 위치
            geometries.append(sphere)
        
        elif vis_mode == VIS_MODE_FRAME:
            transform = np.eye(4)
            transform[:3, :3] = grip_rot     # ← 그리퍼 회전
            transform[:3, 3]  = grip_pos     # ← 그리퍼 위치
            
            alpha = FRAME_ALPHA_MIN + (FRAME_ALPHA_MAX - FRAME_ALPHA_MIN) * (1 - i / max(1, num_steps - 1))
            frame = create_frame_geometry(transform, alpha)
            geometries.append(frame)
            
    return geometries



def eef_to_grip_from_action_pose(pose_10d):
    pos_eef = pose_10d[:3].astype(np.float64)
    rot6d   = pose_10d[3:9].astype(np.float32)

    rot_mat_eef = rotation_6d_to_matrix(
        torch.from_numpy(rot6d).unsqueeze(0)
    ).squeeze(0).numpy()

    T_world_eef = np.eye(4, dtype=np.float64)
    T_world_eef[:3, :3] = rot_mat_eef
    T_world_eef[:3, 3]  = pos_eef

    T_eef_to_grip = np.eye(4, dtype=np.float64)
    T_eef_to_grip[:3, 3] = EEF_TO_GRIP_TRANSLATION

    T_world_grip = T_world_eef @ T_eef_to_grip

    pos_grip = T_world_grip[:3, 3]
    rot_grip = T_world_grip[:3, :3]
    return pos_grip, rot_grip



def main():
    dataset_path = DATASET_PATH
    checkpoint_path = CHECKPOINT_PATH
    episode_idx = EPISODE_IDX
    step_size = STEP_SIZE
    vis_mode = VIS_MODE

    if "/path/to/" in checkpoint_path or "/path/to/" in dataset_path:
        print("="*60)
        print("경고: 스크립트 상단의 CHECKPOINT_PATH와 DATASET_PATH를")
        print("      실제 파일 경로로 수정해야 합니다.")
        print("="*60)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, cfg = load_model(checkpoint_path, device)
    episode_data = load_episode_data(dataset_path, episode_idx)
    
    low_dim_preprocessor = LowDimPreprocessor(**cfg.task.dataset.low_dim_preprocessor_config)
    pc_preprocessor = PointCloudPreprocessor(**cfg.task.dataset.pc_preprocessor_config)
    
    controller = VisController(len(episode_data['pointcloud']), step_size)
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(f"Action Denoising Visualizer | Episode {episode_idx} | Model: {MODEL_TYPE}", width=1280, height=720)
    opt = vis.get_render_option()
    opt.point_size = 3.0

    is_running = True
    def cb_exit(v):
        nonlocal is_running
        is_running = False
        return False

    vis.register_key_callback(ord("N"), controller.next_scene)
    vis.register_key_callback(ord("M"), controller.next_denoising_step)
    vis.register_key_callback(ord("Q"), cb_exit)

    pcd = o3d.geometry.PointCloud()
    action_geometries = []
    
    denoising_state = {"global_cond": None, "trajectory": None, "timesteps": None}

    print("\n" + "="*20 + " CONTROLS " + "="*20)
    print("  [N] : Go to next scene (skip steps)")
    print("  [M] : Run one step of action denoising")
    print("  [Q] : Quit")
    print("="*50 + "\n")

    first_render = True
    while is_running:
        if controller.vis_changed:
            pc_raw = episode_data['pointcloud'][controller.current_step].astype(np.float32)
            pc_processed = pc_preprocessor.process(pc_raw)
            pcd.points = o3d.utility.Vector3dVector(pc_processed[:, :3])
            pcd.colors = o3d.utility.Vector3dVector(pc_processed[:, 3:6])
            
            if first_render:
                vis.add_geometry(pcd)
            else:
                vis.update_geometry(pcd)
            
            controller.vis_changed = False
            controller.start_new_prediction = True

        if controller.start_new_prediction:
            for geom in action_geometries:
                vis.remove_geometry(geom, reset_bounding_box=False)
            action_geometries.clear()
            
            # ================== CORRECTED LOGIC ==================
            # Preprocess the point cloud to add the 7th channel
            
            # For RISE, we need 6D input, for SPEC_mono, 7D
            if MODEL_TYPE == 'RISE':
                pc_input = pc_processed[:, :6]
            else:
                pc_input = pc_processed

            coords = np.floor(pc_input[:, :3] / VOXEL_SIZE).astype(np.int32)
            coords = np.ascontiguousarray(coords)
            feats = pc_input.astype(np.float32)
            # =====================================================

            s_coords, s_feats = ME.utils.sparse_collate([coords], [feats])
            s_tensor = ME.SparseTensor(features=s_feats.to(device), coordinates=s_coords.to(device))

            feats, coords = s_tensor.F, s_tensor.C
            colors = feats[:, 3:6]
            c = feats[:, 6:7]
            colors = model.normalizer['pointcloud_color'].normalize(colors)
            new_feats = torch.cat([feats[:, :3], colors, c], dim=1)
            cloud=ME.SparseTensor(features=new_feats, coordinates=coords, device=s_tensor.device)

            with torch.no_grad():
                global_cond = None
                if MODEL_TYPE == 'SPEC_mono':
                    robot_eef_pose_raw = episode_data['robot_eef_pose'][controller.current_step].astype(np.float64)
                    robot_gripper_raw = episode_data['robot_gripper'][controller.current_step].flatten().astype(np.float64)
                    robot_obs_raw = np.concatenate([robot_eef_pose_raw, robot_gripper_raw[:1]])

                    transformed_obs_7d = low_dim_preprocessor.TF_process(robot_obs_raw[np.newaxis, :]).squeeze(0)
                    obs_pose_euler = transformed_obs_7d[:6]
                    obs_gripper = transformed_obs_7d[6:]
                    obs_9d = rise_tf.xyz_rot_transform(obs_pose_euler, from_rep='euler_angles', to_rep='rotation_6d', from_convention='ZYX')
                    obs_10d = np.concatenate([obs_9d, obs_gripper], axis=-1)
                    robot_obs_tensor = torch.from_numpy(obs_10d).to(device).float().unsqueeze(0)
                    if robot_obs_tensor.dim() == 2:  # [B,10] -> [B,1,10]
                        robot_obs_tensor = robot_obs_tensor.unsqueeze(1)
                    obs_trans = robot_obs_tensor[:, :, :3]
                    obs_rot = robot_obs_tensor[:, :, 3:9]
                    obs_grip = robot_obs_tensor[:, :, 9:10]
                    norm_obs_trans = model.normalizer['obs_translation'].normalize(obs_trans)
                    norm_obs_grip = model.normalizer['obs_gripper'].normalize(obs_grip)
                    robot_obs = torch.cat([norm_obs_trans, obs_rot, norm_obs_grip], dim=-1)
                    
                    src, pos, pad = model.encoder(cloud, batch_size=1, max_num_token=100)
                    readout_seq = model.transformer(src, pad, model.readout_embed.weight, pos)[-1]  # [B,1,D]
                    readout = readout_seq[:, 0, :] 
                    global_cond = torch.cat([readout, robot_obs[:,0,:]], dim =-1)
                else: # RISE
                    src, pos, src_padding_mask = model.sparse_encoder(cloud, batch_size=1)
                    readout = model.transformer(src, src_padding_mask, model.readout_embed.weight, pos)[-1]
                    global_cond = readout[:, 0]
                denoising_state["global_cond"] = global_cond

            scheduler = model.action_decoder.noise_scheduler
            scheduler.set_timesteps(model.action_decoder.num_inference_steps, device=device)
            denoising_state["timesteps"] = scheduler.timesteps
            
            noisy_shape = (1, model.action_decoder.horizon, model.action_decoder.action_dim)
            denoising_state["trajectory"] = torch.randn(noisy_shape, device=device)
            
            controller.denoising_step = 0
            controller.denoising_total_steps = len(scheduler.timesteps)
            controller.start_new_prediction = False
            
            traj_np = denoising_state["trajectory"].squeeze(0).cpu().numpy()
            action_geometries = create_action_geometries(traj_np, vis_mode)
            for geom in action_geometries:
                vis.add_geometry(geom, reset_bounding_box=False)
            
            print(f"Scene {controller.current_step} loaded. Press 'M' to start denoising.")

        if controller.run_denoising_step:
            step_idx = controller.denoising_step
            t = denoising_state["timesteps"][step_idx]
            
            with torch.no_grad():
                model_output = model.action_decoder.model(
                    denoising_state["trajectory"], t, global_cond=denoising_state["global_cond"])
                
                output = model.action_decoder.noise_scheduler.step(
                    model_output, t, denoising_state["trajectory"])
                
                denoising_state["trajectory"] = output.prev_sample

            for geom in action_geometries:
                vis.remove_geometry(geom, reset_bounding_box=False)
            action_geometries.clear()
            
            norm_traj = denoising_state["trajectory"]
            
            action_trans = norm_traj[:, :, :3]
            action_rot = norm_traj[:, :, 3:9]
            action_grip = norm_traj[:, :, 9:10]
            
            unnorm_trans = model.normalizer['action_translation'].unnormalize(action_trans)
            unnorm_grip = model.normalizer['action_gripper'].unnormalize(action_grip)
            
            unnorm_traj_np = torch.cat([unnorm_trans, action_rot, unnorm_grip], dim=-1).squeeze(0).cpu().numpy()
            
            action_geometries = create_action_geometries(unnorm_traj_np, vis_mode)
            for geom in action_geometries:
                vis.add_geometry(geom, reset_bounding_box=False)

            controller.denoising_step += 1
            controller.run_denoising_step = False

        if first_render:
            vis.reset_view_point(True)
            first_render = False

        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)

    vis.destroy_window()

if __name__ == "__main__":
    main()
