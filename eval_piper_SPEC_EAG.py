# eval_piper_SPEC_EAG.py
# -*- coding: utf-8 -*-
"""
SPEC EAG Evaluation Script
--------------------------
Drop-in variant of eval_piper_SPEC.py that adds SAIL-style EAG (Error-Adaptive Guidance).

Key additions:
  - Maintains last predicted chunk and executed index.
  - Extracts previous tail of length Hc as *condition* (not hard copy).
  - Computes tracking errors (position/orientation) and toggles CFG guidance_w with hysteresis.
  - Calls SPECPolicyEAG which supports (prev_action_tail, guidance_w).

Assumptions / Integration notes:
  - Point cloud preprocessor delivers Nx7 pc: [x,y,z,r,g,b,c] in *base/world* frame.
  - robot_obs is 10-D: [pos(3), rot6D(6), grip(1)] consistent with training.
  - You MUST keep train/eval identical in: voxel_size, encoder_max_num_token, channels order, normalizer.
  - Normalizer is loaded from ckpt (preferred). If not found, exit with a clear error.

Author: PCDP (SPEC_mono + EAG)
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME

import click

from termcolor import cprint

# ---- Project imports (adjust paths to your repo layout) ----
from pcdp.real_world.real_env_piper import RealEnv
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.common.precise_sleep import precise_wait
import pcdp.common.mono_time as mono_time

from pcdp.policy.diffusion_SPEC_EAG_policy import SPECPolicyEAG
from pcdp.model.common.normalizer import LinearNormalizer


# =======================
# Utility: Rot conversions
# =======================
def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert 6D rotation (Zhou et al.) to 3x3 rotation matrix. Works for (6,) or (N,6)."""
    r = rot6d.reshape(-1, 6)
    a1 = r[:, 0:3]
    a2 = r[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-9)
    b2 = a2 - (b1 * (a2 * b1).sum(axis=1, keepdims=True))
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=-1)  # (N,3,3)
    if rot6d.ndim == 1:
        return R[0]
    return R


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] or [x,y,z,w] to 3x3 R. Heuristically detect ordering."""
    q = np.asarray(q).reshape(4,)
    # heuristic: if |q[0]| >= max(|q[1:]|) assume w-first, else w-last
    if abs(q[0]) >= np.max(np.abs(q[1:])):
        w, x, y, z = q
    else:
        x, y, z, w = q
    # normalize
    n = np.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n
    # rotmat
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=np.float64)
    return R


def rotation_angle(R_des: np.ndarray, R_curr: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices."""
    M = R_des.T @ R_curr
    tr = np.clip((np.trace(M) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))


# =======================
# Utility: ME SparseTensor
# =======================
def pc7_to_sparse_tensor(pc7: np.ndarray, voxel_size: float) -> ME.SparseTensor:
    """
    Convert Nx7 [x,y,z,r,g,b,c] to MinkowskiEngine SparseTensor.
    - Coordinates quantized by voxel_size (meters → voxel index).
    - Features = [r,g,b,c] (float32).
    - Returns batched SparseTensor (B=1).
    """
    assert pc7.shape[1] == 7, "pc7 must be Nx7"
    xyz = pc7[:, :3].astype(np.float32)
    feats = pc7[:, 3:].astype(np.float32)  # (N,4): rgb+c

    # Quantize coords
    coords = np.floor(xyz / voxel_size + 0.5).astype(np.int32)

    # Minkowski requires batch index in coords[:,0]
    bcoords = np.pad(coords, ((0,0),(1,0)), constant_values=0)  # prepend zeros for batch index

    # ME will internally coalesce duplicate coords averaging feats; use features as provided
    coords_th = torch.from_numpy(bcoords)
    feats_th = torch.from_numpy(feats)

    sp = ME.SparseTensor(features=feats_th, coordinates=coords_th, device=feats_th.device)
    return sp


# =======================
# EAG decision with hysteresis
# =======================
@dataclass
class EAGSwitch:
    e_pos_on: float = 0.015   # 1.5 cm
    e_ori_on: float = 0.04    # 2.3 deg
    e_pos_off: float = 0.025  # 2.5 cm
    e_ori_off: float = 0.06   # 3.4 deg
    guidance_w_max: float = 1.0
    w_decay_tau: float = 0.0  # if >0, exp decay across step k in a new chunk

    _guided: bool = False  # internal state

    def decide(self, e_pos: float, e_ori: float, k_in_chunk: int) -> float:
        """
        Returns guidance_w for the *current* step index k_in_chunk (0-based).
        - Applies hysteresis on/off thresholds.
        - Optionally decays w over k via exp(-k/tau).
        """
        # hysteresis
        if self._guided:
            if (e_pos >= self.e_pos_off) or (e_ori >= self.e_ori_off):
                self._guided = False
        else:
            if (e_pos <= self.e_pos_on) and (e_ori <= self.e_ori_on):
                self._guided = True

        if not self._guided:
            return 0.0

        w = self.guidance_w_max
        if self.w_decay_tau and self.w_decay_tau > 0.0:
            w = float(w * np.exp(-float(k_in_chunk) / float(self.w_decay_tau)))
        return w


# =======================
# Normalizer loader
# =======================
def load_policy_and_normalizer(ckpt_path: str,
                               device: torch.device,
                               horizon: int,
                               action_dim: int,
                               hidden_dim: int,
                               encoder_max_num_token: int,
                               robot_obs_dim: int,
                               eag_Hc: int,
                               eag_tail_embed_dim: int,
                               ) -> SPECPolicyEAG:
    """
    Load SPECPolicyEAG and (preferentially) its normalizer from ckpt.
    The ckpt is expected to contain a 'state_dicts' mapping with 'model' and 'normalizer' keys.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dicts = ckpt.get('state_dicts', ckpt)

    normalizer_sd = state_dicts.get('normalizer', None)
    if normalizer_sd is None:
        raise RuntimeError("Checkpoint has no 'normalizer' state. Please re-train/save with normalizer included.")

    normalizer = LinearNormalizer.from_state_dict(normalizer_sd)

    policy = SPECPolicyEAG(
        horizon=horizon,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        encoder_max_num_token=encoder_max_num_token,
        robot_obs_dim=robot_obs_dim,
        enable_eag=True,
        eag_Hc=eag_Hc,
        eag_tail_embed_dim=eag_tail_embed_dim,
        eag_p_uncond=0.0,   # only used for training
        normalizer=normalizer,
        device=device
    ).to(device)

    model_sd = state_dicts.get('model', None)
    if model_sd is None:
        # try flat state_dict fallback
        model_sd = ckpt.get('model', None)
    if model_sd is None:
        raise RuntimeError("Checkpoint missing model weights (state_dicts['model']).")

    policy.load_state_dict(model_sd, strict=True)
    policy.eval()
    return policy


# =======================
# Main evaluation loop
# =======================
@click.command()
@click.option('--ckpt', type=click.Path(exists=True, dir_okay=False), required=True, help='Path to trained checkpoint (.pt).')
@click.option('--voxel-size', type=float, default=0.01, show_default=True, help='Voxel size (m). Must match training.')
@click.option('--encoder-max-num-token', type=int, default=100, show_default=True, help='Max tokens to keep per frame.')
@click.option('--device', type=str, default='cuda', show_default=True, help='Device to run inference.')
@click.option('--H', type=int, default=20, show_default=True, help='Plan horizon (must match training).')
@click.option('--He', type=int, default=1, show_default=True, help='Steps executed before replanning.')
@click.option('--Hc', type=int, default=1, show_default=True, help='Tail length for EAG condition (<= He).')
@click.option('--guidance-w', type=float, default=1.0, show_default=True, help='Max guidance strength when EAG is ON.')
@click.option('--w-decay-tau', type=float, default=0.0, show_default=True, help='If >0, decay guidance w across new chunk steps.')
@click.option('--e-pos-on', type=float, default=0.015, show_default=True, help='EAG ON threshold for position error (m).')
@click.option('--e-ori-on', type=float, default=0.04, show_default=True, help='EAG ON threshold for orientation error (rad).')
@click.option('--e-pos-off', type=float, default=0.025, show_default=True, help='EAG OFF threshold for position error (m).')
@click.option('--e-ori-off', type=float, default=0.06, show_default=True, help='EAG OFF threshold for orientation error (rad).')
def main(ckpt, voxel_size, encoder_max_num_token, device,
         H, He, Hc, guidance_w, w_decay_tau,
         e_pos_on, e_ori_on, e_pos_off, e_ori_off):
    assert Hc <= He, f"Hc ({Hc}) must be <= He ({He})"
    device = torch.device(device if torch.cuda.is_available() and device.startswith('cuda') else 'cpu')

    # --- create env/controller (project-specific; adjust if your API differs) ---
    env = RealEnv()                     # Provides pointcloud & current EEF pose
    controller = TeleoperationPiper()   # Provides execute(action) or similar

    # --- load policy & normalizer from ckpt (strict) ---
    policy = load_policy_and_normalizer(
        ckpt_path=ckpt,
        device=device,
        horizon=H,
        action_dim=10,
        hidden_dim=512,
        encoder_max_num_token=encoder_max_num_token,
        robot_obs_dim=10,
        eag_Hc=Hc,
        eag_tail_embed_dim=128
    )

    # --- EAG hysteresis switch ---
    eag_switch = EAGSwitch(
        e_pos_on=e_pos_on, e_ori_on=e_ori_on,
        e_pos_off=e_pos_off, e_ori_off=e_ori_off,
        guidance_w_max=guidance_w, w_decay_tau=w_decay_tau
    )

    last_plan = None          # (H, 10)
    executed_in_chunk = 0     # 0..He-1
    dt = 0.1                  # 100 ms; align with your control loop
    t_last = mono_time.time()

    cprint("[EAG] Eval loop start. Press Ctrl+C to stop.", "cyan")

    try:
        while True:
            t_now = mono_time.time()
            elapsed = t_now - t_last
            if elapsed < dt:
                precise_wait(dt - elapsed)
                continue
            t_last = mono_time.time()

            # ---- 1) Read observation ----
            # Expect env to expose pc7 (N,7) and robot_obs 10D & current EEF pose
            pc7 = env.get_latest_pc7()                    # np.ndarray (N,7): [x,y,z,r,g,b,c]
            robot_obs = env.get_robot_obs_10d()           # np.ndarray (10,)
            eef_curr_pos, eef_curr_R = env.get_current_eef_pose()  # (3,), (3,3)

            # Convert to tensors
            sp = pc7_to_sparse_tensor(pc7, voxel_size=voxel_size).to(device)
            robot_obs_th = torch.from_numpy(robot_obs).float().to(device)[None, :]  # (1,10)

            # ---- 2) Replan if needed ----
            need_replan = (last_plan is None) or (executed_in_chunk >= He)

            if need_replan:
                # Build prev tail (condition only)
                prev_tail = None
                if last_plan is not None and Hc > 0:
                    prev_tail = last_plan[He - Hc:He, :]     # (Hc, 10)
                    prev_tail = torch.from_numpy(prev_tail).float().to(device)[None, :, :]

                # Decide EAG guidance based on tracking error between the last desired and current
                # If there was a previous plan and at least one step executed, use the last executed as "desired"
                if last_plan is not None and executed_in_chunk > 0:
                    des = last_plan[executed_in_chunk - 1]     # (10,)
                    des_pos = des[:3]
                    des_R = rot6d_to_matrix(des[3:9])
                else:
                    # If no previous execution, treat current as on-target -> turn EAG ON (will decay within chunk)
                    des_pos = eef_curr_pos.copy()
                    des_R = eef_curr_R.copy()

                e_pos = float(np.linalg.norm(des_pos - eef_curr_pos))
                e_ori = rotation_angle(des_R, eef_curr_R)

                # Hysteretic decision & per-chunk step index k=0
                w_now = eag_switch.decide(e_pos, e_ori, k_in_chunk=0)

                # ---- Policy inference (EAG if w_now>0 and prev_tail exists) ----
                with torch.no_grad():
                    pred = policy(
                        pc_sp_tensor=sp,
                        robot_obs=robot_obs_th,
                        actions=None,
                        prev_action_tail=prev_tail,
                        guidance_w=float(w_now)
                    )  # (1,H,10) unnormalized
                last_plan = pred.squeeze(0).detach().cpu().numpy()
                executed_in_chunk = 0

            # ---- 3) Execute next action in current chunk ----
            action_now = last_plan[executed_in_chunk]  # (10,)
            # Send to controller (convert 6D rot to controller's format if needed)
            controller.execute_10d(action_now)

            # ---- 4) Update executed index and (optional) decay w within chunk ----
            executed_in_chunk += 1

            # (Optional) we may update EAGSwitch for within-chunk decay at the next replan only.
            # If you want per-step decay inside a chunk, you can recompute w and call a light-weight
            # replan with cached readout (advanced). For now we keep it per-chunk.

    except KeyboardInterrupt:
        cprint("\n[EAG] Stopped by user.", "yellow")
    finally:
        try:
            controller.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()