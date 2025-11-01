# train_diffusion_SPEC_EAG_workspace.py
# -*- coding: utf-8 -*-
"""
Minimal trainer for SPEC + EAG.
- Loads config YAML (OmegaConf)
- Instantiates the *original* SPEC dataset via dynamic import, then wraps with SPEC_EAG_Wrapper
- Builds SPECPolicyEAG
- Trains with a simple loop (AdamW + cosine) and saves ckpt with normalizer included

This script is intentionally self-contained and conservative:
- Collate function accepts either (coords, feats) *or* raw pc7 and voxelizes on-the-fly.
- If dataset exposes `normalizer` or `get_normalizer()`, we use it; otherwise training still runs
  (but we strongly recommend providing a proper normalizer to match eval).

Author: PCDP
"""
import importlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from omegaconf import OmegaConf

from termcolor import cprint

from pcdp.policy.diffusion_SPEC_EAG_policy import SPECPolicyEAG
from pcdp.dataset.SPEC_EAG_stack_dataset import SPEC_EAG_Wrapper


# ===============
# Helper: dynamic import from a string like "pcdp.dataset.SPEC_stack_dataset.SPEC_RealStackPointCloudDataset"
# ===============
def import_from_string(target: str):
    module_name, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, cls_name)


# ===============
# Collate: supports either (coords, feats) or pc7
# ===============
def build_sparse_tensor_from_batch(batch_list, voxel_size: float) -> Tuple[ME.SparseTensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Returns:
      sp_tensor: ME.SparseTensor (B-merged)
      robot_obs: (B, 10)
      actions:   (B, H, 10) if present else None
      teacher_tail: (B, Hc, 10) if present else None
    """
    coords_batch = []
    feats_batch  = []
    batch_ids    = []
    robot_obs_list = []
    actions_list   = []
    teacher_tail_list = []

    # First pass: detect mode
    has_coords = ('coords' in batch_list[0]) and ('feats' in batch_list[0])
    has_pc7    = ('pc7' in batch_list[0]) or ('pointcloud' in batch_list[0])

    for b, sample in enumerate(batch_list):
        if has_coords:
            coords = sample['coords']  # (N,3) int32
            feats  = sample['feats']   # (N,C) float32
            if isinstance(coords, np.ndarray):
                coords = torch.from_numpy(coords)
            if isinstance(feats, np.ndarray):
                feats = torch.from_numpy(feats)
            # add batch index
            bcoords = torch.cat([torch.full_like(coords[:, :1], b), coords], dim=1)
            coords_batch.append(bcoords)
            feats_batch.append(feats)
        elif has_pc7:
            key = 'pc7' if 'pc7' in sample else 'pointcloud'
            pc7 = sample[key]
            if isinstance(pc7, np.ndarray):
                pc7 = torch.from_numpy(pc7)
            pc7 = pc7.float()  # (N,7) [x,y,z,r,g,b,c]
            xyz = pc7[:, :3]
            rgbc = pc7[:, 3:]
            # quantize coords
            coords = torch.floor(xyz / voxel_size + 0.5).to(torch.int32)
            bcoords = torch.cat([torch.full_like(coords[:, :1], b), coords], dim=1)
            coords_batch.append(bcoords)
            feats_batch.append(rgbc)
        else:
            raise RuntimeError("Batch sample must contain either ('coords','feats') or 'pc7'/'pointcloud'.")

        # robot obs
        rob = sample.get('robot_obs', None)
        if rob is None:
            raise RuntimeError("Sample missing 'robot_obs' (10D).")
        if isinstance(rob, np.ndarray):
            rob = torch.from_numpy(rob)
        robot_obs_list.append(rob.float().reshape(1, -1))

        # actions (optional for eval, required for training)
        act = sample.get('action', None)
        if act is not None:
            if isinstance(act, np.ndarray):
                act = torch.from_numpy(act)
            actions_list.append(act.float().unsqueeze(0))

        # teacher_tail (optional)
        tt = sample.get('teacher_tail', None)
        if tt is not None:
            if isinstance(tt, np.ndarray):
                tt = torch.from_numpy(tt)
            teacher_tail_list.append(tt.float().unsqueeze(0))

    # stack
    coords_all = torch.cat(coords_batch, dim=0)
    feats_all  = torch.cat(feats_batch, dim=0)

    sp_tensor = ME.SparseTensor(feats_all, coordinates=coords_all)

    robot_obs = torch.cat(robot_obs_list, dim=0)  # (B,10)

    actions = torch.cat(actions_list, dim=0) if len(actions_list) == len(batch_list) else None
    teacher_tail = torch.cat(teacher_tail_list, dim=0) if len(teacher_tail_list) == len(batch_list) else None

    return sp_tensor, robot_obs, actions, teacher_tail


# ===============
# Main train
# ===============
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='train_diffusion_SPEC_EAG_workspace.yaml')
    p.add_argument('--workdir', type=str, default='./outputs_eag')
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)

    # ---- Instantiate base dataset ----
    ds_target = cfg.dataset._target_
    ds_kwargs = {k: v for k, v in cfg.dataset.items() if k != '_target_'}
    DSClass = import_from_string(ds_target)
    base_ds = DSClass(**ds_kwargs)

    # ---- Wrap with EAG teacher-tail provider ----
    Hc = int(cfg.eag.Hc)
    train_ds = SPEC_EAG_Wrapper(base_ds, Hc=Hc)

    # ---- DataLoader ----
    bs = int(cfg.training.batch_size)
    voxel_size = float(cfg.dataset.voxel_size if 'voxel_size' in cfg.dataset else cfg.get('voxel_size', 0.01))

    def collate(batch):
        return build_sparse_tensor_from_batch(batch, voxel_size=voxel_size)

    loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=int(cfg.training.num_workers),
                        collate_fn=collate, drop_last=True, pin_memory=True)

    # ---- Build policy ----
    pol = SPECPolicyEAG(
        horizon=int(cfg.policy.horizon),
        action_dim=int(cfg.policy.action_dim),
        hidden_dim=int(cfg.policy.hidden_dim),
        encoder_max_num_token=int(cfg.policy.encoder_max_num_token),
        robot_obs_dim=int(cfg.policy.robot_obs_dim),
        enable_eag=bool(cfg.policy.enable_eag),
        eag_Hc=int(cfg.eag.Hc),
        eag_tail_embed_dim=int(cfg.policy.eag_tail_embed_dim),
        eag_p_uncond=float(cfg.policy.eag_p_uncond)
    ).cuda()

    # Try attach normalizer if dataset provides
    normalizer = None
    if hasattr(base_ds, 'get_normalizer'):
        normalizer = base_ds.get_normalizer()
    elif hasattr(base_ds, 'normalizer'):
        normalizer = base_ds.normalizer
    if normalizer is not None:
        pol.set_normalizer(normalizer)
        cprint("[train] Attached normalizer from dataset.", "green")
    else:
        cprint("[train] WARNING: No normalizer found in dataset; training will proceed but eval mismatch may occur.", "yellow")

    # ---- Optimizer/Scheduler ----
    optim = torch.optim.AdamW(pol.parameters(), lr=float(cfg.training.lr), weight_decay=float(cfg.training.weight_decay))
    total_steps = int(cfg.training.num_epochs) * (len(loader))
    warmup_steps = int(cfg.training.warmup_steps)
    min_lr = float(cfg.training.min_lr)

    def cosine_lr(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr / cfg.training.lr + 0.5 * (1 - min_lr / cfg.training.lr) * (1 + math.cos(math.pi * p))

    # ---- Train loop ----
    os.makedirs(args.workdir, exist_ok=True)
    step = 0
    for epoch in range(int(cfg.training.num_epochs)):
        for sp, rob, act, tt in loader:
            sp = sp.cuda()
            rob = rob.cuda()
            act = act.cuda() if act is not None else None
            tt  = tt.cuda() if tt is not None else None

            pol.train()
            loss = pol(sp, rob, actions=act, teacher_tail=tt)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), max_norm=float(cfg.training.grad_clip_norm))
            # lr schedule
            for g in optim.param_groups:
                g['lr'] = float(cfg.training.lr) * cosine_lr(step)
            optim.step()

            if step % int(cfg.training.log_every) == 0:
                cprint(f"[epoch {epoch}] step {step} loss {float(loss):.4f}", "cyan")
            step += 1

        # ---- Save ckpt each epoch ----
        save_path = Path(args.workdir) / f"ckpt_epoch_{epoch:03d}.pt"
        state = {
            'state_dicts': {
                'model': pol.state_dict(),
                'normalizer': pol.normalizer.state_dict() if pol.normalizer is not None else None
            },
            'cfg': OmegaConf.to_container(cfg, resolve=True)
        }
        torch.save(state, save_path)
        cprint(f"[train] Saved {save_path}", "green")


if __name__ == "__main__":
    main()