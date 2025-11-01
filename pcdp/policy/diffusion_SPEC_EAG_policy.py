# diffusion_SPEC_EAG_policy.py
# -*- coding: utf-8 -*-
"""
EAG-enabled policy wrapper for SPEC_mono.

This file mirrors diffusion_SPEC_policy_mono.py but adds:
  - Tail embedding from the previous chunk (length Hc)
  - Classifier-Free Guidance (EAG) plumbing (training cond-drop + inference CFG)
  - Forward() signature that accepts prev_action_tail & guidance_w during inference

NOTE:
  - This policy expects the EAG-aware Diffusion decoder to live at:
        pcdp.model.diffusion.RISE_diffusionUNet_EAG.DiffusionUNetPolicyEAG
    which provides:
        - compute_loss(readout, actions, tail_embed=None, p_uncond: float=0.0, return_aux=False)
        - predict_action(readout)                                    # fallback (no EAG)
        - predict_action_eag(readout_base, tail_embed, guidance_w)   # CFG per-step (EAG)
  - You will create that in step 2 (RISE_diffusionUNet_EAG.py).

Author: PCDP (SPEC_mono + EAG)
"""
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from termcolor import cprint

# Project-local imports: keep identical to your mono policy
from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from pcdp.policy.RISE_transformer import Transformer
from pcdp.model.common.normalizer import LinearNormalizer

# EAG-capable diffusion decoder (to be added in step 2)
from pcdp.model.diffusion.RISE_diffusionUNet_EAG import DiffusionUNetPolicyEAG


class SPECPolicyEAG(BasePointCloudPolicy):
    """
    Point-cloud → Sparse3DEncoder → RISE Transformer (readout) →
    concatenate robot_obs (+ optional tail_embed) → EAG-capable Diffusion UNet.

    Training:
      - If actions is not None: computes diffusion loss.
      - Tail teacher-forcing: uses actions[:, :Hc] as prev tail by default OR dataset-provided tail.
      - Cond-drop with p_uncond to learn unconditional branch.

    Inference:
      - If actions is None: returns predicted trajectory (B, H, 10)
      - Accepts prev_action_tail (B, Hc, 10) and guidance_w (float)
      - If guidance_w>0 and prev tail given → EAG CFG branch
      - Else falls back to standard predict_action (no EAG)
    """
    def __init__(self,
                # === Core dims ===
                horizon: int = 20,
                action_dim: int = 10,
                hidden_dim: int = 512,
                # === RISE Transformer ===
                nheads: int = 8,
                num_encoder_layers: int = 4,
                num_decoder_layers: int = 1,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                # === Point cloud encoder & tokens ===
                in_pc_channels: int = 7,        # xyz(3)+rgb(3)+c(1)
                encoder_max_num_token: int = 100,
                enable_c_gate: bool = False,    # if True, rgb *= c (apply in preproc; kept for config symmetry)
                # === Robot obs ===
                robot_obs_dim: int = 10,        # pos3 + rot6 + grip1
                # === EAG (tail) ===
                enable_eag: bool = True,
                eag_Hc: int = 2,
                eag_tail_embed_dim: int = 128,
                eag_p_uncond: float = 0.25,
                # === Misc ===
                normalizer: Optional[LinearNormalizer] = None,
                device: Optional[torch.device] = None,
                ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder_max_num_token = int(encoder_max_num_token)
        self.in_pc_channels = int(in_pc_channels)
        self.robot_obs_dim = int(robot_obs_dim)

        self.enable_c_gate = bool(enable_c_gate)

        # EAG
        self.enable_eag = bool(enable_eag)
        self.eag_Hc = int(eag_Hc)
        self.eag_tail_embed_dim = int(eag_tail_embed_dim)
        self.eag_p_uncond = float(eag_p_uncond)

        # === Modules ===
        # 1) Sparse 3D Encoder (Minkowski ResNet14 + Sparse pos-encoding inside)
        self.pc_encoder = Sparse3DEncoder(input_dim=self.in_pc_channels, output_dim=self.hidden_dim)

        # 2) RISE Transformer → single readout token
        self.readout_transformer = Transformer(
            d_model=self.hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # 3) Tail embedding MLP (flatten Hc*Da → embed)
        if self.enable_eag and self.eag_Hc > 0:
            self.tail_proj = nn.Sequential(
                nn.Linear(self.eag_Hc * self.action_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.eag_tail_embed_dim)
            )
        else:
            self.tail_proj = None

        # 4) Diffusion decoder (EAG-capable UNet)
        #    obs_feature_dim per obs-step equals readout(=hidden_dim) + robot_obs_dim (+ tail_embed if decoder concatenates internally)
        #    Here we pass base cond dim (without tail), and provide tail_embed separately at call-time.
        self.action_decoder = DiffusionUNetPolicyEAG(
            action_dim=self.action_dim,
            horizon=self.horizon,
            n_obs_steps=1,
            obs_feature_dim=self.hidden_dim + self.robot_obs_dim,
            tail_embed_dim=(self.eag_tail_embed_dim if (self.enable_eag and self.tail_proj is not None) else 0),
        )

        # Normalizer holder (LinearNormalizer with named fields)
        self.normalizer = normalizer  # type: Optional[LinearNormalizer]

        # Default device
        self._device = device

    # ---------------------------------------------------------------------
    # Utility: normalizer attachment (kept for compatibility with eval script)
    # ---------------------------------------------------------------------
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer = normalizer
        return self

    # ---------------------------------------------------------------------
    # Internal: build encoder tokens & readout
    # ---------------------------------------------------------------------
    def _encode_pointcloud(self, sp_tensor: ME.SparseTensor, batch_size: int) -> torch.Tensor:
        """
        Args:
            sp_tensor: MinkowskiEngine SparseTensor for the batched pointclouds
            batch_size: B
        Returns:
            readout: (B, hidden_dim) global scene feature
        """
        # Sparse3DEncoder should accept ME.SparseTensor and return (tokens, pos_emb)
        # and then Transformer produces a single readout per batch item.
        # We keep the interface identical to your mono policy.
        tokens, pos_emb = self.pc_encoder(sp_tensor, max_num_token=self.encoder_max_num_token, batch_size=batch_size)
        # Transformer forward expected signature: (tokens, pos_emb) → (B, hidden_dim)
        readout = self.readout_transformer(tokens, pos_emb)  # shape: (B, hidden_dim)
        return readout

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self,
                # Point cloud in ME SparseTensor form
                pc_sp_tensor: ME.SparseTensor,
                # Low-dim robot obs (B, 10)
                robot_obs: torch.Tensor,
                # Training: ground-truth actions (B, H, 10). If provided → returns loss
                actions: Optional[torch.Tensor] = None,
                # Inference-only: previous tail for EAG (B, Hc, 10)
                prev_action_tail: Optional[torch.Tensor] = None,
                # Inference-only: EAG CFG guidance weight
                guidance_w: float = 0.0,
                # Optional: dataset-provided teacher tail in training (B, Hc, 10)
                teacher_tail: Optional[torch.Tensor] = None,
                ) -> Any:
        """
        Returns:
            if actions is not None: scalar loss (or loss dict)
            else: action prediction (B, H, 10) in UNNORMALIZED scale (pos in meters, grip in native)
        """
        assert isinstance(pc_sp_tensor, ME.SparseTensor), "pc_sp_tensor must be a MinkowskiEngine.SparseTensor"
        B = robot_obs.shape[0]
        device = robot_obs.device

        # 1) Encode pointcloud → readout
        readout = self._encode_pointcloud(pc_sp_tensor, batch_size=B)   # (B, hidden_dim)

        # 2) Build base global condition per obs-step (To=1): concat readout + robot_obs
        assert robot_obs.shape[-1] == self.robot_obs_dim, f"robot_obs dim {robot_obs.shape[-1]} != {self.robot_obs_dim}"
        global_cond_base = torch.cat([readout, robot_obs], dim=-1)     # (B, hidden_dim + robot_obs_dim)

        # 3) TRAINING BRANCH
        if actions is not None:
            # Prepare GT actions in normalized space (translation, grip normalized; rot6D raw)
            action_trans = actions[:, :, :3]
            action_rot6d = actions[:, :, 3:9]
            action_grip = actions[:, :, 9:10]

            if self.normalizer is not None:
                action_trans = self.normalizer['action_translation'].normalize(action_trans)
                action_grip = self.normalizer['action_gripper'].normalize(action_grip)

            actions_norm = torch.cat([action_trans, action_rot6d, action_grip], dim=-1)

            # Teacher tail (B, Hc, Da)
            if (self.enable_eag and self.eag_Hc > 0):
                if teacher_tail is not None:
                    tail_src = teacher_tail
                else:
                    # Fallback: use the first Hc steps of the same sequence as teacher tail
                    tail_src = actions[:, :self.eag_Hc, :]
                tail_embed = self.tail_proj(tail_src.reshape(B, -1))
            else:
                tail_embed = None

            # EAG cond-drop probability
            p_uncond = self.eag_p_uncond if (self.enable_eag and tail_embed is not None) else 0.0

            # Compute diffusion loss
            loss = self.action_decoder.compute_loss(
                readout=global_cond_base,           # base cond per obs-step
                actions=actions_norm,
                tail_embed=tail_embed,
                p_uncond=p_uncond,
                return_aux=False
            )
            return loss

        # 4) INFERENCE BRANCH
        # Build tail embed if provided
        tail_embed_inf = None
        if (self.enable_eag and self.eag_Hc > 0 and prev_action_tail is not None):
            assert prev_action_tail.shape[1] == self.eag_Hc, \
                f"prev_action_tail shape mismatch: expected Hc={self.eag_Hc}, got {prev_action_tail.shape[1]}"
            tail_embed_inf = self.tail_proj(prev_action_tail.reshape(B, -1))

        with torch.no_grad():
            if self.enable_eag and tail_embed_inf is not None and guidance_w > 0.0:
                # EAG: per-step CFG inside the decoder
                action_pred = self.action_decoder.predict_action_eag(
                    readout_base=global_cond_base,   # (B, Do_base)
                    tail_embed=tail_embed_inf,       # (B, tail_embed_dim)
                    guidance_w=float(guidance_w)
                )
            else:
                # Fallback: vanilla conditional sampling (no EAG)
                action_pred = self.action_decoder.predict_action(
                    readout=global_cond_base
                )

        # 5) Unnormalize translation & gripper back to real units
        if self.normalizer is not None:
            action_trans = action_pred[:, :, :3]
            action_rot6d = action_pred[:, :, 3:9]
            action_grip = action_pred[:, :, 9:10]

            unnorm_trans = self.normalizer['action_translation'].unnormalize(action_trans)
            unnorm_grip = self.normalizer['action_gripper'].unnormalize(action_grip)

            action_pred = torch.cat([unnorm_trans, action_rot6d, unnorm_grip], dim=-1)

        return action_pred