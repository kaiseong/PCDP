# RISE_diffusionUNet_EAG.py
# -*- coding: utf-8 -*-
"""
EAG-capable Diffusion UNet policy (1D) for SPEC_mono.

This mirrors the original RISE_diffusionUNet.DiffusionUNetPolicy but:
  - Expands the global condition to include a tail-embedding (dim = tail_embed_dim)
  - Adds cond-drop during training (p_uncond)
  - Adds predict_action_eag(...) which performs per-step CFG blending between
    an unconditional branch (tail_embed=0) and a conditional branch (tail_embed!=0).

Author: PCDP
"""
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# These imports match your existing project structure
from pcdp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from pcdp.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUNetPolicyEAG(nn.Module):
    def __init__(self,
                 action_dim: int,
                 horizon: int,
                 n_obs_steps: int,
                 obs_feature_dim: int,      # base: readout(=D) + robot_obs(=10), flattened over n_obs_steps
                 tail_embed_dim: int = 0,   # extra dims to be concatenated to global_cond
                 num_inference_steps: int = 20,
                 diffusion_step_embed_dim: int = 256,
                 down_dims=(256, 512),
                 kernel_size: int = 5,
                 n_groups: int = 8,
                 cond_predict_scale: bool = True,
                 # kwargs forwarded to scheduler.step(...)
                 **kwargs):
        super().__init__()

        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.obs_feature_dim = int(obs_feature_dim)
        self.tail_embed_dim = int(tail_embed_dim)
        self.kwargs = kwargs

        # === Model ===
        input_dim = self.action_dim
        global_cond_dim = self.obs_feature_dim * self.n_obs_steps + self.tail_embed_dim

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        # === Noise scheduler (DDIM) ===
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon"
        )
        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # === Mask generator for low-dim action (no action visible during training) ===
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0,
            max_n_obs_steps=self.n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

    # -------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------
    def _build_global_cond(self,
                           readout_base: torch.Tensor,   # (B, Do_base) flattened over n_obs_steps
                           tail_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Returns a global condition of shape (B, Do_base*n_obs_steps + tail_embed_dim).
        If tail_embed is None, zeros of tail_embed_dim are appended.
        """
        B = readout_base.shape[0]
        device = readout_base.device
        dtype = readout_base.dtype

        if self.n_obs_steps == 1:
            base = readout_base.reshape(B, -1)
        else:
            # already flattened upstream, but keep interface parallel to original
            base = readout_base.reshape(B, -1)

        if self.tail_embed_dim > 0:
            if tail_embed is None:
                tail = torch.zeros((B, self.tail_embed_dim), device=device, dtype=dtype)
            else:
                assert tail_embed.shape == (B, self.tail_embed_dim), \
                    f"tail_embed shape {tail_embed.shape} != {(B, self.tail_embed_dim)}"
                tail = tail_embed
            return torch.cat([base, tail], dim=-1)

        return base

    # -------------------------------------------------------------
    # Inference (vanilla; no EAG). Keeps compatibility with existing callers.
    # -------------------------------------------------------------
    def predict_action(self, readout: torch.Tensor) -> torch.Tensor:
        """
        Args:
            readout: (B, Do_base) where Do_base = obs_feature_dim (already includes robot_obs)
        Returns:
            action_pred: (B, H, Da)
        """
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = readout.device
        dtype = readout.dtype

        # Build conditioning tensors
        global_cond = self._build_global_cond(readout, tail_embed=None)  # tail zeros if tail_dim>0

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # Run sampling
        sample = self._conditional_sample_single(
            condition_data=cond_data,
            condition_mask=cond_mask,
            global_cond=global_cond,
            generator=None
        )
        return sample[..., :Da]

    # -------------------------------------------------------------
    # Inference with EAG (CFG between uncond and cond)
    # -------------------------------------------------------------
    def predict_action_eag(self,
                           readout_base: torch.Tensor,      # (B, Do_base)
                           tail_embed: torch.Tensor,        # (B, tail_embed_dim)
                           guidance_w: float = 1.0) -> torch.Tensor:
        """
        Per-step CFG blending between:
          - uncond branch: global_cond = [readout_base || zeros_tail]
          - cond branch  : global_cond = [readout_base || tail_embed]
        """
        B = readout_base.shape[0]
        T = self.horizon
        Da = self.action_dim
        device = readout_base.device
        dtype = readout_base.dtype

        global_cond_uncond = self._build_global_cond(readout_base, tail_embed=None)
        global_cond_cond   = self._build_global_cond(readout_base, tail_embed=tail_embed)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        sample = self._conditional_sample_cfg(
            condition_data=cond_data,
            condition_mask=cond_mask,
            global_cond_uncond=global_cond_uncond,
            global_cond_cond=global_cond_cond,
            guidance_w=float(guidance_w),
            generator=None
        )
        return sample[..., :Da]

    # -------------------------------------------------------------
    # Training loss (supports cond-drop via p_uncond)
    # -------------------------------------------------------------
    def compute_loss(self,
                     readout: torch.Tensor,      # (B, Do_base)
                     actions: torch.Tensor,      # (B, H, Da) normalized
                     tail_embed: Optional[torch.Tensor] = None,  # (B, tail_dim) or None
                     p_uncond: float = 0.0,
                     return_aux: bool = False):
        """
        If p_uncond>0 and tail_embed is provided, randomly zero-out tail embeddings for a subset of the batch
        to learn the unconditional branch (CFG prerequisite).
        """
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim

        device = readout.device
        dtype = readout.dtype

        # Build global condition with (possibly dropped) tail
        if (self.tail_embed_dim > 0) and (tail_embed is not None) and (p_uncond > 0.0):
            # Bernoulli mask per-sample: 1 → uncond (drop), 0 → cond (keep)
            drop_mask = (torch.rand((B, 1), device=device) < p_uncond).to(dtype=dtype)
            kept_tail = tail_embed * (1.0 - drop_mask)
            global_cond = self._build_global_cond(readout, kept_tail)
        else:
            global_cond = self._build_global_cond(readout, tail_embed)

        # Prepare data
        trajectory = actions  # (B, T, Da)
        condition_data = torch.zeros_like(trajectory)  # empty conditioning
        condition_mask = self.mask_generator(trajectory.shape)  # (B, T, Da) bool

        # forward diffusion
        noise = torch.randn(trajectory.shape, device=device, dtype=dtype)
        bsz = trajectory.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # apply conditioning on noisy trajectory
        noisy_trajectory[condition_mask] = condition_data[condition_mask]

        # Predict noise residual
        pred = self.model(noisy_trajectory, timesteps,
                          local_cond=None, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        if return_aux:
            aux = {
                'xt_last': noisy_trajectory[:, -1, :].detach(),
                'xt': noisy_trajectory.detach(),
                't': timesteps.detach()
            }
            return loss, aux
        return loss

    # -------------------------------------------------------------
    # Internal samplers
    # -------------------------------------------------------------
    def _conditional_sample_single(self,
                                   condition_data: torch.Tensor,
                                   condition_mask: torch.Tensor,
                                   global_cond: torch.Tensor,
                                   generator=None) -> torch.Tensor:
        """Single-branch conditional sampling (no CFG)."""
        scheduler = self.noise_scheduler
        model = self.model

        # init x_T
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trajectory.device)
        timesteps = scheduler.timesteps

        for t in timesteps:
            # 1) enforce conditioning at current step
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2) predict model output
            model_output = model(trajectory, t, local_cond=None, global_cond=global_cond)

            # 3) step x_t → x_{t-1}
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **self.kwargs).prev_sample

        # final enforcement
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _conditional_sample_cfg(self,
                                condition_data: torch.Tensor,
                                condition_mask: torch.Tensor,
                                global_cond_uncond: torch.Tensor,
                                global_cond_cond: torch.Tensor,
                                guidance_w: float,
                                generator=None) -> torch.Tensor:
        """Classifier-Free Guidance sampling with tail-based global condition."""
        scheduler = self.noise_scheduler
        model = self.model

        # init x_T
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trajectory.device)
        timesteps = scheduler.timesteps
        pred_type = scheduler.config.prediction_type  # 'epsilon' or 'sample'

        for t in timesteps:
            # enforce hard condition at each step
            trajectory[condition_mask] = condition_data[condition_mask]

            # unconditional & conditional passes
            eps_u = model(trajectory, t, local_cond=None, global_cond=global_cond_uncond)
            eps_c = model(trajectory, t, local_cond=None, global_cond=global_cond_cond)

            # CFG blend in the prediction space (epsilon or sample)
            model_output = eps_u + guidance_w * (eps_c - eps_u)

            # step
            trajectory = scheduler.step(model_output, t, trajectory, generator=generator, **self.kwargs).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory