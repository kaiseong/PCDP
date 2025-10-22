# RISE_diffusionUNet_RTC.py
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from pcdp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from pcdp.model.diffusion.mask_generator import LowdimMaskGenerator


class DiffusionUNetPolicy(nn.Module):
    def __init__(
        self,
        action_dim,
        horizon,
        n_obs_steps,
        obs_feature_dim,
        num_inference_steps=20,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        # parameters passed to step
        **kwargs,
    ):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        # create noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon",  # we assume epsilon pred in guidance
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    # Original-RTC style: soft-masked inpainting + VJP guidance + clipping β
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        rtc_target=None,
        rtc_mask=None,           # soft mask W (same shape as condition_data)
        num_inference_steps=None,
        beta_clip=None,          # guidance clipping β (e.g., 0.1~1.0)
        lambda0=0.5,             # initial guidance strength
        **kwargs,
    ):
        scheduler = self.noise_scheduler
        if num_inference_steps is not None:
            self.num_inference_steps = num_inference_steps
        scheduler.set_timesteps(self.num_inference_steps)

        # sample init
        x = torch.randn_like(condition_data)
        B, T, Da = x.shape

        # soft-mask (W)
        W = rtc_mask if (rtc_mask is not None) else None

        # helper: compute x0 from (x, eps, t) for epsilon prediction
        def predict_x0_from_epsilon(x_cur, eps_pred, t_step):
            # DDIM timesteps are integer indices into [0, num_train_timesteps)
            # Use alphas_cumprod[t] to compute x0
            # x0 = (x - sqrt(1 - alpha_prod_t) * eps) / sqrt(alpha_prod_t)
            if scheduler.config.prediction_type != "epsilon":
                raise NotImplementedError(
                    f"RTC guidance currently supports 'epsilon' prediction, "
                    f"got '{scheduler.config.prediction_type}'"
                )
            # t_step is a scalar/tensor index; fetch alpha_cumprod
            alpha_prod_t = scheduler.alphas_cumprod[t_step.long()].to(x_cur.device, x_cur.dtype)
            sqrt_alpha = torch.sqrt(alpha_prod_t).view(1, 1, 1)
            sqrt_one_minus = torch.sqrt(1.0 - alpha_prod_t).view(1, 1, 1)
            x0_hat = (x_cur - sqrt_one_minus * eps_pred) / (sqrt_alpha + 1e-12)
            return x0_hat

        for step_idx, t in enumerate(scheduler.timesteps):
            # 1) hard inpainting (freeze executed prefix)
            x = torch.where(condition_mask, condition_data, x)

            # 2) predict noise
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad_(True)
                eps = self.model(x, t, local_cond=local_cond, global_cond=global_cond)

                # 3) IIGDM-style guidance (on x0 reconstruction)
                if (rtc_target is not None) and (W is not None):
                    x0 = predict_x0_from_epsilon(x, eps, t)
                    diff = (x0 - rtc_target) * W
                    loss = (diff * diff).mean()

                    # VJP: d(loss)/d(eps)
                    g = torch.autograd.grad(loss, eps, retain_graph=False, create_graph=False)[0]

                    # guidance clipping (β) across (T*Da)
                    if beta_clip is not None:
                        g_norm = g.flatten(1).norm(p=2, dim=1, keepdim=True) + 1e-8
                        scale = torch.clamp(beta_clip / g_norm, max=1.0).view(B, 1, 1)
                        g = g * scale

                    # λ(t): stronger early, weaker later
                    tau = 1.0 - (step_idx / max(1, len(scheduler.timesteps) - 1))  # 1→0
                    lambda_t = lambda0 * tau

                    # nudge eps against gradient
                    eps = eps - lambda_t * g

            # 4) scheduler step
            out = scheduler.step(
                model_output=eps,
                timestep=t,
                sample=x,
                generator=generator,
                **kwargs,
            )
            x = out.prev_sample

        # final hard inpainting
        x = torch.where(condition_mask, condition_data, x)
        return x

    def predict_action(self, readout, rtc_target=None, rtc_mask=None) -> Dict[str, torch.Tensor]:
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = readout.device
        dtype = readout.dtype
        obs_features = readout
        assert obs_features.shape[0] == B * To  # To=1이면 통과

        local_cond = None
        global_cond = obs_features.reshape(B, -1)

        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            rtc_target=rtc_target,
            rtc_mask=rtc_mask,
            **self.kwargs,
        )

        action_pred = sample[..., :Da]
        return action_pred

    # ========= training  ============
    def compute_loss(self, readout, actions, return_aux: bool = False):
        B = readout.shape[0]
        T = self.horizon
        To = self.n_obs_steps

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory
        assert readout.shape[0] == B * To
        # reshape back to B, Do
        global_cond = readout.reshape(B, -1)  # (B, T*C)

        # generate inpainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device, dtype=trajectory.dtype)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Forward diffusion
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps,
            local_cond=local_cond,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        if return_aux:
            aux = {
                "xt_last": noisy_trajectory[:, -1, :].detach(),  # [B, Da]
                "xt": noisy_trajectory.detach(),
                "t": timesteps.detach(),
            }
            return loss, aux
        return loss
