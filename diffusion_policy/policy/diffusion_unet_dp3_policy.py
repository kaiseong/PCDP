# diffusion_unet_dp3_policy.py
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch3d.ops as torch3d_ops
from einops import reduce
from termcolor import cprint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_pointcloud_policy import BasePointCloudPolicy
from diffusion_policy.model.diffusion.dp3_conditional_unet1d import DP3ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.dp3_model_util import print_params

class DP3Policy(BasePointCloudPolicy):
    def __init__(self,
                shape_meta: dict,
                noise_scheduler: DDPMScheduler,
                horizon,
                n_action_steps,
                n_obs_steps,
                num_inference_steps=None,
                obs_as_global_cond=True,
                diffusion_step_embed_dim=256,
                down_dims=(256,512,1024),
                kernel_size=5,
                n_groups=8,
                encoder_output_dim=256,
                condition_type="film",
                use_down_condition=True,
                use_mid_condition=True,
                use_up_condition=True,
                crop_shape=None,
                use_pc_color=False,
                pointnet_type="pointnet",
                pointcloud_encoder_cfg=None,
                **kwargs
                ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
        
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x:x['shape'])

        obs_encoder = DP3Encoder(observation_space = obs_dict,
                                img_crop_shape = crop_shape,
                                out_channel = encoder_output_dim,
                                pointcloud_encoder_cfg = pointcloud_encoder_cfg,
                                use_pc_color = use_pc_color,
                                pointnet_type = pointnet_type
                                )
        
        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        
        
        
        model = DP3ConditionalUnet1D(
            input_dim = input_dim,
            local_cond_dim = None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type = condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.condition_type = condition_type
        self.action_shape = action_shape
        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[Policy] use_pc_color: {self.use_pc_color}", "yellow", attrs=["bold"])
        cprint(f"[Policy] use_pc_color: {self.use_pc_color}", "yellow", attrs=["bold"])
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        print_params(self)
    
    # ========= inference  ============
    def conditional_sample(
            self,
            condition_data, condition_mask,
            condition_data_pc=None, condition_maks_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. appl conditioning
            trajectory[condition_mask] = condition_data[condition_mask]
            
            # 2. predict model output
            model_output = model(
                sample = trajectory,
                timestep=t,
                local_cond=local_cond, 
                global_cond=global_cond
            )
            # 3. compute previous data
            trajectory = scheduler.step(
                model_output, t, trajectory).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory
    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['pointcloud'] = nobs['pointcloud'][..., :3]
        this_n_point_cloud = nobs['pointcloud']

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x:x[:,:To,...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do <= original
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To, ...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True
        
        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond = local_cond,
            global_cond=global_cond,
            **self.kwargs
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    
    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['pointcloud'] = nobs['pointcloud'][..., :3] # remove point cloud color
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                                lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do <= original
                global_cond = nobs_features.reshape(batch_size, -1)
                this_n_point_cloud = this_nobs['pointcloud'].reshape(batch_size, -1, *this_nobs['pointcloud'].shape[1:])
                this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim= -1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device = trajectory.device)

        bsz = trajectory.shape[0]
        # Sample a random timestep for each scene
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz, ), device=trajectory.device
        ).long()

        # Add noise to the clean scene according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # applyy conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(sample = noisy_trajectory,
                        timestep=timesteps,
                        local_cond=local_cond,
                        global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timestpes], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        loss_dict = {
            'bc_loss': loss.item(),
        }

        return loss, loss_dict




            



