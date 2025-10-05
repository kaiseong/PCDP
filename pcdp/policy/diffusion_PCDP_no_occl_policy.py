# diffusion_unet_dp3_policy.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import cprint
from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.model.diffusion.RISE_diffusionUNet import DiffusionUNetPolicy
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from pcdp.policy.RISE_transformer import Transformer
import MinkowskiEngine as ME
from pcdp.model.common.normalizer import LinearNormalizer

class PCDPPolicy(BasePointCloudPolicy):
    def __init__(
            self,
            num_action = 20,
            input_dim = 6,
            obs_feature_dim = 512,
            action_dim = 10,
            hidden_dim = 512,
            nheads = 8,
            num_encoder_layers = 4,
            num_decoder_layers = 1,
            dim_feedforward = 2048,
            dropout = 0.1
    ):
        super().__init__()
        num_obs = 1
        robot_obs_dim = 10
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.normalizer = LinearNormalizer()
        # policy
        self.cond_proj = nn.Sequential(
            nn.LayerNorm(obs_feature_dim + robot_obs_dim), # 522
            nn.Linear(obs_feature_dim + robot_obs_dim, obs_feature_dim), # 522 -> 512
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(obs_feature_dim, obs_feature_dim) # 512 -> 512
        )




    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def forward(self, cloud: ME.SparseTensor, actions=None, robot_obs=None, batch_size=24):
        dev = cloud.F.device if hasattr(cloud, "F") else cloud.device
        # normalizer 디바이스 정렬 (방어적)
        if self.normalizer is not None:
            norm_dev = getattr(self.normalizer, "device", None)
            if norm_dev != dev:
                self.normalizer.to(dev)

        # Normalize inputs if normalizer is set
        if self.normalizer is not None:
            # Normalize point cloud colors
            cloud_feats = cloud.F
            cloud_colors = cloud_feats[:, 3:]
            norm_colors = self.normalizer['pointcloud_color'].normalize(cloud_colors)
            # Create new sparse tensor with normalized colors
            cloud = ME.SparseTensor(
                features=torch.cat([cloud_feats[:, :3], norm_colors], dim=1),
                coordinates=cloud.C,
                device=cloud.device
            )
            
            if robot_obs.dim() == 2: # train: [B, 1, 10], Inference: [B, 10]
                robot_obs = robot_obs.unsqueeze(1)

            # Normalize low-dim robot_obs
            # robot_obs: (B, T, 10) where T=n_obs_steps
            obs_trans = robot_obs[:, :, :3]
            obs_rot = robot_obs[:, :, 3:9]
            obs_grip = robot_obs[:, :, 9:10]
            norm_obs_trans = self.normalizer['obs_translation'].normalize(obs_trans)
            norm_obs_grip = self.normalizer['obs_gripper'].normalize(obs_grip)
            robot_obs = torch.cat([norm_obs_trans, obs_rot, norm_obs_grip], dim=-1)

            if actions is not None:
                # Normalize actions for training
                # actions: (B, H, 10) where H=horizon
                action_trans = actions[:, :, :3]
                action_rot = actions[:, :, 3:9]
                action_grip = actions[:, :, 9:10]
                norm_action_trans = self.normalizer['action_translation'].normalize(action_trans)
                norm_action_grip = (action_grip > 0.5) * 2 - 1 # Binarize and map to -1, 1
                actions = torch.cat([norm_action_trans, action_rot, norm_action_grip], dim=-1)
            

        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size = batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1] # [B, 1, 512]
        readout_raw = readout[:, 0, :]  # [B,512]  (대조 q는 이걸 사용)
        
        
        
        cond_in = torch.cat([readout_raw, robot_obs[:,0,:]], dim =-1)
        global_cond = self.cond_proj(cond_in)
        
        
        if actions is not None:
            # training mode
            loss = self.action_decoder.compute_loss(global_cond, actions)
            return loss
        else:
            # inference mode
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(global_cond)

            # un-normalize action prediction
            if self.normalizer is not None:
                action_trans = action_pred[:, :, :3]
                action_rot = action_pred[:, :, 3:9]
                action_grip = action_pred[:, :, 9:10]

                unnorm_action_trans = self.normalizer['action_translation'].unnormalize(action_trans)
                unnorm_action_grip = (action_grip > 0).float()

                action_pred = torch.cat([unnorm_action_trans, action_rot, unnorm_action_grip], dim=-1)

            return action_pred