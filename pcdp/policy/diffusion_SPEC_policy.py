# diffusion_SPEC_policy.py
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

class CrossAttn(nn.Module):
    "now(Q), mem(KV) cross attn 1 block"
    def __init__(self, d_model=512, nheads=8, dropout=0.1):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln_out = nn.LayerNorm(d_model)

    def forward(self, now, mem, pad_now=None, pad_mem=None):
        attn_out, _ = self.attn(self.ln_q(now), self.ln_kv(mem), self.ln_kv(mem),
                                key_padding_mask=pad_mem)
        x = now + attn_out
        x = x + self.ffn(self.ln_out(x))
        return x


class SPECPolicy(BasePointCloudPolicy):
    def __init__(
            self,
            num_action = 20,
            input_dim = 7,
            obs_feature_dim = 512,
            action_dim = 10,
            hidden_dim = 512,
            nheads = 8,
            num_encoder_layers = 4,
            num_decoder_layers = 1,
            dim_feedforward = 2048,
            dropout = 0.1,
            # spec
            mem_kv_mode: str = "past_only",     # 'past_only' | 'all'
            enable_c_gate: bool = False,        # True면 rgb *= c 게이팅
            c_now_threshold: float = 1.0 - 1e-6,
            c_past_threshold: float = 1.0 - 1e-6,
    ):
        super().__init__()
        num_obs = 1
        robot_obs_dim = 10

        # dual encoder
        self.enc_mem = Sparse3DEncoder(input_dim=input_dim, output_dim=obs_feature_dim)
        self.enc_now = Sparse3DEncoder(input_dim=input_dim, output_dim=obs_feature_dim)
        # cross attention
        self.cross_attn = CrossAttn(d_model=obs_feature_dim, nheads=nheads, dropout=dropout)
        # rise
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim + robot_obs_dim)
        # self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.normalizer = LinearNormalizer()
        # policy
        self.enable_c_gate=enable_c_gate
        self.mem_kv_mode=mem_kv_mode
        self.c_now_threshold = c_now_threshold
        self.c_past_threshold = c_past_threshold
        

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def _make_now_view(self, cloud: ME.SparseTensor) -> ME.SparseTensor:
        """UNION 없이 now 부분좌표만 추출 (c>=1-ε). 빈 경우 fallback로 전체 반환."""
        feats = cloud.F
        coords = cloud.C
        c = feats[:, 6]
        mask = c >= self.c_now_threshold
        if mask.sum() == 0:
            return cloud
        return ME.SparseTensor(features=feats[mask], coordinates=coords[mask], device=cloud.device)
    
    @torch.no_grad()
    def _make_mem_kv_view(self, cloud: ME.SparseTensor) -> ME.SparseTensor:
        """K/V로 쓸 메모리 뷰 선택: 'past_only'면 c<1-ε만, 'all'이면 전체."""
        if self.mem_kv_mode=='all':
            return cloud
        feats, coords = cloud.F, cloud.C
        c = feats[:, 6]
        mask = c < self.c_past_threshold
        if mask.sum() == 0:
            return cloud
        return ME.SparseTensor(features=feats[mask], coordinates=coords[mask], device=cloud.device)
    

    def forward(self, cloud: ME.SparseTensor, actions=None, robot_obs=None, batch_size=24):
        assert robot_obs is not None, "SPECPolicy expects robot_obs (B,10) or (B,T,10)."
        if robot_obs.dim() == 2:  # Inference: [B,10] -> [B,1,10]
            robot_obs = robot_obs.unsqueeze(1)

        dev = cloud.F.device if hasattr(cloud, "F") else cloud.device
        if self.normalizer is not None:
            self.normalizer.to(dev)

        # Normalize inputs if normalizer is set
        if self.normalizer is not None:
            # Normalize point cloud colors
            cloud_feats = cloud.F
            cloud_colors = cloud_feats[:, 3:6]
            confidence = cloud_feats[:, 6:7]
            norm_colors = self.normalizer['pointcloud_color'].normalize(cloud_colors)
            if self.enable_c_gate:
                norm_colors = norm_colors * confidence
            # Create new sparse tensor with normalized colors
            cloud = ME.SparseTensor(
                features=torch.cat([cloud_feats[:, :3], norm_colors, confidence], dim=1),
                coordinates=cloud.C,
                device=cloud.device
            )

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
                norm_action_grip = self.normalizer['action_gripper'].normalize(action_grip)
                actions = torch.cat([norm_action_trans, action_rot, norm_action_grip], dim=-1)
        
        # now / mem 분리 (UNION 불필요)
        cloud_now = self._make_now_view(cloud) # Q
        cloud_mem = self._make_mem_kv_view(cloud) # K/V

        # dual encoding
        mem_src, mem_pos, mem_pad = self.enc_mem(cloud_mem, batch_size=batch_size)
        now_src, now_pos, now_pad = self.enc_now(cloud_now, batch_size=batch_size)

        # cross attention
        now_fused = self.cross_attn(now_src + now_pos, mem_src+mem_pos, pad_now=now_pad, pad_mem=mem_pad)
        readout_seq = self.transformer(now_fused, now_pad, self.readout_embed.weight, now_pos)[-1] # [B, 1, 512]
        readout_raw = readout_seq[:, 0, :]  # [B,512] 
        
        
        global_cond = torch.cat([readout_raw, robot_obs[:,0,:]], dim =-1)
        # cond_in = torch.cat([readout_raw, robot_obs[:,0,:]], dim =-1)
        # global_cond = self.cond_proj(cond_in)
        
        
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
                unnorm_action_grip = self.normalizer['action_gripper'].unnormalize(action_grip)

                action_pred = torch.cat([unnorm_action_trans, action_rot, unnorm_action_grip], dim=-1)

            return action_pred