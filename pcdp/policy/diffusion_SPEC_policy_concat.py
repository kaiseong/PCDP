# pcdp/policy/diffusion_SPEC_policy_concat.py
import torch
import torch.nn as nn
import MinkowskiEngine as ME

from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.model.diffusion.RISE_diffusionUNet import DiffusionUNetPolicy
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from pcdp.policy.RISE_transformer import Transformer
from pcdp.model.common.normalizer import LinearNormalizer


class SPECPolicyConcat(BasePointCloudPolicy):
    """
    Cross-Attn 없이: 듀얼 인코더(now/mem) → 토큰 concat → RISE → UNet
    입력: Nx7 = [x,y,z,r,g,b,c], c∈[0,1]
      - now view : c >= 1 - eps
      - mem view : (기본) c < 1 - eps  (mem_kv_mode='past_only')
    """
    def __init__(
        self,
        num_action: int = 20,
        input_dim: int = 7,
        obs_feature_dim: int = 512,
        action_dim: int = 10,
        hidden_dim: int = 512,
        nheads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        # concat spec
        mem_kv_mode: str = "past_only",     # 'past_only' | 'all'
        mem_token_cap: int | None = None,   # None or int
        enable_c_gate: bool = False,        # True면 rgb *= c
        c_now_threshold: float = 1.0 - 1e-6,
        c_past_threshold: float = 1.0 - 1e-6,
    ):
        super().__init__()
        num_obs = 1
        self.robot_obs_dim = 10

        # 듀얼 인코더 (각 토큰/pos 차원 = obs_feature_dim)
        self.enc_mem = Sparse3DEncoder(input_dim=input_dim, output_dim=obs_feature_dim)
        self.enc_now = Sparse3DEncoder(input_dim=input_dim, output_dim=obs_feature_dim)

        # RISE Transformer(d_model = hidden_dim)
        self.transformer = Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout
        )
        self.readout_embed = nn.Embedding(1, hidden_dim)

        # UNet 입력 차원은 RISE readout(hidden_dim) + robot_obs(10)
        self.action_decoder = DiffusionUNetPolicy(
            action_dim, num_action, num_obs, hidden_dim + self.robot_obs_dim
        )

        self.normalizer = LinearNormalizer()

        # 옵션
        self.enable_c_gate = enable_c_gate
        self.mem_kv_mode = mem_kv_mode
        self.mem_token_cap = mem_token_cap
        self.c_now_threshold = c_now_threshold
        self.c_past_threshold = c_past_threshold

        # obs_feature_dim != hidden_dim 대비: feature/pos 모두 투영(안전장치)
        self.proj_feat = (
            nn.Identity() if obs_feature_dim == hidden_dim else nn.Linear(obs_feature_dim, hidden_dim)
        )
        self.proj_pos = (
            nn.Identity() if obs_feature_dim == hidden_dim else nn.Linear(obs_feature_dim, hidden_dim, bias=False)
        )

        # 모달리티 임베딩은 Transformer 차원에 맞춰(hidden_dim) 부여
        self.mod_embed = nn.Embedding(2, hidden_dim)  # 0: now, 1: mem

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def _make_now_view(self, cloud: ME.SparseTensor) -> ME.SparseTensor:
        feats, coords = cloud.F, cloud.C
        c = feats[:, 6]
        mask = c >= self.c_now_threshold
        if mask.sum() == 0:
            return cloud  # fallback
        return ME.SparseTensor(features=feats[mask], coordinates=coords[mask], device=cloud.device)

    @torch.no_grad()
    def _make_mem_view(self, cloud: ME.SparseTensor) -> ME.SparseTensor:
        if self.mem_kv_mode == "all":
            return cloud
        feats, coords = cloud.F, cloud.C
        c = feats[:, 6]
        mask = c < self.c_past_threshold
        if mask.sum() == 0:
            return cloud  # fallback
        return ME.SparseTensor(features=feats[mask], coordinates=coords[mask], device=cloud.device)

    @torch.no_grad()
    def _cap_mem_tokens(self, cloud_mem: ME.SparseTensor) -> ME.SparseTensor:
        if self.mem_token_cap is None:
            return cloud_mem
        f, c = cloud_mem.F, cloud_mem.C
        if f.shape[0] <= self.mem_token_cap:
            return cloud_mem
        idx = torch.randperm(f.shape[0], device=f.device)[: self.mem_token_cap]
        return ME.SparseTensor(features=f[idx], coordinates=c[idx], device=cloud_mem.device)

    def forward(self, cloud: ME.SparseTensor, actions=None, robot_obs=None, batch_size=24):
        assert robot_obs is not None, "SPECPolicyConcat expects robot_obs (B,10) or (B,T,10)."
        if robot_obs.dim() == 2:  # [B,10] -> [B,1,10]
            robot_obs = robot_obs.unsqueeze(1)

        dev = cloud.F.device if hasattr(cloud, "F") else cloud.device
        if self.normalizer is not None:
            self.normalizer.to(dev)

        # --- PointCloud 정규화: RGB만, c는 그대로 (옵션: 게이팅) ---
        if self.normalizer is not None:
            feats = cloud.F                         # [N, 7]
            rgb = feats[:, 3:6]                    # [N, 3]
            c = feats[:, 6:7]                      # [N, 1]
            rgb_n = self.normalizer["pointcloud_color"].normalize(rgb)
            if self.enable_c_gate:
                rgb_n = rgb_n * c
            cloud = ME.SparseTensor(
                features=torch.cat([feats[:, :3], rgb_n, c], dim=1),  # [N, 7]
                coordinates=cloud.C,
                device=cloud.device,
            )

        # --- now/mem 분리 & (옵션) mem cap ---
        cloud_now = self._make_now_view(cloud)        # now only
        cloud_mem = self._make_mem_view(cloud)        # mem (past_only/all)
        cloud_mem = self._cap_mem_tokens(cloud_mem)

        # --- 듀얼 인코딩 ---
        # mem_src/now_src: [B, L_*, ObsD], mem_pos/now_pos: [B, L_*, ObsD], *_pad: [B, L_*]
        mem_src, mem_pos, mem_pad = self.enc_mem(cloud_mem, batch_size=batch_size)
        now_src, now_pos, now_pad = self.enc_now(cloud_now, batch_size=batch_size)

        # --- 차원 정합: ObsD -> HidD (필요시) ---
        # now_src/mem_src: [B, L_*, HidD], now_pos/mem_pos: [B, L_*, HidD]
        now_src = self.proj_feat(now_src)
        mem_src = self.proj_feat(mem_src)
        now_pos = self.proj_pos(now_pos)
        mem_pos = self.proj_pos(mem_pos)

        # --- 모달리티 임베딩 부여(Transformer 차원) ---
        e_now = self.mod_embed.weight[0].view(1, 1, -1)  # [1,1,HidD]
        e_mem = self.mod_embed.weight[1].view(1, 1, -1)  # [1,1,HidD]
        now_src = now_src + e_now                        # [B, L_now, HidD]
        mem_src = mem_src + e_mem                        # [B, L_mem, HidD] 

        # --- 토큰 concat ---
        src = torch.cat([now_src, now_src.new_zeros(0)], dim=0)  # no-op to hint type
        src = torch.cat([now_src, mem_src], dim=1)    # [B, L_now+L_mem, HidD]
        pos = torch.cat([now_pos, mem_pos], dim=1)    # [B, L_now+L_mem, HidD]
        pad = torch.cat([now_pad, mem_pad], dim=1)    # [B, L_now+L_mem]

        # --- RISE Transformer readout ---
        hs = self.transformer(src, pad, self.readout_embed.weight, pos)  # list of layers, each [B,1,HidD]
        readout_seq = hs[-1]                       # [B, 1, HidD]
        readout_raw = readout_seq[:, 0, :]         # [B, HidD]   (← 버그 수정: [:, 0:, :] 아님)

        # --- 로봇 관측 정규화 ---
        if self.normalizer is not None:
            obs_trans = robot_obs[:, :, :3]
            obs_rot   = robot_obs[:, :, 3:9]
            obs_grip  = robot_obs[:, :, 9:10]
            norm_obs_trans = self.normalizer["obs_translation"].normalize(obs_trans)
            norm_obs_grip  = self.normalizer["obs_gripper"].normalize(obs_grip)
            robot_obs = torch.cat([norm_obs_trans, obs_rot, norm_obs_grip], dim=-1)  # [B,1,10]

        # --- Diffusion UNet condition ---
        global_cond = torch.cat([readout_raw, robot_obs[:, 0, :]], dim=-1)  # [B, HidD+10]

        # --- 학습/추론 경로 ---
        if actions is not None:
            if self.normalizer is not None:
                action_trans = actions[:, :, :3]
                action_rot   = actions[:, :, 3:9]
                action_grip  = actions[:, :, 9:10]
                norm_action_trans = self.normalizer["action_translation"].normalize(action_trans)
                norm_action_grip  = self.normalizer["action_gripper"].normalize(action_grip)
                actions = torch.cat([norm_action_trans, action_rot, norm_action_grip], dim=-1)
            loss = self.action_decoder.compute_loss(global_cond, actions)
            return loss
        else:
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(global_cond)  # [B,H,10]
            if self.normalizer is not None:
                action_trans = action_pred[:, :, :3]
                action_rot   = action_pred[:, :, 3:9]
                action_grip  = action_pred[:, :, 9:10]
                unnorm_action_trans = self.normalizer["action_translation"].unnormalize(action_trans)
                unnorm_action_grip  = self.normalizer["action_gripper"].unnormalize(action_grip)
                action_pred = torch.cat([unnorm_action_trans, action_rot, unnorm_action_grip], dim=-1)
            return action_pred
