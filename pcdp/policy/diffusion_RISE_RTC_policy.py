# diffusion_RISE_RTC_policy.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import cprint
from pcdp.policy.base_pointcloud_policy import BasePointCloudPolicy
from pcdp.model.diffusion.RISE_diffusionUNet_RTC import DiffusionUNetPolicy
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from pcdp.policy.RISE_transformer import Transformer
import MinkowskiEngine as ME
from pcdp.model.common.normalizer import LinearNormalizer

class RISEPolicy(BasePointCloudPolicy):
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
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        # (기존) num_inference_steps=12은 실시간 프리셋
        self.action_decoder = DiffusionUNetPolicy(action_dim, num_action, num_obs, obs_feature_dim, num_inference_steps=12)
        self.readout_embed = nn.Embedding(1, hidden_dim)
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, cloud, actions=None, batch_size=24, rtc_target=None, rtc_mask=None):
        dev = cloud.F.device if hasattr(cloud, "F") else cloud.device
        if self.normalizer is not None:
            norm_dev = getattr(self.normalizer, "device", None)
            if norm_dev != dev:
                self.normalizer.to(dev)

        # point cloud 정규화 (색상)
        if self.normalizer is not None:
            cloud_feats = cloud.F
            cloud_colors = cloud_feats[:, 3:]
            norm_colors = self.normalizer['pointcloud_color'].normalize(cloud_colors)
            cloud = ME.SparseTensor(
                features=torch.cat([cloud_feats[:, :3], norm_colors], dim=1),
                coordinates=cloud.C,
                device=cloud.device
            )
            if actions is not None:
                # 학습 시 액션 정규화
                action_trans = actions[:, :, :3]
                action_rot = actions[:, :, 3:9]
                action_grip = actions[:, :, 9:10]
                norm_action_trans = self.normalizer['action_translation'].normalize(action_trans)
                norm_action_grip = self.normalizer['action_gripper'].normalize(action_grip)
                actions = torch.cat([norm_action_trans, action_rot, norm_action_grip], dim=-1)

        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]

        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        else:
            # ========================= [RTC 수정 1] =========================
            # Original RTC는 conditional_sample 내부에서 autograd(VJP)를 사용하므로
            # 여기서는 torch.no_grad()를 쓰면 안 됨. (기존 no_grad 삭제)
            # ================================================================
            # ========================= [RTC 수정 2] =========================
            # rtc_target는 현재(외부) 좌표계 스케일(unnormalized)일 수 있으므로
            # 디퓨전 샘플링 공간과 스케일을 맞추기 위해 정규화해서 전달.
            rtc_target_norm = None
            if rtc_target is not None:
                # 기대 shape: (B, T, 10)
                rtc_target = rtc_target.to(dev, dtype=torch.float32)
                t = self.normalizer['action_translation'].normalize(rtc_target[..., :3])
                r = rtc_target[..., 3:9]  # 회전 6D는 학습 시 정규화 없이 사용했다면 그대로
                g = self.normalizer['action_gripper'].normalize(rtc_target[..., 9:10])
                rtc_target_norm = torch.cat([t, r, g], dim=-1)

            # rtc_mask dtype/device 정렬(가중치라 정규화 불필요)
            if rtc_mask is not None:
                rtc_mask = rtc_mask.to(dev, dtype=torch.float32)

            # 디퓨전 UNet로 예측 (RTC soft-guidance는 UNet 내부에서 수행)
            action_pred = self.action_decoder.predict_action(
                readout,
                rtc_target=rtc_target_norm,   # [RTC 수정 2] 정규화된 타깃 전달
                rtc_mask=rtc_mask
            )

            # un-normalize (외부로 내보낼 때 복원)
            if self.normalizer is not None:
                action_trans = action_pred[:, :, :3]
                action_rot = action_pred[:, :, 3:9]
                action_grip = action_pred[:, :, 9:10]
                unnorm_action_trans = self.normalizer['action_translation'].unnormalize(action_trans)
                unnorm_action_grip = self.normalizer['action_gripper'].unnormalize(action_grip)
                action_pred = torch.cat([unnorm_action_trans, action_rot, unnorm_action_grip], dim=-1)

            return action_pred
