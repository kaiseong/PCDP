# pcdp/model/sub/latent_goal_predictor.py
from __future__ import annotations
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGoalPredictor3Head(nn.Module):
    """
    현재 상태 -> 세 개의 '성공 직후' 잠재 목표 임베딩을 예측하는 경량 모듈.
      - z_g_grab   : 파란 큐브를 '잡은 직후' 장면의 잠재 벡터
      - z_g_place  : '초록 큐브 위에 놓은 직후' 장면의 잠재 벡터
      - z_g_return : '홈으로 복귀 완료 직후' 장면의 잠재 벡터

    입력은 z_t(현재 비주얼 임베딩), S_t(로봇 상태), z_hist(최근 N프레임 임베딩).
    z_hist는 프레임별 L2 정규화 후 단순 평균(mean)으로 요약(mu)한다.
    하드 게이트 없음. 예측된 세 목표는 GoalMixer에서 prior와 함께 소프트하게 섞인다.
    """

    def __init__(
        self,
        z_dim: int,              # 임베딩 차원 D (Sparse3DEncoder 출력 차원)
        s_dim: int = 10,         # 로봇 상태 차원 S (x,y,z, rot6d, gripper_width)
        hidden: int = 512,
        dropout: float = 0.1,
        use_hist: bool = True,   # 히스토리 요약(mu) 사용할지
        norm_latents: bool = True,  # 입력/출력 L2 정규화 여부
        out_norm: bool = True,      # 출력 z_g_* L2 정규화
        enable_compat: bool = False # (옵션) 적합도 예측 헤드 사용
    ) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.use_hist = bool(use_hist)
        self.norm_latents = bool(norm_latents)
        self.out_norm = bool(out_norm)
        self.enable_compat = bool(enable_compat)

        # 입력 피처: [ norm(z_t), mu(mean summary), S_t ] => 2*D + S
        in_dim = (2 * z_dim) + s_dim
        self.pre_ln = nn.LayerNorm(in_dim)

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 3개의 목표 임베딩 헤드 (각각 D차원)
        self.head_grab   = nn.Linear(hidden, z_dim)
        self.head_place  = nn.Linear(hidden, z_dim)
        self.head_return = nn.Linear(hidden, z_dim)

        # (옵션) 적합도 헤드: 현재 상태에서 각 목표의 실행 적합도(0~1)
        if self.enable_compat:
            self.compat_head = nn.Linear(hidden, 3)

    # --------------------- helpers ---------------------

    @staticmethod
    def _l2(x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1)

    def _as_BND(self, z_hist: torch.Tensor) -> torch.Tensor:
        """ z_hist: [B,N*D] or [B,N,D] -> [B,N,D] """
        if z_hist.dim() == 2:
            B, ND = z_hist.shape
            if ND % self.z_dim != 0:
                raise ValueError(f"z_hist last dim {ND} not divisible by z_dim {self.z_dim}")
            N = ND // self.z_dim
            return z_hist.view(B, N, self.z_dim)
        elif z_hist.dim() == 3:
            if z_hist.size(-1) != self.z_dim:
                raise ValueError(f"z_hist feature dim {z_hist.size(-1)} != z_dim {self.z_dim}")
            return z_hist
        else:
            raise ValueError(f"z_hist must be [B,N*D] or [B,N,D], got {tuple(z_hist.shape)}")

    def _mean_summary(self, z_hist: torch.Tensor) -> torch.Tensor:
        """ 프레임별 L2 정규화 후 시간 평균 → mu:[B,D] """
        zh = self._as_BND(z_hist)    # [B,N,D]
        zh = self._l2(zh) if self.norm_latents else zh
        mu = zh.mean(dim=1)          # [B,D]
        return mu

    # --------------------- forward ---------------------

    def forward(
        self,
        z_t: torch.Tensor,                 # [B, D]  현재 비주얼 임베딩
        S_t: torch.Tensor,                 # [B, S]  로봇 상태
        z_hist: Optional[torch.Tensor] = None,  # [B,N*D] or [B,N,D]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if z_t.dim() != 2 or S_t.dim() != 2:
            raise ValueError("z_t must be [B,D], S_t must be [B,S]")
        if z_t.size(0) != S_t.size(0):
            raise ValueError("Batch mismatch between z_t and S_t")
        if z_t.size(1) != self.z_dim:
            raise ValueError(f"z_t feature dim {z_t.size(1)} != z_dim {self.z_dim}")

        # 1) 현재 임베딩 정규화
        z_cur = self._l2(z_t) if self.norm_latents else z_t  # [B,D]

        # 2) 히스토리 요약(mu): 없으면 z_cur로 대체
        if self.use_hist and (z_hist is not None):
            mu = self._mean_summary(z_hist)                  # [B,D]
        else:
            mu = z_cur

        # 3) 피처 결합 → MLP
        x = torch.cat([z_cur, mu, S_t], dim=-1)              # [B, 2*D+S]
        x = self.pre_ln(x)
        h = self.trunk(x)                                    # [B,H]

        # 4) 세 목표 임베딩 예측
        z_g_grab   = self.head_grab(h)                       # [B,D]
        z_g_place  = self.head_place(h)                      # [B,D]
        z_g_return = self.head_return(h)                     # [B,D]
        if self.out_norm:
            z_g_grab   = self._l2(z_g_grab)
            z_g_place  = self._l2(z_g_place)
            z_g_return = self._l2(z_g_return)

        # 5) (옵션) 적합도
        compat = None
        if self.enable_compat:
            compat = torch.sigmoid(self.compat_head(h))      # [B,3] in (0,1)

        aux: Dict[str, torch.Tensor] = {"mu": mu}
        return z_g_grab, z_g_place, z_g_return, compat, aux


# --------------------- losses ---------------------

def cosine_goal_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ 1 - cos(pred, target)의 배치 평균 (내부 L2 정규화). """
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    return (1.0 - (pred_n * targ_n).sum(dim=-1)).mean()


def info_nce_goal_loss(
    pred: torch.Tensor,  # [B,D]
    target: torch.Tensor,# [B,D]
    temperature: float = 0.07,
) -> torch.Tensor:
    """ pred_i 와 target_i 만 양성, 나머지는 음성으로 보는 배치 대조 손실. """
    pred_n = F.normalize(pred, dim=-1)
    targ_n = F.normalize(target, dim=-1)
    logits = pred_n @ targ_n.t() / temperature  # [B,B]
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)


def combined_lgp_loss3(
    pred_g: torch.Tensor,  targ_g: torch.Tensor,
    pred_p: torch.Tensor,  targ_p: torch.Tensor,
    pred_r: torch.Tensor,  targ_r: torch.Tensor,
    lambda_nce: float = 0.5,
    temperature: float = 0.07,
) -> Dict[str, torch.Tensor]:
    """ 3개 헤드에 대해 cosine + λ·InfoNCE 합산, 항목별로도 반환. """
    cos_g = cosine_goal_loss(pred_g, targ_g)
    cos_p = cosine_goal_loss(pred_p, targ_p)
    cos_r = cosine_goal_loss(pred_r, targ_r)
    nce_g = info_nce_goal_loss(pred_g, targ_g, temperature)
    nce_p = info_nce_goal_loss(pred_p, targ_p, temperature)
    nce_r = info_nce_goal_loss(pred_r, targ_r, temperature)
    total = (cos_g + cos_p + cos_r) + lambda_nce * (nce_g + nce_p + nce_r)
    return {
        "loss_total": total,
        "loss_cos_grab": cos_g, "loss_cos_place": cos_p, "loss_cos_return": cos_r,
        "loss_nce_grab": nce_g, "loss_nce_place": nce_p, "loss_nce_return": nce_r,
    }
