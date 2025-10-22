# pcdp/model/sub/stage_prior_net_seq.py
from __future__ import annotations
from typing import Optional, Tuple, Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StagePriorNetSeq(nn.Module):
    """
    Sequence-based, gate-free soft prior network:
    입력으로 z_seq, s_seq만 받으면 내부에서 정규화, 요약(mean/EMA/Attn/GRU), 방향(Δ)까지 처리하여
    q=[q_grab,q_place]를 출력한다.

    Inputs
    ------
    z_seq : [B, N, D]      # 최근 N 프레임의 PointCloud 임베딩(비주얼만)
    s_seq : [B, N, S] or None  # 동일 길이의 로봇 상태 시퀀스(없으면 자동 zero)
    mask  : [B, N] or None     # 1=유효, 0=패딩 (없으면 전부 유효로 가정)

    Outputs
    -------
    q      : [B, 2]        # soft prior (sum=1)
    logits : [B, 2]        # loss 계산용
    aux    : dict          # {'mu':[B,D], 'dz_last':[B,D], 'dS_last':[B,S], 'w':[B,N](attn/ema)}

    Notes
    -----
    - hist_mode: {'mean','ema','attn','gru'}
      * mean   : 프레임별 L2 정규화 후 평균
      * ema    : 지수이동평균(최신 프레임 가중↑)
      * attn   : 현재 프레임(z_t)을 쿼리로 soft-attention 요약(옵션 posenc)
      * gru    : GRU로 순서/방향성을 직접 학습하여 마지막 hidden 사용
    - 방향성은 항상 Δz_last(정규화된 z_t - z_{t-1})와 (있으면) ΔS_last를 자동 추가
    - 하드 게이트/룰 없음. 모두 연속, 학습 기반 신호.
    """

    def __init__(
        self,
        z_dim: int,
        s_dim: int,
        hidden: int = 512,
        dropout: float = 0.1,
        hist_mode: str = "attn",   # {'mean','ema','attn','gru'}
        ema_alpha: float = 0.6,    # EMA에서 최신 프레임 비중(↑ 최근 강조)
        attn_beta: float = 6.0,    # Attn softmax 샤프니스(↑ 날카로움)
        use_posenc: bool = False,  # Attn에서 간단한 sin-cos positional encoding 사용
        norm_latents: bool = True,
        use_delta_state: bool = True,
        gru_hidden: int = 512,     # hist_mode='gru'에서 사용
    ) -> None:
        super().__init__()
        assert hist_mode in {"mean", "ema", "attn", "gru"}
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.hist_mode = hist_mode
        self.ema_alpha = float(ema_alpha)
        self.attn_beta = float(attn_beta)
        self.use_posenc = bool(use_posenc)
        self.norm_latents = bool(norm_latents)
        self.use_delta_state = bool(use_delta_state)

        # GRU 경로 준비
        if self.hist_mode == "gru":
            self.pre_ln_seq = nn.LayerNorm(z_dim + s_dim)
            self.gru = nn.GRU(
                input_size=z_dim + s_dim,
                hidden_size=gru_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.gru_proj = nn.Linear(gru_hidden, z_dim)  # 최종 요약을 z_dim로 사상

        # 최종 피처 구성: [norm(z_t), mu, Δz_last, S_t, (ΔS_last)]
        in_dim = z_dim + z_dim + z_dim + s_dim
        if self.use_delta_state:
            in_dim += s_dim

        self.pre_ln = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 2)

    # ---------- helpers ----------

    def _l2norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1) if self.norm_latents else x

    def _apply_mask(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return x
        # mask: [B,N] -> [B,N,1]
        m = mask.unsqueeze(-1).to(x.dtype)
        return x * m

    def _sinusoidal_posenc(self, N: int, D: int, device, dtype) -> torch.Tensor:
        # [N, D], 표준 Transformer 스타일
        pe = torch.zeros(N, D, device=device, dtype=dtype)
        position = torch.arange(0, N, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=dtype) * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [N,D]

    def _summary_mean(self, zh: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # zh: [B,N,D] (정규화 완료)
        if mask is None:
            mu = zh.mean(dim=1)
            return mu, {}
        m = mask.to(zh.dtype).unsqueeze(-1)          # [B,N,1]
        denom = m.sum(dim=1).clamp_min(1.0)          # [B,1,1] -> [B,1,1] but later broadcast
        mu = (zh * m).sum(dim=1) / denom             # [B,D]
        return mu, {}

    def _summary_ema(self, zh: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, D = zh.shape
        idx = torch.arange(N, device=zh.device, dtype=zh.dtype)
        power = (N - 1 - idx)
        base = (1.0 - self.ema_alpha)
        w = (self.ema_alpha * (base ** power)).clamp(min=1e-8)  # [N]
        if mask is not None:
            # 마스크된 프레임은 가중치 0
            w = w * mask.to(zh.dtype).mean(0)  # 간단 처리(배치 평균 마스크)
        w = (w / w.sum()).view(1, N, 1)
        mu = (zh * w).sum(dim=1)               # [B,D]
        aux = {"w": w.squeeze(-1).expand(B, -1)}  # [B,N]
        return mu, aux

    def _summary_attn(self, zh: torch.Tensor, zt: torch.Tensor, mask: Optional[torch.Tensor], use_posenc: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # zh: [B,N,D] (정규화 완료), zt: [B,D] (정규화 완료)
        B, N, D = zh.shape
        if use_posenc:
            pe = self._sinusoidal_posenc(N, D, zh.device, zh.dtype)  # [N,D]
            zh = zh + pe.view(1, N, D)

        sim = torch.einsum("bd,bnd->bn", zt, zh)         # cosine (정규화되어 있음)
        if mask is not None:
            sim = sim.masked_fill(mask == 0, float("-inf"))
        w = torch.softmax(self.attn_beta * sim, dim=-1)  # [B,N]
        mu = torch.einsum("bn,bnd->bd", w, zh)           # [B,D]
        return mu, {"w": w}

    def _summary_gru(self, z_seq: torch.Tensor, s_seq: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # z_seq: [B,N,D] (정규화 전), s_seq: [B,N,S]
        x = torch.cat([self._l2norm(z_seq), s_seq], dim=-1)  # [B,N,D+S]
        x = self.pre_ln_seq(x)
        if mask is not None:
            lengths = mask.sum(dim=1).clamp_min(1).to(torch.long)  # [B]
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hN = self.gru(packed)
        else:
            _, hN = self.gru(x)  # hN: [1,B,H]
        h = hN.squeeze(0)                     # [B,H]
        mu = self.gru_proj(h)                 # [B,D] -> z_dim로 사상
        return mu, {}

    # ---------- forward ----------

    def forward(
        self,
        z_seq: torch.Tensor,           # [B,N,D]
        s_seq: Optional[torch.Tensor] = None,   # [B,N,S] or None
        mask: Optional[torch.Tensor] = None,    # [B,N] or None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if z_seq.dim() != 3: raise ValueError(f"z_seq must be [B,N,D], got {tuple(z_seq.shape)}")
        B, N, D = z_seq.shape
        if D != self.z_dim: raise ValueError(f"z_dim mismatch: {D} != {self.z_dim}")

        if s_seq is None:
            s_seq = torch.zeros(B, N, self.s_dim, device=z_seq.device, dtype=z_seq.dtype)
        else:
            if s_seq.shape[:2] != z_seq.shape[:2] or s_seq.size(-1) != self.s_dim:
                raise ValueError(f"s_seq must be [B,N,S={self.s_dim}] and share [B,N] with z_seq")

        # 정상화
        zh = self._l2norm(z_seq)                # [B,N,D]
        zt = zh[:, -1, :]                       # 현재 프레임 (정규화된) [B,D]
        St = s_seq[:, -1, :]                    # 현재 상태 [B,S]

        # 요약(mu)
        aux: Dict[str, torch.Tensor] = {}
        if self.hist_mode == "mean":
            mu, a = self._summary_mean(zh, mask)
        elif self.hist_mode == "ema":
            mu, a = self._summary_ema(zh, mask)
        elif self.hist_mode == "attn":
            mu, a = self._summary_attn(zh, zt, mask, self.use_posenc)
        else:  # 'gru'
            mu, a = self._summary_gru(z_seq, s_seq, mask)
        aux.update(a)

        # 방향(Δ) — 마지막 두 프레임 차분(없으면 0)
        if N >= 2:
            dz_last = zt - zh[:, -2, :]         # [B,D]  (정규화 기반)
            dS_last = St - s_seq[:, -2, :]      # [B,S]
        else:
            dz_last = torch.zeros_like(zt)
            dS_last = torch.zeros_like(St)

        feats = [zt, mu, dz_last, St]
        if self.use_delta_state:
            feats.append(dS_last)

        x = torch.cat(feats, dim=-1)            # [B, in_dim]
        x = self.pre_ln(x)
        h = self.mlp(x)                         # [B,H]
        logits = self.head(h)                   # [B,2]
        q = F.softmax(logits, dim=-1)           # [B,2]

        aux.update({"mu": mu, "dz_last": dz_last, "dS_last": dS_last})
        return q, logits, aux
