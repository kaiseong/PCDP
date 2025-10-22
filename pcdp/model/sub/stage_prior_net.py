from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class StagePriorNet(nn.Module):
    """
    3-way soft prior (grab/place/return). No hard gates.
    Inputs
      z_t   : [B,D]  (PointCloud latent ONLY)
      S_t   : [B,S]  (robot low-dim state; e.g., S=10: x,y,z, rot6d, gripper_width)
      z_hist: [B,N*D] or [B,N,D] (optional)  -- mean summary after per-frame L2-norm
    Outputs
      q     : [B,3]  (softmax) -> [grab, place, return]
      logits: [B,3]
    """
    def __init__(self, z_dim: int, 
                s_dim: int,
                hidden: int = 512, 
                dropout: float = 0.1,
                 use_hist: bool = True, 
                 norm_latents: bool = True) -> None:
        super().__init__()
        self.z_dim, self.s_dim = int(z_dim), int(s_dim)
        self.use_hist, self.norm_latents = bool(use_hist), bool(norm_latents)

        in_dim = z_dim + s_dim + (z_dim if use_hist else z_dim)  # [z_cur, mu(or z_cur), S_t]
        self.pre_ln = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), 
            nn.GELU(), 
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 3)  # 3-way

    def _norm(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=-1) if self.norm_latents else z

    def _as_BND(self, z_hist: torch.Tensor) -> torch.Tensor:
        if z_hist.dim() == 2:
            B, ND = z_hist.shape
            if ND % self.z_dim != 0:
                raise ValueError(f"z_hist {ND} not divisible by z_dim {self.z_dim}")
            N = ND // self.z_dim
            return z_hist.view(B, N, self.z_dim)
        if z_hist.dim() == 3:
            if z_hist.size(-1) != self.z_dim:
                raise ValueError(f"z_hist feat {z_hist.size(-1)} != z_dim {self.z_dim}")
            return z_hist
        raise ValueError(f"z_hist must be [B,N*D] or [B,N,D]")

    def _mean_summary(self, z_hist: torch.Tensor) -> torch.Tensor:
        zh = self._as_BND(z_hist)        # [B,N,D]
        zh = self._norm(zh)              # per-frame L2
        return zh.mean(dim=1)            # [B,D]

    def forward(self, z_t: torch.Tensor, S_t: torch.Tensor,
                z_hist: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if z_t.dim()!=2 or S_t.dim()!=2: raise ValueError("z_t [B,D], S_t [B,S]")
        if z_t.size(0)!=S_t.size(0):     raise ValueError("Batch mismatch")
        if z_t.size(1)!=self.z_dim:      raise ValueError("z_t dim mismatch")

        z_cur = self._norm(z_t)          # [B,D]
        mu = z_cur if (not self.use_hist or z_hist is None) else self._mean_summary(z_hist)
        x = torch.cat([z_cur, mu, S_t], dim=-1)
        x = self.pre_ln(x)
        h = self.net(x)
        logits = self.head(h)            # [B,3]
        q = F.softmax(logits, dim=-1)    # [B,3]
        return q, logits

def prior_ce_loss(logits: torch.Tensor, target: torch.Tensor,
                  label_smoothing: float = 0.1) -> torch.Tensor:
    """
    3-way CE with optional smoothing.
    target: [B] int{0,1,2} or [B,3] soft
    """
    if target.dim()==1:
        B,C = logits.size(0), logits.size(1)
        eps = float(label_smoothing)
        with torch.no_grad():
            soft = torch.zeros(B, C, device=logits.device, dtype=logits.dtype)
            soft.scatter_(1, target.view(-1,1), 1.0)
            soft = soft*(1.0-eps) + eps/C
        tgt = soft
    elif target.dim()==2:
        tgt = target
    else:
        raise ValueError("target must be [B] or [B,3]")
    logp = F.log_softmax(logits, dim=-1)
    return -(tgt*logp).sum(dim=-1).mean()
