# pcdp/model/sub/goal_mixer.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalMixer(nn.Module):
    """
    GoalMixer: 여러 잠재 목표(z_g_i)와 소프트 prior(q_prior), (옵션) 적합도(compat)를 결합해
    단일 목표 임베딩 z_g_mix를 만드는 경량 모듈. 하드 게이트 없음.

    기본 아이디어
    ------------
    score_i = (use_compat ? log(compat_i+eps) : 0)
              + eta * q_prior_i
              + (use_sim ? sim_scale * cos(z_cur, z_g_i) : 0)

    w = softmax( beta * score ),           # [B, N]
    z_g_mix = Σ_i w_i * z_g_i (정규화 후)  # [B, D]

    매개변수
    -------
    beta : softmax 샤프니스 (↑ 클수록 '가장 그럴듯한' 목표로 더 강하게 쏠림)
    eta  : prior(q_prior)의 영향 가중치
    use_compat : LGP가 뱉은 compat(0~1)를 사용할지
    use_sim    : 현재 z_cur와 각 z_g의 코사인 유사도를 점수에 포함할지
    sim_scale  : 유사도 항의 스케일
    normalize_output : z_g_mix를 L2 정규화해서 반환할지
    eps : 수치 안정성(compat 로그에 쓰임)
    """

    def __init__(
        self,
        beta: float = 6.0,
        eta: float = 1.0,
        use_compat: bool = False,
        use_sim: bool = False,
        sim_scale: float = 1.0,
        normalize_output: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.beta = float(beta)
        self.eta = float(eta)
        self.use_compat = bool(use_compat)
        self.use_sim = bool(use_sim)
        self.sim_scale = float(sim_scale)
        self.normalize_output = bool(normalize_output)
        self.eps = float(eps)

    @staticmethod
    def _stack_goals(z_goals: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        z_goals: list of [B,D] or a single [B,N,D] tensor  ->  [B,N,D]
        """
        if isinstance(z_goals, list):
            assert len(z_goals) >= 2, "Need at least two goals to mix."
            B, D = z_goals[0].shape
            for z in z_goals:
                if z.shape != (B, D):
                    raise ValueError(f"Inconsistent goal shape {tuple(z.shape)} vs {(B, D)}")
            return torch.stack(z_goals, dim=1)  # [B,N,D]
        elif isinstance(z_goals, torch.Tensor):
            if z_goals.dim() != 3:
                raise ValueError(f"z_goals must be [B,N,D], got {tuple(z_goals.shape)}")
            return z_goals
        else:
            raise TypeError("z_goals must be List[Tensor] or Tensor")

    def forward(
        self,
        z_cur: torch.Tensor,                       # [B, D]
        z_goals: Union[List[torch.Tensor], torch.Tensor],  # [B,N,D] or list of [B,D]
        q_prior: torch.Tensor,                     # [B, N]  (from StagePriorNet)
        compat: Optional[torch.Tensor] = None,     # [B, N] in (0,1), optional (from LGP)
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        z_g_mix : [B, D]    # mixed goal embedding
        w       : [B, N]    # mixing weights (soft assignment over goals)
        aux     : dict      # debug info: {'score', 'compat_term', 'sim', 'prior'}
        """
        if z_cur.dim() != 2:
            raise ValueError(f"z_cur must be [B,D], got {tuple(z_cur.shape)}")
        z_stack = self._stack_goals(z_goals)       # [B,N,D]
        B, N, D = z_stack.shape

        if q_prior.shape != (B, N):
            raise ValueError(f"q_prior must be [B,{N}], got {tuple(q_prior.shape)}")

        # normalize for cosine stability if similarity is used or for safe mixing
        z_cur_n = F.normalize(z_cur, dim=-1)
        z_goals_n = F.normalize(z_stack, dim=-1)

        # 1) prior term
        prior_term = q_prior  # [B,N]  (확률 그대로; 스케일은 eta가 담당)

        # 2) compat term (optional, in log-space for dynamic range)
        if self.use_compat and (compat is not None):
            if compat.shape != (B, N):
                raise ValueError(f"compat must be [B,{N}], got {tuple(compat.shape)}")
            compat_term = torch.log(compat.clamp_min(self.eps))  # [B,N]
        else:
            compat_term = torch.zeros(B, N, device=z_stack.device, dtype=z_stack.dtype)

        # 3) similarity term (optional)
        if self.use_sim:
            # cos(z_cur, z_g_i) per goal: [B,N]
            sim = torch.einsum("bd,bnd->bn", z_cur_n, z_goals_n)
        else:
            sim = torch.zeros(B, N, device=z_stack.device, dtype=z_stack.dtype)

        # Combine to score & softmax to weights
        score = compat_term + self.eta * prior_term + self.sim_scale * sim  # [B,N]
        w = F.softmax(self.beta * score, dim=-1)                            # [B,N]

        # Weighted sum of goal embeddings
        z_mix = (w.unsqueeze(-1) * z_goals_n).sum(dim=1)                    # [B,D]
        if self.normalize_output:
            z_mix = F.normalize(z_mix, dim=-1)

        aux: Dict[str, torch.Tensor] = {
            "score": score.detach(),
            "prior": prior_term.detach(),
            "compat_term": compat_term.detach(),
            "sim": sim.detach(),
            "weights": w.detach(),
        }
        return z_mix, w, aux


# --- Functional wrapper for quick use (kept for convenience) ---
def mix_goals(
    z_cur: torch.Tensor,
    z_goals: Union[List[torch.Tensor], torch.Tensor],
    q_prior: torch.Tensor,
    compat: Optional[torch.Tensor] = None,
    beta: float = 6.0,
    eta: float = 1.0,
    use_compat: bool = False,
    use_sim: bool = False,
    sim_scale: float = 1.0,
    normalize_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mixer = GoalMixer(beta=beta, eta=eta, use_compat=use_compat,
                      use_sim=use_sim, sim_scale=sim_scale,
                      normalize_output=normalize_output)
    z_mix, w, _ = mixer(z_cur, z_goals, q_prior, compat)
    return z_mix, w
