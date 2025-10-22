# pcdp/dataset/goal_target_miner.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EpisodeTargets:
    """
    에피소드 단위의 타깃 번들.
    - z_g_* : 각 스테이지의 '성공 직후' 목표 잠재벡터 [D] (L2 정규화됨)
    - prior_soft : [T, 3] 프레임별 prior 타깃 (grab/place/return)
    - stage_ranges : 각 스테이지의 연속 구간 인덱스 (디버깅용)
    """
    z_g_grab: torch.Tensor      # [D]
    z_g_place: torch.Tensor     # [D]
    z_g_return: torch.Tensor    # [D]
    prior_soft: torch.Tensor    # [T, 3]
    stage_ranges: Dict[int, List[Tuple[int, int]]]  # {0:[(s,e),...], 1:[...], 2:[...]}


# ---------- 유틸 ----------

def _l2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim)

def _as_BND(z_seq: torch.Tensor) -> torch.Tensor:
    """
    z_seq: [T, D] or [1, T, D] -> [T, D]
    """
    if z_seq.dim() == 3:
        assert z_seq.size(0) == 1, f"Expect batch=1 for episode mining, got {z_seq.size(0)}"
        return z_seq[0]
    if z_seq.dim() != 2:
        raise ValueError(f"z_seq must be [T,D] or [1,T,D], got {tuple(z_seq.shape)}")
    return z_seq


def find_stage_ranges(stage_ids: torch.Tensor, num_classes: int = 3) -> Dict[int, List[Tuple[int, int]]]:
    """
    stage_ids: [T] (int, 0/1/2 ...)
    연속 구간을 (start, end) 포맷으로 반환. end는 포함 인덱스.
    예: [0,0,0,1,1,2,2] -> {0:[(0,2)], 1:[(3,4)], 2:[(5,6)]}
    """
    if stage_ids.dim() != 1:
        raise ValueError(f"stage_ids must be [T], got {tuple(stage_ids.shape)}")
    T = stage_ids.numel()
    out: Dict[int, List[Tuple[int, int]]] = {k: [] for k in range(num_classes)}
    if T == 0:
        return out

    s = int(stage_ids[0].item())
    start = 0
    for t in range(1, T):
        cur = int(stage_ids[t].item())
        if cur != s:
            out[s].append((start, t - 1))
            s = cur
            start = t
    out[s].append((start, T - 1))
    return out


def _avg_l2(z_seq: torch.Tensor, s: int, e: int) -> torch.Tensor:
    """
    [s, e] (포함) 범위의 프레임을 L2 정규화 후 평균 -> 다시 L2 정규화하여 [D] 반환.
    """
    chunk = z_seq[s : e + 1]            # [K, D]
    chunk = _l2(chunk, dim=-1)          # per-frame
    z = chunk.mean(dim=0)               # [D]
    return _l2(z, dim=-1)


def mine_goal_targets_from_ranges(
    z_seq: torch.Tensor,              # [T, D]
    stage_ranges: Dict[int, List[Tuple[int, int]]],
    window: int = 3,                  # 각 스테이지의 '마지막 window' 평균
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    각 스테이지의 마지막 구간 끝부분(window 프레임)을 모아 타깃 z_g_*을 만든다.
    - 키보드 전환 직전 정지 구간이 존재하므로, 끝에서 window개 프레임 평균이 튼튼함.
    - 구간 길이가 window보다 짧으면 전체 구간 사용.
    """
    z_seq = _as_BND(z_seq)            # -> [T, D]
    T, D = z_seq.shape

    def last_window_avg(ranges: List[Tuple[int, int]]) -> torch.Tensor:
        if len(ranges) == 0:
            # 스테이지가 없다면 빈 텐서 반환(호출측에서 대체)
            return torch.empty(0, device=z_seq.device, dtype=z_seq.dtype)
        s, e = ranges[-1]
        k = max(1, min(window, e - s + 1))
        return _avg_l2(z_seq, e - k + 1, e)

    z_g0 = last_window_avg(stage_ranges.get(0, []))  # grab
    z_g1 = last_window_avg(stage_ranges.get(1, []))  # place
    z_g2 = last_window_avg(stage_ranges.get(2, []))  # return

    return z_g0, z_g1, z_g2


def make_prior_soft_labels(
    stage_ids: torch.Tensor,     # [T] int (0/1/2)
    num_classes: int = 3,
    label_smoothing: float = 0.1,
    edge_width: int = 2,         # 전환 경계에서 양쪽 클래스를 선형 혼합하는 프레임 폭
) -> torch.Tensor:
    """
    3-way soft 라벨 생성:
    - 기본은 one-hot + label smoothing
    - 전환 경계에서 edge_width 프레임 동안 양쪽 클래스를 선형 혼합하여 완충
      (예: ... 0 0 (0->1 경계) 1 1 ... 인 경우, 경계 양쪽 k=1..edge_width에 대해
       [α, 1-α, 0] 식으로 부드럽게 변화)
    반환: [T, 3]
    """
    if stage_ids.dim() != 1:
        raise ValueError(f"stage_ids must be [T], got {tuple(stage_ids.shape)}")
    T = stage_ids.numel()
    y = torch.zeros(T, num_classes, dtype=torch.float32, device=stage_ids.device)

    # 기본 one-hot
    y.scatter_(1, stage_ids.view(-1, 1).long(), 1.0)

    # 경계 부드럽게
    # 경계 인덱스: t-1 != t
    for t in range(1, T):
        a = int(stage_ids[t - 1].item())
        b = int(stage_ids[t].item())
        if a == b:
            continue
        # 왼쪽(이전 클래스 a) 측 완충
        for k in range(1, edge_width + 1):
            idx = t - k
            if idx < 0 or stage_ids[idx] != a:  # 같은 구간 안에서만
                break
            alpha = (edge_width - k + 1) / (edge_width + 1)  # 1에 가까울수록 a 비중↑
            y[idx, :] = 0.0
            y[idx, a] = alpha
            y[idx, b] = 1.0 - alpha
        # 오른쪽(다음 클래스 b) 측 완충
        for k in range(0, edge_width):
            idx = t + k
            if idx >= T or stage_ids[idx] != b:
                break
            alpha = (edge_width - k) / (edge_width + 1)      # 1에 가까울수록 b 비중↑
            y[idx, :] = 0.0
            y[idx, a] = 1.0 - alpha
            y[idx, b] = alpha

    # label smoothing 적용
    if label_smoothing > 0:
        eps = float(label_smoothing)
        y = y * (1.0 - eps) + eps / num_classes

    return y


# ---------- 메인 엔트리 ----------

def mine_episode_targets(
    z_seq: torch.Tensor,           # [T,D] or [1,T,D]
    stage_ids: torch.Tensor,       # [T] int (0/1/2)
    window: int = 3,
    label_smoothing: float = 0.1,
    edge_width: int = 2,
) -> EpisodeTargets:
    """
    한 에피소드에 대해:
      1) stage 연속 구간을 파싱
      2) 각 stage의 '마지막 window 프레임 평균'으로 z_g_* 생성 (L2 정규화)
      3) prior 학습용 soft 타깃 [T,3] 생성(경계 완충 + smoothing)
    """
    z_seq = _as_BND(z_seq)  # -> [T,D]
    stage_ids = stage_ids.view(-1).to(torch.long)

    stage_ranges = find_stage_ranges(stage_ids, num_classes=3)
    z_g0, z_g1, z_g2 = mine_goal_targets_from_ranges(z_seq, stage_ranges, window=window)

    # 반환 안전장치: 누락된 스테이지가 있으면 마지막 프레임 기반으로 대체
    D = z_seq.size(1)
    if z_g0.numel() == 0: z_g0 = _avg_l2(z_seq, 0, min(z_seq.size(0)-1, window-1))
    if z_g1.numel() == 0: z_g1 = _avg_l2(z_seq, 0, min(z_seq.size(0)-1, window-1))
    if z_g2.numel() == 0: z_g2 = _avg_l2(z_seq, max(0, z_seq.size(0)-window), z_seq.size(0)-1)

    prior_soft = make_prior_soft_labels(
        stage_ids=stage_ids,
        num_classes=3,
        label_smoothing=label_smoothing,
        edge_width=edge_width,
    )

    return EpisodeTargets(
        z_g_grab=z_g0, z_g_place=z_g1, z_g_return=z_g2,
        prior_soft=prior_soft,
        stage_ranges=stage_ranges,
    )
