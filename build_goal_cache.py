# build_goal_cache.py
# -*- coding: utf-8 -*-
"""
원본 please/episode_*/obs_replay_buffer.zarr 에
  - goal_raw/z_seq        : [T_raw, D]  (fp16, L2-normalized)
  - goal_raw/prior_full   : [T_raw, 3]  (경계 램프 + 라벨 스무딩)
  - goal_raw/z_g3         : [3, D]      (각 stage '성공 직후' window 평균, L2)
를 '추가' 저장하는 오프라인 스크립트.

입력 (딱 2개):
  --encoder_ckpt  : 학습된 Sparse3DEncoder ckpt 경로
  --dataset_root  : please 루트 (episode_* 하위 포함)

주의:
  - 여기서는 downsample_factor=1 (프레임 삭제 X, 정렬만 O)
  - 캐시(학습용) 생성 단계에서 이 값을 다운샘플 타임라인에 맞춰 리샘플/인덱싱해서 사용
  - 전처리(좌표계/C2D/crop/voxel/token-limit)와 Encoder 스냅샷은 학습·런타임과 동일해야 함
"""

import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import zarr
import MinkowskiEngine as ME

# 프로젝트 파일 (경로가 다르면 import 경로만 바꿔줘)
from pcdp.real_world.real_data_pc_conversion import process_single_episode, PointCloudPreprocessor, LowDimPreprocessor
from pcdp.model.vision.Sparse3DEncoder import Sparse3DEncoder
from termcolor import cprint


# ===== 하이퍼(필요하면 여기만 수정) =====
VOXEL_SIZE      = 0.005     # 학습 전처리와 동일
MAX_NUM_TOKEN   = 100       # Sparse3DEncoder forward arg (학습과 동일)
GOAL_WINDOW     = 3         # 각 stage '성공 직후' 윈도 길이
EDGE_WIDTH      = 2         # prior 경계 램프 폭(프레임)
LABEL_SMOOTH    = 0.1       # prior 라벨 스무딩
BATCH_PC        = 1800        # 포인트클라우드 인코딩 배치 크기
DEVICE_DEFAULT  = "cuda"    # 없으면 자동 CPU


# ===== 유틸 =====
@torch.inference_mode()
def _masked_mean_tokens(tokens: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    """ tokens[B,Ttok,D], pad_mask[B,Ttok](True=pad) -> z[B,D] (L2) """
    valid = (~pad_mask).float()
    denom = valid.sum(1, keepdim=True).clamp_min(1.0)
    z = (tokens * valid.unsqueeze(-1)).sum(1) / denom
    return F.normalize(z, dim=-1)

def _encoder_fingerprint(model: torch.nn.Module) -> str:
    """가벼운 지문: 파라미터 shape+mean 기반 16자 요약"""
    h = 0
    with torch.no_grad():
        for p in model.parameters():
            h ^= hash((tuple(p.shape), float(p.detach().cpu().float().mean())))
    return hex((h & ((1<<64)-1)))[2:]

def _make_prior_soft(stage_ids: np.ndarray,
                     edge_width: int = EDGE_WIDTH,
                     label_smooth: float = LABEL_SMOOTH) -> np.ndarray:
    """ stage_ids[T] -> prior[T,3] (경계 램프 + 라벨 스무딩) """
    stage_ids = stage_ids.astype(np.int64)
    T, K = stage_ids.shape[0], 3
    prior = np.zeros((T, K), dtype=np.float32)
    prior[np.arange(T), stage_ids] = 1.0
    # 경계 램프
    for b in range(1, T):
        sp, sn = int(stage_ids[b-1]), int(stage_ids[b])
        if sp == sn:
            continue
        L, R = max(0, b - edge_width), min(T, b + edge_width)
        if R - L <= 1:
            continue
        for t in range(L, R):
            a = (t - L) / max(1, (R - L - 1))  # 0..1
            mix = np.zeros(K, dtype=np.float32); mix[sp] = 1.0 - a; mix[sn] = a
            prior[t] = mix
    # 라벨 스무딩
    eps = float(label_smooth)
    prior = prior * (1.0 - eps) + eps / K
    return prior

def _compute_zg3(z_seq: torch.Tensor, stage_ids: np.ndarray, window: int = GOAL_WINDOW) -> np.ndarray:
    """ z_seq[T,D](L2) + stage_ids[T] -> z_g3[3,D] (각 stage 마지막 window 평균, L2) """
    sid_t = torch.from_numpy(stage_ids.astype(np.int64))
    T, D = z_seq.shape
    out = torch.zeros(3, D, dtype=z_seq.dtype, device=z_seq.device)
    for s in range(3):
        idx = torch.nonzero(sid_t == s).flatten()
        if idx.numel() == 0:
            out[s] = F.normalize(z_seq.mean(0, keepdim=True), dim=-1).squeeze(0)
            continue
        end = int(idx.max().item())
        st  = max(0, end - window + 1)
        z   = F.normalize(z_seq[st:end+1], dim=-1)
        out[s] = F.normalize(z.mean(0, keepdim=True), dim=-1).squeeze(0)
    return out.cpu().numpy()


# ===== 인코딩 경로 =====
def _build_sparse_batch(clouds: List[np.ndarray], voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    clouds: list of (N_i,6) [x,y,z,r,g,b] per frame
    -> (coords_batch, feats_batch) for MinkowskiEngine SparseTensor
    """
    coords_list, feats_list = [], []
    for pc in clouds:
        coords = np.ascontiguousarray(pc[:, :3] / voxel_size, dtype=np.int32)  # grid index
        feats  = pc.astype(np.float32)                                         # use XYZRGB as features
        coords_list.append(coords)
        feats_list.append(feats)
    coords_b, feats_b = ME.utils.sparse_collate(coords=coords_list, feats=feats_list)
    coords_b = coords_b.int()
    feats_b = feats_b.float()
    return coords_b, feats_b

@torch.inference_mode()
def _encode_pc_list(encoder: Sparse3DEncoder,
                    pc_list: List[np.ndarray],
                    voxel_size: float,
                    max_num_token: int,
                    device: torch.device) -> np.ndarray:
    """ pc_list(list of Nx6) → z_seq[T,D] (float32, L2) """
    z_all = []
    for s in range(0, len(pc_list), BATCH_PC):
        chunk = pc_list[s:s+BATCH_PC]
        coords_b, feats_b = _build_sparse_batch(chunk, voxel_size)
        sinput = ME.SparseTensor(features=feats_b, coordinates=coords_b, device=device)
        cprint(f"[ME] sinput.F.device={sinput.F.device}, coords on CPU={sinput.C.device}", "cyan", attrs=["bold"])
        # Sparse3DEncoder.forward 시그니처 대응
        out = encoder(sinput, max_num_token=max_num_token, batch_size=len(chunk))
        if isinstance(out, (tuple, list)):
            # (tokens, pos?, pad_mask) 형태 가정
            tokens = out[0]
            pad    = out[-1]
            z = _masked_mean_tokens(tokens, pad)   # [B,D]
        else:
            # 이미 [B,D] 임베딩을 돌려주는 경우
            z = F.normalize(out, dim=-1)           # [B,D]
        z_all.append(z.cpu())
    z_seq = torch.cat(z_all, dim=0)  # [T,D]
    return z_seq.numpy()


# ===== 저장 =====
def _save_goal_raw(obs_zarr_path: Path,
                   z_seq_fp16: np.ndarray,
                   prior_full: np.ndarray,
                   z_g3: np.ndarray,
                   meta: Dict[str, Any]):
    g = zarr.open_group(str(obs_zarr_path), mode="a")
    grp = g.require_group("goal_raw")
    # overwrite-safe
    for key, arr in [("z_seq", z_seq_fp16), ("prior_full", prior_full), ("z_g3", z_g3)]:
        if key in grp:
            del grp[key]
        grp.create_dataset(key, data=arr, chunks=True, overwrite=True)
    for k, v in meta.items():
        if isinstance(v, (np.generic,)):
            v = v.item()
        grp.attrs[k] = v


def extract_state_dict_like(obj):
    # (a) 흔한 케이스: 'state_dict'
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        # (b) Lightning이나 커스텀: 'model', 'net', 'encoder' 등
        for k in ["model", "network", "net", "encoder", "backbone", "cloud_encoder"]:
            if k in obj and isinstance(obj[k], dict):
                # 이 안에 바로 파라미터 키들이 있을 수도 있고
                # 다시 'state_dict'가 있을 수도 있음
                inner = obj[k]
                if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                    return inner["state_dict"]
                return inner
        # (c) 'state_dicts' 같은 복수 저장형
        if "state_dicts" in obj and isinstance(obj["state_dicts"], dict):
            # 가장 큰(파라미터 개수가 많은) dict를 고름
            cand = max(
                (v for v in obj["state_dicts"].values() if isinstance(v, dict)),
                key=lambda d: sum(hasattr(x, "shape") for x in d.values()),
                default=None
            )
            if cand is not None:
                return cand
        # (d) 최후: 혹시 top-level이 곧 파라미터 dict인 경우
        tensor_like = [k for k, v in obj.items() if hasattr(v, "shape")]
        if len(tensor_like) > 10:  # 파라미터처럼 보이면 그대로 리턴
            return obj
    raise RuntimeError("Could not locate a parameter state_dict in the checkpoint.")


# ===== 메인 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder_ckpt", type=str, default="data/outputs/2025.10.10/PCDP_3/checkpoints/latest.ckpt" ,help="학습된 Sparse3DEncoder ckpt")
    ap.add_argument("--dataset_root", type=str, default="data/please/recorder_data" ,help="please 루트 (episode_* 하위 포함)")
    ap.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) 엔코더 로드·고정
    enc = Sparse3DEncoder()
    ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
    sd_raw = extract_state_dict_like(ckpt)
    from collections import OrderedDict
    sd = OrderedDict()
    for k, v in sd_raw.items():
        nk = k

        # DataParallel
        if nk.startswith("module."):
            nk = nk[len("module."):]

        # <핵심> ckpt: 'sparse_encoder.cloud_encoder.*'  →  모델: 'cloud_encoder.*'
        if nk.startswith("sparse_encoder."):
            nk = nk[len("sparse_encoder."):]

        # (옵션) 혹시 다른 저장체계면 여기도 매핑 가능
        # if nk.startswith("encoder."):
        #     nk = "cloud_encoder." + nk[len("encoder."):]

        # 모델에 없는 잡키는 건너뛰기 (예: '_dummy_variable', 'pickles', ...)
        if not nk.startswith("cloud_encoder."):
            continue

    sd[nk] = v
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    print(f"[enc] remapped ckpt loaded. missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print("  (missing sample)", missing[:10])
    if unexpected:
        print("  (unexpected sample)", unexpected[:10])

    for p in enc.parameters(): p.requires_grad = False
    enc.to(device).eval()
    cprint(f"[device] torch.cuda.is_available={torch.cuda.is_available()}, using={device}", "cyan", attrs=["bold"])
    enc_fp = _encoder_fingerprint(enc)

    # 2) 에피소드 순회
    root = Path(args.dataset_root)
    ep_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    if not ep_dirs:
        raise FileNotFoundError(f"No 'episode_*' under: {root}")


    # 정렬만 수행(다운샘플 X) — 학습과 동일한 전처리 셋업 필요
    pc_prep  = PointCloudPreprocessor(enable_transform=True, enable_cropping=True, enable_sampling=False, enable_filter=False, verbose=False)
    low_prep = LowDimPreprocessor()

    for ep in ep_dirs:
        obs_zarr = ep / "obs_replay_buffer.zarr"
        act_zarr = ep / "action_replay_buffer.zarr"
        if not (obs_zarr.exists() and act_zarr.exists()):
            print(f"[skip] {ep.name}: missing zarr")
            continue

        epi = process_single_episode(
            episode_path=ep,
            pc_preprocessor=pc_prep,
            lowdim_preprocessor=low_prep,
            downsample_factor=1  # ★ 프레임 삭제 없이 '정렬'만 수행
        )
        if epi is None or "pointcloud" not in epi:
            print(f"[warn] {ep.name}: process_single_episode failed or missing 'pointcloud'")
            continue

        # stage 키 찾기(프로젝트별 키가 다를 수 있으니 fallback)
        stage = None
        for k in ["stage", "stage_ids", "stage_id", "stages"]:
            if k in epi:
                stage = np.asarray(epi[k]).astype(np.int64)
                break
        if stage is None:
            print(f"[warn] {ep.name}: no 'stage' found in episode_data. Skipped.")
            continue
        
        

        pc_list   = list(epi["pointcloud"])  # list of (N,6)
        T_raw     = len(pc_list)
        if T_raw == 0 or T_raw != stage.shape[0]:
            print(f"[warn] {ep.name}: length mismatch: pointcloud({T_raw}) vs stage({stage.shape[0]}). Skipped.")
            continue

        # 3) 인코딩 → z_seq[T_raw,D]
        z_seq_f32 = _encode_pc_list(enc, pc_list, VOXEL_SIZE, MAX_NUM_TOKEN, device)  # float32
        z_norm = np.linalg.norm(z_seq_f32, axis=1)
        cprint(f"[z] norm mean={z_norm.mean():.3f}, std={z_norm.std():.3f} (expect ~1.0)", "green", attrs=["bold"])
        # 4) 타깃 생성
        prior_full = _make_prior_soft(stage, EDGE_WIDTH, LABEL_SMOOTH)                # [T_raw,3]
        s = prior_full.sum(axis=1)
        cprint(f"[prior] min={prior_full.min():.3f}, max={prior_full.max():.3f}, sum≈1 mean={s.mean():.3f}", "green", attrs=["bold"])
        z_g3       = _compute_zg3(torch.from_numpy(z_seq_f32), stage, GOAL_WINDOW)    # [3,D]
        
        
        sid = torch.from_numpy(stage)
        T = len(z_seq_f32); D = z_seq_f32.shape[1]
        z_t = torch.from_numpy(z_seq_f32).float()
        for s in range(3):
            idx = torch.nonzero(sid == s).flatten()
            if idx.numel()==0: continue
            end = int(idx.max().item())
            st  = max(0, end - GOAL_WINDOW + 1)
            win = F.normalize(z_t[st:end+1], dim=-1)
            ref = F.normalize(win.mean(0, keepdim=True), dim=-1)     # 윈도 평균
            zg  = torch.from_numpy(z_g3[s]).float().unsqueeze(0)
            cos = float(F.cosine_similarity(ref, F.normalize(zg, dim=-1)).item())
            cprint(f"[zg] stage{s} cos(mean(last{GOAL_WINDOW}), z_g) = {cos:.3f}","green", attrs=["bold"])
            

        # 5) 저장 (메모리/용량 절감 위해 z_seq는 fp16로)
        z_seq_fp16 = z_seq_f32.astype(np.float16)
        meta = dict(
            encoder_fingerprint=enc_fp,
            z_dim=int(z_seq_f32.shape[-1]),
            goal_window=int(GOAL_WINDOW),
            edge_width=int(EDGE_WIDTH),
            label_smoothing=float(LABEL_SMOOTH),
            voxel_size=float(VOXEL_SIZE),
            max_num_token=int(MAX_NUM_TOKEN),
            timeline_length_raw=int(T_raw),
            note="goal_raw/* are BEFORE any dataset downsampling; resample/index at cache build."
        )
        _save_goal_raw(obs_zarr, z_seq_fp16, prior_full, z_g3, meta)
        print(f"[ok] {ep.name}: goal_raw/z_seq({z_seq_fp16.shape}, fp16), prior_full({prior_full.shape}), z_g3({z_g3.shape}) saved")

        

    print("[done] all episodes processed")


if __name__ == "__main__":
    main()

"""
  --encoder_ckpt  : 학습된 Sparse3DEncoder ckpt 경로
  --dataset_root  : please 루트 (episode_* 하위 포함)
"""