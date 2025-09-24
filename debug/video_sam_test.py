import numpy as np
import cv2

def _to_hw_bool_mask(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3 and m.shape[0] == 1:     # (1,H,W)
        m = m[0]
    if m.ndim == 3 and m.shape[-1] == 1:    # (H,W,1)
        m = m[..., 0]
    if m.ndim != 2:
        # (W,H)일 수도 있으니 전치 시도, 실패하면 리사이즈
        if m.shape == (W, H):
            m = m.T
        else:
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            return m.astype(bool)
    if m.shape != (H, W):
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        return m.astype(bool)
    return m.astype(bool)

def compute_pct_from_segments(
    segments: dict,           # {frame_idx: {obj_id: mask_bool(H,W)}}
    H: int,
    W: int,
    num_frames: int,
    robot_obj_ids: list[int] | None = None  # None이면 해당 프레임의 모든 mask 합집합
) -> np.ndarray:
    """프레임별 로봇 마스크 비율[pct]을 길이 num_frames의 float32 배열로 반환."""
    tot = H * W
    pcts = np.zeros((num_frames,), dtype=np.float32)
    for fi in range(num_frames):
        mset = segments.get(fi, {})
        if not mset:
            continue
        union = np.zeros((H, W), dtype=bool)
        if robot_obj_ids is None:
            for _, m in mset.items():
                union |= _to_hw_bool_mask(m, H, W)
        else:
            for rid in robot_obj_ids:
                if rid in mset:
                    union |= _to_hw_bool_mask(mset[rid], H, W)
        pcts[fi] = (union.sum() / tot) if tot > 0 else 0.0
    return pcts


from pathlib import Path
import zarr, numcodecs
import numpy as np

def write_occlusion_to_episode_zarr(
    episode_id: int,
    pcts: np.ndarray,
    recorder_root: str = "/home/nscl/diffusion_policy/data/recorder_data",
    dataset_name: str = "occlusion",
    check_with_key: str = "pointcloud",   # 길이 정합 확인용 기준 키
    overwrite: bool = True
):
    ep_dir = Path(recorder_root) / f"episode_{episode_id:04d}"
    zarr_path = ep_dir / "obs_replay_buffer.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr not found: {zarr_path}")

    root = zarr.open(str(zarr_path), mode='a')
    g = root["data"]

    # Zarr 길이와 정합
    n_steps = int(g[check_with_key].shape[0]) if check_with_key in g else len(pcts)
    arr = np.asarray(pcts, dtype=np.float32)
    if len(arr) != n_steps:
        # 보수적으로 trim/pad
        out = np.zeros((n_steps,), dtype=np.float32)
        m = min(n_steps, len(arr))
        out[:m] = arr[:m]
        arr = out

    compressor = numcodecs.Blosc(cname='zstd', clevel=2, shuffle=numcodecs.Blosc.BITSHUFFLE)
    chunks = (min(2048, len(arr)),)
    g.create_dataset(dataset_name, data=arr, chunks=chunks, compressor=compressor, overwrite=overwrite)

    # 간단 메타데이터
    g[dataset_name].attrs.put({
        "desc": "Per-frame occlusion ratio (robot_mask / total_pixels) from SAM2",
        "source": "SAM2-propagate",
    })
    print(f"[OK] {dataset_name} saved to {zarr_path}/data ({len(arr)} steps)")
