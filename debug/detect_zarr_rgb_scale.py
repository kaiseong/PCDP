#!/usr/bin/env python3
import os, zarr, argparse, numpy as np
from tqdm import tqdm

def classify_rgb(arr, assume_uint8_if_max_gt=1.5):
    """Classify an RGB array by scale: 'RAW_255', 'UNIT_0_1', or 'ZSCORE_LIKE'."""
    if arr.size == 0:
        return "EMPTY", {"min": None, "max": None, "mean": None, "std": None}
    a = arr.astype(np.float64, copy=False)
    a = a[np.isfinite(a).all(axis=1)]
    if a.size == 0:
        return "EMPTY", {"min": None, "max": None, "mean": None, "std": None}
    mn, mx = float(np.min(a)), float(np.max(a))
    mean = a.mean(axis=0).tolist()
    std  = a.std(axis=0).tolist()
    if mx > assume_uint8_if_max_gt:
        cls = "RAW_255"
    else:
        # Heuristic: if z-score-like (mean near 0, std near 1) within a tolerance
        mean_abs = np.mean(np.abs(a.mean(axis=0)))
        std_mean = float(np.mean(a.std(axis=0)))
        if mean_abs < 0.2 and 0.6 <= std_mean <= 1.6:
            cls = "ZSCORE_LIKE"
        else:
            cls = "UNIT_0_1"
    return cls, {"min": mn, "max": mx, "mean": mean, "std": std}

def scan_zarr(path, max_frames=50):
    """Scan a .zarr.zip archive and classify its RGB scale by sampling up to max_frames frames."""
    results = []
    with zarr.ZipStore(path, mode='r') as store:
        g = zarr.group(store=store)
        if 'data' not in g or 'meta' not in g:
            return {"file": path, "error": "Missing 'data' or 'meta' group."}
        data_group, meta_group = g['data'], g['meta']
        if 'pointcloud' not in data_group or 'episode_ends' not in meta_group:
            return {"file": path, "error": "Missing 'data/pointcloud' or 'meta/episode_ends'."}
        pcs = data_group['pointcloud']
        episode_ends = meta_group['episode_ends'][:]
        n_ep = g.attrs.get('n_episodes', len(episode_ends))
        start = 0
        frames_checked = 0
        classes = []
        stats_sample = []
        for i in range(min(n_ep, len(episode_ends))):
            end = int(episode_ends[i])
            for frame in pcs[start:end]:
                if isinstance(frame, np.ndarray) and frame.ndim == 2 and frame.shape[1] >= 6:
                    rgb = frame[:, 3:6]
                    cls, stat = classify_rgb(rgb)
                    classes.append(cls)
                    stats_sample.append(stat)
                    frames_checked += 1
                    if frames_checked >= max_frames:
                        break
            start = end
            if frames_checked >= max_frames:
                break
        if frames_checked == 0:
            return {"file": path, "error": "No valid frames with >=6 columns found."}
        # Majority vote of classes excluding EMPTY
        classes = [c for c in classes if c != "EMPTY"]
        if not classes:
            final = "EMPTY"
        else:
            final = max(set(classes), key=classes.count)
        # Aggregate min/max over sample for quick view
        mins = [s["min"] for s in stats_sample if s["min"] is not None]
        maxs = [s["max"] for s in stats_sample if s["max"] is not None]
        return {
            "file": path,
            "frames_checked": frames_checked,
            "class": final,
            "global_min": float(np.min(mins)) if mins else None,
            "global_max": float(np.max(maxs)) if maxs else None,
            "note": "RAW_255 -> divide by 255 before normalization; UNIT_0_1 -> use as-is; ZSCORE_LIKE -> likely already normalized."
        }

def main():
    ap = argparse.ArgumentParser(description="Detect RGB scale of Zarr archives (RAW_255 / UNIT_0_1 / ZSCORE_LIKE).")
    ap.add_argument("--dataset_path", required=True, help="Directory containing .zarr.zip archives.")
    ap.add_argument("--max_frames", type=int, default=50, help="Max frames to sample per archive.")
    args = ap.parse_args()
    files = [os.path.join(args.dataset_path, f) for f in os.listdir(args.dataset_path) if f.endswith(".zarr.zip")]
    if not files:
        print("No .zarr.zip files found in", args.dataset_path)
        return
    summaries = []
    for f in tqdm(files, desc="Scanning archives"):
        try:
            summaries.append(scan_zarr(f, args.max_frames))
        except Exception as e:
            summaries.append({"file": f, "error": str(e)})
    # Pretty print
    for s in summaries:
        print("\\n" + "="*80)
        print("File:", s.get("file"))
        if "error" in s:
            print("ERROR:", s["error"])
            continue
        print("Frames checked:", s["frames_checked"])
        print("Class:", s["class"])
        print("Global RGB min/max:", s["global_min"], s["global_max"])
        print("Hint:", s["note"])
    # Also dump a JSON alongside for programmatic use
    out_json = os.path.join(args.dataset_path, "rgb_scale_report.json")
    try:
        import json
        with open(out_json, "w") as f:
            json.dump(summaries, f, indent=2)
        print("\\nSaved report to:", out_json)
    except Exception:
        pass

if __name__ == "__main__":
    from tqdm import tqdm  # local import to avoid CLI missing tqdm if not installed
    main()
