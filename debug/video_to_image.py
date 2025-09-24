#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import click
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

# SAM2 (공식 패키지 구조 가정)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ----------------------------
# 유틸: SAM2 config 경로/이름 자동 정규화
# ----------------------------
def normalize_sam2_cfg(cfg: str) -> str:
    """
    사용자가 .yaml 절대/상대경로를 주든, hydra config 이름을 주든 모두 허용.
    - 경로가 실제 존재하고 .yaml이면 → 'parent_dir_name/file_stem' 으로 변환
    - 그 외는 그대로 반환 (이미 config name이라고 간주)
    """
    p = Path(cfg)
    if p.suffix == ".yaml" and p.exists():
        # ex) .../sam2/configs/sam2.1/sam2.1_hiera_l.yaml -> sam2.1/sam2.1_hiera_l
        return f"{p.parent.name}/{p.stem}"
    return cfg


def _cuda_amp_dtype():
    # bf16 지원되면 bf16, 아니면 fp16
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return None


# ----------------------------
# SAM 백엔드 래퍼
# ----------------------------
class SegBackend:
    def __init__(self,
                 device: str = "cuda",
                 sam2_cfg: str | None = None,
                 sam2_ckpt: str | None = None,
                 sam1_ckpt: str | None = None,
                 sam1_type: str = "vit_h"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.impl = None  # ("sam2", predictor) or ("sam1_amg", amg)

        # 1) SAM2 시도
        if sam2_cfg and sam2_ckpt:
            try:
                cfg_name = normalize_sam2_cfg(sam2_cfg)
                print(f"[INFO] Building SAM2 predictor: cfg={cfg_name}, ckpt={sam2_ckpt}")
                model = build_sam2(cfg_name, sam2_ckpt)
                model.to(self.device)
                model.eval()
                predictor = SAM2ImagePredictor(model)
                self.impl = ("sam2", predictor)
                print("[INFO] SAM2 ready.")
            except Exception as e:
                print(f"[WARN] SAM2 load failed: {e}")

        # 2) 폴백: SAM v1 AMG
        if self.impl is None:
            try:
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                assert sam1_ckpt is not None, "SAM1 checkpoint가 없어 폴백 불가합니다."
                print(f"[INFO] Fallback to SAM v1 AMG: type={sam1_type}")
                sam = sam_model_registry[sam1_type](checkpoint=sam1_ckpt)
                sam.to(device=self.device)
                amg = SamAutomaticMaskGenerator(
                    sam,
                    points_per_side=32,
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.92,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                )
                self.impl = ("sam1_amg", amg)
                print("[INFO] SAM v1 AMG ready.")
            except Exception as e:
                raise RuntimeError(
                    f"No usable segmentation backend. "
                    f"Provide valid SAM2 cfg/ckpt or SAM1 ckpt. Error: {e}"
                )

    @torch.no_grad()
    def segment_best(self, rgb: np.ndarray) -> np.ndarray:
        """
        한 장의 RGB 프레임에 대해 최고 점수 마스크 1장을 uint8(0/255)로 반환.
        """
        kind, obj = self.impl
        H, W = rgb.shape[:2]

        if kind == "sam2":
            predictor: SAM2ImagePredictor = obj
            amp_dtype = _cuda_amp_dtype()
            # 풀 프레임 박스 프롬프트
            full_box = np.array([[0, 0, W - 1, H - 1]], dtype=np.float32)

            # CUDA면 autocast 사용
            class _Null:
                def __enter__(self): return None
                def __exit__(self, *a): return False

            amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if (amp_dtype is not None) else _Null()
            with torch.inference_mode(), amp_ctx:
                predictor.set_image(rgb)  # RGB uint8
                masks, scores, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=full_box,
                    multimask_output=True,
                    return_logits=False,
                )

            # 정규화
            if isinstance(masks, torch.Tensor):
                masks = masks.detach().cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()

            if masks.ndim == 2:
                best = masks
            elif masks.ndim == 3 and scores is not None and scores.ndim == 1 and scores.shape[0] == masks.shape[0]:
                best = masks[int(np.argmax(scores))]
            elif masks.ndim == 3:
                best = masks[0]
            else:
                best = np.zeros((H, W), dtype=np.uint8)

            best = (best > 0.5).astype(np.uint8) * 255
            return best

        elif kind == "sam1_amg":
            amg = obj
            masks = amg.generate(rgb)  # list of dicts
            if not masks:
                return np.zeros((H, W), dtype=np.uint8)
            # 최고 iou 점수 선택
            j = int(np.argmax([m.get("predicted_iou", 0.0) for m in masks]))
            seg = masks[j]["segmentation"]
            if seg.dtype != np.uint8:
                seg = (seg.astype(np.uint8) * 255)
            return seg

        else:
            raise RuntimeError(f"Unknown backend: {kind}")


@click.command()
@click.option('--video-path', '-v', required=True, help='Path to the input video file.')
@click.option('--output-dir', '-o', required=True, help='Directory to save the output frames and masks.')
@click.option('--sam2-config', default="/home/nscl/diffusion_policy/dependencies/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
              show_default=True, help="SAM2 hydra config name or .yaml path")
@click.option('--sam2-checkpoint', default="/home/nscl/diffusion_policy/dependencies/sam2/checkpoints/sam2.1_hiera_large.pt",
              show_default=True, help="Path to SAM2 checkpoint")
@click.option('--sam1-checkpoint', default=None, help="(Fallback) Path to SAM v1 checkpoint if SAM2 not available")
@click.option('--sam1-model-type', default="vit_h", show_default=True, help="(Fallback) SAM v1 model type")
@click.option('--device', default="cuda", show_default=True)
@click.option('--stride', default=1, type=int, show_default=True, help="Process every Nth frame")
@click.option('--max-frames', default=None, type=int, help="Limit number of processed frames")
@click.option('--no-save-frames', is_flag=True, help="Do not save raw frames")
def main(video_path, output_dir, sam2_config, sam2_checkpoint, sam1_checkpoint, sam1_model_type,
         device, stride, max_frames, no_save_frames):
    """Extract frames from a video and apply SAM2 (or fallback SAM v1) per frame."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frames_dir = output_dir / 'frames'
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    if not no_save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # 백엔드 자동 빌드 (SAM2 우선, 실패시 SAM1 폴백)
    backend = SegBackend(
        device=device,
        sam2_cfg=sam2_config,
        sam2_ckpt=sam2_checkpoint,
        sam1_ckpt=sam1_checkpoint,
        sam1_type=sam1_model_type,
    )

    print(f"[INFO] Loading video: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] frames={frame_count}, fps={fps}, stride={stride}, device={device}")

    processed = 0
    frame_idx = 0
    with tqdm(total=frame_count if frame_count > 0 else None, desc="Processing frames") as pbar:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if (frame_idx % stride) != 0:
                frame_idx += 1
                if frame_count > 0:
                    pbar.update(1)
                continue

            # BGR -> RGB
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # 세그멘테이션
            try:
                mask = backend.segment_best(rgb)
            except Exception as e:
                cap.release()
                raise RuntimeError(f"SAM inference failed at frame {frame_idx}: {e}")

            # 저장
            fname = f"{frame_idx:05d}.png"
            if not no_save_frames:
                cv2.imwrite(str(frames_dir / fname), frame_bgr)
            cv2.imwrite(str(masks_dir / fname), mask)

            processed += 1
            frame_idx += 1
            if frame_count > 0:
                pbar.update(1)
            if (max_frames is not None) and (processed >= max_frames):
                break

    cap.release()
    print(f"\n[SUCCESS] Done. frames_saved={not no_save_frames}, "
          f"frames_dir={frames_dir if not no_save_frames else 'skipped'}, "
          f"masks_dir={masks_dir}, processed={processed})")


if __name__ == '__main__':
    main()
