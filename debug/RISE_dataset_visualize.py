"""
    Dataset class를 통해 불러온
    batch 데이터 검증용 코드 (RISE version)

"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch, open3d as o3d, hydra
import pytorch3d.ops as torch3d_ops
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pcdp.dataset.RISE_stack_pc_dataset import collate_fn # RISE collate_fn
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor


camera_to_base = np.array([
                [ 0.0,        -0.9063,      0.4226,    0.110],
                [ -1.0,        0.,          0.,          0.],
                [0.0,          -0.4226,      -0.9063,     0.510       ],
                [ 0.,          0.,          0.,          1.         ]
            ])

workspace_bounds = np.array([
    [0.100, 0.800],    # X range (milli meters)
    [-0.400, 0.400],    # Y range (milli meters)
    [-0.100, 0.350]     # Z range (milli meters)
])

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ────────────────────────────────────────────────────────────
# 설정 – 프로젝트에 맞게 YAML 경로/이름만 바꿔 주세요
CONFIG_DIR  = "../pcdp/config"    # your yaml folder
CONFIG_NAME = "train_diffusion_RISE_workspace"
BATCH_SIZE_VIS = 10       # cfg.dataloader.batch_size 와 동일하게
# ────────────────────────────────────────────────────────────

def main():
    preprocess = PointCloudPreprocessor(camera_to_base,
                                        workspace_bounds,
                                        enable_sampling=False)
    # 1) YAML 로드 & Dataset 인스턴스
    with hydra.initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = hydra.compose(config_name=CONFIG_NAME)

    ds = hydra.utils.instantiate(cfg.task.dataset, use_cache=True)
    print(f"[INFO] dataset sequences: {len(ds)}")

    # 3) DataLoader (시각화용)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_VIS,
                    num_workers=10, shuffle=False, collate_fn=collate_fn)
    batch_iter = iter(dl)
    batch = None
    sample_idx = 0
    
    # 4) Open3D 시각화 창
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("RISE Dataset inspector", 1280, 720, visible=True)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0, 0, 0])
    pcd = o3d.geometry.PointCloud()
    
    first_loaded = False

    def load_next_sample():
        nonlocal batch, sample_idx, first_loaded
        try:
            if batch is None or sample_idx >= BATCH_SIZE_VIS:
                print("\nLoading next batch...")
                batch = next(batch_iter)
                sample_idx = 0

            # collated batch에서 현재 sample_idx에 해당하는 포인트들 추출
            coords = batch['input_coords_list']
            feats = batch['input_feats_list']
            
            # coords의 첫번째 열이 batch index임
            point_indices = (coords[:, 0] == sample_idx)
            
            if not torch.any(point_indices):
                print(f"[WARN] No points found for sample_idx {sample_idx} in this batch. Skipping.")
                sample_idx += 1
                if sample_idx >= BATCH_SIZE_VIS:
                    load_next_sample() # 다음 배치 시도
                return

            pc = feats[point_indices].numpy()
            
            print(f"\n[Sample {sample_idx}] Original point count: {len(pc)}")

            # feats: [x, y, z, r, g, b, ...]
            xyz, rgb = pc[:, :3], pc[:, 3:6]

            # 색상 정규화 해제 (dataset에서 (x - mean) / std 했으므로)
            # RISE_stack_pc_dataset.py 참조
            IMG_MEAN = np.array([0.0217, 0.0217, 0.0217])
            IMG_STD = np.array([0.1474, 0.1474, 0.1474])
            rgb = rgb * IMG_STD + IMG_MEAN
            rgb = np.clip(rgb, 0, 1)

            
            sampled_xyz = xyz
            sampled_rgb = rgb

            pcd.points = o3d.utility.Vector3dVector(sampled_xyz.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(sampled_rgb.astype(np.float32))
            
            if not first_loaded:
                vis.add_geometry(pcd, reset_bounding_box=True)
                # 카메라 시점 설정
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                if not bbox.is_empty():
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.5)
                first_loaded = True
            else:
                vis.update_geometry(pcd)
            
            vis.poll_events()
            vis.update_renderer()

            sample_idx += 1

        except StopIteration:
            print("✔ 모든 샘플 확인 완료!  Q 로 창을 닫아 종료하세요.")
            # 루프를 멈추기 위해 아무것도 하지 않음
            pass

    # 첫 샘플 로드
    load_next_sample()
    
    # 콜백 등록
    vis.register_key_callback(ord("N"), lambda v: load_next_sample())
    vis.register_key_callback(ord("Q"), lambda v: v.close())

    print("\n[N] → 다음 샘플 | [Q] → 종료")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
