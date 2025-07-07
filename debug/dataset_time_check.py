"""
    Dataset class를 통해 불러온
    batch 데이터 검증용 코드

"""
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pathlib, random, math, sys, time
import numpy as np
import torch, open3d as o3d, tqdm, hydra
import csv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from diffusion_policy.common.pytorch_util import dict_apply
OmegaConf.register_new_resolver("eval", eval, replace=True)

# ────────────────────────────────────────────────────────────
# 설정 – 프로젝트에 맞게 YAML 경로/이름만 바꿔 주세요
CONFIG_DIR  = "../diffusion_policy/config"    # your yaml folder
CONFIG_NAME = "train_diffusion_unet_real_pointcloud_workspace"
SAMPLE_LIMIT = 1000       # 사전 검사 시 최대 스캔 샘플 수 (None = 전부)
BATCH_SIZE_VIS = 32       # cfg.dataloader.batch_size 와 동일하게
DEVICE_SCAN  = "cuda"      # scan 통계 계산용 디바이스(cpu/cuda)
# ────────────────────────────────────────────────────────────

save_align_timestamp = []
save_action_timestamp = []
def save_csv(path, data):
    """index, timestamp 형식으로 CSV 저장"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "align_timestamp"])
        for idx, ts in enumerate(data):
            w.writerow([idx, ts])

def main():
    # 1) YAML 로드 & Dataset 인스턴스
    with hydra.initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = hydra.compose(CONFIG_NAME)

    ds = hydra.utils.instantiate(cfg.task.dataset, use_cache=True)
    print(f"[INFO] dataset sequences: {len(ds)}")

    # 3) DataLoader (시각화용)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_VIS,
                    num_workers=0, shuffle=False)
    batch_iter = iter(dl)
    batch = None
    sample_idx = 0
    normalizer = ds.get_normalizer()
    device = torch.device(cfg.training.device)

    """
    # 4) Open3D 시각화 창
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Dataset inspector", 1280, 720, visible=True)
    opt = vis.get_render_option()
    opt.point_size = 1.0
    pcd = o3d.geometry.PointCloud()
    """
    first_loaded = False

    def load_next_sample():
        nonlocal batch, sample_idx, first_loaded
        try:
            if batch is None or sample_idx >= batch['action'].size(0):
                batch = next(batch_iter)
                batch_cpu = dict_apply(batch, lambda t: t.detach().cpu())
                batch.clear(); batch.update(batch_cpu)
                sample_idx = 0

            pts = batch['obs']['pointcloud'][sample_idx, -1].numpy()
            eef = batch['obs']['robot_eef_pose'][sample_idx].numpy()
            align_timestamp = batch['obs']['align_timestamp'][sample_idx].numpy()
            action_timestamp = batch['action_time'][sample_idx].numpy()

            align_timestamp_ = batch['obs']['align_timestamp'][sample_idx].numpy()
            action_timestamp_ = batch['action_time'][sample_idx].numpy()
            save_align_timestamp.append(align_timestamp_)
            save_action_timestamp.append(action_timestamp_)

            
            act = batch['action'][sample_idx].numpy()

            """
            xyz, rgb = pts[:, :3], pts[:, 3:]
            if rgb.max() > 1.0:
                rgb = rgb / 255.0
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))

            if not first_loaded:
                vis.add_geometry(pcd, reset_bounding_box=True)
                # 카메라 시점 설정
                ctr = vis.get_view_control()
                bbox = pcd.get_axis_aligned_bounding_box()
                if not bbox.is_empty():
                    ctr.set_lookat(bbox.get_center())
                    ctr.set_front([0.0, 0.0, -1.0])
                    ctr.set_up([0.0, -1.0, 0.0])
                    ctr.set_zoom(0.4)
                first_loaded = True
            else:
                vis.update_geometry(pcd)
            """
            print(f"\n[batch {len(dl)} idx {sample_idx}]")
            # print("robot_eef_pose:", np.round(eef, 4))
            print("align_timestamp:\n", np.round(align_timestamp, 4))
            print("action_timestamp:\n", np.round(action_timestamp, 4))
            # print("action (16×A):\n", np.round(act, 4))
            sample_idx += 1
        except StopIteration:
            with open("debugging.csv", "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["index", "timestamp"])
                for idx, ts in enumerate(save_action_timestamp):
                    w.writerow([f"align_timestamp {idx}", save_align_timestamp[idx][0], save_align_timestamp[idx][1], " ",
                                save_action_timestamp[idx][0], save_action_timestamp[idx][1], save_action_timestamp[idx][2],
                                save_action_timestamp[idx][3], save_action_timestamp[idx][4], save_action_timestamp[idx][5],
                                save_action_timestamp[idx][6], save_action_timestamp[idx][7], save_action_timestamp[idx][8],
                                save_action_timestamp[idx][9], save_action_timestamp[idx][10], save_action_timestamp[idx][11],
                                save_action_timestamp[idx][12],save_action_timestamp[idx][13], save_action_timestamp[idx][14],
                                save_action_timestamp[idx][15]])
                    


            save_csv("timestamp.csv", save_align_timestamp)
            save_csv("action_timestamp.csv", save_action_timestamp)
            print("✔ 모든 샘플 확인 완료!  Q 로 창을 닫아 종료하세요.")
            

    # 첫 샘플
    load_next_sample()

    while True:
        load_next_sample()
        time.sleep(0.001)

    """
    # 콜백 등록
    vis.register_key_callback(ord("N"), lambda v: (load_next_sample(), False)[1])
    vis.register_key_callback(ord("Q"), lambda v: (v.close(), True)[1])

    print("\n[N] → 다음 샘플 | [Q] → 종료")
    vis.run()
    vis.destroy_window()
    """


if __name__ == "__main__":
    main()