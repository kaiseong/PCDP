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
import torch, tqdm, hydra
import csv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pcdp.common.pytorch_util import dict_apply
from pcdp.dataset.RISE_stack_dataset import collate_fn # RISE collate_fn
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor


camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
    ])

workspace_bounds = np.array([
    [-0.000, 0.740],    # X range (m)
    [-0.400, 0.350],    # Y range (m)
    [-0.100, 0.400]     # Z range (m)
])

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ────────────────────────────────────────────────────────────
# 설정 – 프로젝트에 맞게 YAML 경로/이름만 바꿔 주세요
CONFIG_DIR  = "../pcdp/config"    # your yaml folder
CONFIG_NAME = "train_diffusion_RISE_workspace"
BATCH_SIZE_VIS = 32       # cfg.dataloader.batch_size 와 동일하게
# ────────────────────────────────────────────────────────────


def save_csv(path, header, data):
    """지정된 형식으로 CSV 저장"""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx, row_data in enumerate(data):
            w.writerow([idx] + row_data.tolist())

def main():
    # 1) YAML 로드 & Dataset 인스턴스
    with hydra.initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = hydra.compose(config_name=CONFIG_NAME)

    ds = hydra.utils.instantiate(cfg.task.dataset, use_cache=True)
    print(f"[INFO] dataset sequences: {len(ds)}")

    # 3) DataLoader
    dl = DataLoader(ds, batch_size=BATCH_SIZE_VIS,
                    num_workers=20, shuffle=False, collate_fn=collate_fn)
    
    batch_iter = iter(dl)
    pbar = tqdm.tqdm(total=len(dl), desc="Processing batches")
    
    save_action = []
    save_action_normalized = []

    try:
        while True:
            batch = next(batch_iter)
            
            # CPU로 데이터 이동
            batch_cpu = dict_apply(batch, lambda t: t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t)

            # 데이터 저장
            save_action.extend(batch_cpu['action'])
            save_action_normalized.extend(batch_cpu['action_normalized'])
            
            pbar.update(1)

    except StopIteration:
        pbar.close()
        print("✔ 모든 샘플 확인 완료!")

        # CSV 파일로 저장
        # action: (x,y,z, rot_6d, gripper) - 10 dim
        # action_normalized: normalized version
        save_csv("RISE_action_data.csv", 
                 ["index", "x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"],
                 save_action)
        save_csv("RISE_action_normalized_data.csv",
                 ["index", "x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"],
                 save_action_normalized)
        print(f"✔ 데이터 저장 완료: RISE_action_data.csv, RISE_action_normalized_data.csv")


if __name__ == "__main__":
    main()
