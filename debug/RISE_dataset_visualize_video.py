"""
    Dataset class를 통해 불러온
    batch 데이터 검증용 코드 (RISE version) - 비디오 재생 버전

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
import time

from pcdp.dataset.RISE_stack_pc_dataset import collate_fn # RISE collate_fn
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.common import RISE_transformation as rise_tf

camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0. ,        0. ,        1.      ]
    ])

workspace_bounds = np.array([
    [-0.800, 0.800],    # X range (milli meters)
    [-0.500, 0.500],    # Y range (milli meters)
    [-0.100, 0.350]     # Z range (milli meters)
])

robot_to_base = np.array([
    [1.,         0.,         0.,          0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ────────────────────────────────────────────────────────────
# 설정 – 프로젝트에 맞게 YAML 경로/이름만 바꿔 주세요
CONFIG_DIR  = "../pcdp/config"    # your yaml folder
CONFIG_NAME = "train_diffusion_RISE_workspace"
BATCH_SIZE_VIS = 10       # cfg.dataloader.batch_size 와 동일하게
PLAYBACK_SPEED_FPS = 30   # 영상 재생 속도 (Hz)
# ────────────────────────────────────────────────────────────

def main():
    # 1) YAML 로드 & Dataset 인스턴스
    with hydra.initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = hydra.compose(config_name=CONFIG_NAME)

    ds = hydra.utils.instantiate(cfg.task.dataset, use_cache=True)
    print(f"[INFO] dataset sequences: {len(ds)}")

    # 2) DataLoader (시각화용)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_VIS,
                    num_workers=10, shuffle=False, collate_fn=collate_fn)
    batch_iter = iter(dl)
    batch = None
    sample_idx = 0
    
    # 3) Open3D 시각화 창
    vis = o3d.visualization.Visualizer()
    vis.create_window("RISE Dataset Video", 1280, 720, visible=True)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0, 0, 0])
    
    pcd = o3d.geometry.PointCloud()
    eef_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame.transform(camera_to_base)
    last_eef_transform = np.identity(4)

    first_loaded = False
    running = True

    while running:
        try:
            if batch is None or sample_idx >= BATCH_SIZE_VIS:
                batch = next(batch_iter)
                sample_idx = 0

            # collated batch에서 현재 sample_idx에 해당하는 데이터 추출
            coords = batch['input_coords_list']
            feats = batch['input_feats_list']
            action_euler = batch['action_euler']
            
            point_indices = (coords[:, 0] == sample_idx)
            
            if not torch.any(point_indices):
                sample_idx += 1
                continue

            pc = feats[point_indices].numpy()
            action = action_euler[sample_idx, 0].numpy()
            
            # 데이터 처리
            xyz, rgb = pc[:, :3], pc[:, 3:6]
            IMG_MEAN = np.array([0.0217, 0.0217, 0.0217])
            IMG_STD = np.array([0.1474, 0.1474, 0.1474])
            rgb = np.clip(rgb * IMG_STD + IMG_MEAN, 0, 1)

            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))
            
            # EEF 좌표계 업데이트
            z_offset = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.01], [0, 0, 0, 1]])
            eef_to_robot_base = rise_tf.rot_trans_mat(action[:3], action[3:6])
            current_eef_transform = robot_to_base @ eef_to_robot_base @ z_offset
            
            transform_to_apply = current_eef_transform @ np.linalg.inv(last_eef_transform)
            eef_frame.transform(transform_to_apply)
            last_eef_transform = current_eef_transform

            if not first_loaded:
                vis.add_geometry(pcd)
                vis.add_geometry(base_frame)
                vis.add_geometry(camera_frame)
                vis.add_geometry(eef_frame)
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
                vis.update_geometry(eef_frame)
            
            sample_idx += 1

        except StopIteration:
            print("✔ 모든 샘플 확인 완료!")
            running = False # 모든 배치를 다 돌았으면 종료
        
        if not vis.poll_events():
            running = False
        vis.update_renderer()
        time.sleep(1 / PLAYBACK_SPEED_FPS)

    vis.destroy_window()

if __name__ == "__main__":
    main()
