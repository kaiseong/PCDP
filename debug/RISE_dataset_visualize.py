"""
    RISE dataset를 로드한 뒤,
    시점 t의 포인트클라우드/카메라/원점 좌표계를 시각화하고,
    EEF 좌표계를 t..t+H_FUTURE까지(기본 16) 동시에 표시하는 도구.
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch, open3d as o3d, hydra
import pytorch3d.ops as torch3d_ops  # (필요시 사용)
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from pcdp.dataset.RISE_stack_dataset import collate_fn  # RISE collate_fn
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor
from pcdp.common import RISE_transformation as rise_tf

# ────────────────────────────────────────────────────────────
# 고정된 외부/내부 좌표계 변환(당신 코드의 기존 상수 유지)
camera_to_base = np.array([
    [  0.007131,  -0.91491,    0.403594,  0.05116],
    [ -0.994138,   0.003833,   0.02656,  -0.00918],
    [ -0.025717,  -0.403641,  -0.914552, 0.50821 ],
    [  0.,         0.,         0.,        1.      ]
])

workspace_bounds = np.array([
    [-0.000, 0.715],    # X range (m)
    [-0.400, 0.350],    # Y range (m)
    [-0.100, 0.400]     # Z range (m)
])

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])


z_offset = np.array([
    [1, 0, 0, 0], 
    [0, 1, 0, 0], 
    [0, 0, 1, 0.07], 
    [0, 0, 0, 1]])

# ────────────────────────────────────────────────────────────

# Open3D 표시용 상수
H_FUTURE = 16             # t..t+16 (총 17 프레임)
EEF_FRAME_SIZE = 0.045    # EEF 좌표축 크기(겹침 완화용)
POINT_SIZE = 2.0

# 색상 역정규화(당신 데이터의 mean/std 유지)
IMG_MEAN = np.array([0.0217, 0.0217, 0.0217])
IMG_STD  = np.array([0.1474, 0.1474, 0.1474])

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ────────────────────────────────────────────────────────────
# 설정 – 프로젝트에 맞게 YAML 경로/이름만 바꾸세요
CONFIG_DIR  = "../pcdp/config"                 # your yaml folder
CONFIG_NAME = "train_diffusion_PCDP_workspace" # your yaml name
BATCH_SIZE_VIS = 10                            # cfg.dataloader.batch_size 와 동일 권장
NUM_WORKERS = 10
# ────────────────────────────────────────────────────────────


def main():
    # 필요시 전처리기 사용 (지금은 미사용)
    preprocess = PointCloudPreprocessor(extrinsics_matrix=camera_to_base,
                                        workspace_bounds=workspace_bounds,
                                        enable_sampling=False)

    # 1) YAML 로드 & Dataset 인스턴스
    with hydra.initialize(version_base=None, config_path=CONFIG_DIR):
        cfg = hydra.compose(config_name=CONFIG_NAME)

    ds = hydra.utils.instantiate(cfg.task.dataset, use_cache=True)
    print(f"[INFO] dataset sequences: {len(ds)}")

    # 2) DataLoader (시각화용)
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE_VIS,
                    num_workers=NUM_WORKERS,
                    shuffle=False,
                    collate_fn=collate_fn)
    batch_iter = iter(dl)
    batch = None
    sample_idx = 0

    # 3) Open3D 시각화 창
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("RISE Dataset inspector", 1280, 720, visible=True)

    opt = vis.get_render_option()
    opt.point_size = POINT_SIZE
    opt.background_color = np.array([0, 0, 0])

    # 지오메트리
    pcd = o3d.geometry.PointCloud()
    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10, origin=[0, 0, 0])
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
    camera_frame.transform(camera_to_base)

    # 여러 개의 EEF 좌표축을 관리할 컨테이너
    eef_frames = []

    first_loaded = False

    def load_next_sample():
        nonlocal batch, sample_idx, first_loaded, eef_frames
        try:
            # 배치가 없거나 이번 배치의 끝이면 다음 배치 로드
            if batch is None or sample_idx >= BATCH_SIZE_VIS:
                print("\nLoading next batch...")
                batch = next(batch_iter)
                sample_idx = 0

            # collated batch에서 현재 sample_idx에 해당하는 포인트들 추출
            coords = batch['input_coords_list']   # [N, 4] (batch_idx, x, y, z or etc.)
            feats = batch['input_feats_list']     # [N, C] (x,y,z,r,g,b,...)
            action_euler = batch['action_euler']  # [B, T, 6]  (rpyxyz)

            # coords의 첫 컬럼이 batch index
            point_indices = (coords[:, 0] == sample_idx)

            if not torch.any(point_indices):
                print(f"[WARN] No points for sample_idx {sample_idx} in this batch. Skipping.")
                sample_idx += 1
                if sample_idx >= BATCH_SIZE_VIS:
                    load_next_sample()  # 다음 배치 시도
                return

            pc = feats[point_indices].numpy()
            print(f"\n[Sample {sample_idx}] Original point count: {len(pc)}")

            # feats: [x, y, z, r, g, b, ...]
            xyz, rgb = pc[:, :3], pc[:, 3:6]

            # 색상 역정규화
            rgb = rgb * IMG_STD + IMG_MEAN
            rgb = np.clip(rgb, 0, 1)

            # 포인트 설정 (t 시점)
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float32))
            pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float32))

            # 이전 샘플에서 추가된 EEF 좌표계를 모두 제거
            for g in eef_frames:
                vis.remove_geometry(g, reset_bounding_box=False)
            eef_frames.clear()

            # EEF 좌표계: t..t+H_FUTURE (시퀀스 길이에 맞춰 안전하게 자름)
            T_total = action_euler.shape[1]
            k_max = int(min(H_FUTURE, T_total - 1))

            # 살짝 띄워 가시성 높이기
            z_offset = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0.07],
                                 [0, 0, 0, 1]])

            for k in range(0, k_max + 1):
                act_k = action_euler[sample_idx, k].numpy()  # [x, y, z, roll, pitch, yaw]
                eef_to_robot_base_k = rise_tf.rot_trans_mat(act_k[:3], act_k[3:6])
                # T_k = robot_to_base @ eef_to_robot_base_k @ z_offset
                T_k = eef_to_robot_base_k 

                axis_k = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=EEF_FRAME_SIZE, origin=[0, 0, 0]
                )
                axis_k.transform(T_k)
                eef_frames.append(axis_k)
                vis.add_geometry(axis_k, reset_bounding_box=False)
                
                # t_ratio = k_max if k_max > 0 else 1
                # w = k / t_ratio  # 0.0 at t  → 1.0 at t+H_FUTURE
                # # green→red: (r,g,b) = (w, 1-w, 0)
                # color_k = [float(w), float(1.0 - w), 0.0]


                # sphere_k = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
                # sphere_k.compute_vertex_normals()
                # sphere_k.paint_uniform_color(color_k)

                # # 구는 회전 필요 없음. 위치만 반영
                # center_k = T_k[:3, 3]  # (x,y,z)
                # sphere_k.translate(center_k)

                # eef_frames.append(sphere_k)
                # vis.add_geometry(sphere_k, reset_bounding_box=False)
                

            # 장면에 지오메트리 배치/업데이트
            if not first_loaded:
                vis.add_geometry(pcd, reset_bounding_box=True)
                # vis.add_geometry(base_frame, reset_bounding_box=False)   # origin at t
                # vis.add_geometry(camera_frame, reset_bounding_box=False) # camera at t

                # 카메라 시점(초기)
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
                # eef_frames는 새로 add 되었으므로 update 불필요

            vis.poll_events()
            vis.update_renderer()

            sample_idx += 1

        except StopIteration:
            print("✔ 모든 샘플 확인 완료!  [Q]로 창을 닫아 종료하세요.")
            # 반복 종료

    # 첫 샘플 로드
    load_next_sample()

    # 콜백
    vis.register_key_callback(ord("N"), lambda v: load_next_sample())
    vis.register_key_callback(ord("Q"), lambda v: v.close())

    print("\n[N] → 다음 샘플 | [Q] → 종료")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
