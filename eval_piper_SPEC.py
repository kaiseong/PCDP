# eval_piper_SPEC_mono.py
# -*- coding: utf-8 -*-
import time
from multiprocessing.managers import SharedMemoryManager
import click
import os
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
import MinkowskiEngine as ME
from termcolor import cprint

from pcdp.real_world.real_env_piper import RealEnv
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from pcdp.common.precise_sleep import precise_wait
import pcdp.common.mono_time as mono_time
from pcdp.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from pcdp.policy.diffusion_SPEC_policy_mono import SPECPolicyMono
from pcdp.common.RISE_transformation import xyz_rot_transform
from pcdp.model.common.normalizer import LinearNormalizer
from SPEC_Processor import SPECProcessors  # ← 추가

robot_to_base = np.array([
    [1., 0., 0., -0.04],
    [0., 1., 0., -0.29],
    [0., 0., 1., -0.03],
    [0., 0., 0.,  1.0]
])

OmegaConf.register_new_resolver("eval", eval, replace=True)

from pcdp.common import RISE_transformation as rise_tf

def revert_action_transformation(transformed_action_6d, robot_to_base_matrix):
    base_to_robot_matrix = np.linalg.inv(robot_to_base_matrix)
    original_poses = []
    for action in transformed_action_6d:
        translation = action[:3]
        rotation = action[3:6]
        transformed_matrix = rise_tf.rot_trans_mat(translation, rotation)
        original_matrix = base_to_robot_matrix @ transformed_matrix
        original_pose_6d = rise_tf.mat_to_xyz_rot(
            original_matrix, rotation_rep='euler_angles', rotation_rep_convention='ZYX'
        )
        original_poses.append(original_pose_6d)
    return np.array(original_poses, dtype=np.float32)

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save evaluation results.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--save-data', is_flag=True, default=False, help="Enable saving episode data (pointcloud, robot state).")
def main(input, output, match_episode, frequency, save_data):
    # 체크포인트 및 설정 로드
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # 정책 모델 초기화 및 가중치 로드
    policy: SPECPolicyMono = hydra.utils.instantiate(cfg.policy)
    
    state_dict = payload['state_dicts']['model']
    policy.load_state_dict(state_dict)
    
    device = torch.device(cfg.training.device)
    policy.to(device).eval()
    cprint(f"Policy loaded on {device}", "green")

    # =========================
    # Normalizer 설정 (재발 방지)
    # 1) 체크포인트 폴더에 normalizer.pt가 있으면 로드
    # 2) 없으면 학습 YAML의 translation 범위를 dataset에 강제 세팅 후 fit
    # =========================
    # ckpt_dir = os.path.dirname(ckpt_path)
    # n_path = os.path.join(ckpt_dir, "normalizer.pt")
    # if os.path.exists(n_path):
    #     state = torch.load(n_path, map_location=device)
    #     n = LinearNormalizer(); n.load_state_dict(state)
    #     policy.set_normalizer(n)
    #     cprint(f"Loaded normalizer: {n_path}", "green")
    # else:
    #     # ⬇️ 학습과 같은 translation 정규화 범위를 강제 (오프셋 방지)
    #     cfg.task.dataset._target_ = 'pcdp.dataset.PCDP_stack_dataset.PCDP_RealStackPointCloudDataset'
    #     dataset = hydra.utils.instantiate(cfg.task.dataset)
    #     if 'translation' in cfg.training:
    #         dataset.set_translation_norm_config(cfg.training.translation)
    #     normalizer = dataset.get_normalizer(device=device)
    #     policy.set_normalizer(normalizer)
    #     cprint("Built normalizer from dataset with training.translation enforced.", "green")

    # 주기 등
    dt = 1.0 / frequency
    voxel_size = cfg.task.dataset.voxel_size
    max_pts = int(cfg.task.pointcloud_shape[0])  # e.g., 92160

    # URDF/Env
    urdf_path = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/pcdp/dependencies"
    ee_link_name = "link6"

    # 프로세서 시작 (전처리 병렬화)
    pc_cfg = OmegaConf.to_container(cfg.task.dataset.pc_preprocessor_config, resolve=True)
    # 실기 성능 위해 필터 OFF 권장 (캐시 생성시에만 ON)
    pc_cfg["enable_filter"] = False
    ld_cfg = OmegaConf.to_container(cfg.task.dataset.low_dim_preprocessor_config, resolve=True)
    processors = SPECProcessors(max_points=max_pts,
                                pc_preprocessor_config=pc_cfg,
                                lowdim_preproc_config=ld_cfg)
    processors.start()

    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                 TeleoperationPiper(shm_manager=shm_manager) as teleop, \
                 RealEnv(
                    output_dir=output,
                    urdf_path=urdf_path, mesh_dir=mesh_dir, ee_link_name=ee_link_name,
                    joints_to_lock_names=[],
                    frequency=frequency, init_joints=True,
                    orbbec_mode="C2D", shm_manager=shm_manager
                 ) as env:

                cprint('Press "C" to start, "S" to stop, "Q" to quit.', "yellow")

                # 초기 자세
                target_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
                env.exec_actions([target_pose], [mono_time.now_s() + 2.0])
                time.sleep(2.0)

                t_start = mono_time.now_s()
                iter_idx = 0
                is_evaluating = False
                last_infer_ms = mono_time.now_ms()

                while True:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    obs = env.get_obs()

                    # 키 처리
                    for key_stroke in key_counter.get_press_events():
                        if key_stroke == KeyCode(char='q'):
                            if is_evaluating: env.end_episode()
                            return
                        elif key_stroke == KeyCode(char='c'):
                            if not is_evaluating:
                                env.start_episode()
                                is_evaluating = True
                                cprint("Evaluation started!", "cyan")
                        elif key_stroke == KeyCode(char='s'):
                            if is_evaluating:
                                env.end_episode()
                                is_evaluating = False
                                cprint("Evaluation stopped.", "yellow")

                    if is_evaluating:
                        # --- 1) RAW를 프로세서에 밀어넣기 (아주 가벼움) ---
                        pc_raw = obs['main_pointcloud'][-1].astype(np.float32, copy=False)   # (N,6)
                        ts_obs = float(obs['timestamp'][-1])
                        processors.push_pc_raw(pc_raw, ts_obs)

                        pose_raw = obs['robot_eef_pose'][-1].astype(np.float64, copy=False)
                        grip_raw = obs['robot_gripper'][-1].flatten().astype(np.float64, copy=False)
                        robot7d = np.concatenate([pose_raw, grip_raw[:1]], axis=0).astype(np.float32, copy=False)
                        processors.push_state7d(robot7d, ts_obs)

                        # --- 2) 최신 fused/state를 당겨오기 ---
                        fused, ts_pc, seq_pc = processors.read_fused_pc()     # (N,7) or None
                        st10, ts_st, seq_st = processors.read_state10d()      # (1,10) or None

                        if fused is None or st10 is None:
                            # 아직 워커 첫 산출이 없으면 스킵
                            precise_wait(t_cycle_end)
                            iter_idx += 1
                            continue

                        # --- 3) SparseTensor로 구성 ---
                        coords = np.floor(fused[:, :3] / voxel_size).astype(np.int32, copy=False)
                        coords = np.ascontiguousarray(coords)
                        feats = fused.astype(np.float32, copy=False)
                        coords_b, feats_b = ME.utils.sparse_collate([coords], [feats])
                        cloud = ME.SparseTensor(features=feats_b, coordinates=coords_b, device=device)

                        robot_obs_tensor = torch.from_numpy(st10.astype(np.float32)).to(device)
                        # st10: (1,10)

                        # --- 4) 추론 ---
                        t0 = mono_time.now_ms()
                        with torch.no_grad():
                            pred_action_10d = policy(cloud, robot_obs=robot_obs_tensor, batch_size=1).cpu()
                        infer_ms = mono_time.now_ms() - t0

                        # --- 5) 액션 후처리 ---
                        pred = pred_action_10d.squeeze(0).numpy()  # (H,10)
                        pos   = pred[:, :3]
                        rot6d = pred[:, 3:9]
                        grip  = (pred[:, 9:] ).astype(np.int32)
                        xyz_euler_base = xyz_rot_transform(
                            np.concatenate([pos, rot6d], axis=-1),
                            from_rep="rotation_6d", to_rep="euler_angles", to_convention="ZYX"
                        )
                        xyz_euler_robot = revert_action_transformation(xyz_euler_base, robot_to_base)
                        action_seq_7d = np.concatenate([xyz_euler_robot, grip], axis=-1)

                        # --- 6) 지연 보상 타임스탬프 (학습 dt=0.1s 기준) ---
                        L = len(action_seq_7d)
                        infer_latency = mono_time.now_s() - ts_pc   # 실측 지연
                        action_timestamps = (np.arange(L, dtype=np.float64) * dt) + ts_pc + infer_latency

                        # 과거 타임스탬프 제거(최소 실행지연 10ms)
                        min_exec = mono_time.now_s() + 0.01
                        is_new = action_timestamps > min_exec
                        if not np.any(is_new):
                            next_step = mono_time.now_s() + dt
                            env.exec_actions([action_seq_7d[-1]], [next_step])
                        else:
                            env.exec_actions(actions=action_seq_7d[is_new],
                                            timestamps=action_timestamps[is_new])

                        # 로깅
                        print(f"infer_ms: {infer_ms:5.1f} | loop_ms: {mono_time.now_ms()-last_infer_ms:5.1f} | pts:{fused.shape[0]}")
                        last_infer_ms = mono_time.now_ms()
                    else:
                        target_pose = teleop.get_motion_state()
                        env.exec_actions(actions=[target_pose], timestamps=[mono_time.now_s() + dt])

                    precise_wait(t_cycle_end)
                    iter_idx += 1
    finally:
        processors.stop()

if __name__ == '__main__':
    main()
