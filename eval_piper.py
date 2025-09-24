# eval_piper.py
import time
from multiprocessing.managers import SharedMemoryManager
import click
import open3d as o3d
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
from pcdp.policy.diffusion_PCDP_policy import PCDPPolicy
from pcdp.common.RISE_transformation import xyz_rot_transform
from pcdp.dataset.RISE_util import *
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor


robot_to_base = np.array([
    [1.,         0.,         0.,          -0.04],
    [0.,         1.,         0.,         -0.29],
    [0.,         0.,         1.,          -0.03],
    [0.,         0.,         0.,          1.0]
])


# Hydra 설정 등록
OmegaConf.register_new_resolver("eval", eval, replace=True)


import numpy as np
from pcdp.common import RISE_transformation as rise_tf


def revert_action_transformation(transformed_action_6d, robot_to_base_matrix):
    
    base_to_robot_matrix = np.linalg.inv(robot_to_base_matrix)
    original_poses = []
    for action in transformed_action_6d:
        translation = action[:3]
        rotation = action[3:6]
        transformed_matrix = rise_tf.rot_trans_mat(translation, rotation)

        # 순방향: T_k = robot_to_base @ eef_to_robot_base_k
        # 역방향: eef_to_robot_base_k = inv(robot_to_base) @ T_k
        original_matrix = base_to_robot_matrix @ transformed_matrix

        # 5. 원래의 4x4 행렬을 다시 6D pose [x,y,z,r,p,y] 벡터로 변환합니다.
        original_pose_6d = rise_tf.mat_to_xyz_rot(
            original_matrix,
            rotation_rep='euler_angles',
            rotation_rep_convention='ZYX'
        )
        original_poses.append(original_pose_6d)
    return np.array(original_poses, dtype=np.float32)


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save evaluation results.')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode for initial condition visualization')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
def main(input, output, match_episode, frequency):
    # 체크포인트 및 설정 로드
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    # 정책 모델 초기화 및 가중치 로드
    policy: PCDPPolicy = hydra.utils.instantiate(cfg.policy)
    
    # EMA 가중치가 있으면 사용하고, 없으면 일반 모델 가중치를 사용
    state_dict = payload['state_dicts']['model']
    if 'ema_model' in payload['state_dicts']:
        state_dict = payload['state_dicts']['ema_model']
        cprint("Loading EMA model weights for evaluation.", "green")
    else:
        cprint("Loading non-EMA model weights for evaluation.", "yellow")
    policy.load_state_dict(state_dict)
    
    device = torch.device(cfg.training.device)
    policy.to(device).eval()
    cprint(f"Policy loaded on {device}", "green")

    dt = 1.0 / frequency
    voxel_size = cfg.task.dataset.voxel_size

    # URDF 및 IK 파라미터 설정
    urdf_path = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/pcdp/dependencies"
    ee_link_name = "link6"

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            TeleoperationPiper(shm_manager=shm_manager) as teleop, \
            RealEnv(
                output_dir=output, 
                urdf_path=urdf_path,
                mesh_dir=mesh_dir,
                ee_link_name=ee_link_name,
                joints_to_lock_names=[],
                frequency=frequency,
                init_joints=True,
                orbbec_mode="C2D",
                shm_manager=shm_manager
            ) as env:
            
            cprint('Ready! Press "C" to start evaluation, "S" to stop, "Q" to quit.', "yellow")
            
            # YAML 설정에서 preprocessor_config를 가져와 PointCloudPreprocessor 인스턴스화
            pc_preprocessor = PointCloudPreprocessor(**cfg.task.dataset.pc_preprocessor_config)
            low_preprocessor = LowDimPreprocessor(**cfg.task.dataset.low_dim_preprocessor_config)
            
            target_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([target_pose], [plan_time])
            time.sleep(2.0)
            
            t_start = mono_time.now_s()
            iter_idx = 0
            is_evaluating = False
            t2=0
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                obs = env.get_obs()
                # 키보드 입력 처리
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        if is_evaluating:
                            env.end_episode()
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
                            cprint("Evaluation stopped. Human in control.", "yellow")

                if is_evaluating:
                    with torch.no_grad():
                        # 1. 관측 데이터 전처리 (학습 파이프라인과 일치시킴)
                        pc_raw = obs['pointcloud'][-1]
                        pose_raw = obs['robot_eef_pose'][-1].astype(np.float64)
                        grip_raw = obs['robot_gripper'][-1].flatten().astype(np.float64)
                        robot_obs_raw_7d = np.concatenate([pose_raw, grip_raw])

                        # 로봇 관측값 좌표계 변환 (학습 데이터와 동일하게)
                        transformed_obs_7d = low_preprocessor.TF_process(robot_obs_raw_7d[np.newaxis, :]).squeeze(0)
                        
                        # Policy 입력을 위해 10D 텐서로 변환
                        obs_pose_euler = transformed_obs_7d[:6]
                        obs_gripper = transformed_obs_7d[6:]
                        obs_9d = xyz_rot_transform(obs_pose_euler, from_rep='euler_angles', to_rep='rotation_6d', from_convention='ZYX')
                        obs_10d = np.concatenate([obs_9d, obs_gripper], axis=-1)
                        robot_obs_tensor = torch.from_numpy(obs_10d).to(device).float().unsqueeze(0)

                        # 포인트클라우드 전처리
                        pc = pc_preprocessor.process(pc_raw)
                        coords = np.ascontiguousarray(pc[:, :3] / voxel_size, dtype=np.int32)
                        feats = pc.astype(np.float32)
                        coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
                        cloud_data = ME.SparseTensor(
                            features=feats_batch, 
                            coordinates=coords_batch,
                            device=device)
                        
                        t1 = mono_time.now_ms()
                        
                        # 2. 액션 추론 (Policy가 Normalizer 상태까지 포함)
                        pred_action_10d = policy(cloud_data, robot_obs=robot_obs_tensor, batch_size=1).cpu()
                        
                        print(f"inference: {mono_time.now_ms() - t1}")
                        print(f"loop time: {mono_time.now_ms() - t2}")
                        t2 = mono_time.now_ms()

                        # 3. 액션 후처리
                        pred = pred_action_10d
                        if pred.ndim == 3:
                            pred = pred.squeeze(0)

                        pos   = pred[:, :3].cpu().numpy()
                        rot6d = pred[:, 3:9].cpu().numpy()
                        grip = (pred[:, 9:].cpu().numpy() >= 0.5).astype(np.int32)
                        xyz_rot6d = np.concatenate([pos, rot6d], axis=-1)

                        # 6D 회전 표현을 오일러 각도로 변환
                        xyz_euler_base_frame = xyz_rot_transform(
                            xyz_rot6d,
                            from_rep="rotation_6d",
                            to_rep="euler_angles",
                            to_convention="ZYX"
                        )
                        
                        # 액션을 "base" 좌표계에서 로봇의 실제 실행 좌표계로 역변환
                        xyz_euler_robot_frame = revert_action_transformation(xyz_euler_base_frame, robot_to_base)
                        
                        action_sequence_7d = np.concatenate([xyz_euler_robot_frame, grip], axis=-1)
                        
                        # 4. 로봇 제어
                        now = mono_time.now_s()
                        timestamps = now + np.arange(len(action_sequence_7d)) * dt
                        env.exec_actions(
                            actions=action_sequence_7d,
                            timestamps=timestamps
                        )
                else:
                    # ===== 사람 제어 (Human Control) =====
                    target_pose = teleop.get_motion_state()
                    env.exec_actions(actions=[target_pose], timestamps=[mono_time.now_s() + dt])

                precise_wait(t_cycle_end)
                iter_idx += 1

if __name__ == '__main__':
    main()