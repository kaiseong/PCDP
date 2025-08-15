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
from pcdp.policy.diffusion_RISE_policy import RISEPolicy
from pcdp.common.RISE_transformation import xyz_rot_transform
from pcdp.dataset.RISE_util import *
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor


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
            rotation_rep_convention='XYZ'
        )
        original_poses.append(original_pose_6d)


    return original_poses


def _unnormalize_action(action_6d):
    """
    주어진 6D 회전 표현 액션을 역정규화합니다.
    action_6d: (N, 10) 크기의 배열 - [x, y, z, rot_6d(6), gripper(1)]
    """
    # TRANS_MIN/MAX는 (3,) 크기이므로 앞 3개 요소에만 적용
    trans_min_exp = torch.from_numpy(TRANS_MIN).to(action_6d.device).float()
    trans_max_exp = torch.from_numpy(TRANS_MAX).to(action_6d.device).float()
    
    # 위치 역정규화: [-1, 1] -> [min, max]
    action_6d[..., :3] = (action_6d[..., :3] + 1) / 2.0 * (trans_max_exp - trans_min_exp) + trans_min_exp
    
    # 그리퍼 역정규화: [-1, 1] -> [0, 1] (0: closed, 1: open)
    # RISE의 그리퍼 값은 너비가 아닌 상태를 나타내므로 여기서는 0과 1 사이로 매핑
    action_6d[..., -1] = (action_6d[..., -1] + 1) / 2.0 
    return action_6d

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
    policy: RISEPolicy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload['state_dicts']['model'])
    
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
            preprocessor = PointCloudPreprocessor(**cfg.task.dataset.preprocessor_config)
            
            target_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([target_pose], [plan_time])
            time.sleep(2.0)
            
            # ... (main 함수 내부) ...
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
                    # ===== 정책 기반 제어 (Policy Control) =====
                    with torch.no_grad():
                        # 1. 관측 데이터 전처리 (수정됨)
                        # 가장 마지막 관측 프레임 하나만 사용
                        pc_raw = obs['pointcloud'][-1]
                        
                        # 좌표 변환, 작업 공간 필터링 등 (config 기반)
                        pc = preprocessor(pc_raw)

                        # 색상 정규화 (학습 때와 동일하게)
                        colors = pc[:, 3:6] / 255.0  # [0, 255] -> [0, 1]
                        colors = (colors - IMG_MEAN) / IMG_STD
                        pc[:, 3:6] = colors

                        # Voxelize
                        coords = np.ascontiguousarray(pc[:, :3] / voxel_size, dtype=np.int32)
                        feats = pc.astype(np.float32)

                        # MinkowskiEngine SparseTensor 생성 (단일 프레임)
                        coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
                        coords_batch = coords_batch.to(device)
                        feats_batch = feats_batch.to(device)
                        cloud_data = ME.SparseTensor(feats_batch, coords_batch)
                        t1= mono_time.now_ms()
                        pred_action_10d_normalized = policy(cloud_data, batch_size=1).cpu()
                        print(f"inference: {mono_time.now_ms() - t1}")

                        print(f"loop time: {mono_time.now_ms() - t2}")
                        t2= mono_time.now_ms()
                        # 2. 액션 추론
                        pred = _unnormalize_action(pred_action_10d_normalized.squeeze(0))
                        if pred.ndim == 1:
                            pred = pred.unsqueeze(0)

                        pos   = pred[:, :3].cpu().numpy()
                        rot6d = pred[:, 3:9].cpu().numpy()
                        grip  = pred[:, 9:].cpu().numpy()

                        xyz_rot6d = np.concatenate([pos, rot6d], axis=-1)  # (N, 9)

                        # (xyz + rot6d) → (xyz + euler)
                        xyz_euler = xyz_rot_transform(
                            xyz_rot6d,
                            from_rep="rotation_6d",
                            to_rep="euler_angles",
                            to_convention="XYZ"
                        )

                        TF_xyz_euler = revert_action_transformation(xyz_euler, robot_to_base)
                        action_sequence_7d = np.concatenate([TF_xyz_euler, grip], axis=-1)
                        
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