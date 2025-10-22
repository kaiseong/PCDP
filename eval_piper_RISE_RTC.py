# eval_piper_RISE_RTC.py

import threading
import time
from multiprocessing.managers import SharedMemoryManager
import click
import open3d as o3d
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
from pcdp.policy.diffusion_RISE_RTC_policy import RISEPolicy
from pcdp.common.RISE_transformation import xyz_rot_transform
from pcdp.dataset.RISE_util import *
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor
from pcdp.model.common.normalizer import LinearNormalizer


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


class RTCChunkManager:
    def __init__(self, policy, horizon, device, dt):
        self.policy = policy
        self.horizon = horizon
        self.device = device
        self.dt = dt

        self.chunk = None           # (H, A) 액션 시퀀스
        self.timestamp = None       # 이 청크의 기준 시간 (now_s)
        self.exec_idx = 0           # 현재 실행 중인 인덱스

        self.latest_obs = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._inference_loop)
        self.thread.daemon = True
        self.thread.start()

    def set_obs(self, obs):
        with self.lock:
            self.latest_obs = obs

    def get_action_sequence(self):
        with self.lock:
            if self.chunk is None:
                return None, None

            t_now = mono_time.now_s()
            t_offset = t_now - self.timestamp  # 실행된 시간
            idx_offset = int(t_offset / self.dt)
            idx_offset = np.clip(idx_offset, 0, self.horizon - 1)

            future = self.chunk[idx_offset:]  # 남은 액션 시퀀스
            ts = t_now + self.dt * np.arange(len(future))

            self.exec_idx = idx_offset
            return future, ts

    def stop(self):
        self.running = False
        self.thread.join()

    def _inference_loop(self):
        while self.running:
            time.sleep(0.01)
            with self.lock:
                if self.latest_obs is None:
                    continue
                obs = self.latest_obs  # 추론용 복사

            # 1) 관측은 메인 스레드에서 SparseTensor로 전처리하여 전달됨
            cloud_data = obs['cloud']  # ME.SparseTensor

            # 2) RTC용 이전 청크 참조값 구성 (Original RTC의 hard freeze + soft overlap 램프)
            prev_chunk = self.chunk if self.chunk is not None else None
            exec_idx = self.exec_idx

            if prev_chunk is not None:
                # --- [RTC 수정] 지연(d) 기반 프리즈 길이 보정 ---
                d = max(0.0, mono_time.now_s() - (self.timestamp or mono_time.now_s()))
                freeze_extra = int(d / self.dt)                 # 지연만큼 prefix 추가 고정
                freeze_len = min(exec_idx + freeze_extra, self.horizon)

                # --- soft ramp mask (코사인 램프) ---
                rtc_target = torch.tensor(prev_chunk, device=self.device, dtype=torch.float32).unsqueeze(0)  # (1,T,Da)
                W = torch.ones_like(rtc_target)                  # 기본 1(유도 적용)
                W[:, :freeze_len] = 0.0                          # 이미 실행된 prefix는 유도 off

                # 겹치는 꼬리 K 구간을 서서히 0→1로 (부드러운 접합)
                K = max(4, min(12, self.horizon // 4))           # 권장: 4~12
                tail_start = max(freeze_len - K, 0)
                if tail_start < freeze_len:
                    ramp = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, steps=(freeze_len - tail_start), device=self.device)))
                    # ramp: 0→1 (freeze_len-K .. freeze_len-1)
                    W[:, tail_start:freeze_len] = ramp.view(1, -1, 1)

                rtc_mask = W
            else:
                rtc_target = rtc_mask = None

            # 3) 정책 호출 (★ 중요: no_grad 금지 — UNet 내부에서 VJP 사용)
            pred = self.policy(
                cloud_data,
                batch_size=1,
                rtc_target=rtc_target,
                rtc_mask=rtc_mask,
            )
            pred = pred.squeeze(0).detach().cpu().numpy()  # (H, A)

            # 4) 최신 청크로 교체 및 시간 기준 갱신
            with self.lock:
                self.chunk = pred
                self.timestamp = mono_time.now_s()
                self.exec_idx = 0


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

        # 다시 6D pose [x,y,z,r,p,y]
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
    policy: RISEPolicy = hydra.utils.instantiate(cfg.policy)

    # EMA 미사용 가정: 곧바로 model 가중치 사용
    state_dict = payload['state_dicts']['model']
    policy.load_state_dict(state_dict)

    device = torch.device(cfg.training.device)
    policy.to(device).eval()
    cprint(f"Policy loaded on {device}", "green")

    # (선택) 정규화기 중심 확인
    try:
        pd = policy.normalizer['action_translation'].params_dict  # SingleFieldLinearNormalizer
        center = (-pd['offset'] / pd['scale']).detach().cpu().numpy()
        cprint(f"action_translation center (m): {center}", "cyan")
    except Exception as e:
        cprint(f"[warn] center print failed: {e}", "yellow")

    dt = 1.0 / frequency
    voxel_size = cfg.task.dataset.voxel_size

    # URDF 및 IK 파라미터 설정
    urdf_path = "/home/moai/pcdp/dependencies/piper_description/urdf/piper_no_gripper_description.urdf"
    mesh_dir = "/home/moai/pcdp/dependencies"
    ee_link_name = "link6"

    horizon = getattr(getattr(policy, "action_decoder", None), "horizon", None)
    if horizon is None:
        horizon = getattr(getattr(cfg, "policy", None), "num_action", 20)

    rtc_mgr = RTCChunkManager(
        policy=policy,
        horizon=horizon,
        device=device,
        dt=dt
    )

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

            target_pose = [0.054952, 0.0, 0.493991, 0.0, np.deg2rad(85.0), 0.0, 0.0]
            plan_time = mono_time.now_s() + 2.0
            env.exec_actions([target_pose], [plan_time])
            time.sleep(2.0)

            t_start = mono_time.now_s()
            iter_idx = 0
            is_evaluating = False
            test_start = 0

            while True:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                obs = env.get_obs()

                # 키보드 입력 처리
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        if is_evaluating:
                            env.end_episode()
                        try:
                            rtc_mgr.stop()   # RTC 스레드 정리
                        except Exception:
                            pass
                        return
                    elif key_stroke == KeyCode(char='c'):
                        if not is_evaluating:
                            env.start_episode()
                            is_evaluating = True
                            test_start = mono_time.now_ms()
                            cprint("Evaluation started!", "cyan")
                    elif key_stroke == KeyCode(char='s'):
                        if is_evaluating:
                            env.end_episode()
                            is_evaluating = False
                            cprint("Evaluation stopped. Human in control.", "yellow")

                if is_evaluating:
                    # (주의) 여기의 no_grad는 메인 스레드 전처리만 해당.
                    # RTCChunkManager의 정책 호출은 별도 스레드에서 autograd를 사용함.
                    with torch.no_grad():
                        # 1) 관측 데이터 전처리 (학습 파이프라인과 일치시킴)
                        pc_raw = obs['main_pointcloud'][-1]
                        pc = pc_preprocessor.process(pc_raw)
                        coords = np.ascontiguousarray(pc[:, :3] / voxel_size, dtype=np.int32)
                        feats = pc.astype(np.float32)
                        coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])
                        cloud_data = ME.SparseTensor(
                            features=feats_batch,
                            coordinates=coords_batch,
                            device=device
                        )

                        # 2) RTC: 관측 전달 + 액션 청크 조회
                        rtc_mgr.set_obs({'cloud': cloud_data})
                        seq10, ts = rtc_mgr.get_action_sequence()
                        if seq10 is None:
                            precise_wait(t_cycle_end)
                            iter_idx += 1
                            continue

                        # 3) 10D → pos, rot6d, grip
                        pred = seq10  # (L, 10) numpy
                        if pred.ndim == 3:
                            pred = pred.squeeze(0)

                        pos   = pred[:, :3]
                        rot6d = pred[:, 3:9]
                        # (권장) 그리퍼 이진화: RTC 안정성 및 하드웨어 호환
                        grip  = pred[:, 9:]

                        xyz_rot6d = np.concatenate([pos, rot6d], axis=-1)

                        # 4) 회전 변환 및 좌표계 역변환
                        xyz_euler_base_frame = xyz_rot_transform(
                            xyz_rot6d,
                            from_rep="rotation_6d",
                            to_rep="euler_angles",
                            to_convention="ZYX"
                        )
                        xyz_euler_robot_frame = revert_action_transformation(xyz_euler_base_frame, robot_to_base)

                        action_sequence_7d = np.concatenate([xyz_euler_robot_frame, grip], axis=-1)

                        # 5) 실행 (RTC 매니저가 만든 ts 그대로 사용)
                        env.exec_actions(actions=action_sequence_7d, timestamps=ts)

                        # 6) 60초 제한
                        if mono_time.now_ms() - test_start > 60000:
                            env.end_episode()
                            is_evaluating = False
                else:
                    # ===== 사람 제어 (Human Control) =====
                    target_pose = teleop.get_motion_state()
                    env.exec_actions(actions=[target_pose], timestamps=[mono_time.now_s() + dt])

                # ★ 공통 타이밍 처리: if/else 밖에서 수행 (버그 방지)
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()
