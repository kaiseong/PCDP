# pcdp/real_world/SPEC_Processor.py
import multiprocessing as mp
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from dataclasses import dataclass
from pcdp.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from pcdp.shared_memory.shared_memory_util import ArraySpec
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor
from pcdp.common.RISE_transformation import xyz_rot_transform  # eval에서 쓰던 그대로
import pcdp.common.mono_time as mono_time
import time
from pcdp.common.precise_sleep import precise_wait

@dataclass
class SpecProcConfig:
    voxel_size: float = 0.01
    max_points: int = 90000
    feats_dim: int = 7             # [x,y,z,r,g,b,c]
    grid_origin: tuple = (-1.0,-1.0,0.0)

def quantize_to_me_coords(xyz, origin, voxel):  # 동일
    co = np.floor((xyz - np.asarray(origin, np.float32)) / voxel).astype(np.int32)
    return co

def pack_robot10_from_env_obs(obs_last, low_preproc):
    # eval_piper_SPEC.py에서 하던 것 그대로 재사용
    pose_raw = obs_last['robot_eef_pose'].astype(np.float64)         # [x,y,z,rx,ry,rz]
    grip_raw = obs_last['robot_gripper'].flatten().astype(np.float64) # [angle,...] → 1D 스칼라만 사용
    robot7 = np.concatenate([pose_raw, grip_raw[:1]], axis=0)        # [x,y,z,rx,ry,rz,g]

    robot7_base = low_preproc.TF_process(robot7[None, :]).squeeze(0) # 학습과 동일 변환(로봇→base)
    obs_9d = xyz_rot_transform(
        robot7_base[:6],
        from_rep='euler_angles', to_rep='rotation_6d', from_convention='ZYX'
    )  # [xyz(3)+rot6d(6)]
    robot10 = np.concatenate([obs_9d, robot7_base[6:7]], axis=0).astype(np.float32)
    return robot10

class SPECProcessor(mp.Process):
    def __init__(self, shm_manager: SharedMemoryManager, env, cfg: SpecProcConfig, put_fps=10):
        super().__init__(daemon=True)
        self.shm_manager = shm_manager
        self.env = env
        self.cfg = cfg
        self.stop_event = mp.Event()

        specs = [
            ArraySpec('pc7',      (cfg.max_points, cfg.feats_dim), np.float32), 
            ArraySpec('n_points', (), np.int32),
            ArraySpec('robot10',  (10,), np.float32),
            ArraySpec('timestamp',(), np.float64),
            ArraySpec('step_idx', (), np.int32),
        ]
        self.proc_rb = SharedMemoryRingBuffer(
            shm_manager=shm_manager,
            array_specs=specs, get_max_k=2, put_desired_frequency=put_fps
        )

        self.pcproc = PointCloudPreprocessor(
            enable_sampling=True, target_num_points=min(cfg.max_points, 90000),
            enable_transform=True,  # 카메라→base 변환(Extrinsics 세팅되어 있어야 함)
            enable_crop=True, enable_temporal=True, export_mode='current', occlusion_prune=False
        )
        self.low_preproc = LowDimPreprocessor()  # 로봇→base 변환

    def get_ringbuffer(self):
        return self.proc_rb

    def stop(self):
        self.stop_event.set()

    def run(self):
        step = 0
        dt = 0.001  # 10Hz
        pre_time = 0
        while not self.stop_event.is_set():
            t0 = mono_time.now_s()

            # ✅ RealEnv에서 '정렬된' 관측 묶음을 받고 마지막 프레임만 사용
            obs_seq = self.env.get_obs()    # dict of arrays (len = n_obs_steps)
            pc6  = obs_seq['main_pointcloud'][-1]          # (N,6) xyzrgb (카메라 프레임)
            ts   = float(obs_seq['timestamp'][-1])         # 정렬된 타임스탬프(카메라 기준)
            obs_last = {k: v[-1] for k, v in obs_seq.items() if isinstance(v, np.ndarray)}

            if ts <= pre_time:
                time.sleep(0.001)
                continue
                
            pre_time = ts
            # 로봇 10D 생성(학습과 동일 규약)
            robot10 = pack_robot10_from_env_obs(obs_last, self.low_preproc)

            # 포인트클라우드 전처리(→ base 프레임, crop, sampling, c 채널)
            pts7 = self.pcproc.process(pc6)                 # (M,7) [x,y,z,r,g,b,c] in base
            n = int(pts7.shape[0])
            if n == 0:
                continue
            if n > self.cfg.max_points:
                pts7 = pts7[:self.cfg.max_points]

            pc7_buf = np.zeros((self.cfg.max_points, self.cfg.feats_dim), dtype=np.float32)
            pc7_buf[:n, :] = pts7.astype(np.float32, copy=False)

            self.proc_rb.put({
                'pc7': pc7_buf,
                'n_points': np.int32(n),
                'robot10': robot10,
                'timestamp': np.float64(ts),
                'step_idx': np.int32(step)
            }, wait=False)
            
            step += 1

            # 10Hz 페이싱(RealEnv의 주기와 맞춤)
            sleep = max(0.0, dt - (mono_time.now_s() - t0))
            precise_wait(mono_time.now_s() + sleep)
