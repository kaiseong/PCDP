# SPEC_Processor.py
# -*- coding: utf-8 -*-
"""
센서(RGBD→PointCloud)와 로봇 state 전처리를 별도 프로세스로 분리.
공유메모리에 "항상 최신 한 장"만 유지하고, eval은 최신만 즉시 꺼내 쓴다.
"""

import time
import math
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# PCDP 모듈
from pcdp.real_world.real_data_pc_conversion import PointCloudPreprocessor, LowDimPreprocessor
from pcdp.common.RISE_transformation import xyz_rot_transform

# -----------------------------
# 공유 메모리: 최신 1샷 버퍼
# -----------------------------
@dataclass
class LatestBufferSpec:
    max_items: Tuple[int, ...]  # e.g., (92160, 7)
    dtype: Any                   # e.g., np.float32


class LatestBuffer:
    """
    프로세스간 '항상 최신 한 샷'만 공유하는 단순 버퍼.
    - writer: write(data, ts)
    - reader: read_latest() -> (np.ndarray or None, ts, seq)
    """
    def __init__(self, spec: LatestBufferSpec):
        self.spec = spec
        size = int(np.prod(spec.max_items))
        # 공유 배열/값들
        self._arr = mp.Array('f', size, lock=False)   # float32 고정
        self._n   = mp.Value('i', 0)                  # 유효 행수
        self._ts  = mp.Value('d', 0.0)                # 초 단위 타임스탬프
        self._seq = mp.Value('L', 0)                  # 갱신 시퀀스
        self._lock = mp.Lock()

    def write(self, data: np.ndarray, ts: float):
        assert data.dtype == np.float32, f"dtype must be float32, got {data.dtype}"
        assert data.ndim == len(self.spec.max_items), f"shape must be {self.spec.max_items}, got {data.shape}"
        max_n = self.spec.max_items[0]
        n = min(max_n, data.shape[0])
        flat = np.frombuffer(self._arr, dtype=np.float32, count=int(np.prod(self.spec.max_items)))
        with self._lock:
            # 앞쪽 n행만 덮어쓰기
            flat[: n * self.spec.max_items[1]] = data[:n].ravel()
            self._n.value = n
            self._ts.value = float(ts)
            self._seq.value += 1

    def read_latest(self) -> Tuple[Optional[np.ndarray], float, int]:
        """복사본 반환 (None이면 아직 데이터 없음)"""
        with self._lock:
            n = self._n.value
            ts = self._ts.value
            seq = self._seq.value
            if n <= 0:
                return None, 0.0, seq
            # 복사
            flat = np.frombuffer(self._arr, dtype=np.float32, count=int(np.prod(self.spec.max_items)))
            out = flat[: n * self.spec.max_items[1]].copy().reshape((n, self.spec.max_items[1]))
            return out, ts, seq


# -----------------------------
# 워커 루프
# -----------------------------
def _pc_worker_loop(run_flag: mp.Value,
                    in_buf: LatestBuffer,
                    out_buf: LatestBuffer,
                    pc_preprocessor_config: Dict[str, Any]):
    """
    PointCloud 전처리(좌표변환/크롭/필터/누적mem) → fused (N,7) 생성
    in_buf: (N,6) XYZRGB raw
    out_buf: (N,7) XYZRGBc fused
    """
    # 실기 성능: 필터는 eval에서 끄는 것을 권장(캐시 생성시에만 ON)
    cfg = dict(pc_preprocessor_config)
    # cfg['enable_filter'] = False  # 필요시 강제 OFF
    pcproc = PointCloudPreprocessor(**cfg)

    last_seq = -1
    while True:
        if run_flag.value == 0:
            break
        # 최신 raw를 읽어들임
        raw, ts, seq = in_buf.read_latest()
        if raw is None or seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = seq
        try:
            fused = pcproc.process(raw.astype(np.float32, copy=False))  # (N,7) or (N,6)
            # (N,6) 방어: temporal fused가 아닐 경우 c 채널을 1.0으로 채움
            if fused.shape[1] == 6:
                c = np.ones((fused.shape[0], 1), dtype=np.float32)
                fused = np.concatenate([fused.astype(np.float32, copy=False), c], axis=1)
            out_buf.write(fused.astype(np.float32, copy=False), ts)
        except Exception as e:
            # 워커는 죽지 않고 스킵
            # (로그는 여기서 찍어도 되고 조용히 넘어가도 됨)
            time.sleep(0.001)


def _state_worker_loop(run_flag: mp.Value,
                       in_buf: LatestBuffer,
                       out_buf: LatestBuffer,
                       lowdim_preproc_config: Dict[str, Any]):
    """
    로봇 state 7D(3 trans + eulerZYX + gripper) → base frame 정렬 → rot6D + grip (10D)
    in_buf: (1,7)
    out_buf: (1,10)
    """
    lowproc = LowDimPreprocessor(**lowdim_preproc_config)
    last_seq = -1
    rot6_len = 6

    while True:
        if run_flag.value == 0:
            break
        raw, ts, seq = in_buf.read_latest()
        if raw is None or seq == last_seq:
            time.sleep(0.001)
            continue
        last_seq = seq
        try:
            # raw shape: (1,7)
            raw7 = raw.reshape(-1, 7)
            base7 = lowproc.TF_process(raw7)[0]  # (7,)
            pose6 = base7[:6]
            grip1 = base7[6:7]
            rot6 = xyz_rot_transform(pose6, from_rep='euler_angles',
                                     to_rep='rotation_6d',
                                     from_convention='ZYX').reshape(rot6_len,)
            out10 = np.concatenate([pose6[:3], rot6, grip1], axis=0).astype(np.float32)[None, :]
            out_buf.write(out10, ts)
        except Exception:
            time.sleep(0.001)


# -----------------------------
# 컨트롤 클래스
# -----------------------------
class SPECProcessors:
    """
    - pc_raw_in  : (MAX_PTS, 6) XYZRGB raw
    - pc_fused_out: (MAX_PTS, 7) XYZRGBc fused
    - state_in   : (1, 7) robot 7D (x,y,z, r,p,y, grip)
    - state_out  : (1,10) robot 10D (x,y,z, rot6D, grip)
    """
    def __init__(self,
                 max_points: int,
                 pc_preprocessor_config: Dict[str, Any],
                 lowdim_preproc_config: Dict[str, Any]):
        self.max_points = int(max_points)
        self.pc_cfg = dict(pc_preprocessor_config)
        self.ld_cfg = dict(lowdim_preproc_config)

        self.pc_raw_in = LatestBuffer(LatestBufferSpec((self.max_points, 6), np.float32))
        self.pc_fused_out = LatestBuffer(LatestBufferSpec((self.max_points, 7), np.float32))
        self.state_in = LatestBuffer(LatestBufferSpec((1, 7), np.float32))
        self.state_out = LatestBuffer(LatestBufferSpec((1, 10), np.float32))

        self._run_flag = mp.Value('i', 0)
        self._pc_proc: Optional[mp.Process] = None
        self._st_proc: Optional[mp.Process] = None

    def start(self):
        if self._run_flag.value == 1:
            return
        self._run_flag.value = 1
        self._pc_proc = mp.Process(
            target=_pc_worker_loop,
            args=(self._run_flag, self.pc_raw_in, self.pc_fused_out, self.pc_cfg),
            daemon=True
        )
        self._st_proc = mp.Process(
            target=_state_worker_loop,
            args=(self._run_flag, self.state_in, self.state_out, self.ld_cfg),
            daemon=True
        )
        self._pc_proc.start()
        self._st_proc.start()

    def stop(self, timeout: float = 2.0):
        self._run_flag.value = 0
        for p in (self._pc_proc, self._st_proc):
            if p is not None:
                p.join(timeout=timeout)

    # --- Producer API (eval이 raw를 밀어넣음) ---
    def push_pc_raw(self, pc_xyzrgb: np.ndarray, ts_sec: float):
        """(N,6) float32"""
        if pc_xyzrgb is None or len(pc_xyzrgb) == 0:
            return
        self.pc_raw_in.write(pc_xyzrgb.astype(np.float32, copy=False), ts_sec)

    def push_state7d(self, robot7d: np.ndarray, ts_sec: float):
        """(7,) or (1,7) float32/float64"""
        x = robot7d.reshape(1, 7).astype(np.float32, copy=False)
        self.state_in.write(x, ts_sec)

    # --- Consumer API (eval이 최신만 읽음) ---
    def read_fused_pc(self) -> Tuple[Optional[np.ndarray], float, int]:
        """(N,7) or None, ts, seq"""
        return self.pc_fused_out.read_latest()

    def read_state10d(self) -> Tuple[Optional[np.ndarray], float, int]:
        """(1,10) or None, ts, seq"""
        return self.state_out.read_latest()
