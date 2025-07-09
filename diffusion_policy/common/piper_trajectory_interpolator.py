# piper_trajectory_interpolator.py (Simple Logic, Drop-in Replacement Version)

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import time
import numbers
from typing import Union

class PoseTrajectoryInterpolator:
    """
    기존 PoseTrajectoryInterpolator와 호환성을 유지하면서,
    단순 키프레임 보간 로직으로 동작하는 새로운 Interpolator.
    
    - 'schedule_waypoint'로 새로운 목표가 주어지면, 현재 지점과 목표 지점 사이를
      정해진 시간 동안 부드럽게 보간합니다.
    - max_pos_speed, max_rot_speed 파라미터는 호환성을 위해 받지만, 내부 로직에서는 사용하지 않습니다.
    """

    def __init__(self, times: np.ndarray, poses: np.ndarray, logger=None):
        """
        기존과 동일한 형식으로 초기화하지만, 내부적으로는 단순한 상태만 설정합니다.
        가장 마지막의 Pose와 Time을 초기 상태로 사용합니다.
        """
        # 가장 마지막 Pose를 현재 상태로 초기화
        initial_pose = np.array(poses[-1])
        initial_time = times[-1]

        self.start_pose = initial_pose.copy()
        self.target_pose = initial_pose.copy()
        self.start_time = initial_time
        self.end_time = initial_time
        self.logger = logger

    @property
    def times(self) -> np.ndarray:
        """호환성을 위한 속성. 현재 경로의 시작과 끝 시간을 반환합니다."""
        return np.array([self.start_time, self.end_time])

    @property
    def poses(self) -> np.ndarray:
        """호환성을 위한 속성. 현재 경로의 시작과 끝 Pose를 반환합니다."""
        return np.array([self.start_pose, self.target_pose])

    def schedule_waypoint(self,
                          pose: np.ndarray,
                          time: float,
                          curr_time: float,
                          # 아래 파라미터들은 호환성을 위해 존재하지만 사용되지 않음
                          max_pos_speed: float = np.inf,
                          max_rot_speed: float = np.inf,
                          last_waypoint_time: float = None
                          ) -> "PoseTrajectoryInterpolator":
        """
        새로운 목표 Pose를 설정하여 경로를 업데이트합니다.
        기존의 복잡한 속도 기반 시간 계산 대신, 주어진 시간을 그대로 사용합니다.
        """
        # 현재 보간이 진행 중인 위치를 새로운 경로의 시작점으로 설정
        current_interpolated_pose = self(curr_time)
        
        self.start_pose = current_interpolated_pose
        self.start_time = curr_time
        
        self.target_pose = np.array(pose)
        self.end_time = float(time)
        
        # end_time이 start_time보다 빠른 경우, 최소한의 시간을 보장 (예: 1ms)
        if self.end_time <= self.start_time:
            self.end_time = self.start_time + 0.001

        # 기존 API와 같이 객체 자신을 반환하여, pose_interp = pose_interp.schedule_waypoint(...) 구문이 동작하도록 함
        return self

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        특정 시간 t에 해당하는 보간된 Pose를 계산하여 반환합니다.
        """
        is_single = isinstance(t, numbers.Number)
        t_arr = np.array([t]) if is_single else np.array(t)
        
        duration = self.end_time - self.start_time
        
        # 시간이 0이거나 매우 작은 경우, 목표 지점을 바로 반환
        if duration <= 1e-6:
            # 입력 형태에 맞춰서 반환
            if is_single:
                return self.target_pose
            else:
                return np.tile(self.target_pose, (len(t_arr), 1))

        # 보간 계수 (alpha) 계산, 0.0 ~ 1.0
        alpha = (t_arr - self.start_time) / duration
        alpha = np.clip(alpha, 0.0, 1.0)

        # 각 alpha 값에 대해 보간 수행
        interpolated_poses = np.array([self._interpolate_pose(self.start_pose, self.target_pose, a) for a in alpha])
        
        return interpolated_poses[0] if is_single else interpolated_poses

    def _interpolate_pose(self, p1: np.ndarray, p2: np.ndarray, alpha: float) -> np.ndarray:
        """ 두 Pose 사이를 보간하는 내부 헬퍼 함수 (위치: Lerp, 회전: Slerp) """
        # 위치 선형 보간
        interp_pos = p1[:3] * (1 - alpha) + p2[:3] * alpha
        
        # 회전 구면 선형 보간 (Slerp)
        key_rotations = Rotation.from_rotvec([p1[3:], p2[3:]])
        slerp = Slerp([0, 1], key_rotations)
        interp_rot = slerp(alpha)
        interp_rot_vec = interp_rot.as_rotvec()
        
        return np.concatenate((interp_pos, interp_rot_vec))

    # --- 아래는 호환성을 위한 더미(Dummy) 또는 단순화된 메서드 ---
    def drive_to_waypoint(self,
                          pose,
                          time,
                          curr_time,
                          max_pos_speed=np.inf,
                          max_rot_speed=np.inf
                          ) -> "PoseTrajectoryInterpolator":
        """ 'schedule_waypoint'와 동일하게 동작하도록 단순화 """
        return self.schedule_waypoint(pose, time, curr_time)
        
    def trim(self, start_t: float, end_t: float) -> "PoseTrajectoryInterpolator":
        """
        호환성을 위한 메서드. 이 로직에서는 trim이 큰 의미가 없으므로
        단순히 현재 상태를 유지하며 자신을 반환합니다.
        """
        # 복잡한 trim 로직 대신, 요청된 시간 범위의 시작과 끝 Pose로 새 객체를 만들어 반환할 수 있으나,
        # 현재 제어 루프에서는 사용되지 않으므로 가장 간단하게 처리.
        # print("Warning: trim() is called but not fully implemented in this simplified interpolator.")
        return self