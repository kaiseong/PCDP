import numpy as np
from typing import Union
import numbers

class NoPoseTrajectoryInterpolator:
    """
    A dummy interpolator that satisfies the PoseTrajectoryInterpolator interface
    but does not perform any interpolation. It simply stores and returns the
    most recently scheduled waypoint. This is used for debugging and to mimic
    a direct command pass-through.
    """
    def __init__(self, times: np.ndarray, poses: np.ndarray, **kwargs):
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)
        
        # Store only the last waypoint
        self._times = times[-1:]
        self._poses = poses[-1:]

    @property
    def poses(self) -> np.ndarray:
        return self._poses

    @property
    def times(self) -> np.ndarray:
        return self._times

    def schedule_waypoint(self, pose, time, **kwargs) -> "NoPoseTrajectoryInterpolator":
        """
        Discards the old waypoint and creates a new interpolator
        with the new waypoint.
        """
        new_times = np.array([time])
        new_poses = np.array([pose])
        return NoPoseTrajectoryInterpolator(times=new_times, poses=new_poses)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        Ignores the time t and always returns the single stored pose.
        """
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
        
        pose = self._poses[0]

        if not is_single:
            t_len = len(t)
            pose = np.tile(pose, (t_len, 1))

        return pose
    
    def drive_to_waypoint(self, pose, time, curr_time, **kwargs) -> "NoPoseTrajectoryInterpolator":
        return self.schedule_waypoint(pose, time)

    def trim(self, start_t: float, end_t: float) -> "NoPoseTrajectoryInterpolator":
        return self
