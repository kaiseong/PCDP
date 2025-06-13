from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.real_world.piper_interpolation_controller import PiperInterpolationController
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.real_world.point_recorder import PointCloudRecorder
from diffusion_policy.real_world.async_point_recorder import AsyncPointCloudRecorder
from diffusion_policy.real_world.single_orbbec import SingleOrbbec
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator, 
    TimestampActionAccumulator,
    align_timestamps
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

DEFAULT_OBS_KEY_MAP = {
    # robot
    'ArmEndPoseMsgs': 'robot_eef_pose',
    'ArmJointMsgs': 'robot_joint',
    'ArmGripperMsgs': 'robot_gripper',
    'TargetEndPose': 'robot_eef_target',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}

class RealEnv:
    def __init__(self, 
            # required params
            output_dir,
            # env params
            frequency=10,
            n_obs_steps=2,
            # obs
            max_obs_buffer_size=30,
            obs_key_map=DEFAULT_OBS_KEY_MAP,
            # action
            max_pos_speed=0.1,
            max_rot_speed=0.12,
            # robot
            init_joints=False,
            # video capture params
            capture_fps=30,
            # saving params
            record_raw_video=True,
            thread_per_video=2,
            video_crf=21,
            # vis params
            orbbec_mode = "C2D",
            # shared memory
            shm_manager=None
            ):
        assert frequency <= capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        orbbec_video_dir = output_dir.joinpath('orbbec_videos')
        orbbec_video_dir.mkdir(parents=True, exist_ok=True)
        orbbec_point_dir = output_dir.joinpath('orbbec_points.zarr')
        orbbec_point_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        
        recording_fps = capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        
        orbbec_video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf = video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        orbbec_point_recorder = AsyncPointCloudRecorder(
            fps=recording_fps,
            compression_level=3,
            queue_size=60
        )
        
        orbbec = SingleOrbbec(
            shm_manager=shm_manager,
            rgb_resolution = (1280, 720),
            capture_fps = capture_fps,
            put_fps = capture_fps,
            record_fps = recording_fps,
            get_max_k = max_obs_buffer_size,
            mode=orbbec_mode,
            recording_transform = None,
            video_recorder=orbbec_video_recorder,
            point_recorder=orbbec_point_recorder,
            verbose=True
        )

        cube_diag = np.linalg.norm([1,1,1])
        j_init = np.array([0.142506, 0.580776, -1.23754, 0.116763, 0.6878, 0.0]) 
        if not init_joints:
            j_init = None

        robot = PiperInterpolationController(
            shm_manager=shm_manager,
            frequency=200, # Piper frequency
            lookahead_time=0.1,
            gain=300,
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            launch_timeout=3,
            payload_mass=None,
            joints_init=j_init,
            soft_real_time=False,
            debug=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size
            )
        
        self.orbbec = orbbec
        self.robot = robot
        self.capture_fps = capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        # recording
        self.output_dir = output_dir
        self.orbbec_video_dir = orbbec_video_dir
        self.orbbec_point_dir = orbbec_point_dir
        self.replay_buffer = replay_buffer
        # temp memory buffers
        self.last_orbbec_data = None
        # recording buffers
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
    
    # ======== start-stop API =============
    @property
    def is_ready(self):
        ready = self.orbbec.is_ready and self.robot.is_ready
        return ready
    
    def start(self, wait=True):
        self.orbbec.start(wait=False)
        self.robot.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.end_episode()
        self.robot.stop(wait=False)
        self.orbbec.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.orbbec.start_wait()
        self.robot.start_wait()
    
    def stop_wait(self):
        self.robot.stop_wait()
        self.orbbec.join()
        # self.realsense.stop_wait()
        
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.frequency))
        self.last_orbbec_data = self.orbbec.get(
            k=k, 
            out=self.last_orbbec_data)
        
        
        # 200 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = self.last_orbbec_data['timestamp'][-1] 
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)


        orbbec_obs = dict()
        orbbec_timestamps = self.last_orbbec_data['timestamp']
        orbbec_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(orbbec_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            orbbec_idxs.append(this_idx)
        orbbec_obs['pointcloud'] = self.last_orbbec_data['pointcloud'][orbbec_idxs]

        # align robot obs
        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        
        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]
        
        # print("robot_obs_raw")
        # for key in robot_obs_raw.keys():
        #     print(f"    {key}: {robot_obs_raw[key].shape}")
        # print(f"    robot_timestamps: {robot_timestamps.shape}")
        # robot_times = np.asarray(robot_timestamps, dtype=np.float64)
        # if self.start_time is not None:
        #     print(robot_times - self.start_time)
        # print("robot_obs")
        # for key in robot_obs.keys():
        #     print(f"    {key}: {robot_obs[key].shape}")

        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )
        
            

            # print("orbbec_obs")
            # for key in orbbec_obs.keys():
            #     print(f"    {key}: {orbbec_obs[key].shape}")
            # print("last_orbbec_data")
            # for key in self.last_orbbec_data.keys():
            #     print(f"    {key}: {self.last_orbbec_data[key].shape}")

        # return obs
        obs_data = dict(orbbec_obs)
        obs_data.update(robot_obs)

        
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data
    
    def exec_actions(self, 
            actions: np.ndarray, 
            timestamps: np.ndarray, 
            stages: Optional[np.ndarray]=None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # convert action to pose
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        for i in range(len(new_actions)):
            self.robot.schedule_waypoint(
                pose=new_actions[i],
                target_time=new_timestamps[i]
            )
        
        # record actions
        if self.action_accumulator is not None:
            self.action_accumulator.put(
                new_actions,
                new_timestamps
            )
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                new_stages,
                new_timestamps
            )
    
    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        
        # start recording on orbbec
        orbbec_video_dir = self.orbbec_video_dir.joinpath(str(episode_id))
        orbbec_video_dir.mkdir(parents=True, exist_ok=True)
        orbbec_video_path = str(orbbec_video_dir.joinpath('orbbec_view.mp4').absolute())
        orbbec_point_dir = self.orbbec_point_dir.joinpath(str(episode_id))
        orbbec_point_dir.mkdir(parents=True, exist_ok=True)
        orbbec_point_path = str(orbbec_point_dir.joinpath())
        self.orbbec.restart_put(start_time=start_time)
        self.orbbec.start_recording(
            video_path=orbbec_video_path, 
            point_path=orbbec_point_path,
            start_time=start_time
        )

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')
        print(f"start_time: {self.start_time}")
    
    def end_episode(self):
        "Stop recording"
        assert self.is_ready
        
        # stop video recorder
        self.orbbec.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps
            print(f"robot timestamp")
            for i in range(5):
                print(f"    {i} timestamp: {obs_timestamps[i]}")
            print(f"    last timestamp: {obs_timestamps[-1]}")
            # print(f"Obs data keys: {obs_data.keys()}")

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            print(f"obs_timestamps: {obs_timestamps.shape}, "
                  f"action_timestamps: {action_timestamps.shape}, ")
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            print(f"n_steps: {n_steps}, ")
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')
            
            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.orbbec_video_dir.joinpath(str(episode_id))
        this_point_dir = self.orbbec_point_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        if this_point_dir.exists():
            shutil.rmtree(str(this_point_dir))
        print(f'Episode {episode_id} dropped!')

