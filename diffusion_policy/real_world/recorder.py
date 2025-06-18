# recorder.py
from typing import Optional, Callable, Dict
import pathlib
import os
import time
import threading
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import zarr
import numcodecs
from collections import deque

from diffusion_policy.common.timestamp_accumulator import align_timestamps
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.single_orbbec import SingleOrbbec
from diffusion_policy.real_world.piper_interpolation_controller import PiperInterpolationController
from diffusion_policy.common.replay_buffer import ReplayBuffer
import diffusion_policy.common.mono_time as mono_time
from termcolor import cprint

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

class RecorderCommand:
    START = 0
    STOP = 1
    DROP = 2


class Recorder(mp.Process):
    """
    Unified data recorder that replaces async_point_recorder and point_recorder.
    Reads from both orbbec and robot ring_buffers, synchronizes timestamps,
    and saves paired data to zarr format.
    """
    
    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        orbbec: SingleOrbbec,
        robot: PiperInterpolationController,
        output_dir: str,
        compression_level: int = 2,
        frequency: float = 100.0,  # 10ms polling
        max_buffer_size: int = 30
    ):
        super().__init__(name="Recorder")
        
        self.shm_manager = shm_manager
        self.orbbec = orbbec
        self.robot = robot
        self.output_dir = pathlib.Path(output_dir)
        self.compression_level = compression_level
        self.polling_dt = 1.0 / frequency  # 0.01s = 10ms
        self.max_buffer_size = max_buffer_size
        
        # Recording state
        self.recording = False
        self.start_time = None
        self.episode_id = None
        
        # Zarr storage
        obs_zarr_path = str(self.output_dir.joinpath('obs_replay_buffer.zarr').absolute())
        action_zarr_path = str(self.output_dir.joinpath('action_replay_buffer.zarr').absolute())
        self.obs_replay_buffer = ReplayBuffer.create_from_path(zarr_path=obs_zarr_path, mode ='a')
        self.action_replay_buffer = ReplayBuffer.create_from_path(zarr_path=action_zarr_path, mode ='a')

        # Data buffers for synchronization
        self.robot_buffer = deque(maxlen=max_buffer_size)
        self.action_buffer = deque(maxlen=max_buffer_size)
        self.orbbec_buffer = deque(maxlen=30)


        # Frame counters
        self.last_orbbec_timestamp = -1
        self.last_robot_timestamp = -1
        
        # Control events
        self.stop_event = mp.Event()
        self.recording_event = mp.Event()
        
        # Command queue for start/stop recording
        command_examples = {
            'cmd': RecorderCommand.START,  # 'start' or 'stop' 'drop'
            'start_time': 0.0,
            'episode_id': 0
        }
        self.command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=command_examples,
            buffer_size=64
        )
            
    def start_recording(self, start_time: float, episode_id: int):
        """Start recording from the specified start_time"""
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.START,
                'start_time': start_time,
                'episode_id': episode_id
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])
    
    def stop_recording(self):
        """Stop current recording"""
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.STOP,
                'start_time': 0.0,
                'episode_id': 0
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])
    
    def drop_episode(self):
        try:
            self.command_queue.put({
                'cmd': RecorderCommand.DROP,
                'start_time': 0.0,
                'episode_id': 0
            })
        except Full:
            cprint("[Recorder] Warning: Command queue full", "cyan", attrs=["bold"])

    def add_action(self, action: np.ndarray, timestamp: float, stage: int =0):
        """Add action to the action buffer"""
        if self.recording:
            action_data = {
                'action': action,
                'timestamp': timestamp,
                'stage': stage
            }
            self.action_buffer.append(action_data)
    
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
    
    def start_wait(self):
        time.sleep(0.1)
    
    def stop(self, wait=True):
        self.stop_process()
        if wait:
            self.join()


    def stop_process(self):
        self.stop_event.set()

    def _read_orbbec_data(self) -> bool: 
        try:
            data=self.orbbec.get()
            if data is not None and 'timestamp' in data:
                timestamp = data['timestamp']
                if timestamp > self.last_orbbec_timestamp:
                    self.last_orbbec_timestamp = timestamp
                    orbbec_data = {
                        'pointcloud': data['pointcloud'],
                        'timestamp': timestamp,
                        'camera_capture_timestamp': data['camera_capture_timestamp']
                    }
                    self.orbbec_buffer.append(orbbec_data)
                    return True
        except Exception as e:
            cprint(f"[Recorder] Error reading orbbec data: {e}", "cyan", attrs=["bold"])
        return False


    def _read_robot_data(self) -> bool:
        """Read latest data from robot ring buffer"""
        try:
            # Get latest data from robot
            all_robot_data = self.robot.get_all_state()
            if all_robot_data is not None and 'robot_receive_timestamp' in all_robot_data:
                timestamps = all_robot_data['robot_receive_timestamp']
                
                new_data_added = False
                for i, timestamp in enumerate(timestamps):
                    if timestamp > self.last_robot_timestamp:
                        robot_data = {
                            'robot_eef_pose': all_robot_data.get('ArmEndPoseMsgs', [np.zeros(6)])[i],
                            'robot_joint': all_robot_data.get('ArmJointMsgs', [np.zeros(6)])[i],
                            'robot_gripper': all_robot_data.get('ArmGripperMsgs', [np.zeros(3)])[i],
                            'robot_eef_target': all_robot_data.get('TargetEndPose', [np.zeros(6)])[i],
                            'robot_receive_timestamp': timestamp
                        }
                        self.robot_buffer.append(robot_data)
                        new_data_added = True
                        self.last_robot_timestamp = timestamp
                return new_data_added
        except Exception as e:
            cprint(f"[Recorder] Error reading robot data: {e}", "cyan")
        return False
    
    def _find_previous_robot_data(self, orbbec_timestamps: np.ndarray) -> np.ndarray:
        if len(self.robot_buffer) == 0:
            return np.array([])

        robot_timestamps = np.array([r['robot_receive_timestamp'] for r in self.robot_buffer])
        indices = np.searchsorted(robot_timestamps, orbbec_timestamps, side='right') - 1
        indices = np.maximum(indices, 0)  # Ensure indices are not negative
        
        return indices

    

    def _save_actions(self):
        if not self.action_buffer:
            return
        
        try:
            actions = np.array([action['action'] for action in self.action_buffer])
            timestamps = np.array([action['timestamp'] for action in self.action_buffer])
            stages = np.array([action['stage'] for action in self.action_buffer])
            action_data = {
                'action': actions,
                'timestamp': timestamps,
                'stage': stages
            }
            self.action_replay_buffer.add_episode(action_data, compressors='disk')
        except Exception as e:
            cprint(f"[Recorder] Error saving actions: {e}", "cyan", attrs=["bold"])

    def _process_new_data(self):
        processed = 0
        MAX_BATCH = 3

        
        orbbec_data_list =[]
        while processed < MAX_BATCH:
            try:
                orbbec_data = self.orbbec_buffer.popleft()
            except IndexError:
                break
            if orbbec_data['timestamp'] >= self.start_time:
                orbbec_data_list.append(orbbec_data)
                processed += 1
        
        if not orbbec_data_list:
            return
        
        orbbec_timestamps = np.array([od['timestamp'] for od in orbbec_data_list])
        robot_indices = self._find_previous_robot_data(orbbec_timestamps)

        paired_obs =[]
        for i, orbbec_data in enumerate(orbbec_data_list):
            if i < len(robot_indices):
                robot_idx = robot_indices[i]
                if robot_idx <len(self.robot_buffer):
                    robot_data = self.robot_buffer[robot_idx]
                    obs_data={
                        'pointcloud': orbbec_data['pointcloud'],
                        'robot_eef_pose': robot_data['robot_eef_pose'],
                        'robot_joint': robot_data['robot_joint'],
                        'robot_gripper': robot_data['robot_gripper'],
                        'robot_eef_target': robot_data['robot_eef_target'],
                        'align_timestamp': orbbec_data['timestamp'],
                        'robot_timestamp': robot_data['robot_receive_timestamp'],
                        'capture_timestamp': orbbec_data['camera_capture_timestamp']
                    }
                    paired_obs.append(obs_data)

        if paired_obs:
            self._save_obs_batch(paired_obs)
    
    def _save_obs_batch(self, obs_batch: list):
        if not obs_batch:
            return
        
        try:
            batch_data = {}
            for key in ['pointcloud', 'robot_eef_pose', 'robot_joint', 'robot_gripper', 'robot_eef_target',
                        'align_timestamp', 'robot_timestamp', 'capture_timestamp']:
                batch_data[key] = np.array([obs[key] for obs in obs_batch])
            
            self.obs_replay_buffer.add_episode(batch_data, compressors='disk')
            del batch_data
            obs_batch.clear()
            
        except Exception as e:
            cprint(f"[Recorder] Error saving observation batch: {e}", "cyan", attrs=["bold"])
    
    def _process_commands(self):
        """Process recording commands"""
        try:
            commands = self.command_queue.get_all()
            n_cmd = len(commands['cmd'])
            
            for i in range(n_cmd):
                cmd = commands['cmd'][i]
                
                if cmd == RecorderCommand.START:
                    if not self.recording:
                        self.start_time = commands['start_time'][i]
                        self.episode_id = commands['episode_id'][i]
                        self.recording = True
                        self.recording_event.set()
                        
                        self.orbbec_buffer.clear()
                        self.robot_buffer.clear()
                        self.action_buffer.clear()
                        cprint(f"[Recorder] Started recording episode {self.episode_id} from time {self.start_time}", 
                                "cyan", attrs=["bold"])
                
                elif cmd == RecorderCommand.STOP:
                    if self.recording:
                        self._process_new_data()
                        self._save_actions()
                        self.recording = False
                        self.recording_event.clear()
                        cprint(f"[Recorder] saved episode {self.episode_id}.", "cyan", attrs=["bold"])
                
                elif cmd == RecorderCommand.DROP:
                    self.recording = False
                    self.recording_event.clear()
                    if self.obs_replay_buffer.n_episodes > 0:
                        self.obs_replay_buffer.drop_episode()
                    if self.action_replay_buffer.n_episodes > 0:
                        self.action_replay_buffer.drop_episode()
                        
        except Empty:
            pass
        except Exception as e:
            cprint(f"[Recorder] Error processing commands: {e}", "cyan")
    

    def run(self):
        """Main recorder loop"""
        cprint(f"[Recorder] Starting recorder process", "cyan", attrs=["bold"])
        cprint(f"[Recorder] now episode: {self.obs_replay_buffer.n_episodes}", "cyan", attrs=["bold"])
        
        try:
            while not self.stop_event.is_set():
                start_time = mono_time.now_s()
                
                # Process commands
                self._process_commands()
                
                # Read data from both sources
                orbbec_updated = self._read_orbbec_data()
                robot_updated = self._read_robot_data()
                
                # If recording and we have new orbbec data
                if self.recording and orbbec_updated:
                    self._process_new_data()
                    
                # Maintain polling frequency (10ms)
                elapsed = mono_time.now_s() - start_time
                sleep_time = max(0, self.polling_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            cprint(f"[Recorder] Error in main loop: {e}", "cyan", attrs=["bold"])
            import traceback
            traceback.print_exc()
        finally:
            if self.recording:
                pass
            cprint("[Recorder] Recorder process ended", "cyan", attrs=["bold"])
    
    @property
    def is_recording(self):
        """Check if currently recording"""
        return self.recording_event.is_set()