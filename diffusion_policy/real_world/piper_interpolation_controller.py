# PiperInterpolationController.py
import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from typing import (Optional,)
from piper_sdk import *
from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

class Command(enum.Enum):
    """Command enum for the interpolation controller."""
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class PiperInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """
    
    def __init__(self,
            shm_manager: SharedMemoryManager,
            frequency = 200,
            lookahead_time=0.1,
            gain = 300,
            max_pos_speed=0.25,
            max_rot_speed=0.16,
            launch_timeout=3,
            payload_mass=None,
            joints_init=None,
            soft_real_time=False,
            debug=False,
            receive_keys=None,
            get_max_k=128,
            ):
        """
        frequency: piper 200 Hz
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: rad/s
        max_rot_speed: rad/s
        payload_mass: float
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 0 < max_pos_speed and 0 < max_rot_speed

        if payload_mass is not None:
            assert 0 <= payload_mass <= 1.5
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="PiperInterpolationController")
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.payload_mass = payload_mass
        self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.verbose = debug
        
        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
        }

        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ArmEndPoseMsgs',
                'ArmJointMsgs',
                'ArmGripperMsgs',
                'TargetEndPose',
            ]
        shape_map = {
            'ArmEndPoseMsgs': (6,),   # X Y Z RX RY RZ
            'ArmJointMsgs':   (6,),   # j1~j6
            'ArmGripperMsgs': (3,),   # angle effort status
            'TargetEndPose': (6,)
        }
        example = {k: np.zeros(shape_map[k], dtype=np.float64) for k in receive_keys}
        example['robot_receive_timestamp'] = time.time()

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
    
    # =========== launch method ============ 
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[PiperPositionalController] Controller process spawned at {self.pid}")
        
    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # =========== unit convert method ============
    def _sdk_pose_to_vec(self, raw):
        # raw is [mm] and [deg]
        v = np.asarray(raw, dtype=float)
        v[:3] *= 1e-3
        v[3:] = np.deg2rad(v[3:])
        return v

    def _vec_to_sdk_pose(self, vec):
        # vec is [m] and [rad]
        pose_int = np.empty(6, dtype=int)
        pose_int[:3] = np.round(vec[:3] * 1e6).astype(int)
        pose_int[3:] = np.round(np.rad2deg(vec[3:]) * 1e3).astype(int)
        return pose_int.tolist()

    # =========== context method ============
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, etc_val, exc_tb):
        self.stop()
    
    # =========== command method ============
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert (duration >= (1/self.frequency))
        pose = np.asarray(pose, dtype=float)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': float(duration),
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.asarray(pose,dtype=float)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': float(target_time)
        }   
        self.input_queue.put(message)

    # =========== receive APIs ============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # =========== main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
        
        # start rtde
        try:
            piper = C_PiperInterface_V2("can0")
            piper.ConnectPort()
        except Exception as e:
            print(f"[PiperPositionalController] Failed to connect to Piper: {e}")
            self.ready_event.set()
            return
        piper.EnableArm(7)

        enable_flag = False
        timeout = 5
        start_time = time.time()
        elapsed_time_flag = False
        while not (enable_flag):
            elapsed_time = time.time() - start_time
            print("--------------------")
            enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
            print("상태 활성화:",enable_flag)
            piper.EnableArm(7)
            piper.GripperCtrl(0,5000,0x01, 0)
            print("--------------------")
            if elapsed_time > timeout:
                print("시간초과....")
                elapsed_time_flag = True
                enable_flag = True
                break
            time.sleep(1)
            pass
        if(elapsed_time_flag):
            print("프로그램 timeout으로 종료합니다")
            exit(0)

        try:
            if self.verbose:
                print(f"[PiperPoisionalController] Connect to robot: {piper.GetCanFps()} Hz")

            #set parameters
            if self.payload_mass is not None:
                if self.payload_mass <= 0.75:
                    piper.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0x00, 0x01)
                elif self.payload_mass >= 1.35:
                    piper.ArmParamEnquiryAndConfig(0x00, 0x00, 0x00, 0x00, 0x02)
            
            # init pose
            if self.joints_init is not None:
                piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
                tgt = (np.rad2deg(self.joints_init)*1e3).astype(int)
                piper.JointCtrl(tgt)
            
            # main loop
            dt = 1. / self.frequency
            curr_pose = self._sdk_pose_to_vec(piper.GetArmEndPoseMsgs())
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            iter_idx = 0
            keep_running=True
            
            while keep_running:
                # start control iteration
                loop_start=time.perf_counter()

                # send command to robot
                t_now = time.monotonic()
                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                pose_command = pose_interp(t_now) 
                target_pose_vec = pose_command.copy()
                vel = max(1, int(self.max_rot_speed / 1.57 * 100))
                piper.MotionCtrl_2(0x01, 0x00, vel, 0x00)
                piper.EndPoseCtrl(self._vec_to_sdk_pose(pose_command))
                # print(f"command: {self._vec_to_sdk_pose(pose_command)}")
                
                # update robot state
                state = dict()
                for k in self.receive_keys:
                    if k == 'TargetEndPose':
                        continue
                    raw = getattr(piper, 'Get'+k)()
                    if k == 'ArmEndPoseMsgs':
                        state[k] = self._sdk_pose_to_vec(raw)
                    else:
                        state[k] = np.asarray(raw)
                
                state["ArmJointMsgs"] = np.deg2rad(state["ArmJointMsgs"])
                state["TargetEndPose"] = target_pose_vec
                state["robot_receive_timestamp"] = time.time()
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                
                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        # stop immediately, ignore later commands
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose = target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time=t_insert
                        if self.verbose:
                            if iter_idx % 100==0:
                                print("[PiperPositionalController] New pose target: {} duration: {}s".format(
                                    target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time=target_time
                    else:
                        keep_running = False
                        break
                if iter_idx==0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[PiperPositionalController] Actual frequency {1/(time.perf_counter() - loop_start)}")
                spent = time.perf_counter() - loop_start
                if spent < dt:
                    time.sleep(dt - spent)
        
        finally:
            # manditory cleanup
            # decelerate
            piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
            piper.JointCtrl([0, 100, 0, 0, 0, 0])
            time.sleep(5)
            piper.DisableArm(7)
            time.sleep(1)

            piper.DisconnectPort()
            self.ready_event.set()

            if self.verbose:
                print(f"[PiperPositionalController] Disconnect from robot")