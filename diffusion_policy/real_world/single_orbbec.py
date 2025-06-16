# single_orbbec2.py
from typing import Optional, Callable, Dict
import os
import enum
import time
import json
import numpy as np
import multiprocessing as mp
import cv2
import open3d as o3d
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager

# debug
import csv
def save_timestamp_duration_to_csv(timestamps, filename):
    """타임스탬프 배열을 CSV 파일로 저장"""
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'timestamp'])
        for i, ts in enumerate(timestamps):
            writer.writerow([i, ts])


import pyorbbecsdk as ob

from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from diffusion_policy.shared_memory.shared_ndarray import SharedNDArray
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Full, Empty
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.real_world.point_recorder import PointCloudRecorder
from diffusion_policy.real_world.async_point_recorder import AsyncPointCloudRecorder
from diffusion_policy.common.orbbec_util import frame_to_bgr_image


class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4


class SingleOrbbec(mp.Process):
    MAX_PATH_LENGTH = 4096

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        rgb_resolution=(1280, 720),
        put_fps=None,
        put_downsample=True,
        video_record_fps=30,
        get_max_k=30,
        mode="C2D",
        recording_transform: Optional[Callable[[Dict], Dict]] = None,
        video_recorder: Optional[VideoRecorder] = None,
        point_recorder: Optional[AsyncPointCloudRecorder]=None,
        verbose=False
    ):
        super().__init__()
        
        
        if put_fps is None:
            put_fps = video_record_fps

        # create ring buffer
        resolution = tuple(rgb_resolution)
        examples = dict()
        
        if mode == "D2C":
            examples['pointcloud'] = np.empty(
                shape=(921600, 6), dtype=np.float32)  # XYZ + RGB
        elif mode == "C2D":
            examples['pointcloud'] = np.empty(
                shape=(368640, 6), dtype=np.float32)
        else:
            raise RuntimeError("mode is wrong")
        
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            get_max_k=get_max_k,
            get_time_budget=0.016 ,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': 0,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'point_path': np.array('a'*self.MAX_PATH_LENGTH),
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
            mem_mgr=shm_manager,
            shape=(6,), 
            dtype=np.float64)
        intrinsics_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            video_recorder = VideoRecorder.create_h264(
                fps=video_record_fps,
                codec='h264',
                input_pix_fmt='bgr24',
                crf=18,
                thread_type='FRAME',
                thread_count=1)
        
        if point_recorder is None:
            point_recorder = AsyncPointCloudRecorder(
                compression_level=1,
                queue_size=60
            )

        # copied variables
        self.resolution = resolution
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.mode = mode
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.point_recorder = point_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array

    @staticmethod
    def get_connected_devices():
        """Get list of connected Orbbec devices"""
        if ob is None:
            return []
        
        try:
            ctx = ob.Context()
            devices = ctx.query_devices()
            device_list = []
            for i in range(devices.get_count()):
                device = devices.get_device_by_index(i)
                device_info = device.get_device_info()
                device_list.append({
                    'index': i,
                    'name': device_info.get_name(),
                    'serial': device_info.get_serial_number(),
                    'pid': device_info.get_pid(),
                    'vid': device_info.get_vid()
                })
            return device_list
        except Exception as e:
            print(f"Error querying Orbbec devices: {e}")
            return []

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time=time.time()
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)

    def get_intrinsics(self):
        assert self.ready_event.is_set()
        fx, fy, cx, cy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = cx
        mat[1,2] = cy
        return mat

    def start_recording(self, video_path: str, point_path: str, start_time: float=-1):
        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'point_path': point_path,
            'recording_start_time': start_time
        })

    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })

    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    # ========= internal API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        
        pipeline = ob.Pipeline()
        cfg = ob.Config()

        depth_res = (640, 576)
        color_res = self.resolution
        
        if self.mode=="D2C":
            align_to = ob.OBStreamType.COLOR_STREAM
        elif self.mode=="C2D":
            align_to = ob.OBStreamType.DEPTH_STREAM
        
        w, h = self.resolution
        fps = self.put_fps

        depth_profile = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)\
                        .get_video_stream_profile(*depth_res,ob.OBFormat.Y16,fps)
        color_profile = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)\
                        .get_video_stream_profile(*color_res,ob.OBFormat.RGB,fps)
        cfg.enable_stream(depth_profile)
        cfg.enable_stream(color_profile)
        pipeline.enable_frame_sync()
        pipeline.start(cfg)
        align = ob.AlignFilter(align_to_stream=align_to)
        pc_filter = ob.PointCloudFilter()
        cam_param = pipeline.get_camera_param()
        pc_filter.set_camera_param(cam_param)
        pc_filter.set_create_point_format(ob.OBFormat.RGB_POINT)
        
        try:
            if self.mode == "D2C":
                intr = cam_param.rgb_intrinsic
            elif self.mode == "C2D":
                intr = cam_param.depth_intrinsic

            self.intrinsics_array.get()[0] = intr.fx
            self.intrinsics_array.get()[1] = intr.fy
            self.intrinsics_array.get()[2] = intr.cx
            self.intrinsics_array.get()[3] = intr.cy
            self.intrinsics_array.get()[4] = intr.height
            self.intrinsics_array.get()[5] = intr.width

            if self.verbose:
                print(f"[SingleOrbbec] Initialized with resolution {w}x{h} at {fps} FPS.")
                print(f"[SingleOrbbec] Main loop started.")

            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()
            
            iter_idx =0
            t_start = time.time_ns()


            # debuging
            if self.verbose:
                pre_depth_time= 0
                delay_cnt=0
                no_cnt=0
                normal_cnt=0
                checking=False

                normal_duration = np.array([], dtype=np.float32)
                record_duration = np.array([], dtype=np.float32)
                timestamp_duration = np.array([],dtype=np.float32)
                record_end = 0
            
            while not self.stop_event.is_set():
                frames = pipeline.wait_for_frames(10)
                if frames is None:
                    continue
                
                depth, color = frames.get_depth_frame(), frames.get_color_frame()
                if depth is None or color is None:
                    continue
                
                frame = align.process(frames)
                pc_filter.set_position_data_scaled(depth.get_depth_scale())
                point_cloud = pc_filter.calculate(pc_filter.process(frame))  # (N,6) float32
                receive_time = time.time()

                if self.verbose:
                    ################# user ################
                    record_start=time.time_ns()
                    if checking and self.verbose:
                        if (depth.get_timestamp() - pre_depth_time) > 70:
                            delay_cnt += 1
                        if (depth.get_timestamp() == pre_depth_time):
                            no_cnt += 1
                        normal_cnt += 1
                        pre_depth_time = depth.get_timestamp()
                        record_duration=np.append(record_duration, (record_start-record_end)/1e6)
                        timestamp_duration= np.append(timestamp_duration, receive_time)
                    record_end =record_start
                    #######################################

                if point_cloud is not None:
                    points_data = np.asarray(point_cloud, dtype=np.float32)
                    if len(points_data.shape) == 1:
                        points_data = points_data.reshape(-1, 6)
                else:
                    if self.mode == "D2C":
                        points_data = np.zeros((921600, 6), dtype=np.float32)
                    elif self.mode == "C2D":
                        points_data = np.zeros((368640, 6), dtype=np.float32)

                data = dict()
                data['camera_receive_timestamp'] = receive_time
                data['camera_capture_timestamp'] = depth.get_timestamp() / 1000.0
                 
                data['pointcloud']=points_data

                color_bgr = np.asarray(frame_to_bgr_image(color), dtype=np.uint8)

                # data['pointcloud] shape is (368640, 6) or (921600, 6)
                put_data = data
                
                if self.put_downsample:
                    
                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt =1/self.put_fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )
                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        put_data['timestamp'] = receive_time
                        try:
                            self.ring_buffer.put(put_data, wait=False)
                        except TimeoutError:
                            if self.verbose:
                                print("[SingleOrbbec] Ring buffer full, dropping frame.")
                            break
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    try:
                        self.ring_buffer.put(put_data, wait=False)
                    except TimeoutError:
                            if self.verbose:
                                print("[SingleOrbbec] Ring buffer full, dropping frame.")
                            continue
                
                if iter_idx ==0:
                    self.ready_event.set()
                
                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(color_bgr, frame_time=receive_time)
                if self.point_recorder.is_ready():
                    self.point_recorder.write_frame(points_data, frame_time=receive_time)
                
                

                if self.verbose:
                    t_end = time.time_ns()
                    if iter_idx>2:
                        normal_duration=np.append(normal_duration, (t_end - t_start)/1e6)
                    t_start=t_end
                
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd =0
                
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path']).rstrip('\x00')
                        point_path = str(command['point_path']).rstrip('\x00')
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        # self.video_recorder.start(video_path, start_time=start_time)
                        self.point_recorder.start(point_path, start_time=start_time)
                        if self.verbose:
                            checking=True
                    elif cmd == Command.STOP_RECORDING.value:
                        # self.video_recorder.stop()
                        self.point_recorder.stop()
                        if self.verbose:
                            checking=False
                            print(f"delay_cnt: {delay_cnt}, normal_cnt: {normal_cnt}, no_cnt: {no_cnt}")
                            print("normal_duration")
                            print(f"    mean: {normal_duration.mean()}ms\n"
                                  f"    max: {normal_duration.max()}ms\n"
                                  f"    min: {normal_duration.min()}ms\n"
                                  f"    std: {normal_duration.std()}ms")
                            print("record_duration")
                            print(f"    mean: {record_duration.mean()}ms\n"
                                  f"    max: {record_duration.max()}ms\n"
                                  f"    min: {record_duration.min()}ms\n"
                                  f"    std: {record_duration.std()}ms")
                            print(f"recorder_frame_count: {self.point_recorder.frame_count}")
                            for i in range(10):
                                print(f"{i} record_timestamp: {timestamp_duration[i]}")
                            print(f"last record time: {timestamp_duration[-1]}")
                            # debug
                            timestamp=time.time()
                            csv_filename = f"timestamp_duration_{timestamp}.csv"
                            save_timestamp_duration_to_csv(timestamp_duration, csv_filename)
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                iter_idx += 1
                

                
        except Exception as e:
            print(f"[SingleOrbbec] Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.video_recorder.stop()
            self.point_recorder.stop()
            pipeline.stop()
            self.ready_event.set()
            if self.verbose:
                print("[SingleOrbbec] Main loop ended, resources cleaned up.")




