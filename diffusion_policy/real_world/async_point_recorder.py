import threading
from queue import Queue
import time
import numpy as np
from diffusion_policy.real_world.point_recorder import PointCloudRecorder

class AsyncPointCloudRecorder:
    def __init__(self, fps=30, compression_level=1, queue_size=60):
        
        self.recorder = PointCloudRecorder(fps=fps, compression_level=compression_level)
        self.write_queue = Queue(maxsize=queue_size)
        self.writer_thread = None
        self.stop_flag = threading.Event()
        self.recording = False
        self.last_timestamp = None
        
    def start(self, file_path, start_time=None):
        """녹화 시작"""
        self.recorder.start(file_path, start_time)
        self.recording = True
        self.stop_flag.clear()
        
        # 별도 스레드에서 저장 처리
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
    def write_frame(self, pointcloud, frame_time=None):
        """메인 스레드: 큐에 넣고 즉시 리턴 (논블로킹)"""
        if not self.recording:
            return
        
        if frame_time is not None and frame_time == self.last_timestamp:
            return
        
        try:
            # 큐가 가득 차면 오래된 프레임 버리기
            if self.write_queue.full():
                try:
                    self.write_queue.get_nowait()  # 오래된 프레임 제거
                except:
                    pass
            
            # 새 프레임 추가 (복사본 사용)
            self.write_queue.put_nowait((pointcloud.copy(), frame_time))
            self.last_timestamp = frame_time
            
        except:
            pass  # 큐 오류 시 프레임 스킵
            
    def _writer_loop(self):
        """별도 스레드: 큐에서 꺼내서 저장 (블로킹)"""
        while not self.stop_flag.is_set() or not self.write_queue.empty():
            try:
                pointcloud, frame_time = self.write_queue.get(timeout=0.1)
                self.recorder.write_frame(pointcloud, frame_time)
                self.frame_count+=1
            except:
                continue
                
    def stop(self):
        """녹화 종료"""
        self.recording = False
        self.stop_flag.set()
        
        # 남은 데이터 모두 저장될 때까지 대기
        if self.writer_thread and self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5)
            
        self.recorder.stop()
        
    def is_ready(self):
        return self.recorder.is_ready()
        
    @property
    def frame_count(self):
        return self.recorder.frame_count
