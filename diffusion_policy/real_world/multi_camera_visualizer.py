import time
import multiprocessing as mp
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from diffusion_policy.real_world.multi_realsense import MultiRealsense
from diffusion_policy.real_world.single_orbbec import SingleOrbbec
from queue import Empty

class MultiCameraVisualizer(mp.Process):
    def __init__(self,
        realsense: MultiRealsense,
        orbbec: SingleOrbbec,
        row, col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True
        ):
        super().__init__()
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr=rgb_to_bgr
        self.realsense = realsense
        self.orbbec = orbbec
        # shared variables
        self.stop_event = mp.Event()

    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):
        cv2.setNumThreads(1)
        threadpool_limits(1)
        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None,None,-1)

        vis_data = None
        vis_img = None
        orb_vis_data=None
        orb_vis_pc=None
        while not self.stop_event.is_set():
            try:
                vis_data = self.realsense.get_vis(out=vis_data)
            except Empty:
                vis_data = None
            if self.orbbec is not None:
                try:
                    orb_vis_data = self.orbbec.get_vis(out=orb_vis_data)
                    point = orb_vis_data['pointcloud']
                    # 20ms duration
                except Empty:
                    orb_vis_data = None
            if vis_data is None:
                continue

            color = vis_data['color']
            N, H, W, C = color.shape
            assert C == 3
            oh = H * self.row
            ow = W * self.col
            if vis_img is None:
                vis_img = np.full((oh, ow, 3), 
                    fill_value=self.fill_value, dtype=np.uint8)
            for row in range(self.row):
                for col in range(self.col):
                    idx = col + row * self.col
                    h_start = H * row
                    h_end = h_start + H
                    w_start = W * col
                    w_end = w_start + W
                    if idx < N:
                        # opencv uses bgr
                        vis_img[h_start:h_end,w_start:w_end
                            ] = color[idx,:,:,channel_slice]
            cv2.imshow(self.window_name, vis_img)
            cv2.pollKey()
            
            time.sleep(1.0 / self.vis_fps)
