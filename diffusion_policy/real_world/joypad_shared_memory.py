# joypad_shared_memory.py
import multiprocessing as mp
import numpy as np
import time
import pygame
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

class JoypadSpacemouse(mp.Process):
    def __init__(self,
                shm_manager,
                get_max_k=30,
                frequency=200,
                max_value=500,
                deadzone=(0.02,0.02,0,0,0,0),
                dtype=np.float32,
                n_buttons=12):
        """
        Continuously listen to gamepad/joystick events
        and update the latest state to mimic SpaceMouse behavior.

        max_value: Maximum value for axis input normalization
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) equivalent space mouse orientation
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()
        
        # 멤버 변수들
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        
        # SpaceMouse 용 ring buffer와 동일 구조
        example = {
            'motion_event': np.zeros((7,), dtype=np.int64),  # [Tx, Ty, Tz, Rx, Ry, Rz, period]
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # 프로세스 제어용
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        # 좌표계 변환 행렬(스페이스마우스 코드와 동일)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

    def get_motion_state(self):
        state = self.ring_buffer.get()
        # [Tx, Ty, Tz, Rx, Ry, Rz]
        arr = np.array(state['motion_event'][:6], dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < arr) & (arr < self.deadzone)
        arr[is_dead] = 0
        return arr

    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        raw = self.get_motion_state()
        tf_state = np.zeros_like(raw)
        tf_state[:3] = self.tx_zup_spnav @ raw[:3]
        tf_state[3:] = self.tx_zup_spnav @ raw[3:]
        return tf_state

    def get_button_state(self):
        return self.ring_buffer.get()['button_state']

    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def run(self):
        # Pygame 초기화
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        # 에러 처리
        if joystick_count == 0:
            print("No Joystick Found!")
            self.ready_event.set()
            return
        js = pygame.joystick.Joystick(0)
        js.init()
        
        print(f"Joystick: {js.get_name()}")
        print(f"Axes: {js.get_numaxes()}, Buttons: {js.get_numbuttons()}")

        # 초기 값 (Tx, Ty, Tz, Rx, Ry, Rz, period=0)
        motion_event = np.zeros((7,), dtype=np.int64)
        button_state = np.zeros((self.n_buttons,), dtype=bool)

        # 초기 레코드 전송(서버 쪽에서 바로 읽도록)
        self.ring_buffer.put({
            'motion_event': motion_event,
            'button_state': button_state,
            'receive_timestamp': time.time()
        })
        self.ready_event.set()

        # main loop
        freq_dt = 1.0 / self.frequency
        last_time = time.time()
        while not self.stop_event.is_set():
            now = time.time()

            # pygame 이벤트 폴링
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # 축 매핑 완성 - 6자유도 매핑
                    # 일반적인 게임패드 매핑 (조정 필요할 수 있음):
                    # 왼쪽 스틱: Tx, Ty (좌우, 상하)
                    # 오른쪽 스틱: Tx, Ty 
                    axis = event.axis
                    val = event.value
                    
                    # 일반 게임패드 구성에 맞게 조정 (조이스틱에 따라 달라질 수 있음)
                    if axis == 0:  # 왼쪽 스틱 좌우
                        motion_event[0] = int(-val * self.max_value/4)  # Tx
                    elif axis == 1:  # 왼쪽 스틱 상하
                        motion_event[2] = int(val * self.max_value/4)  # Ty (반전)
                    elif axis == 2:  # 오른쪽 스틱 좌우
                        motion_event[0] = int(-val * self.max_value)  # Tx
                    elif axis == 3:  # 오른쪽 스틱 상하
                        motion_event[2] = int(val * self.max_value)  # Ty (반전)
                
                elif event.type == pygame.JOYBUTTONDOWN:
                    # 버튼 매핑 - demo_real_robot.py에서 사용하는 좌/우 버튼 기능
                    # 사용 중인 조이스틱에 맞게 조정 필요할 수 있음
                    if event.button < self.n_buttons:
                        button_state[event.button] = True
                
                elif event.type == pygame.JOYBUTTONUP:
                    if event.button < self.n_buttons:
                        button_state[event.button] = False
            
            # period 계산 (밀리초 단위) - SpaceMouse와 유사하게
            period = int((now - last_time) * 1000)
            motion_event[6] = period
            last_time = now
            
            # ring buffer에 put
            self.ring_buffer.put({
                'motion_event': motion_event.copy(),
                'button_state': button_state.copy(),
                'receive_timestamp': now
            })

            sleep_time = freq_dt - (time.time() - now)
            if sleep_time > 0:
                time.sleep(sleep_time)

        pygame.joystick.quit()
        pygame.quit()