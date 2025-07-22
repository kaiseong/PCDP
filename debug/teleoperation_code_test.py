import sys
import os

# 프로젝트 루트를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
from pcdp.real_world.teleoperation_piper import TeleoperationPiper
from multiprocessing.managers import SharedMemoryManager

def main():
    with SharedMemoryManager() as shm_manager:
        with TeleoperationPiper(shm_manager=shm_manager) as sm:
            cnt = 0
            while cnt < 2000:
                try:
                    pose = sm.get_motion_state()
                    print(pose)
                    # print(f"cnt: {cnt}")
                    cnt += 1
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Error: {e}")
                    break


if __name__ == "__main__":
    main()