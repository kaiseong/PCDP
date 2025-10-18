#!/usr/bin/env python3
# -*-coding:utf8-*-

"""
해당 파일은 PiPER 컨트롤러의 펌웨어 버전을 확인합니다.
추후 PiPER 문제 발생 시, 펌웨어 버전 정보도 전달주시면 더욱 빠른 기술지원이 가능합니다.
"""

from typing import Optional
import time
from piper_sdk import *

if __name__ == "__main__":
    piper = C_PiperInterface_V2("can_slave")
    piper.ConnectPort()
    time.sleep(0.025) # 펌웨어 피드백 프레임을 읽는데 시간이 걸립니다. 그렇지 않으면 피드백이-0x4AF됩니다.
    print(piper.GetPiperFirmwareVersion())
    while True:
        piper.SearchPiperFirmwareVersion()
        time.sleep(0.025)
        a= piper.GetArmGripperMsgs()
        print(a)
        
        