from piper_sdk import *
import time

def main():
    piper = C_PiperInterface_V2("can_slave")
    piper.ConnectPort()
    # piper.EnableArm(7)
    time.sleep(3)
    # piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    # piper.EndPoseCtrl(87979, -2790, 378643, -83184, 79328, -85173)
    # print(f"Joints: {piper.GetArmJointMsgs()}")
    # print(f"endpose: {piper.GetArmEndPoseMsgs()}")
    time.sleep(5)
    cnt=0
    while cnt<500:
        print(piper.GetArmEndPoseMsgs())
        time.sleep(0.1)
        
    # print(piper.GetArmJointMsgs())
    time.sleep(3)
    piper.DisconnectPort()

if __name__ == "__main__":
    main()
