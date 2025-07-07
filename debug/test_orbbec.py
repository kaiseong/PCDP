from piper_sdk import *
import time
import pytorch3d
import numpy as np

piper = C_PiperInterface_V2()
piper.ConnectPort()

try:
    while True:
        test = np.asarray(piper.GetArmJointMsgs()) * 1000
        print(test)
        print(f"read EndPose: {piper.GetArmEndPoseMsgs()}")
        print(f"read Joints: {piper.GetArmJointMsgs()}")
        print(f"read_gripper: {piper.GetArmGripperMsgs()}")
        print(f"read status: {piper.GetArmStatus()}")
        time.sleep(0.01)
finally:
    piper.DisableArm(7)
    piper.DisconnectPort()