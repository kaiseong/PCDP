from piper_sdk import *
import time
import pytorch3d
import numpy as np




piper = C_PiperInterface_V2("can_slave")
piper.ConnectPort()
# piper.EnableArm(7)
time.sleep(2)
# piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
# piper.JointCtrl([0, 0, 0, 0, 0, 0])
time.sleep(2)
try:
    while True:
        test = np.asarray(piper.GetArmJointMsgs())
        print(test)
        end_pose = piper.GetArmEndPoseMsgs()
        x= end_pose[0]
        y= end_pose[1]
        z= end_pose[2]
        roll= end_pose[3]
        pitch= end_pose[4]
        yaw= end_pose[5]
        print(f"read EndPose: {x:.4f}, {y:.4f}, {z:.4f}, {roll:.4f}, {pitch:.4f}, {yaw:.4f}")
        # print(f"read Joints: {piper.GetArmJointMsgs()}")
        # print(f"read_gripper: {piper.GetArmGripperMsgs()}")
        # print(f"read status: {piper.GetArmStatus()}")
        time.sleep(0.01)
finally:
    piper.DisableArm(7)
    piper.DisconnectPort()