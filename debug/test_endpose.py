from piper_sdk import *
import numpy as np
from pcdp.real_world.pinocchio_ik_controller import PinocchioIKController
from pcdp.real_world.trac_ik_controller import TracIKController
import time
import scipy.spatial.transform as st
import pinocchio as pin
import pcdp.common.mono_time as mono_time


if __name__ == "__main__":
    
    piper_master = C_PiperInterface_V2("can_master")
    piper_slave = C_PiperInterface_V2("can_slave")
    piper_master.ConnectPort()
    piper_slave.ConnectPort()
    duration = np.array([])
    try:

        piper_slave.EnableArm(7)
        print("Waiting for slave arm to enable...")
        time.sleep(2)
        
        piper_slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper_slave.JointCtrl([0, 0, 0, 0, 0, 0])
        print("Moving to initial joint state. Waiting for 2 seconds...")
        time.sleep(2)
        cnt=0
        tick=0
        
        t_du =None
        while True:
            start_time = mono_time.now_s()
            # [mm, mm, mm, deg, deg, deg]
            if tick % 20 == 0:
                EndPose = np.asarray(piper_master.GetArmEndPoseMsgs())
            
            command_pose = (EndPose *1e3).astype(int).tolist()
            

            t1=mono_time.now_ms()
            duration = np.append(duration, mono_time.now_ms()-t1)
            piper_slave.MotionCtrl_2(0x01, 0x02, 100, 0x00)
            piper_slave.EndPoseCtrl(command_pose)
            print(command_pose)
            cnt+=1
            
            elapsed = mono_time.now_s() - start_time
            sleep_time = max(0, 0.005 - elapsed)
            if sleep_time >0:
                time.sleep(sleep_time)
                
            tick+=1
            
    finally:
        duration = duration[1:]
        print(f"mean: {duration.mean()}\n \
                max: {duration.max()}\n \
                min: {duration.min()}")
        piper_slave.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        piper_slave.JointCtrl([0, 0, 0, 0, 0, 0])
        time.sleep(6)
        piper_slave.DisableArm(7)
        piper_slave.DisconnectPort()
        piper_master.DisconnectPort()
        print("Disconnected.")
    






        


