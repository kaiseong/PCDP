from piper_sdk import *
import time

def main():
    piper = C_PiperInterface_V2("can_slave", start_sdk_joint_limit=True)
    piper.ConnectPort()
    piper.SetSDKJointLimitParam('j4', -1.7453, 1.7977)
    piper.SetSDKJointLimitParam('j5', -1.3265, 1.2741)
    piper.SetSDKJointLimitParam('j6', -2.0071, 2.2166)
    time.sleep(0.5)
    print(piper.GetSDKJointLimitParam('j4'))
    print(piper.GetSDKJointLimitParam('j5'))
    print(piper.GetSDKJointLimitParam('j6'))
    piper.EnableArm(7)

    time.sleep(3)

    piper.DisableArm(7)
    piper.DisconnectPort()



if __name__=="__main__":
    main()