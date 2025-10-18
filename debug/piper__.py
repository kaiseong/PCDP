from piper_sdk import *
import time

def main():
    piper_slave = C_PiperInterface_V2("can_slave")
    piper_master = C_PiperInterface_V2("can_master")
    piper_slave.ConnectPort()
    piper_master.ConnectPort()
    
    piper_slave.EnableArm(7)
    enable_flag = False
    elapsed_time_flag = False
    timeout = 5  # 5초의 타임아웃 설정
    start_time = time.time()
    
    # 모터가 준비될 때까지 대기
    while not enable_flag:
        elapsed_time = time.time() - start_time
        enable_flag = piper_slave.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                piper_slave.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                piper_slave.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                piper_slave.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                piper_slave.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                piper_slave.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        piper_slave.EnableArm(7)  # 로봇 팔 활성화
        piper_slave.GripperCtrl(0, 00, 0x01, 0)  # 그리퍼 초기화
        
        if elapsed_time > timeout:  # 타임아웃 초과 시 종료
            print("시간초과")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
    if elapsed_time_flag:
        print("프로그램을 종료합니다")
        exit(0)
    time.sleep(2)

    

    print("Setting slave gripper zero point...")
    piper_slave.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=0x01, set_zero=0xAE)
    time.sleep(1)
    piper_slave.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=0x01, set_zero=0)
    time.sleep(1)
    print("Setting master gripper zero point...")
    piper_master.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=0x01, set_zero=0xAE)
    time.sleep(1)
    piper_master.GripperCtrl(gripper_angle=0, gripper_effort=0, gripper_code=0x01, set_zero=0x0)
    time.sleep(1)
    print("Gripper zero points set.")
    # piper.MotionCtrl_2(0x01, 0x00, 50, 0x00)
    # piper.EndPoseCtrl(87979, -2790, 378643, -83184, 79328, -85173)
    # print(f"Joints: {piper.GetArmJointMsgs()}")
    # print(f"endpose: {piper.GetArmEndPoseMsgs()}")
    cnt=0
    while cnt <100:
        # print(piper.GetArmEndPoseMsgs())
        target = int(piper_master.GetArmGripperMsgs()[0])
        print(f"master: {target}")
        print(f"slaev: {piper_slave.GetArmGripperMsgs()[0]}")
        piper_slave.GripperCtrl(target, 1000, 0x00)
        # piper_slave.GripperCtrl(0, 00, 0x00, 0xAE)
        # piper_master.GripperCtrl(0, 000, 0x00, 0xAE)
        time.sleep(0.1)
        cnt+=1
        
    # print(piper.GetArmJointMsgs())
    time.sleep(1)
    piper_slave.DisableArm(7)
    piper_slave.DisconnectPort()

if __name__ == "__main__":
    main()
