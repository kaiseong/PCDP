from piper_sdk import *
import time
import numpy as np

def main():
    master_piper = C_PiperInterface_V2("can_master")
    slave_piper = C_PiperInterface_V2("can_slave")

    master_piper.ConnectPort()
    slave_piper.ConnectPort()
    slave_piper.EnableArm(7)
    time.sleep(3)
    enable_flag = False
    elapsed_time_flag = False
    timeout = 5  # 5초의 타임아웃 설정
    start_time = time.time()
    # 모터가 준비될 때까지 대기
    while not enable_flag:
        elapsed_time = time.time() - start_time
        enable_flag = slave_piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                slave_piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                slave_piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                slave_piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                slave_piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                slave_piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        slave_piper.EnableArm(7)  # 로봇 팔 활성화
        slave_piper.GripperCtrl(0, 00, 0x01, 0)  # 그리퍼 초기화
        
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
    slave_piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)

    print("Master and Slave Piper connected. Starting end pose following...")

    try:
        while True:
            # 1. Read master's end pose
            master_end_pose_raw = master_piper.GetArmEndPoseMsgs()
            # GetArmEndPoseMsgs returns X,Y,Z in 0.001mm and RX,RY,RZ in 0.001 degrees
            master_x, master_y, master_z, master_rx, master_ry, master_rz = master_end_pose_raw
            master_grip = master_piper.GetArmGripperMsgs()[0]
            # 2. Command slave to master's end pose
            # Assuming EndPoseCtrl takes the same 0.001mm and 0.001 degrees units
            slave_piper.EndPoseCtrl(int(master_x*1e3), int(master_y*1e3), int(master_z*1e3), int(master_rx*1e3), int(master_ry*1e3), int(master_rz*1e3))
            slave_piper.GripperCtrl(int(master_grip*1e3), 1000, 0x01)
            time.sleep(0.1) # Control frequency

            # 3. Read slave's end pose for error calculation
            slave_end_pose_raw = slave_piper.GetArmEndPoseMsgs()
            slave_x, slave_y, slave_z, slave_rx, slave_ry, slave_rz = slave_end_pose_raw
            slave_grip = slave_piper.GetArmGripperMsgs()[0]

            # 4. Calculate and print error
            # Convert to consistent units for error calculation (e.g., mm and degrees)
            master_pos = np.array([master_x, master_y, master_z]) / 1000.0 # m
            master_rot = np.array([master_rx, master_ry, master_rz]) # degrees
            master_gri = np.array([master_grip])# degrees

            slave_pos = np.array([slave_x, slave_y, slave_z]) / 1000.0 # m
            slave_rot = np.array([slave_rx, slave_ry, slave_rz])# degrees
            slave_gri = np.array([slave_grip])# degrees

            pos_error = np.linalg.norm(master_pos - slave_pos)
            rot_error = np.linalg.norm(master_rot - slave_rot)
            gri_error = np.linalg.norm(master_gri - slave_gri)

            print("----------------------------------------")

            print(f"  Master Pos (m): {master_pos[0]:.3f}, {master_pos[1]:.3f}, {master_pos[2]:.3f}")
            print(f"  Slave Pos (m):  {slave_pos[0]:.3f}, {slave_pos[1]:.3f}, {slave_pos[2]:.3f}")
            print(f"  Master grip (m):  {master_gri[0]:.3f}")
            print(f"  Slave grip (m):  {slave_gri[0]:.3f}")
            print(f"  Position Error (m): {pos_error:.3f}")
            print(f"  Rotation Error (deg): {rot_error:.3f}")
            print(f"  gripper Error (deg): {gri_error:.3f}")
            print("----------------------------------------")

    except KeyboardInterrupt:
        print("Stopping end pose following.")
    finally:
        slave_piper.DisableArm(7)
        master_piper.DisconnectPort()
        slave_piper.DisconnectPort()
        print("Pipers disconnected.")

if __name__ == "__main__":
    main()