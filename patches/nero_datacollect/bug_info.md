python -c "
import asyncio, sys, os
sys.path.insert(0, 'brainco-hand-sdk/python')
from common_imports import logger, libstark, modbus_open

async def check():
    client = await libstark.modbus_open('/dev/ttyUSB0', libstark.Baudrate.Baud460800)
    await client.set_hardware_type(0x7E, libstark.StarkHardwareType.Revo2Touch)
    await client.set_finger_unit_mode(0x7E, libstark.FingerUnitMode.Normalized)

    # 读每根手指的配置
    for i, name in enumerate(['拇指','食指','中指','无名指','小指','手腕']):
        fid = list(libstark.FingerId)[i]
        settings = await client.get_finger_settings(0x7E, fid)
        print(f'{name}: min_pos={settings.min_position}, max_pos={settings.max_position}, max_speed={settings.max_speed}')

    # 读当前位置
    status = await client.get_motor_status(0x7E)
    print(f'当前位置: {list(status.positions)}')
    print(f'当前速度: {list(status.speeds)}')

    libstark.modbus_close(client)

asyncio.run(check())
"



pw6@jetson-3:~/Documents/test$ python diagnose.py 
2026-06-27T21:29:50.083349Z[INFO][src/python/py_mod.rs:24] bc_stark_sdk version: 2.0.2
2026-06-27T21:29:50.085355Z[INFO][src/utils/logging_desktop.rs:17] bc_stark_sdk version: 2.0.2
可用端口: ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3']

--- /dev/ttyUSB0 ---
2026-06-27T21:29:50.093204Z[INFO][src/python/py_device.rs:84] Set hardware type for slave 126: Revo2Basic
2026-06-27T21:29:50.094950Z[INFO][src/stark/api/motor_config.rs:44] set_finger_unit_mode, slave_id: 126, mode: Normalized
2026-06-27T21:29:50.097445Z[INFO][src/stark/api/device_info.rs:33] slave[126] get_device_info...
2026-06-27T21:29:50.117349Z[INFO][src/stark/api/device_info.rs:163] TouchVendor for slave 126: Capacitive
2026-06-27T21:29:50.119503Z[INFO][src/stark/api/device_info.rs:189] DeviceInfo: "{\"sku_type\":\"MediumLeft\",\"hand_type\":\"Left\",\"hardware_type\":\"Revo2Touch\",\"serial_number\":\"BCXTL2313J2500008\",\"firmware_version\":\"1.0.11.U\",\"hardware_version\":\"\"}"
  *** 成功! /dev/ttyUSB0, baud=Baudrate.Baud460800, id=126
  *** DeviceInfo { sku_type: MediumLeft, hand_type: Left, hardware_type: Revo2Touch, serial_number: BCXTL2313J2500008, firmware_version: 1.0.11.U, hardware_version:  }



cd ~/Documents/test/pyAgxArm-master

# 读取当前机械臂 7 个关节
python3 read_leader_joints.py -m nero -f v112 -c can1 --mode follower

# 如果要读取示教/主臂关节
python3 read_leader_joints.py -m nero -f v112 -c can1 --mode leader

angles = robot.get_joint_angles()
if angles:
    print(angles.msg)  # 7 个关节，单位 rad
    0x2A5 → J1、J2
0x2A6 → J3、J4
0x2A7 → J5、J6
0x2A9 → J7


python3 arm_hand_bridge.py --execute