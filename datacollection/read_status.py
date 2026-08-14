"""读取 Revo2 当前关节状态"""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainco-hand-sdk", "python"))
from common_imports import logger, libstark, modbus_open

async def read():
    client = await libstark.modbus_open('/dev/ttyUSB0', libstark.Baudrate.Baud460800)
    await client.set_hardware_type(0x7F, libstark.StarkHardwareType.Revo2Touch)

    status = await client.get_motor_status(0x7F)
    fingers = ['拇辅', '拇指', '食指', '中指', '无名指', '小指']

    print("\n===== 当前关节状态 =====")
    for i, name in enumerate(fingers):
        print(f"  {name}:  位置={status.positions[i]:>5},  速度={status.speeds[i]:>5},  电流={status.currents[i]:>5},  状态={status.states[i]}")
    print("========================\n")

    libstark.modbus_close(client)

asyncio.run(read())