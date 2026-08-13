"""
Revo2Touch 握紧/张开控制 (RS-485 Modbus-RTU)

设备: BCXTL2313J2500008 / Revo2Touch / 左手 / 固件 1.0.11.U
端口: /dev/ttyUSB0 / 460800bps / Slave ID 126
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainco-hand-sdk", "python"))
from common_imports import logger, libstark, modbus_open

libstark.init_logging()

# 诊断结果 — 已确认
SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = libstark.Baudrate.Baud460800
SLAVE_ID = 0x7F
HW_TYPE = libstark.StarkHardwareType.Revo2Touch  # 触觉版！

FULLY_OPEN = 0
FULLY_CLOSED = 1000


async def connect():
    client = await libstark.modbus_open(port_name=SERIAL_PORT, baudrate=BAUDRATE)
    await client.set_hardware_type(SLAVE_ID, HW_TYPE)
    await client.set_finger_unit_mode(SLAVE_ID, libstark.FingerUnitMode.Normalized)
    info = await client.get_device_info(SLAVE_ID)
    logger.info(f"设备: {info.description}")
    # 触觉版特有：读一下触觉传感器状态
    try:
        touch = await client.get_touch_sensor_status(SLAVE_ID)
        logger.info(f"触觉传感器: online")
    except:
        pass
    return client


async def show_status(client):
    status = await client.get_motor_status(SLAVE_ID)
    logger.info(f"手指状态: {status.description}")
    return status


async def grip_fist(client, strength=1000, duration_ms=500):
    logger.info(f"握紧中...")
    await client.set_finger_positions_and_durations(SLAVE_ID, [strength]*6, [duration_ms]*6)
    await asyncio.sleep(duration_ms / 1000.0 + 0.2)
    await show_status(client)
    logger.info("握紧完成!")


async def open_hand(client, duration_ms=500):
    logger.info(f"张开中...")
    await client.set_finger_positions_and_durations(SLAVE_ID, [FULLY_OPEN]*6, [duration_ms]*6)
    await asyncio.sleep(duration_ms / 1000.0 + 0.2)
    await show_status(client)
    logger.info("张开完成!")


async def main():
    client = None
    try:
        client = await connect()

        logger.info("===== 初始状态 =====")
        await show_status(client)

        input("\n按 Enter 握紧拳头...")
        await grip_fist(client)

        input("\n按 Enter 张开手掌...")
        await open_hand(client)

        logger.info("\n完成!")

    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.error(f"错误: {e}", exc_info=True)
    finally:
        if client:
            libstark.modbus_close(client)
            logger.info("连接已关闭")


if __name__ == "__main__":
    asyncio.run(main())
