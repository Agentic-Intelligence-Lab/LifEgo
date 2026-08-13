import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainco-hand-sdk", "python"))
from common_imports import logger, libstark, modbus_open

async def check():
    client = await libstark.modbus_open('/dev/ttyUSB0', libstark.Baudrate.Baud460800)
    await client.set_hardware_type(0x7F, libstark.StarkHardwareType.Revo2Touch)
    await client.set_finger_unit_mode(0x7F, libstark.FingerUnitMode.Normalized)

    # 6指: [拇辅, 拇指, 食指, 中指, 无名指, 小指]
    fingers = [
        (libstark.FingerId.Thumb,    "拇辅"),
        (libstark.FingerId.ThumbAux, "拇指"),
        (libstark.FingerId.Index,    "食指"),
        (libstark.FingerId.Middle,   "中指"),
        (libstark.FingerId.Ring,     "无名指"),
        (libstark.FingerId.Pinky,    "小指"),
    ]

    for fid, name in fingers:
        settings = await client.get_finger_settings(0x7F, fid)
        print(f"{name}: min={settings.min_position}, max={settings.max_position}, max_spd={settings.max_speed}, max_cur={settings.max_current}")

    status = await client.get_motor_status(0x7F)
    print(f"\n6指位置: {list(status.positions)}")
    print(f"6指速度: {list(status.speeds)}")
    print(f"6指电流: {list(status.currents)}")

    libstark.modbus_close(client)

asyncio.run(check())