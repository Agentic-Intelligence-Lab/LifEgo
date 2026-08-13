"""逐指验证: 每个手指单独闭合再张开, 确认实物对应"""
import asyncio, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "brainco-hand-sdk", "python"))
from common_imports import logger, libstark, modbus_open

async def verify():
    client = await libstark.modbus_open('/dev/ttyUSB0', libstark.Baudrate.Baud460800)
    await client.set_hardware_type(0x7F, libstark.StarkHardwareType.Revo2Touch)
    await client.set_finger_unit_mode(0x7F, libstark.FingerUnitMode.Normalized)

    # 6指: [拇辅, 拇指, 食指, 中指, 无名指, 小指]
    tests = [
        (libstark.FingerId.Thumb,    "拇辅"),
        (libstark.FingerId.ThumbAux, "拇指"),
        (libstark.FingerId.Index,    "食指"),
        (libstark.FingerId.Middle,   "中指"),
        (libstark.FingerId.Ring,     "无名指"),
        (libstark.FingerId.Pinky,    "小指"),
    ]

    for fid, name in tests:
        input(f"\n=== 测试 {name} === 看好了按 Enter...")

        # 先到500
        print(f"  {name} → 500")
        await client.set_finger_position_with_millis(0x7F, fid, 500, 300)
        await asyncio.sleep(0.6)
        status = await client.get_motor_status(0x7F)
        print(f"  位置: {list(status.positions)}")

        input("  看实物: 哪根手指动了? 按 Enter 复原...")

        # 回到0
        await client.set_finger_position_with_millis(0x7F, fid, 0, 300)
        await asyncio.sleep(0.6)
        status = await client.get_motor_status(0x7F)
        print(f"  位置: {list(status.positions)}")

    libstark.modbus_close(client)
    print("验证完毕")

asyncio.run(verify())