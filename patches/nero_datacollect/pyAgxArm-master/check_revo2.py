#!/usr/bin/env python3
"""检查强脑 Revo2 灵巧手是否通过机械臂 CAN 通信正常。"""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

CHANNEL = "can1"

cfg = create_agx_arm_config(
    robot=ArmModel.NERO, firmeware_version=NeroFW.DEFAULT, channel=CHANNEL
)
robot = AgxArmFactory.create_arm(cfg)
hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)

print("=== 强脑 Revo2 灵巧手通信检查 ===")
print(f"CAN: {CHANNEL}")

robot.connect()
print("机械臂 connect: OK")

t0 = time.monotonic()
while not robot.enable():
    robot.set_normal_mode()
    if time.monotonic() - t0 > 5:
        print("机械臂 enable: FAIL")
        print("请先激活 can1：")
        print("  sudo ip link set can1 type can bitrate 1000000")
        print("  sudo ip link set can1 up")
        raise SystemExit(1)
    time.sleep(0.01)
print("机械臂 enable: OK")

time.sleep(0.5)
print(f"is_ok():     {hand.is_ok()}")
print(f"get_fps():   {hand.get_fps()}")

hs = hand.get_hand_status()
fp = hand.get_finger_pos()
print(f"hand_status: {hs.msg if hs else None}")
print(f"finger_pos:  {fp.msg if fp else None}")

if hand.is_ok() and hs is not None:
    print("\n结论: Revo2 已连接，CAN 通信正常")
elif hs is not None or (fp is not None and hand.get_fps() > 0):
    print("\n结论: 有反馈，灵巧手可能已连接（检查 is_ok / 供电）")
else:
    print("\n结论: 未收到 Revo2 反馈")
    print("请检查: 手是否装在法兰上、线缆、手部供电、机械臂 CAN 数据是否正常")
