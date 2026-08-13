#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读连接自检：验证能否收到 Nero 机械臂数据。

不会使能、不会让机械臂运动，只读取关节角。

Windows（candleLight / gs_usb）推荐：
    python win_can_selftest.py

本脚本也可跨平台使用：
    python selftest_connect.py
"""

import os
import time
from platform import system

from safe_arm import _auto_channel, _install_gs_usb_adapter
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

interface, channel = _auto_channel()
if interface == "gs_usb":
    _install_gs_usb_adapter()

print(f"使用 interface={interface}, channel={channel}")
cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.V112 if system() != "Darwin" else NeroFW.DEFAULT,
    interface=interface,
    channel=channel,
)
robot = AgxArmFactory.create_arm(cfg)
robot.connect()
print("总线已连接，等待机械臂数据（最多 3 秒）...")

ok = False
t0 = time.monotonic()
while time.monotonic() - t0 < 3.0:
    ja = robot.get_joint_angles()
    if ja is not None:
        print("✅ 收到关节角:", ja.msg)
        ok = True
        break
    time.sleep(0.01)

if not ok:
    print("❌ 未收到机械臂数据。请检查：机械臂是否上电、CAN 线是否接好、波特率是否 1Mbps。")

# gs_usb 关闭时会 segfault，强制退出规避（不影响功能）
os._exit(0 if ok else 1)
