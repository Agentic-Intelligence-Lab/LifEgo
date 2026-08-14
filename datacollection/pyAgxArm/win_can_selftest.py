#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Windows candleLight / gs_usb CAN 一键自检。

步骤：
1. 打开 gs_usb，嗅探若干秒原始帧；
2. 再用 pyAgxArm 只读连接 Nero，打印关节角。

不会使能电机，也不会下发运动指令。

用法：
    python win_can_selftest.py
    python win_can_selftest.py --sniff 3 --wait 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from platform import system

from safe_arm import _auto_channel, _install_gs_usb_adapter


NERO_FOLLOWER_JOINT_IDS = {0x2A5, 0x2A6, 0x2A7, 0x2A9}
NERO_LEADER_V112_IDS = {0x155, 0x156, 0x157, 0x170}
BITRATE = 1_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows Nero CAN self-test")
    parser.add_argument(
        "--sniff",
        type=float,
        default=0.0,
        help="Optional extra raw sniff seconds after joint read (default: 0; Windows often cannot reopen in-process)",
    )
    parser.add_argument("--wait", type=float, default=3.0, help="Joint-read wait seconds")
    parser.add_argument(
        "--firmware",
        default="v112",
        choices=["default", "v111", "v112"],
        help="Nero firmware profile",
    )
    return parser.parse_args()


def sniff(duration: float) -> Counter[int]:
    import can

    counts: Counter[int] = Counter()
    print(f"== 嗅探 gs_usb {duration:.1f}s @ {BITRATE} bps ==")
    bus = can.interface.Bus(interface="gs_usb", channel=0, bitrate=BITRATE)
    start = time.monotonic()
    printed = 0
    try:
        while time.monotonic() - start < duration:
            msg = bus.recv(0.2)
            if msg is None:
                continue
            counts[msg.arbitration_id] += 1
            if printed < 12:
                printed += 1
                data = bytes(msg.data).hex(" ")
                print(f"  id=0x{msg.arbitration_id:03X} dlc={msg.dlc} data={data}")
    finally:
        try:
            bus.shutdown()
        except Exception:
            pass
    return counts


def summarize(counts: Counter[int]) -> bool:
    total = sum(counts.values())
    print(f"\n共 {total} 帧，{len(counts)} 个 CAN ID")
    for cid in sorted(counts):
        print(f"  0x{cid:03X}: {counts[cid]}")
    if total == 0:
        print(
            "\n[FAIL] 无帧。请检查：机械臂上电、CAN 线、candleLight 插入、波特率 1Mbps。"
        )
        return False
    follower = counts.keys() & NERO_FOLLOWER_JOINT_IDS
    leader = counts.keys() & NERO_LEADER_V112_IDS
    print(f"\nFollower 关节 ID: {sorted(f'0x{i:03X}' for i in follower) or '无'}")
    print(f"Leader 关节 ID:   {sorted(f'0x{i:03X}' for i in leader) or '无'}")
    print("[OK] 物理层有数据")
    return True


def read_joints(wait: float, firmware: str) -> bool:
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    fw = {
        "default": NeroFW.DEFAULT,
        "v111": NeroFW.V111,
        "v112": NeroFW.V112,
    }[firmware]
    interface, channel = _auto_channel()
    print(f"\n== pyAgxArm 只读连接 interface={interface} channel={channel} ==")
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=fw,
        interface=interface,
        channel=channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    print(f"已连接，等待关节角最多 {wait:.1f}s ...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait:
        ja = robot.get_joint_angles()
        if ja is not None:
            vals = [round(float(x), 4) for x in ja.msg]
            print("[OK] 关节角 (rad):", vals)
            return True
        time.sleep(0.01)
    print("[FAIL] 未读到关节角")
    return False


def main() -> int:
    if system() != "Windows":
        print("本脚本面向 Windows + candleLight。其它系统请用 selftest_connect.py / can_check.py。")
        return 2

    args = parse_args()
    print("准备 libusb + gs_usb ...")
    try:
        _install_gs_usb_adapter()
    except Exception as exc:
        print(f"[FAIL] gs_usb 初始化失败: {exc}")
        print('请确认已安装: pip install "python-can[gs-usb]" libusb-package')
        return 1

    # 先做 SDK 读关节；同一进程里先 sniff 再二次打开，Windows 上偶发 Access denied。
    try:
        joints_ok = read_joints(args.wait, args.firmware)
    except Exception as exc:
        print(f"[FAIL] SDK 连接失败: {exc}")
        os._exit(1)

    if not joints_ok:
        os._exit(1)

    if args.sniff > 0:
        print("\n释放总线后开始原始帧嗅探 ...")
        time.sleep(1.0)
        try:
            # 第二次打开前再装一次补丁（idempotent），避免上一轮 shutdown 搅乱状态
            _install_gs_usb_adapter()
            counts = sniff(args.sniff)
            summarize(counts)
        except Exception as exc:
            print(f"[WARN] 嗅探失败（关节读取已成功）: {exc}")

    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
