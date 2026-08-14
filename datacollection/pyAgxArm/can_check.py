#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson / Linux socketcan 快速诊断：接口状态 + 5 秒收帧统计。

用法:
    python3 can_check.py          # 默认 can1
    python3 can_check.py can0
"""
from __future__ import annotations

import subprocess
import sys
import time

DURATION = 5.0
BITRATE = 1_000_000


def iface_up(name: str) -> bool:
    try:
        out = subprocess.check_output(
            ["ip", "link", "show", name], text=True, stderr=subprocess.DEVNULL
        )
        return "state UP" in out
    except subprocess.CalledProcessError:
        return False


def iface_driver(name: str) -> str:
    try:
        return subprocess.check_output(
            ["ethtool", "-i", name], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "(无法读取 ethtool -i)"


def sniff(channel: str) -> tuple[int, dict[int, int]]:
    import can

    bus = can.interface.Bus(interface="socketcan", channel=channel, bitrate=BITRATE)
    counts: dict[int, int] = {}
    n = 0
    t0 = time.time()
    try:
        while time.time() - t0 < DURATION:
            msg = bus.recv(0.2)
            if msg is None:
                continue
            n += 1
            counts[msg.arbitration_id] = counts.get(msg.arbitration_id, 0) + 1
            if n <= 15:
                print(
                    f"  id=0x{msg.arbitration_id:03X}  "
                    f"dlc={msg.dlc}  data={bytes(msg.data).hex()}"
                )
    finally:
        bus.shutdown()
    return n, counts


def main() -> int:
    channel = sys.argv[1] if len(sys.argv) > 1 else "can1"

    print(f"=== CAN 诊断: {channel} ===\n")
    try:
        brief = subprocess.check_output(
            ["ip", "-br", "link", "show", channel], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        brief = f"{channel} 不存在"
    print(f"接口: {brief}")
    print(f"驱动:\n{iface_driver(channel)}\n")

    if not iface_up(channel):
        print(
            "❌ 接口未 UP。请在【系统终端】执行（Cursor 内 sudo 不可用）：\n"
            f"  sudo ip link set {channel} down\n"
            f"  sudo ip link set {channel} type can bitrate {BITRATE}\n"
            f"  sudo ip link set {channel} up\n"
            "或: sudo bash scripts/jetson/can1_up.sh"
        )
        return 1

    print(f"监听 {DURATION:.0f} 秒（机械臂请保持上电）...\n")
    try:
        n, counts = sniff(channel)
    except OSError as e:
        print(f"❌ 打开 {channel} 失败: {e}")
        return 1

    print(f"\n共收到 {n} 帧。")
    if counts:
        print("CAN ID 统计:")
        for cid in sorted(counts):
            print(f"  0x{cid:03X}: {counts[cid]} 帧")
        print("\n✅ 总线有数据 → 物理层 OK，若 go_home 仍无反馈，检查固件/SDK 配置。")
        return 0

    print(
        "❌ 0 帧 → connect OK 但机械臂无反馈，通常是：\n"
        "  1) 机械臂/控制柜未上电或未就绪\n"
        "  2) USB-CAN 插在 Jetson，但 CANH/CANL 未接到机械臂 CAN 口\n"
        "  3) CANH/CANL 接反、未共地、缺 120Ω 终端电阻\n"
        "  4) 波特率不是 1 Mbps（Nero 默认 1000000）\n"
        "  5) 误用 can0（板载）而非 can1（USB gs_usb）"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
