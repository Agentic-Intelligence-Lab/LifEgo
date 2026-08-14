#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 Nero 末端位姿（法兰位姿 + TCP 位姿），只读，不使能、不下发运动指令。

用法：
  python read_pose.py                              # Windows 默认 gs_usb / channel 0
  python read_pose.py --hz 10
  python read_pose.py --tcp-offset 0.13             # 法兰局部 +X 方向 TCP 偏置（米，法兰→夹爪尖端），默认 0.13（AgxGripper）
  python read_pose.py --once                        # 只读一次就退出
  python read_pose.py --firmware v112 --channel 0
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from platform import system

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import format_pose_line, format_web_ui_pose, sdk_to_web_ui_pose


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read Nero end-effector pose (flange + TCP)")
    p.add_argument("-f", "--firmware", default="v112",
                   choices=["default", "v111", "v112"])
    p.add_argument("-c", "--channel", default=None, help="CAN channel (default: platform default)")
    p.add_argument("--hz", type=float, default=10.0, help="Read/print rate (default 10 Hz)")
    p.add_argument("--tcp-offset", type=float, default=0.13,
                   help="TCP offset along flange-local +X in meters (flange->gripper tip direction; "
                        "default 0.13, AgxGripper). 0 disables offset (TCP == flange).")
    p.add_argument("--once", action="store_true", help="Read once and exit")
    return p.parse_args()


def build_config(firmware: str, channel: str | None) -> dict:
    plat = system()
    # candleLight/gs_usb adapters are native USB devices, not a serial port --
    # Darwin uses the same gs_usb interface as Windows (see collect_apriltag_corners.py).
    if channel is None:
        channel = {"Windows": "0", "Linux": "can1", "Darwin": "0"}.get(plat, "can0")
    interface = {"Windows": "gs_usb", "Linux": "socketcan", "Darwin": "gs_usb"}.get(plat, "socketcan")

    if interface == "gs_usb":
        from safe_arm import _install_gs_usb_adapter
        _install_gs_usb_adapter()

    fw = {"default": NeroFW.DEFAULT, "v111": NeroFW.V111, "v112": NeroFW.V112}.get(firmware, firmware)
    return create_agx_arm_config(
        robot=ArmModel.NERO, firmeware_version=fw, interface=interface, channel=channel,
    )


def main() -> int:
    args = parse_args()
    cfg = build_config(args.firmware, args.channel)
    print(f"firmware={args.firmware}  interface={cfg['comm']['can']['interface']}  "
          f"channel={cfg['comm']['can']['channel']}  tcp_offset_x={args.tcp_offset}m")

    robot = AgxArmFactory.create_arm(cfg)
    print("Connecting ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Connect failed: {e}")
        return 1
    print(f"Connected! joint_nums={robot.joint_nums}")

    if args.tcp_offset:
        robot.set_tcp_offset([args.tcp_offset, 0.0, 0.0, 0.0, 0.0, 0.0])

    running = True

    def on_int(sig, frame):
        nonlocal running
        print("\nStopping ...")
        running = False

    signal.signal(signal.SIGINT, on_int)

    interval = 1.0 / max(args.hz, 0.1)
    print("Reading end-effector pose, Ctrl+C to stop" if not args.once else "Reading pose once")
    print("-" * 70)

    while running:
        t0 = time.monotonic()

        fp = robot.get_flange_pose()
        tcp = robot.get_tcp_pose()

        if fp is not None:
            print(format_pose_line(fp.msg, "法兰(flange)"), f"  hz={fp.hz:.1f}")
        else:
            print("法兰(flange): (无数据，检查 CAN 连接)")

        if tcp is not None:
            print(format_pose_line(tcp.msg, "TCP        "), f"  hz={tcp.hz:.1f}")
            xyz_mm, rpy_deg = sdk_to_web_ui_pose(tcp.msg)
            print(" " * 4 + format_web_ui_pose(xyz_mm, rpy_deg, "  上位机风格: "))
        else:
            print("TCP        : (无数据)")

        print()

        if args.once:
            break

        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
