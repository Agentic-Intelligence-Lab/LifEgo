#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read arm joint angles via CAN bus.

Modes:
  --mode follower  -> get_joint_angles()        (your own arm, DEFAULT)
  --mode leader    -> get_leader_joint_angles()  (the teaching/drag arm)

Usage:
  python read_leader_joints.py -m nero -f v112 -c can1                 # follower
  python read_leader_joints.py -m nero -f v112 -c can1 --mode leader   # leader
  python read_leader_joints.py -m nero -f v112 -c can1 --enable        # enable motors first
  python read_leader_joints.py -m nero -f v112 -c can1 --enable --require-enable
"""

import argparse, math, signal, sys, time
from platform import system

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW, NeroFW


def parse_args():
    p = argparse.ArgumentParser(description="Read arm joint angles via CAN")
    p.add_argument("-m", "--model",
                   choices=["piper","piper_h","piper_l","piper_x","nero"],
                   default="nero")
    p.add_argument("-f", "--firmware", default="default")
    p.add_argument("-c", "--channel", default=None)
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--mode", choices=["follower","leader"], default="follower",
                   help="follower=your own arm (default); leader=teaching arm")
    p.add_argument("--enable", action="store_true",
                   help="Enable all motors before reading")
    p.add_argument("--enable-timeout", type=float, default=10.0,
                   help="Seconds to wait for --enable before continuing")
    p.add_argument("--require-enable", action="store_true",
                   help="Exit if --enable cannot be confirmed")
    return p.parse_args()


def build_config(model, firmware, channel=None):
    robot_map = {
        "piper": ArmModel.PIPER, "piper_h": ArmModel.PIPER_H,
        "piper_l": ArmModel.PIPER_L, "piper_x": ArmModel.PIPER_X,
        "nero": ArmModel.NERO,
    }
    plat = system()
    if channel is None:
        channel = {"Windows": "0", "Linux": "can1", "Darwin": "/dev/ttyACM0"}.get(
            plat, "can0"
        )
    iface = {
        "Windows": "gs_usb",
        "Linux": "socketcan",
        "Darwin": "slcan",
    }.get(plat, "socketcan")
    if iface == "gs_usb":
        from safe_arm import _install_gs_usb_adapter

        _install_gs_usb_adapter()

    if model == "nero":
        fw = {"default":NeroFW.DEFAULT,"v111":NeroFW.V111,"v112":NeroFW.V112}.get(firmware,firmware)
    else:
        fw = {"default":PiperFW.DEFAULT,"v183":PiperFW.V183,"v188":PiperFW.V188}.get(firmware,firmware)

    return create_agx_arm_config(
        robot=robot_map[model], firmeware_version=fw, interface=iface, channel=channel)


def enable_status_text(robot):
    try:
        states = robot.get_joints_enable_status_list()
    except Exception as e:
        return f"unavailable ({e})"
    return " ".join(f"J{i+1}={'ON' if state else 'off'}"
                    for i, state in enumerate(states))


def enable_with_timeout(robot, timeout):
    deadline = time.monotonic() + max(timeout, 0.0)
    next_report = 0.0

    while True:
        if robot.enable():
            return True

        now = time.monotonic()
        if now >= deadline:
            return False

        if now >= next_report:
            print(f"  waiting... enable status: {enable_status_text(robot)}")
            next_report = now + 2.0

        time.sleep(0.05)


def main():
    args = parse_args()
    cfg = build_config(args.model, args.firmware, args.channel)
    print(f"model={args.model}  fw={args.firmware}  mode={args.mode}")
    print(f"channel={cfg['comm']['can']['channel']}  interface={cfg['comm']['can']['interface']}")

    robot = AgxArmFactory.create_arm(cfg)
    print("Connecting ...")
    try:
        robot.connect()
    except Exception as e:
        print(f"Connect failed: {e}")
        sys.exit(1)

    print(f"Connected! joint_nums={robot.joint_nums}")

    fw_info = robot.get_firmware()
    if fw_info is not None:
        print(f"Firmware: {fw_info}")

    if args.enable:
        print(f"Enabling all motors (timeout {args.enable_timeout:.1f}s) ...")
        if enable_with_timeout(robot, args.enable_timeout):
            print("Motors enabled!")
        else:
            print(f"Enable not confirmed: {enable_status_text(robot)}")
            if args.require_enable:
                print("Exiting because --require-enable was set.")
                sys.exit(2)
            print("Continuing to read feedback without confirmed enable.")

    mode_label = "Leader (teaching arm)" if args.mode == "leader" else "Follower (your arm)"
    read_fn = robot.get_leader_joint_angles if args.mode == "leader" else robot.get_joint_angles

    running = True
    def on_int(sig, frame):
        nonlocal running
        print("\nStopping ...")
        running = False
    signal.signal(signal.SIGINT, on_int)

    interval = 1.0 / max(args.hz, 0.1)
    print(f"Reading {mode_label} joint angles at ~{args.hz} Hz, Ctrl+C to stop")
    print("-" * 70)

    last_print = 0.0
    print_interval = 0.5

    while running:
        t0 = time.monotonic()
        try:
            result = read_fn()
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
            continue

        if result is not None:
            rad = result.msg
            deg = [math.degrees(a) for a in rad]
            now = time.time()
            if now - last_print >= print_interval:
                last_print = now
                rad_s = "  ".join(f"J{i+1}={a: 8.4f}" for i, a in enumerate(rad))
                deg_s = "  ".join(f"J{i+1}={d: 7.1f} deg" for i, d in enumerate(deg))
                print(f"[{result.timestamp:.3f}s] hz={result.hz:5.1f} | {rad_s}")
                print(f"          {'':7s} | {deg_s}")
                print()
        else:
            now = time.time()
            if now - last_print >= print_interval:
                last_print = now
                print(f"[No data] {mode_label} joint angles unavailable - "
                      "check CAN frames/channel/firmware mode")

        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print("Done.")


if __name__ == "__main__":
    main()
