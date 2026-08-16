#!/usr/bin/env python3
"""Move real Nero TCP to the first corrected ego_hand1 EEF target."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[2]
PYAGX_ROOT = REPO_ROOT / "datacollection" / "pyAgxArm"
if str(PYAGX_ROOT) not in sys.path:
    sys.path.insert(0, str(PYAGX_ROOT))


TARGET_TCP_POS_M = np.array(
    [-0.290893998990047, -0.2651488208158831, 0.22013958142257684],
    dtype=np.float64,
)
TARGET_TCP_QUAT_XYZW = np.array(
    [0.6866902645667813, -0.2746157150626934, 0.6079597085157775, -0.2888385057626845],
    dtype=np.float64,
)
TCP_OFFSET_M = [0.13, 0.0, 0.0]
FIRMWARE_CHOICES = ("default", "v111", "v112")


def format_pose(pose: Sequence[float]) -> str:
    x, y, z, roll, pitch, yaw = [float(v) for v in pose[:6]]
    return (
        f"xyz=[{x:.4f}, {y:.4f}, {z:.4f}] m  "
        f"rpy=[{math.degrees(roll):.2f}, {math.degrees(pitch):.2f}, {math.degrees(yaw):.2f}] deg"
    )


def target_tcp_pose() -> list[float]:
    rpy = R.from_quat(TARGET_TCP_QUAT_XYZW).as_euler("xyz")
    return [*TARGET_TCP_POS_M.tolist(), *rpy.tolist()]


def connect_and_enable(args: argparse.Namespace):
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    firmwares = {
        "default": NeroFW.DEFAULT,
        "v111": NeroFW.V111,
        "v112": NeroFW.V112,
    }
    cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=firmwares[args.firmware],
        interface="socketcan",
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(cfg)
    robot.connect()
    deadline = time.monotonic() + args.enable_timeout
    while not robot.enable():
        if time.monotonic() >= deadline:
            robot.disconnect()
            raise TimeoutError(f"NERO enable timed out after {args.enable_timeout:.1f}s")
        time.sleep(0.01)
    return robot


def read_tcp_pose(robot) -> list[float]:
    msg = robot.get_tcp_pose()
    if msg is None:
        raise RuntimeError("No TCP feedback")
    pose = list(msg.msg)[:6]
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RuntimeError(f"Invalid TCP feedback: {pose}")
    return pose


def wait_until_tcp_reached(robot, target: Sequence[float], args: argparse.Namespace) -> bool:
    target_pos = np.asarray(target[:3], dtype=np.float64)
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        current = read_tcp_pose(robot)
        pos_err = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
        if pos_err <= args.pos_tol_m:
            print(f"reached: tcp position error {pos_err * 1000.0:.1f} mm")
            print(f"actual tcp: {format_pose(current)}")
            return True
        time.sleep(0.1)
    current = read_tcp_pose(robot)
    pos_err = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
    print(f"timeout: tcp position error {pos_err * 1000.0:.1f} mm")
    print(f"actual tcp: {format_pose(current)}")
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="can1")
    parser.add_argument("--firmware", choices=FIRMWARE_CHOICES, default="v112")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--enable-timeout", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--pos-tol-m", type=float, default=0.04)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--yes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_tcp = target_tcp_pose()
    print(f"target tcp: {format_pose(target_tcp)}")
    print(f"tcp offset in flange: {TCP_OFFSET_M} m")
    if not args.execute:
        print("dry-run only. Add --execute to command the robot.")
        return 0
    if not args.yes:
        text = input("Type EXECUTE to move real Nero to this single EEF > ").strip()
        if text != "EXECUTE":
            print("cancelled")
            return 1

    robot = None
    try:
        robot = connect_and_enable(args)
        robot.set_tcp_offset([*TCP_OFFSET_M, 0.0, 0.0, 0.0])
        current_tcp = read_tcp_pose(robot)
        target_flange = robot.get_tcp2flange_pose(target_tcp)
        print(f"current tcp:       {format_pose(current_tcp)}")
        print(f"target flange cmd: {format_pose(target_flange)}")
        robot.set_speed_percent(max(1, min(100, args.speed_percent)))
        robot.move_p(list(target_flange))
        return 0 if wait_until_tcp_reached(robot, target_tcp, args) else 1
    except KeyboardInterrupt:
        print("\nInterrupted. Use the physical emergency stop if needed.")
        return 130
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
