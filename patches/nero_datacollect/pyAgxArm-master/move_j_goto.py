#!/usr/bin/env python3
"""关节运动 + 灵巧手序列。

流程：
  home（起点）→ 张开手
  位置1       → 握拳
  位置2       → 张开手
  home（终点）

每站停留 3 秒。灵巧手走 USB 串口（bc-stark-sdk）。
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import (
    connect_and_enable,
    format_pose_line,
    format_web_ui_pose,
    joint_angle_error,
    print_arm_status,
    sdk_to_web_ui_pose,
    wait_motion_done,
)

# 灵巧手模块（brainco_hand_service/revo2_hand.py）
sys.path.insert(0, str(Path(__file__).resolve().parent / "brainco_hand_service"))
from revo2_hand import Revo2Hand  # noqa: E402

POSE1_DEG = [-101.0, -66.0, 143.0, 108.0, 67.0, -5.0, -67.0]
POSE2_DEG = [-53.216, 53.812, 40.118, 98.660, -46.083, -3.870, -23.198]

SPEED_PERCENT = 20
HOLD_SECONDS = 3.0
HAND_WAIT = 1.0  # 手动作后等待到位 (s)
JOINT_TOL_RAD = math.radians(2.0)

# 每步: (关节角 rad, 名称, 到位后手动作: "open"|"close"|None)
Step = tuple[list[float], str, str | None]


def deg_list_to_rad(joints_deg):
    return [math.radians(j) for j in joints_deg]


def format_joints_deg(joints_rad, label: str = "") -> str:
    deg = [math.degrees(j) for j in joints_rad]
    body = ", ".join(f"J{i+1}={d:.3f}°" for i, d in enumerate(deg))
    return f"{label}{body}" if label else body


def print_pose_feedback(robot) -> None:
    ja = robot.get_joint_angles()
    fp = robot.get_flange_pose()
    if ja is not None:
        print(format_joints_deg(ja.msg, "  关节: "))
    if fp is not None:
        xyz, rpy = sdk_to_web_ui_pose(fp.msg)
        print(format_web_ui_pose(xyz, rpy, "  法兰: "))
        print(format_pose_line(fp.msg, "  SDK: "))


def move_joints(robot, target, label: str) -> bool:
    print(f"\n===== {label} =====")
    print(format_joints_deg(target, "目标: "))
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(target)
    if wait_motion_done(
        robot,
        timeout=40.0,
        target_joints=target,
        joint_tol_rad=JOINT_TOL_RAD,
    ):
        return True
    err = joint_angle_error(robot, target)
    if err is not None and err <= JOINT_TOL_RAD:
        print(f"关节已到位（最大偏差 {math.degrees(err):.2f}°）")
        return True
    print(f"{label} 未完成")
    print_arm_status(robot, "  ")
    return False


def hand_action(hand: Revo2Hand, action: str | None) -> None:
    if action == "open":
        hand.open_hand(HAND_WAIT)
    elif action == "close":
        hand.close_hand(HAND_WAIT)


def dwell(label: str) -> None:
    print(f"停留 {HOLD_SECONDS:.0f}s @ {label} ...")
    time.sleep(HOLD_SECONDS)


def main() -> int:
    zero = [0.0] * 7
    steps: list[Step] = [
        (zero, "home（起点）", "open"),
        (deg_list_to_rad(POSE1_DEG), "位置1", "close"),
        (deg_list_to_rad(POSE2_DEG), "位置2", "open"),
        (zero, "home（终点）", None),
    ]

    robot_cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        channel="can1",
    )
    robot = AgxArmFactory.create_arm(robot_cfg)

    print("===== 使能 =====")
    try:
        connect_and_enable(robot, channel="can1", enable_timeout=15.0)
    except TimeoutError as e:
        print(e)
        return 1

    robot.set_normal_mode()
    robot.clear_joint_error(255)
    robot.set_speed_percent(SPEED_PERCENT)

    try:
        with Revo2Hand() as hand:
            for i, (joints, name, hand_cmd) in enumerate(steps):
                if not move_joints(robot, joints, f"{i + 1}. {name}"):
                    return 1
                print_pose_feedback(robot)
                hand_action(hand, hand_cmd)
                dwell(name)
    except RuntimeError as e:
        print(f"灵巧手错误: {e}")
        return 1

    print("\n序列完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
