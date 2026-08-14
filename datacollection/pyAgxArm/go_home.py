#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""先使能，再恢复关节零位（7 轴全 0 rad）。

Nero 无专用 go_home API，用 move_j 实现。
固件 1.10 在 move_j 后常报 motion_status=1，脚本以关节角偏差判断到位。
"""
from __future__ import annotations

import argparse
import sys
import time

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import (
    connect_and_enable,
    joint_angle_error,
    print_arm_status,
    wait_motion_done,
)

DEFAULT_SPEED = 20
JOINT_TOL_RAD = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description="先使能，再 move_j 回关节零位")
    parser.add_argument(
        "--speed",
        type=int,
        default=DEFAULT_SPEED,
        help=f"速度百分比 1~100（默认 {DEFAULT_SPEED}）",
    )
    parser.add_argument(
        "--channel",
        default="can1",
        help="CAN 接口名（默认 can1）",
    )
    args = parser.parse_args()

    robot_cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        channel=args.channel,
    )
    robot = AgxArmFactory.create_arm(robot_cfg)

    print("\n===== 使能 =====")
    try:
        connect_and_enable(robot, channel=args.channel, enable_timeout=15.0)
    except TimeoutError as e:
        print(e)
        return 1

    robot.set_normal_mode()
    robot.clear_joint_error(255)
    time.sleep(0.2)

    zero = [0.0] * robot.joint_nums
    ja = robot.get_joint_angles()
    if ja is not None:
        print("当前关节角:", ja.msg)

    err = joint_angle_error(robot, zero)
    if err is not None and err <= JOINT_TOL_RAD:
        print("\n===== 恢复零位 =====")
        print("已在零位，无需运动。")
        print_arm_status(robot)
        return 0

    print("\n===== 恢复零位 =====")
    print("目标关节角（零位）:", zero)
    robot.set_speed_percent(max(1, min(100, args.speed)))
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.J)
    robot.move_j(zero)

    if wait_motion_done(
        robot,
        timeout=30.0,
        target_joints=zero,
        joint_tol_rad=JOINT_TOL_RAD,
    ):
        ja = robot.get_joint_angles()
        print("回零后关节角:", ja.msg if ja else None)
        print("回零完成（保持使能）")
        return 0

    err = joint_angle_error(robot, zero)
    if err is not None and err <= JOINT_TOL_RAD:
        print("回零完成（关节角已到位，motion_status 可能仍为 1）")
        return 0

    print("回零未完成。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
