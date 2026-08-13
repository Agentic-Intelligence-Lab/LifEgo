#!/usr/bin/env python3
"""运动到指定法兰位姿（上位机坐标，笛卡尔 6 维）。

目标（与网页操控区一致，X/Y 均为负）：
  X=-412 mm, Y=-246 mm, Z=81 mm
  RX=90°, RY=12°, RZ=0°
"""
import sys

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import (
    connect_and_enable,
    format_pose_line,
    format_web_ui_pose,
    print_arm_status,
    sdk_to_web_ui_pose,
    wait_motion_done,
    web_ui_to_sdk_pose,
)

# 上位机坐标（mm / °）—— X、Y 填负数，与网页显示一致
TARGET_XYZ_MM = (-412.0, -246.0, 81.0)
TARGET_RPY_DEG = (90.0, 12.0, 0.0)  # RX, RY, RZ

SPEED_PERCENT = 20
POS_TOL_M = 0.08  # 位置容差 80 mm（固件 1.10 move_p）
RETURN_TO_START = False


def main() -> int:
    target = web_ui_to_sdk_pose(TARGET_XYZ_MM, TARGET_RPY_DEG)

    robot_cfg = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.DEFAULT,
        channel="can1",
    )
    robot = AgxArmFactory.create_arm(robot_cfg)
    connect_and_enable(robot)

    robot.set_normal_mode()
    robot.clear_joint_error(255)

    fp = robot.get_flange_pose()
    if fp is None:
        print("无法读取法兰位姿")
        return 1

    home = list(fp.msg)
    home_xyz, home_rpy = sdk_to_web_ui_pose(home)
    print(format_web_ui_pose(home_xyz, home_rpy, "当前（上位机）: "))
    print(format_web_ui_pose(TARGET_XYZ_MM, TARGET_RPY_DEG, "目标（上位机）: "))
    print(format_pose_line(target, "目标（SDK）: "))

    robot.set_speed_percent(SPEED_PERCENT)
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
    robot.move_p(target)

    if not wait_motion_done(robot, target_pose=target, pos_tol_m=POS_TOL_M):
        fp2 = robot.get_flange_pose()
        print("\n运动未完成。")
        if fp2 is not None:
            act_xyz, act_rpy = sdk_to_web_ui_pose(fp2.msg)
            print(format_web_ui_pose(act_xyz, act_rpy, "当前（上位机）: "))
            print(format_pose_line(fp2.msg, "当前（SDK）: "))
        print_arm_status(robot, "  ")
        print(
            "提示：若 arm_status=无解 且完全没动，请在上位机网页输入相同 XYZ/RPY，"
            "看「是否有解」是否为「是」。Z=81mm 极低，可能超出工作空间。"
        )
        return 1

    fp3 = robot.get_flange_pose()
    if fp3 is not None:
        act_xyz, act_rpy = sdk_to_web_ui_pose(fp3.msg)
        print(format_web_ui_pose(act_xyz, act_rpy, "到位（上位机）: "))

    if RETURN_TO_START:
        print("回到起点...")
        robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
        robot.move_p(home)
        if not wait_motion_done(robot, target_pose=home, pos_tol_m=POS_TOL_M):
            print("回位未完成")
            return 1

    print("完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
