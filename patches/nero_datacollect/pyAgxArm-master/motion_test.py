#!/usr/bin/env python3
"""Nero 首次低速运动测试：第 7 关节小幅摆动后回原位。"""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import connect_and_enable, wait_motion_done

robot_cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    channel="can1",
)
robot = AgxArmFactory.create_arm(robot_cfg)
connect_and_enable(robot)

robot.set_speed_percent(20)

ja = robot.get_joint_angles()
if ja is None:
    raise RuntimeError("无法读取关节角，请检查 CAN 连接")

home = list(ja.msg)
target = list(home)
target[6] += 0.2618  # 第 7 关节 +15°（0.2618 rad）

print("当前关节角:", home)
print("目标关节角:", target)

robot.move_j(target)
wait_motion_done(robot)

print("回到原位...")
robot.move_j(home)
wait_motion_done(robot)

print("测试完成")
