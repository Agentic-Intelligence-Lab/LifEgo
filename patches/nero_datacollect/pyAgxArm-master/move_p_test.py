#!/usr/bin/env python3
"""Nero 首次笛卡尔运动测试：法兰 z 升高 2 cm 后回到起点。"""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

Z_OFFSET_M = 0.02  # z 方向升高 2 cm


def wait_motion_done(robot, timeout=15.0):
    time.sleep(0.5)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        status = robot.get_arm_status()
        if status is not None and status.msg.motion_status == 0:
            print("运动完成")
            return True
        time.sleep(0.1)
    print(f"等待超时（{timeout:.0f}s）")
    return False


robot_cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    channel="can1",
)
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
print("connect OK")

while not robot.enable():
    time.sleep(0.01)
print("enable OK")

robot.set_speed_percent(20)

fp = robot.get_flange_pose()
if fp is None:
    raise RuntimeError("无法读取法兰位姿，请检查 CAN 连接")

home = list(fp.msg)
target = list(home)
target[2] += Z_OFFSET_M  # z + 2 cm，姿态不变

print("起点法兰位姿 [x,y,z,r,p,y]:", home)
print("目标法兰位姿 [x,y,z,r,p,y]:", target)

robot.move_p(target)
if not wait_motion_done(robot):
    raise RuntimeError("上移运动未完成")

print("回到起点...")
robot.move_p(home)
if not wait_motion_done(robot):
    raise RuntimeError("回位运动未完成")

print("测试完成")
