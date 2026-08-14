#!/usr/bin/env python3
"""Nero 首次连接测试：读固件、关节角、法兰位姿。确认无误后再试运动。"""
import time
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

robot_cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,  # 本机实测 1.10 → DEFAULT
    channel="can1",                 # Jetson USB CAN
)
robot = AgxArmFactory.create_arm(robot_cfg)
robot.connect()
print("connect OK")

while not robot.enable():
    time.sleep(0.01)
print("enable OK")

print("固件:", robot.get_firmware())

ja = robot.get_joint_angles()
fp = robot.get_flange_pose()
print("关节角 (rad):", ja.msg if ja else None)
print("法兰位姿 [x,y,z,r,p,y]:", fp.msg if fp else None)

# 确认读数正常、周围无障碍物后，再取消注释试低速运动：
# robot.set_speed_percent(20)
# current = ja.msg
# robot.move_j([j + 0.05 for j in current])
