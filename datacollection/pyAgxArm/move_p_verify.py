#!/usr/bin/env python3
"""验证 move_p 是否可用：从当前位姿 z 方向下移 5 cm（竖直姿态下有意义）。"""
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import connect_and_enable, wait_motion_done

Z_DELTA_M = -0.05  # 下移 5 cm
SPEED_PERCENT = 20

robot_cfg = create_agx_arm_config(
    robot=ArmModel.NERO,
    firmeware_version=NeroFW.DEFAULT,
    channel="can1",
)
robot = AgxArmFactory.create_arm(robot_cfg)
connect_and_enable(robot)

fp = robot.get_flange_pose()
if fp is None:
    raise RuntimeError("无法读取法兰位姿")

home = list(fp.msg)
target = list(home)
target[2] += Z_DELTA_M

print("当前:", home)
print("目标:", target)

robot.set_speed_percent(SPEED_PERCENT)
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
robot.move_p(target)
if not wait_motion_done(robot, target_pose=target, pos_tol_m=0.03):
    raise RuntimeError("move_p 验证失败")

print("回到起点...")
robot.move_p(home)
wait_motion_done(robot, target_pose=home, pos_tol_m=0.03)
print("move_p 验证通过")
