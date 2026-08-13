#!/usr/bin/env python3
"""运动到指定法兰位姿。

从竖直零位 move_p 时：只改 XYZ、保持当前姿态 往往比同时改 RPY 更容易成功。
（move_p_verify 能成就是因为只动了 z、姿态未变。）
"""
import math
from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
from nero_motion_utils import connect_and_enable, wait_motion_done

# 目标位置（m）
TARGET_XYZ = [-0.15, 0.0, 0.15]

# True：只改 XYZ，RPY 跟当前一致（从竖直零位推荐）
# False：使用下方 TARGET_RPY_DEG 完整 6 维位姿
KEEP_CURRENT_ORIENTATION = True
TARGET_RPY_DEG = (0.0, 0.0, 0.0)  # 仅 KEEP_CURRENT_ORIENTATION=False 时生效

SPEED_PERCENT = 20
RETURN_TO_START = False
POS_TOL_M = 0.08  # 位置容差 80 mm（固件 1.10 move_p）


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
if KEEP_CURRENT_ORIENTATION:
    target = TARGET_XYZ + home[3:6]
    print("模式: 只改 XYZ，保持当前姿态")
else:
    target = TARGET_XYZ + [math.radians(a) for a in TARGET_RPY_DEG]
    print("模式: 完整 6 维位姿（含 RPY）")

print("当前法兰位姿 [x,y,z,r,p,y]:", home)
print("目标法兰位姿 [x,y,z,r,p,y]:", target)

robot.set_speed_percent(SPEED_PERCENT)
robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
robot.move_p(target)

if not wait_motion_done(robot, target_pose=target, pos_tol_m=POS_TOL_M):
    fp2 = robot.get_flange_pose()
    actual = fp2.msg if fp2 else None
    print("\n失败。当前实际位姿:", actual)
    print("若 arm_status=无解 且完全没动：")
    print("  - 竖直零位不要同时设 RPY=0,0,0（与零位姿态差约 180°）")
    print("  - 先试 KEEP_CURRENT_ORIENTATION=True 只改位置")
    print("  - 或先小幅 move_p（如 move_p_verify.py）再加大")
    raise RuntimeError("运动到目标点未完成")

if RETURN_TO_START:
    print("回到起点...")
    robot.set_motion_mode(robot.OPTIONS.MOTION_MODE.P)
    robot.move_p(home)
    if not wait_motion_done(robot, target_pose=home, pos_tol_m=POS_TOL_M):
        raise RuntimeError("回位未完成")

print("完成")
