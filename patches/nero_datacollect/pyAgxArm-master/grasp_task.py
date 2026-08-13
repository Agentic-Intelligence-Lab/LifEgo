#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取任务：AgileX NERO 机械臂 + 强脑 Revo2/RENV2 灵巧手。

完整 pick-and-place 流程（与需求 10 步一一对应）：
  1. 机械臂回到安全初始位 (home)
  2. 灵巧手张开
  3. 移动到物体上方/前方 (pre_grasp)
  4. 慢速靠近物体 (grasp, speed=slow)
  5. 灵巧手闭合（接触自适应）
  6. 等待抓稳
  7. 抬起物体 (lift)
  8. 移动到放置位置 (place)
  9. 张开灵巧手释放
 10. 机械臂撤回 (home)

所有运动目标/反馈位姿均为基座系下 [x, y, z, roll, pitch, yaw]：
  向前 = -x，向右 = +y，向上 = +z；位置 cm，姿态 °（SDK 内部仍为 m/rad）。
home 用关节角，更稳定可复现。

安全特性（全部来自 safe_arm.SafeNeroArm）：
  - 指尖软地板：任何目标都不会让指尖低于 table_z + safe_margin，避免撞桌。
  - 运动中实时监控指尖高度，逼近桌面立即急停。
  - 关节软限位、末端限速、碰撞防护、单步临时调速。
  - 本任务全程保持使能，脚本退出不会调用 disable()。

上机前务必：按实物修改 SafetyConfig 与下方 GraspPoses 中的数值，
先空载、低速、手放急停旁验证轨迹，确认无误再放物体。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Sequence

from nero_motion_utils import POSE_AXIS_HINT, format_pose_line
from safe_arm import SafeNeroArm, SafetyConfig, FloorViolation


# --------------------------------------------------------------------------- #
# 任务位姿配置（按你的工位实测修改！）
# --------------------------------------------------------------------------- #
@dataclass
class GraspPoses:
    """一次抓取任务用到的全部位姿。

    home_joints: 7 个关节角 (rad)，安全初始位。
    其余均为指尖(TCP)位姿 [x, y, z, roll, pitch, yaw]。
    """

    # 安全初始位（关节角，长度 7）—— 一个手臂收拢、远离桌面的姿态
    home_joints: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # 物体上方/前方（预抓取 TCP）。默认占位值，需按工位标定！
    pre_grasp: List[float] = field(
        default_factory=lambda: [-0.40, 0.00, 0.25, -math.pi / 2, 0.0, -math.pi]
    )

    # 抓取位（指尖贴近物体，比 pre_grasp 低一些）
    grasp: List[float] = field(
        default_factory=lambda: [-0.40, 0.00, 0.12, -math.pi / 2, 0.0, -math.pi]
    )

    # 抬起位（抓稳后竖直抬高）
    lift: List[float] = field(
        default_factory=lambda: [-0.40, 0.00, 0.30, -math.pi / 2, 0.0, -math.pi]
    )

    # 放置位（移动到目标点上方）
    place: List[float] = field(
        default_factory=lambda: [-0.20, 0.30, 0.20, -math.pi / 2, 0.0, -math.pi]
    )


@dataclass
class GraspParams:
    """抓取行为参数。"""

    slow_speed: int = 8          # 靠近物体时的慢速 (%)
    normal_speed: int = 20       # 常规移动速度 (%)
    grasp_level: int = 90        # 最大闭合程度 (0~100)
    grasp_current_limit: int = 40  # 接触判定电流阈值
    settle_time: float = 0.5     # 抓稳等待时间 (s)


# --------------------------------------------------------------------------- #
# 抓取任务
# --------------------------------------------------------------------------- #
def run_grasp(
    arm: SafeNeroArm,
    poses: GraspPoses,
    params: GraspParams,
    *,
    skip_home: bool = False,
) -> bool:
    """执行一次完整抓取-放置。返回 True 表示全程成功。"""

    def step(n: int, desc: str) -> None:
        print(f"\n===== 步骤 {n}: {desc} =====")

    def relative_flange(arm: SafeNeroArm, dx=0.0, dy=0.0, dz=0.0) -> List[float]:
        return arm.relative_flange(dx, dy, dz)

    if not skip_home:
        # 1. 回到安全初始位
        step(1, "机械臂回到安全初始位")
        arm.set_speed(params.normal_speed)
        if not arm.move_j(poses.home_joints):
            print("到达 home 失败，终止。")
            return False

    # 2. 灵巧手张开
    step(2, "灵巧手张开")
    arm.open_hand()
    time.sleep(0.3)

    # 3. 预抓取：竖直零位下用【法兰】小幅下移（已验证 move_p_verify 可行）
    step(3, "移动到预抓取位（法兰 z -5cm）")
    pre = relative_flange(arm, dz=-0.05)
    print(format_pose_line(pre, "预抓取法兰"))
    if not arm.move_flange_p(pre):
        print("到达 pre_grasp 失败，终止。")
        return False

    # 4. 慢速下压
    step(4, "慢速靠近物体（法兰 z 再 -3cm）")
    grasp = relative_flange(arm, dz=-0.03)
    print(format_pose_line(grasp, "抓取法兰"))
    if not arm.move_flange_l(grasp, speed_percent=params.slow_speed):
        print("靠近 grasp 失败（可能触发地板/碰撞保护），终止。")
        return False

    # 5. 灵巧手闭合（接触自适应，避免夹太松或过载）
    step(5, "灵巧手闭合")
    arm.close_hand_until_contact(
        target_level=params.grasp_level,
        current_limit=params.grasp_current_limit,
    )

    # 6. 等待抓稳
    step(6, "等待抓稳")
    time.sleep(params.settle_time)

    # 7. 抬起
    step(7, "抬起物体（法兰 z +8cm）")
    lift = relative_flange(arm, dz=0.08)
    print(format_pose_line(lift, "抬起法兰"))
    if not arm.move_flange_l(lift, speed_percent=params.slow_speed):
        print("抬起失败，终止（物体可能未抓稳）。")
        return False

    # 8. 侧移放置（先 y 方向小移，避免 IK 无解）
    step(8, "移动到放置位置（法兰 y +10cm）")
    place = relative_flange(arm, dy=0.10)
    print(format_pose_line(place, "放置法兰"))
    if not arm.move_flange_p(place, speed_percent=params.normal_speed):
        print("移动到 place 失败，终止。")
        return False

    # 9. 张开灵巧手释放
    step(9, "张开灵巧手释放")
    arm.open_hand()
    time.sleep(0.3)

    # 10. 机械臂撤回
    step(10, "机械臂撤回到 home")
    arm.set_speed(params.normal_speed)
    arm.move_j(poses.home_joints)

    print("\n抓取任务完成。")
    return True


def main() -> None:
    # ---- 1) 安全/几何标定（必须按实物修改）----
    safety = SafetyConfig(
        hand_length=0.16,        # 灵巧手指尖沿法兰 +Z 的伸出长度
        table_z=0.0,             # 桌面在基座系的高度
        safe_margin=0.05,        # 指尖距桌面安全余量
        debug_speed_percent=15,  # 初始全局速度
        crash_rating=3,          # 碰撞防护等级
        motion_timeout=25.0,     # 单次运动等待（原 8s 对大动作偏短）
        disable_on_exit=False,   # 抓取任务：永不失能
    )

    # ---- 2) 任务位姿（必须按工位修改）----
    poses = GraspPoses()
    params = GraspParams()

    # ---- 3) 执行（异常急停，但不失能）----
    # Jetson：USB-CAN 为 can1（非板载 can0）
    with SafeNeroArm(safety, channel="can1") as arm:
        print(POSE_AXIS_HINT)
        print(f"软地板高度 z_floor = {safety.z_floor * 100:.1f} cm")
        print("使能策略: 全程保持使能，脚本结束不会失能")

        print("\n===== 使能 =====")
        print("（已在连接时完成）")

        print("\n===== 恢复零位 =====")
        arm.robot.set_normal_mode()
        arm.clear_errors()
        arm.set_speed(params.normal_speed)
        if not arm.move_j(poses.home_joints):
            print("恢复零位失败，终止。")
            return

        try:
            run_grasp(arm, poses, params, skip_home=True)
            print("\n任务结束，机械臂保持使能（未调用 disable）。")
        except FloorViolation as e:
            print(f"[安全拦截] {e}")
            print("机械臂保持使能（未调用 disable）。")
        except KeyboardInterrupt:
            print("\n收到中断，急停（仍保持使能，未 disable）。")
            arm.emergency_stop()


if __name__ == "__main__":
    main()
