"""Nero 运动辅助：等待完成、打印机械臂状态。"""
import math
import time

# 基座坐标系（与用户工位一致）：向前 = -x，向右 = +y，向上 = +z
POSE_AXIS_HINT = "基座系: 向前=-x, 向右=+y, 向上=+z | xyz=cm, rpy=°"

ARM_STATUS_NAMES = {
    0x00: "正常",
    0x01: "急停",
    0x02: "无解(IK失败)",
    0x03: "奇异点",
    0x04: "目标超限",
    0x05: "关节通信异常",
    0x06: "抱闸未释放",
    0x07: "碰撞",
}

CTRL_MODE_NAMES = {
    0x00: "待机",
    0x01: "CAN控制",
    0x02: "示教",
}


def _enum_val(v):
    if hasattr(v, "value"):
        return int(v.value)
    return int(v)


def format_pose(pose, label: str = "") -> str:
    """将 SDK 位姿 [x,y,z,roll,pitch,yaw]（m/rad）格式化为 cm/° 显示。"""
    if pose is None:
        return f"{label}(无数据)"
    p = list(pose)
    if len(p) < 6:
        return f"{label}{p}"
    x, y, z, roll, pitch, yaw = p[:6]
    body = (
        f"x={x * 100:.2f}cm, y={y * 100:.2f}cm, z={z * 100:.2f}cm, "
        f"roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, "
        f"yaw={math.degrees(yaw):.2f}°"
    )
    return f"{label}{body}" if label else body


def format_pose_line(pose, name: str = "") -> str:
    """单行位姿，带名称前缀。"""
    prefix = f"{name}: " if name else ""
    return prefix + format_pose(pose)


def format_web_ui_pose(xyz_mm, rpy_deg, label: str = "") -> str:
    """上位机风格位姿（mm / °），与网页操控区显示一致。"""
    x, y, z = xyz_mm
    rx, ry, rz = rpy_deg
    body = (
        f"X={x:.3f}mm, Y={y:.3f}mm, Z={z:.3f}mm, "
        f"RX={rx:.3f}°, RY={ry:.3f}°, RZ={rz:.3f}°"
    )
    return f"{label}{body}" if label else body


def web_ui_to_sdk_pose(xyz_mm, rpy_deg):
    """上位机坐标 → SDK move_p 位姿 [x,y,z,roll,pitch,yaw]（m/rad）。

    XYZ：上位机 mm 原样除以 1000（X/Y 可为负，与网页显示一致）。
    RPY：固件 ≤1.10 在 CAN 层与 SDK 有符号补偿，需 roll=-RX、pitch=-RY。
    """
    x, y, z = xyz_mm
    rx, ry, rz = rpy_deg
    return [
        x / 1000.0,
        y / 1000.0,
        z / 1000.0,
        -math.radians(rx),
        -math.radians(ry),
        math.radians(rz),
    ]


def sdk_to_web_ui_pose(pose):
    """SDK 位姿 → 上位机 mm/°（与 web_ui_to_sdk_pose 互逆）。"""
    x, y, z, roll, pitch, yaw = pose[:6]
    return (
        (x * 1000.0, y * 1000.0, z * 1000.0),
        (-math.degrees(roll), -math.degrees(pitch), math.degrees(yaw)),
    )


def pose_xyz_error(robot, target_pose):
    fp = robot.get_flange_pose()
    if fp is None or target_pose is None:
        return None
    p = fp.msg
    return math.sqrt(sum((p[i] - target_pose[i]) ** 2 for i in range(3)))


def joint_angle_error(robot, target_joints):
    ja = robot.get_joint_angles()
    if ja is None or target_joints is None:
        return None
    n = min(len(ja.msg), len(target_joints))
    return max(abs(ja.msg[i] - target_joints[i]) for i in range(n))


def print_arm_status(robot, prefix=""):
    st = robot.get_arm_status()
    if st is None:
        print(f"{prefix}arm_status: 无反馈")
        return
    m = st.msg
    arm_i = _enum_val(m.arm_status)
    ctrl_i = _enum_val(m.ctrl_mode)
    mode_i = _enum_val(m.mode_feedback)
    motion_i = _enum_val(m.motion_status)
    arm = ARM_STATUS_NAMES.get(arm_i, f"0x{arm_i:02x}")
    ctrl = CTRL_MODE_NAMES.get(ctrl_i, f"0x{ctrl_i:02x}")
    print(
        f"{prefix}ctrl_mode={ctrl}  arm_status={arm}  "
        f"mode_feedback=0x{mode_i:02x}  motion_status={motion_i}"
    )


def wait_motion_done(
    robot,
    timeout=30.0,
    target_pose=None,
    target_joints=None,
    pos_tol_m=0.03,
    joint_tol_rad=0.05,
):
    """等待运动结束。

    move_j：优先看关节角偏差；无 target_joints 时看 motion_status。
    move_p（固件 1.10）：motion_status 常为 1 但机械臂已动，需传 target_pose，
    以实际法兰位置偏差判断成功与否。
    """
    time.sleep(0.5)
    start = time.monotonic()

    while time.monotonic() - start < timeout:
        if target_joints is not None:
            err = joint_angle_error(robot, target_joints)
            if err is not None and err <= joint_tol_rad:
                print(
                    f"运动完成（关节最大偏差 {math.degrees(err):.2f}° ≤ "
                    f"{math.degrees(joint_tol_rad):.1f}°）"
                )
                return True

        if target_pose is not None:
            err = pose_xyz_error(robot, target_pose)
            if err is not None and err <= pos_tol_m:
                print(
                    f"运动完成（位置偏差 {err * 1000:.1f} mm ≤ "
                    f"{pos_tol_m * 1000:.0f} mm；固件 1.10 move_p 常报 motion_status=1）"
                )
                return True

        st = robot.get_arm_status()
        if st is not None:
            ms = _enum_val(st.msg.motion_status)
            if ms == 0:
                if target_pose is None and target_joints is None:
                    print("运动完成（motion_status=0）")
                    return True
                if target_pose is not None:
                    err = pose_xyz_error(robot, target_pose)
                    if err is not None and err <= pos_tol_m:
                        print("运动完成（motion_status=0 且位置到位）")
                        return True
                if target_joints is not None:
                    err = joint_angle_error(robot, target_joints)
                    if err is not None and err <= joint_tol_rad:
                        print("运动完成（motion_status=0 且关节到位）")
                        return True
            if target_pose is None and target_joints is None and ms == 1:
                print("运动失败（motion_status=1）:")
                print_arm_status(robot, "  ")
                return False

        time.sleep(0.1)

    if target_joints is not None:
        err = joint_angle_error(robot, target_joints)
        if err is not None:
            print(
                f"超时，当前关节最大偏差 {math.degrees(err):.2f}°"
                f"（容差 {math.degrees(joint_tol_rad):.1f}°）"
            )
    elif target_pose is not None:
        err = pose_xyz_error(robot, target_pose)
        if err is not None:
            print(f"超时，当前位置偏差 {err * 1000:.1f} mm（容差 {pos_tol_m * 1000:.0f} mm）")
    else:
        print(f"等待超时（{timeout:.0f}s），当前状态:")
    print_arm_status(robot, "  ")
    return False


def wait_until_enabled(robot, timeout: float = 15.0) -> None:
    """循环 enable() 直到全部关节使能，或超时抛出 TimeoutError。"""
    status_list = robot.get_joints_enable_status_list()
    if status_list and all(status_list):
        print("enable OK（关节已使能）")
        print_arm_status(robot)
        return

    robot.set_normal_mode()
    robot.clear_joint_error(255)
    time.sleep(0.2)

    print("正在使能...", flush=True)
    t0 = time.monotonic()
    last_print = t0

    while time.monotonic() - t0 < timeout:
        if robot.enable():
            print("enable OK")
            print_arm_status(robot)
            return

        now = time.monotonic()
        if now - last_print >= 2.0:
            last_print = now
            elapsed = now - t0
            print(f"  仍在等待使能（{elapsed:.0f}s）...", flush=True)
            print_arm_status(robot, "  ")
            ja = robot.get_joint_angles()
            fp = robot.get_flange_pose()
            if ja is None and fp is None:
                print(
                    "  ⚠ 无关节/位姿反馈 → 检查机械臂上电、CAN 接线、"
                    "candump can1 是否有数据"
                )
            enables = robot.get_joints_enable_status_list()
            if enables:
                print(f"  关节使能: {sum(enables)}/7  {enables}")
            st = robot.get_arm_status()
            if st is not None:
                arm_i = _enum_val(st.msg.arm_status)
                if arm_i == 0x01:
                    print(
                        "  ⚠ arm_status=急停 → 请松开物理急停；"
                        "软件急停后需使能成功才能 reset()"
                    )
                elif arm_i == 0x06:
                    print("  ⚠ arm_status=抱闸未释放 → 检查抱闸/供电")

        time.sleep(0.01)

    print(f"使能超时（{timeout:.0f}s）。")
    print_arm_status(robot, "  ")
    raise TimeoutError(
        "机械臂使能失败。请检查：\n"
        "  1) 物理急停是否松开\n"
        "  2) 机械臂是否上电（系统终端: candump can1 应有数据）\n"
        "  3) can1 是否 UP: ip -br link show can1\n"
        "  4) 若 arm_status=急停，断电重启机械臂后再试"
    )


def connect_and_enable(robot, channel: str = "can1", enable_timeout: float = 15.0):
    from safe_arm import try_activate_can

    try_activate_can(channel)
    robot.connect()
    print("connect OK")
    wait_until_enabled(robot, timeout=enable_timeout)
