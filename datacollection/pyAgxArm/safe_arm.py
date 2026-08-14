#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全控制封装：Nero 七轴机械臂 + 强脑 Revo2 灵巧手。

设计目标：在调用 SDK 原生运动接口的基础上，加一层"软地板（Z 下限）"保护，
避免灵巧手指尖在运动中撞到桌面，同时统一管理低速、关节软限位、碰撞防护、急停。

防撞三道闸（从主到辅）：
  1. TCP 偏置：把控制点从法兰移到灵巧手指尖，所有高度判断都基于"指尖"。
  2. Z 软地板：任何运动指令下发前先把目标指尖高度夹到地板之上；运动中持续监控，
     指尖逼近真实桌面立即急停。
  3. 低速 + 关节软限位 + 碰撞防护：作为底层兜底，撞到东西时自动停。

使用前务必按实物测量并修改 SafetyConfig 中的：
  hand_length / table_z / safe_margin / 坐标系 Z 方向。

依赖：pip3 install python-can  以及已安装 pyAgxArm（本仓库）。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from platform import system
from typing import List, Optional, Sequence

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class SafetyConfig:
    """安全与现场标定参数。所有几何量单位为米 (m)。

    重要：以下默认值仅为占位示例，必须按你的实际安装测量后修改！
    """

    # ---- 现场几何标定（必须按实物修改）----
    hand_length: float = 0.16
    """灵巧手指尖沿法兰局部 +X（法兰指向工具尖端方向；不是 +Z——已按 URDF 几何核对修正）
    的伸出长度。决定 TCP 偏置。"""

    table_z: float = 0.0
    """基座坐标系下桌面所在的 Z 高度（绝对禁区下边界）。"""

    safe_margin: float = 0.05
    """指尖距桌面保留的安全余量。指尖目标不会低于 table_z + safe_margin。"""

    # ---- 运动安全限制 ----
    debug_speed_percent: int = 15
    """全局速度比例（调试期建议 10~20），跑通后再逐步提高。"""

    max_linear_vel: float = 0.10
    """末端最大线速度 (m/s)。"""

    max_linear_acc: float = 0.30
    """末端最大线加速度 (m/s^2)。"""

    max_angular_vel: float = 0.40
    """末端最大角速度 (rad/s)。"""

    max_angular_acc: float = 0.40
    """末端最大角加速度 (rad/s^2)。"""

    crash_rating: int = 3
    """碰撞防护等级 0~8，0=关闭，越大越灵敏（越保守）。"""

    enable_joint_limits: bool = True
    """是否启用关节软限位。"""

    # ---- 固件 ----
    firmware: str = NeroFW.DEFAULT
    """Nero 固件：<=1.10 用 DEFAULT；==1.11 用 V111；>=1.12 用 V112。"""

    # ---- 监控 ----
    motion_timeout: float = 8.0
    """单次运动等待到位的超时 (s)。"""

    poll_interval: float = 0.02
    """运动中监控轮询间隔 (s)。"""

    disable_on_exit: bool = True
    """上下文退出时是否失能。抓取任务等需保持抱闸/使能的场景设为 False。"""

    @property
    def z_floor(self) -> float:
        """指尖允许到达的最低高度（软地板）。"""
        return self.table_z + self.safe_margin


# candleLight / gs_usb USB-CAN 适配器的 USB ID（bytewerk candleLight）
CANDLELIGHT_VID = 0x1D50
CANDLELIGHT_PID = 0x606F


def _auto_channel() -> tuple[str, str]:
    """按操作系统返回 (interface, channel)。

    Windows / macOS 上默认 candleLight ``gs_usb``；Linux 上优先 socketcan（gs_usb 驱动的网卡）。
    """
    plat = system()
    if plat == "Windows":
        # 本仓库现场使用 candleLight；不是松灵 agx_cando 专用棒。
        return "gs_usb", "0"
    if plat == "Linux":
        import subprocess

        try:
            out = subprocess.check_output(
                ["ip", "-br", "link", "show", "type", "can"], text=True
            )
            ifaces = [ln.split()[0] for ln in out.strip().splitlines() if ln.strip()]
            for iface in ifaces:
                try:
                    info = subprocess.check_output(
                        ["ethtool", "-i", iface], text=True, stderr=subprocess.DEVNULL
                    )
                    if "gs_usb" in info:
                        return "socketcan", iface
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            if "can1" in ifaces:
                return "socketcan", "can1"
            if ifaces:
                return "socketcan", ifaces[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        return "socketcan", "can0"
    if plat == "Darwin":
        return "gs_usb", "candlelight"
    raise RuntimeError(
        "仅支持 Linux(socketcan) / Windows(gs_usb) / macOS(gs_usb 或 slcan)，"
        "其它平台请显式传入 interface 与 channel。"
    )


def _ensure_windows_libusb_backend(vid: int = CANDLELIGHT_VID, pid: int = CANDLELIGHT_PID) -> None:
    """Windows 上为 pyusb / gs_usb 提供 libusb DLL，并避免硬编码空后端。"""
    if system() != "Windows":
        return

    import usb.core
    import usb.backend.libusb1 as libusb1
    from gs_usb.gs_usb import GsUsb

    try:
        import libusb_package
    except ImportError as exc:
        raise RuntimeError(
            "Windows 使用 candleLight/gs_usb 需要 libusb。请执行: "
            'pip install "python-can[gs-usb]" libusb-package'
        ) from exc

    backend = libusb1.get_backend(find_library=libusb_package.find_library)
    if backend is None:
        raise RuntimeError(
            "找不到 libusb 后端。请确认已安装: pip install libusb-package"
        )

    if getattr(GsUsb, "_agx_libusb_scan_patched", False):
        return

    @staticmethod
    def _scan_with_backend() -> list:
        return [
            GsUsb(dev)
            for dev in usb.core.find(
                find_all=True,
                idVendor=vid,
                idProduct=pid,
                backend=backend,
            )
        ]

    GsUsb.scan = _scan_with_backend  # type: ignore[method-assign, assignment]
    GsUsb._agx_libusb_scan_patched = True


def _patch_gs_usb_start_for_macos() -> None:
    """修复 macOS 上 gs_usb 库 start() 调用 is_kernel_driver_active 抛错的问题。

    gs_usb 的 start() 为 Linux 写死了内核驱动 detach 逻辑，但 macOS 的 libusb
    后端不支持 is_kernel_driver_active / detach_kernel_driver，会抛
    USBError [Errno 2] Entity not found。macOS 上没有内核驱动 claim candleLight，
    本就不需要 detach，这里用一个安全版 start 覆盖它（把内核驱动操作包在
    try/except 里）。
    """
    import platform as _platform

    if _platform.system() != "Darwin":
        return

    import usb.core
    from gs_usb import gs_usb as _gs_mod
    from gs_usb.gs_usb import GsUsb
    from gs_usb.gs_usb_structures import DeviceMode
    from gs_usb.constants import (
        GS_CAN_MODE_NORMAL,
        GS_CAN_MODE_HW_TIMESTAMP,
        GS_CAN_MODE_LISTEN_ONLY,
        GS_CAN_MODE_LOOP_BACK,
        GS_CAN_MODE_ONE_SHOT,
    )

    if getattr(GsUsb, "_agx_macos_start_patched", False):
        return

    def _safe_start(self, flags=(GS_CAN_MODE_NORMAL | GS_CAN_MODE_HW_TIMESTAMP)):
        try:
            self.gs_usb.reset()
        except Exception:
            pass
        # macOS libusb 不支持内核驱动查询/分离，忽略其错误
        try:
            if self.gs_usb.is_kernel_driver_active(0):
                self.gs_usb.detach_kernel_driver(0)
        except Exception:
            pass
        flags &= self.device_capability.feature
        flags &= (
            GS_CAN_MODE_LISTEN_ONLY
            | GS_CAN_MODE_LOOP_BACK
            | GS_CAN_MODE_ONE_SHOT
            | GS_CAN_MODE_HW_TIMESTAMP
        )
        self.device_flags = flags
        mode = DeviceMode(_gs_mod.GS_CAN_MODE_START, flags)
        self.gs_usb.ctrl_transfer(0x41, _gs_mod._GS_USB_BREQ_MODE, 0, 0, mode.pack())

    GsUsb.start = _safe_start
    GsUsb._agx_macos_start_patched = True


def _install_gs_usb_adapter(vid: int = CANDLELIGHT_VID, pid: int = CANDLELIGHT_PID) -> None:
    """让 pyAgxArm 能用 python-can 的 gs_usb 接口打开 candleLight。

    pyAgxArm 内部会传入 ``receive_own_messages`` / ``local_loopback`` 等 gs_usb
    不支持的参数。Windows 上还要注入 libusb DLL，并用 index=0 打开设备。

    注意：不要读取 ``usb.core.Device.product`` 字符串——在 Windows 上可能占住
    设备句柄，导致随后 open 报 Access denied。
    """
    _patch_gs_usb_start_for_macos()
    _ensure_windows_libusb_backend(vid=vid, pid=pid)
    import can.interface as _ci

    if getattr(_ci, "_agx_gs_usb_patched", False):
        return

    _orig_bus = _ci.Bus

    def _bus(*args, **kwargs):
        if kwargs.get("interface") == "gs_usb":
            # 强制用 index 打开；channel 仅作透传标签
            kwargs["channel"] = 0
            kwargs["index"] = 0
            kwargs.pop("local_loopback", None)
            kwargs.pop("receive_own_messages", None)
        return _orig_bus(*args, **kwargs)

    _ci.Bus = _bus
    _ci._agx_gs_usb_patched = True


class FloorViolation(Exception):
    """目标位姿低于软地板且无法被安全夹紧时抛出。"""


def _is_can_interface_up(channel: str) -> bool:
    import subprocess

    try:
        out = subprocess.check_output(
            ["ip", "link", "show", channel], text=True, stderr=subprocess.DEVNULL
        )
        return "state UP" in out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def try_activate_can(channel: str = "can1", bitrate: int = 1_000_000) -> bool:
    """尝试激活 CAN（需已运行 install_can1_autostart.sh 配置免密 sudo）。"""
    import subprocess

    if _is_can_interface_up(channel):
        return True

    for cmd in (
        ["sudo", "-n", "/usr/local/sbin/agx-can-up.sh", channel, str(bitrate)],
        ["sudo", "-n", "/usr/local/sbin/agx-can-up.sh", channel],
        ["sudo", "-n", "/usr/local/sbin/agx-can-up.sh"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and _is_can_interface_up(channel):
                print(f"[CAN] 已自动激活 {channel}")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return _is_can_interface_up(channel)


# --------------------------------------------------------------------------- #
# 安全机械臂封装
# --------------------------------------------------------------------------- #
class SafeNeroArm:
    """Nero + Revo2 的安全控制封装。

    典型用法::

        cfg = SafetyConfig(hand_length=0.16, table_z=0.0, safe_margin=0.05)
        with SafeNeroArm(cfg) as arm:
            arm.open_hand()
            arm.move_tcp_p([-0.4, 0.0, 0.20, -1.5708, 0.0, -3.14159])
    """

    def __init__(
        self,
        config: Optional[SafetyConfig] = None,
        interface: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> None:
        self.cfg = config or SafetyConfig()
        if interface is None or channel is None:
            auto_if, auto_ch = _auto_channel()
            interface = interface or auto_if
            channel = channel or auto_ch
        self._interface = interface
        self._channel = channel

        # macOS 的 candleLight 需要 gs_usb 适配（修正 channel/index、剔除不支持的参数）
        if interface == "gs_usb":
            _install_gs_usb_adapter()

        arm_cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=self.cfg.firmware,
            interface=interface,
            channel=channel,
        )
        self.robot = AgxArmFactory.create_arm(arm_cfg)
        # init_effector 只能调用一次，且建议在 connect 之前
        self.hand = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.REVO2)
        self._connected = False
        self._current_speed = self.cfg.debug_speed_percent

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def connect(self, enable_timeout: float = 5.0) -> "SafeNeroArm":
        """连接、使能并下发全部安全配置。"""
        # socketcan 才有 ``ip link``；gs_usb / Windows 直接走 SDK 打开。
        if self._interface == "socketcan" and not _is_can_interface_up(self._channel):
            if try_activate_can(self._channel):
                pass
            elif not _is_can_interface_up(self._channel):
                raise RuntimeError(
                    f"CAN 接口 {self._channel} 未激活。\n"
                    "一次性安装自动激活（系统终端执行）：\n"
                    "  cd ~/Downloads/pyAgxArm-master/scripts/jetson\n"
                    "  sudo bash install_can1_autostart.sh\n"
                    "或手动：\n"
                    f"  sudo ip link set {self._channel} type can bitrate 1000000\n"
                    f"  sudo ip link set {self._channel} up"
                )

        print(f"使用 CAN: {self._interface} / {self._channel}")
        self.robot.connect()
        self._connected = True

        from nero_motion_utils import wait_until_enabled

        wait_until_enabled(self.robot, timeout=enable_timeout)

        self._apply_safety_config()
        return self

    def _apply_safety_config(self) -> None:
        """统一下发：速度、软限位、末端限速、碰撞防护、TCP 偏置。"""
        c = self.cfg
        self.set_speed(c.debug_speed_percent)
        self.robot.set_joint_limits_enabled(c.enable_joint_limits)
        self.robot.set_flange_vel_acc_limits(
            max_linear_vel=c.max_linear_vel,
            max_angular_vel=c.max_angular_vel,
            max_linear_acc=c.max_linear_acc,
            max_angular_acc=c.max_angular_acc,
        )
        self.robot.set_crash_protection_rating(joint_index=255, rating=c.crash_rating)
        # 把控制点从法兰移到指尖（法兰局部 +X 方向），后续所有高度判断基于指尖
        self.robot.set_tcp_offset([c.hand_length, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(0.1)

    def __enter__(self) -> "SafeNeroArm":
        return self.connect()

    def __exit__(self, exc_type, exc, tb) -> None:
        # 异常退出时优先急停保护（急停 ≠ 失能）
        if exc_type is not None:
            try:
                self.emergency_stop()
            except Exception:
                pass
        self.shutdown()

    def shutdown(self) -> None:
        """安全收尾。默认失能；disable_on_exit=False 时保持使能。"""
        if not self.cfg.disable_on_exit:
            return
        try:
            self.robot.disable()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 速度控制
    # ------------------------------------------------------------------ #
    def set_speed(self, percent: int) -> None:
        """设置全局速度比例（1~100），并记录当前值。"""
        percent = max(1, min(100, int(percent)))
        self.robot.set_speed_percent(percent)
        self._current_speed = percent

    @contextmanager
    def _temp_speed(self, percent: Optional[int]):
        """临时切换速度，运动完成后恢复原速度。"""
        if percent is None or percent == self._current_speed:
            yield
            return
        prev = self._current_speed
        self.set_speed(percent)
        try:
            yield
        finally:
            self.set_speed(prev)

    # ------------------------------------------------------------------ #
    # 防撞核心逻辑
    # ------------------------------------------------------------------ #
    def clamp_tcp_z(self, tcp_pose: Sequence[float]) -> List[float]:
        """把目标 TCP（指尖）位姿的 Z 夹到软地板之上。"""
        pose = list(tcp_pose)
        if pose[2] < self.cfg.z_floor:
            print(
                f"[安全] 目标指尖 Z={pose[2]:.3f} 低于软地板 {self.cfg.z_floor:.3f}，已夹紧。"
            )
            pose[2] = self.cfg.z_floor
        return pose

    def _fingertip_z_for_joints(self, joints: Sequence[float]) -> float:
        """离线预判：给定关节角，计算指尖在基座系的 Z 高度。"""
        flange_pose = self.robot.fk(list(joints))
        tcp_pose = self.robot.get_flange2tcp_pose(flange_pose)
        return tcp_pose[2]

    def current_fingertip_z(self) -> Optional[float]:
        """读取当前指尖 Z 高度（无数据返回 None）。"""
        tcp = self.robot.get_tcp_pose()
        return None if tcp is None else tcp.msg[2]

    def _pose_xyz_error(
        self, current: Sequence[float], target: Sequence[float]
    ) -> Optional[float]:
        import math

        return math.sqrt(sum((current[i] - target[i]) ** 2 for i in range(3)))

    def _joint_angle_error(
        self, target_joints: Sequence[float]
    ) -> Optional[float]:
        ja = self.robot.get_joint_angles()
        if ja is None:
            return None
        n = min(len(ja.msg), len(target_joints))
        return max(abs(ja.msg[i] - target_joints[i]) for i in range(n))

    def _wait_motion_done_guarded(
        self,
        timeout: Optional[float] = None,
        target_tcp: Optional[Sequence[float]] = None,
        target_flange: Optional[Sequence[float]] = None,
        target_joints: Optional[Sequence[float]] = None,
        pos_tol_m: float = 0.08,
        joint_tol_rad: float = 0.05,
    ) -> bool:
        """等待运动到位，期间持续监控指尖高度；逼近真实桌面立即急停。"""
        timeout = self.cfg.motion_timeout if timeout is None else timeout
        time.sleep(0.3)
        t0 = time.monotonic()
        while True:
            status = self.robot.get_arm_status()
            z = self.current_fingertip_z()

            if z is not None and z < self.cfg.table_z:
                print(f"[安全] 指尖 Z={z:.3f} 已逼近桌面，触发急停！")
                self.emergency_stop()
                return False

            if target_joints is not None:
                err = self._joint_angle_error(target_joints)
                if err is not None and err <= joint_tol_rad:
                    return True

            if target_flange is not None:
                fp = self.robot.get_flange_pose()
                if fp is not None:
                    err = self._pose_xyz_error(fp.msg, target_flange)
                    if err is not None and err <= pos_tol_m:
                        return True

            if target_tcp is not None:
                tcp = self.robot.get_tcp_pose()
                if tcp is not None:
                    err = self._pose_xyz_error(tcp.msg, target_tcp)
                    if err is not None and err <= pos_tol_m:
                        return True

            if status is not None:
                ms = getattr(status.msg.motion_status, "value", status.msg.motion_status)
                if ms == 0:
                    return True

            if time.monotonic() - t0 > timeout:
                print(f"[安全] 等待运动结束超时（{timeout:.1f}s）。")
                self._print_motion_debug(target_tcp, target_flange, pos_tol_m)
                return False

            time.sleep(self.cfg.poll_interval)

    def _print_motion_debug(
        self,
        target_tcp: Optional[Sequence[float]],
        target_flange: Optional[Sequence[float]] = None,
        pos_tol_m: float = 0.08,
    ) -> None:
        st = self.robot.get_arm_status()
        if st is not None:
            m = st.msg
            ms = getattr(m.motion_status, "value", m.motion_status)
            arm = getattr(m.arm_status, "value", m.arm_status)
            arm_names = {0: "正常", 2: "无解(IK失败)", 3: "奇异点", 4: "目标超限"}
            print(
                f"  arm_status={arm_names.get(arm, arm)}  "
                f"motion_status={ms}  ctrl_mode={getattr(m.ctrl_mode, 'value', m.ctrl_mode)}"
            )
        tcp = self.robot.get_tcp_pose()
        fp = self.robot.get_flange_pose()
        if fp is not None:
            from nero_motion_utils import format_pose_line

            print("  " + format_pose_line(fp.msg, "当前法兰"))
        if target_flange is not None:
            print("  " + format_pose_line(list(target_flange), "目标法兰"))
            if fp is not None:
                err = self._pose_xyz_error(fp.msg, target_flange)
                if err is not None:
                    print(f"  法兰位置偏差: {err * 1000:.1f} mm（容差 {pos_tol_m * 1000:.0f} mm）")
        if tcp is not None:
            from nero_motion_utils import format_pose_line

            print("  " + format_pose_line(tcp.msg, "当前 TCP"))
        if target_tcp is not None:
            print("  " + format_pose_line(list(target_tcp), "目标 TCP"))
            if tcp is not None:
                err = self._pose_xyz_error(tcp.msg, target_tcp)
                if err is not None:
                    print(f"  TCP位置偏差: {err * 1000:.1f} mm（容差 {pos_tol_m * 1000:.0f} mm）")
        if st is not None and getattr(st.msg.arm_status, "value", st.msg.arm_status) == 2:
            print("  → IK 无解：从当前姿态无法一步到达该 TCP 位姿，请改小目标或先 move_j 到中间姿态")

    # ------------------------------------------------------------------ #
    # 受保护的运动接口
    # ------------------------------------------------------------------ #

    def relative_flange(
        self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0
    ) -> List[float]:
        """从当前法兰位姿偏移（保持姿态）。固件 1.10 竖直零位下比 TCP 更可靠。"""
        fp = self.robot.get_flange_pose()
        if fp is None:
            raise RuntimeError("无法读取法兰位姿")
        p = list(fp.msg)
        p[0] += dx
        p[1] += dy
        p[2] += dz
        return p

    def _check_flange_tcp_floor(self, flange_pose: Sequence[float]) -> None:
        tcp = self.robot.get_flange2tcp_pose(list(flange_pose))
        if tcp[2] < self.cfg.z_floor:
            raise FloorViolation(
                f"拒绝执行：目标指尖 Z={tcp[2]:.3f} 低于软地板 {self.cfg.z_floor:.3f}"
            )

    def move_flange_p(
        self,
        flange_pose: Sequence[float],
        wait: bool = True,
        timeout: Optional[float] = None,
        speed_percent: Optional[int] = None,
        pos_tol_m: float = 0.05,
    ) -> bool:
        """点到点运动到法兰位姿（竖直姿态下比 move_tcp_p 更稳定）。"""
        target = list(flange_pose)
        self._check_flange_tcp_floor(target)
        with self._temp_speed(speed_percent):
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.P)
            self.robot.move_p(target)
            return (
                self._wait_motion_done_guarded(
                    timeout, target_flange=target, pos_tol_m=pos_tol_m
                )
                if wait
                else True
            )

    def move_flange_l(
        self,
        flange_pose: Sequence[float],
        wait: bool = True,
        timeout: Optional[float] = None,
        speed_percent: Optional[int] = None,
        pos_tol_m: float = 0.05,
    ) -> bool:
        """直线运动到法兰位姿。"""
        target = list(flange_pose)
        self._check_flange_tcp_floor(target)
        with self._temp_speed(speed_percent):
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.L)
            self.robot.move_l(target)
            return (
                self._wait_motion_done_guarded(
                    timeout, target_flange=target, pos_tol_m=pos_tol_m
                )
                if wait
                else True
            )

    def move_tcp_p(
        self,
        tcp_pose: Sequence[float],
        wait: bool = True,
        timeout: Optional[float] = None,
        speed_percent: Optional[int] = None,
    ) -> bool:
        """点到点运动到目标【指尖】位姿（自动做地板保护）。

        tcp_pose: [x, y, z, roll, pitch, yaw]，指的是指尖（TCP）在基座系的位姿。
        speed_percent: 本次运动临时速度（1~100），运动后自动恢复；None=用当前速度。
        """
        safe_tcp = self.clamp_tcp_z(tcp_pose)
        flange_pose = self.robot.get_tcp2flange_pose(safe_tcp)
        with self._temp_speed(speed_percent):
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.P)
            self.robot.move_p(flange_pose)
            return (
                self._wait_motion_done_guarded(timeout, target_tcp=safe_tcp)
                if wait
                else True
            )

    def move_tcp_l(
        self,
        tcp_pose: Sequence[float],
        wait: bool = True,
        timeout: Optional[float] = None,
        speed_percent: Optional[int] = None,
    ) -> bool:
        """直线运动到目标【指尖】位姿（自动做地板保护）。

        注意：move_l 底层每收到新点都会重新规划，不可用于连续高频下发。
        speed_percent: 本次运动临时速度（1~100），运动后自动恢复；None=用当前速度。
        """
        safe_tcp = self.clamp_tcp_z(tcp_pose)
        flange_pose = self.robot.get_tcp2flange_pose(safe_tcp)
        with self._temp_speed(speed_percent):
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.L)
            self.robot.move_l(flange_pose)
            return (
                self._wait_motion_done_guarded(timeout, target_tcp=safe_tcp)
                if wait
                else True
            )

    def move_j(
        self,
        joints: Sequence[float],
        wait: bool = True,
        timeout: Optional[float] = None,
        speed_percent: Optional[int] = None,
    ) -> bool:
        """关节运动，但运动前先用 FK 预判指尖高度，低于软地板则拒绝执行。"""
        predicted_z = self._fingertip_z_for_joints(joints)
        if predicted_z < self.cfg.z_floor:
            raise FloorViolation(
                f"拒绝执行：该关节目标会让指尖到 Z={predicted_z:.3f}，"
                f"低于软地板 {self.cfg.z_floor:.3f}。"
            )
        with self._temp_speed(speed_percent):
            self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.J)
            self.robot.move_j(list(joints))
            return (
                self._wait_motion_done_guarded(timeout, target_joints=list(joints))
                if wait
                else True
            )

    # ------------------------------------------------------------------ #
    # 急停 / 复位
    # ------------------------------------------------------------------ #
    def emergency_stop(self) -> None:
        """电子急停（关节带恒定阻尼缓降，不会硬砸下来）。"""
        self.robot.electronic_emergency_stop()

    def reset(self) -> None:
        """急停后复位（需先处于急停且使能状态）。"""
        self.robot.reset()

    def clear_errors(self) -> None:
        """清除全部关节错误码。"""
        self.robot.clear_joint_error(255)

    # ------------------------------------------------------------------ #
    # Revo2 灵巧手便捷封装（0=张开/伸直，100=闭合/收拢）
    # ------------------------------------------------------------------ #
    def open_hand(self) -> None:
        """五指全部张开/归零。"""
        self.hand.position_ctrl()

    def close_hand(self, level: int = 100) -> None:
        """五指闭合到指定程度（0~100）。"""
        level = self._clip_finger(level)
        self.hand.position_ctrl(
            thumb_tip=level,
            thumb_base=level,
            index_finger=level,
            middle_finger=level,
            ring_finger=level,
            pinky_finger=level,
        )

    def set_fingers(
        self,
        thumb_tip: int = 0,
        thumb_base: int = 0,
        index_finger: int = 0,
        middle_finger: int = 0,
        ring_finger: int = 0,
        pinky_finger: int = 0,
    ) -> None:
        """逐指位置控制（每个值 0~100，自动裁剪到合法范围）。"""
        self.hand.position_ctrl(
            thumb_tip=self._clip_finger(thumb_tip),
            thumb_base=self._clip_finger(thumb_base),
            index_finger=self._clip_finger(index_finger),
            middle_finger=self._clip_finger(middle_finger),
            ring_finger=self._clip_finger(ring_finger),
            pinky_finger=self._clip_finger(pinky_finger),
        )

    def close_hand_until_contact(
        self,
        target_level: int = 100,
        step: int = 5,
        current_limit: int = 40,
        settle: float = 0.06,
    ) -> int:
        """逐步闭合五指，任一指电流超过 current_limit 即停止（判定抓稳）。

        target_level: 最大闭合程度（0~100）。
        step: 每次递增的闭合量。
        current_limit: 电流阈值（0~100），超过即认为接触到物体。
        settle: 每步之间的等待时间 (s)。
        返回最终下发的闭合程度。
        """
        target_level = self._clip_finger(target_level)
        level = 0
        while level < target_level:
            level = min(target_level, level + step)
            self.close_hand(level)
            time.sleep(settle)
            fc = self.hand.get_finger_current()
            if fc is not None:
                peak = max(
                    abs(getattr(fc.msg, name, 0))
                    for name in (
                        "thumb_tip", "thumb_base", "index_finger",
                        "middle_finger", "ring_finger", "pinky_finger",
                    )
                )
                if peak >= current_limit:
                    print(f"[抓取] 接触到物体（峰值电流 {peak} ≥ {current_limit}），停止闭合。level={level}")
                    return level
        print(f"[抓取] 已闭合到目标 {level}（未触发电流阈值）。")
        return level

    @staticmethod
    def _clip_finger(v: int) -> int:
        return max(0, min(100, int(v)))


# --------------------------------------------------------------------------- #
# 示例
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # 按你的实物安装修改这些数值！
    cfg = SafetyConfig(
        hand_length=0.16,   # 灵巧手指尖伸出长度
        table_z=0.0,        # 桌面在基座系的高度
        safe_margin=0.05,   # 指尖离桌面安全余量
        debug_speed_percent=15,
        crash_rating=3,
    )

    with SafeNeroArm(cfg) as arm:
        print("软地板高度 z_floor =", cfg.z_floor)

        # 1) 灵巧手先张开，避免运动中手指剐蹭
        arm.open_hand()
        time.sleep(0.5)

        # 2) 安全移动到桌面上方（指尖坐标）
        arm.move_tcp_p([-0.4, 0.0, 0.20, -1.5708, 0.0, -3.14159])

        # 3) 故意给一个会撞桌的目标（z=-0.10），会被自动夹到软地板之上
        arm.move_tcp_p([-0.4, 0.0, -0.10, -1.5708, 0.0, -3.14159])

        # 4) 抓取动作：靠近后闭合手指
        arm.close_hand(80)
        time.sleep(0.5)

        # 5) 抬起离开
        arm.move_tcp_l([-0.4, 0.0, 0.30, -1.5708, 0.0, -3.14159])

        print("演示结束。")
