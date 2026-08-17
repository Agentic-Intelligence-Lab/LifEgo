#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采集桌面 AprilTag 的 top_left 角点在机器人 base 坐标系下的位置（供相机外参标定用）。

每个 tag 只碰 1 个点（top_left，即从相机方向看标签正面时的左上角），不用碰另外
3 个角——LifEgo/patches/calibrate_realsense_extrinsic_from_apriltags.py 会用固定
边长（--tag-size-m，默认 0.05m/50mm）在标定阶段联合求解相机外参和每个 tag 在桌面
内的朝向角，把另外 3 个角反推出来，不需要你告诉它 tag 具体转了多少度。

两种工作模式：

* 默认模式（主从臂/leader-follower 联动，本脚本连接的是从臂 follower）：
  不改动机械臂的 leader/follower 联动状态（不调用 set_leader_mode/
  set_follower_mode），假定你已经在用手拖动主臂、从臂跟随镜像运动。你只需要
  把从臂末端工具移动到每个 tag 的 top_left 角点，按回车用本脚本读取从臂自己的
  `get_tcp_pose()` 采集一次。若从臂不在默认 CAN 通道上，用 --channel 指定。

* --drag-teach 模式（单臂、无主从联动，直接零力拖动本机械臂）：
  连接后先进入零力拖动 (leader) 模式，操作者直接用手拖动这台机械臂本身
  的末端去碰角点，采集完成后恢复 follower（受控保持）模式。

本脚本不做使能 (enable)，假定机械臂已经处于使能状态（由外部流程/上一步完成）。

⚠️ --tcp-offset 是必填参数：`get_tcp_pose()` 返回的是法兰位姿按这个偏移量
变换后的结果，并不会自动等于工具尖端/实际接触点。必须填成"从法兰到实际碰
角点那一点"的精确偏移（米），沿法兰局部坐标系的 +X 方向（法兰→夹爪尖端
方向；不是 +Z——早期脚本/record_teleop.py 里用的 [0,0,0.13] 沿 +Z 是错的，
根据 LifEgo 那边用 URDF 重新核对法兰/夹爪几何后确认应该是 +X），比如你们
常用的 AgxGripper 默认值 0.13m 是"夹爪中心沿法兰 +X 伸出13cm"实测值，如果
你实际接触角点的位置不是这个夹爪中心（比如用的是指尖、探针，或夹爪张开/
闭合状态不同），必须换成对应的真实偏移，否则每个角点都会带一个系统性
偏差，标定出来的相机外参会整体偏移。

要采集几个 tag：用 -n/--num-tags（比如 -n 5 会依次采集 id 1..5），或者用
--tag-ids 明确指定不连续的 id 列表（如果不知道桌上贴的是哪些 id，先用
annotate_apriltags.py 拍照识别一下）。calibrate_realsense_extrinsic_from_
apriltags.py 那边也已同步支持任意数量 n 的 tag 联合标定，tag 越多、标定的
误差通常越小。

用法（--out 默认直接写到 LifEgo/examples/calib/tag_corners_base.json，即下一步
calibrate_realsense_extrinsic_from_apriltags.py 读取的位置，不用再手动复制；如需
另存别处再传 --out 覆盖）：
  # 主从臂模式（推荐，假设从臂已在跟随主臂运动）—— 采集 5 个 tag（id 1..5）
  python collect_apriltag_corners.py -n 5 --tcp-offset 0.13 --tag-size-m 0.05

  # 指定具体 id（不连续也可以）
  python collect_apriltag_corners.py --tag-ids 1 3 7 --tcp-offset 0.13 --tag-size-m 0.05

  # 单臂零力拖动模式
  python collect_apriltag_corners.py -n 5 --tcp-offset 0.13 --tag-size-m 0.05 --drag-teach

安全提示：--drag-teach 模式下机械臂会进入零力状态，手不扶住工具/末端可能因
重力下坠，脚本会在进入该模式前提示确认。
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from platform import system

import numpy as np

from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW

# .../LifEgo/patches/nero_datacollect/pyAgxArm-master/collect_apriltag_corners.py -> LifEgo
REPO_ROOT = Path(__file__).resolve().parents[3]
# Write straight into the location patches/calibrate_realsense_extrinsic_from_apriltags.py's
# --tag-corners-base example points at, so no manual copy step is needed between the two scripts.
DEFAULT_OUT = REPO_ROOT / "examples" / "calib" / "tag_corners_base.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect each AprilTag's top_left corner position in the robot base frame by touch.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("-f", "--firmware", default="v112", choices=["default", "v111", "v112"])
    p.add_argument("-c", "--channel", default=None, help="CAN channel (default: platform default)")
    p.add_argument(
        "-n",
        "--num-tags",
        type=int,
        default=None,
        help="How many tags to collect (n >= 1) -- more tags gives calibrate_realsense_extrinsic_"
        "from_apriltags.py more correspondences and reduces its solved reprojection error. Shorthand "
        "for --tag-ids 1 2 ... n (sequential ids starting at 1). Ignored if --tag-ids is given "
        "explicitly. Default if neither is given: 2 (tag ids 1, 2).",
    )
    p.add_argument(
        "--tag-ids",
        nargs="+",
        type=int,
        default=None,
        help="AprilTag ids to collect, in order (use annotate_apriltags.py first if you don't know "
        "which ids are printed on your tags). Overrides --num-tags; its length becomes n.",
    )
    p.add_argument(
        "--tag-size-m",
        type=float,
        default=0.05,
        help="Printed tag side length in meters (default 0.05 / 50mm). Stored alongside the collected "
        "top_left point so the calibration script can reconstruct the other 3 corners.",
    )
    p.add_argument(
        "--tcp-offset",
        type=float,
        required=True,
        help="REQUIRED. Exact offset along flange +X (meters, flange-local frame; +X points from flange "
        "toward the gripper tip -- confirmed against the URDF-derived flange/gripper geometry, NOT +Z) "
        "from the flange to whatever point actually touches the corner (fingertip / probe tip / "
        "gripper-center reference point) -- get_tcp_pose() does not know your tool, it just applies "
        "this offset. Wrong value = systematic bias in every point. Use 0 if the point that touches is "
        "the bare flange. The AgxGripper's measured gripper-center offset (0.13) is only correct if "
        "that exact point is what touches the corner.",
    )
    p.add_argument(
        "--drag-teach",
        action="store_true",
        help="Single-arm zero-force drag mode: toggle this arm's own leader/follower linkage "
        "(set_leader_mode() before capture, set_follower_mode() after) so you can drag its own "
        "end effector by hand. Do NOT use this if the arm is already running as the follower in an "
        "existing leader-follower rig -- it would fight/override that linkage. Default (off): don't "
        "touch the linkage at all; just poll this arm's own get_tcp_pose() while a human teleoperates "
        "it (e.g. by hand-driving a separate leader arm that this one mirrors).",
    )
    p.add_argument("--samples-per-point", type=int, default=8, help="Readings averaged per touch.")
    p.add_argument("--sample-interval", type=float, default=0.02, help="Seconds between readings within a touch.")
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output JSON path. Defaults to {DEFAULT_OUT} (where "
        "calibrate_realsense_extrinsic_from_apriltags.py's --tag-corners-base example reads from).",
    )
    args = p.parse_args()

    if args.tag_ids is not None:
        if args.num_tags is not None and args.num_tags != len(args.tag_ids):
            p.error(f"--num-tags {args.num_tags} doesn't match --tag-ids length {len(args.tag_ids)} ({args.tag_ids})")
    else:
        n = args.num_tags if args.num_tags is not None else 2
        if n < 1:
            p.error(f"--num-tags must be >= 1 (got {n})")
        args.tag_ids = list(range(1, n + 1))
    return args


def build_config(firmware: str, channel: str | None) -> dict:
    plat = system()
    # candleLight/gs_usb adapters (the ones this rig uses) are native USB
    # devices, not a serial port -- they never show up as /dev/tty.*, on any
    # OS. Darwin uses the same gs_usb interface as Windows; safe_arm.py's
    # _install_gs_usb_adapter() already carries a macOS-specific libusb patch
    # for it (_patch_gs_usb_start_for_macos).
    if channel is None:
        channel = {"Windows": "0", "Linux": "can1", "Darwin": "0"}.get(plat, "can0")
    interface = {"Windows": "gs_usb", "Linux": "socketcan", "Darwin": "gs_usb"}.get(plat, "socketcan")

    if interface == "gs_usb":
        from safe_arm import _install_gs_usb_adapter

        _install_gs_usb_adapter()

    fw = {"default": NeroFW.DEFAULT, "v111": NeroFW.V111, "v112": NeroFW.V112}.get(firmware, firmware)
    return create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=fw, interface=interface, channel=channel)


def wait_for_tcp_pose(robot, timeout: float = 5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        tcp = robot.get_tcp_pose()
        if tcp is not None:
            return tcp
        time.sleep(0.05)
    return None


def capture_point(robot, samples: int, interval: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Average `samples` TCP readings; returns (mean_xyz, std_xyz) or None on no feedback."""
    xyz = []
    for _ in range(samples):
        tcp = robot.get_tcp_pose()
        if tcp is not None:
            xyz.append(tcp.msg[:3])
        time.sleep(interval)
    if not xyz:
        return None
    arr = np.array(xyz, dtype=np.float64)
    return arr.mean(axis=0), arr.std(axis=0)


def collect(args: argparse.Namespace) -> None:
    cfg = build_config(args.firmware, args.channel)
    print(
        f"firmware={args.firmware}  interface={cfg['comm']['can']['interface']}  "
        f"channel={cfg['comm']['can']['channel']}  tcp_offset_x={args.tcp_offset}m"
    )

    robot = AgxArmFactory.create_arm(cfg)
    print("Connecting ...")
    robot.connect()
    print(f"Connected! joint_nums={robot.joint_nums}")

    if args.tcp_offset:
        robot.set_tcp_offset([args.tcp_offset, 0.0, 0.0, 0.0, 0.0, 0.0])

    aborted = False

    def on_int(sig, frame):
        nonlocal aborted
        aborted = True

    signal.signal(signal.SIGINT, on_int)

    if args.drag_teach:
        print("\n" + "=" * 70)
        print("即将进入零力拖动 (leader) 模式：机械臂会失去主动保持力。")
        print("请先用手扶稳工具/末端，再按回车继续（Ctrl+C 可随时中止并安全退出）。")
        print("=" * 70)
        input("按回车进入拖动模式 > ")
        robot.set_leader_mode()
        print("已进入拖动模式。现在可以用手拖动末端去碰角点。\n")
    else:
        print("\n" + "=" * 70)
        print("主从臂模式：不会修改本机械臂的 leader/follower 联动状态。")
        print("请用手拖动主臂，让本脚本连接的这台（从臂）跟随镜像运动，")
        print("把从臂末端工具移动到 top_left 角点后按回车采集。")
        print("=" * 70 + "\n")

    results: dict[int, list[float] | None] = {}
    stds: dict[int, list[float] | None] = {}

    try:
        for tag_id in args.tag_ids:
            if aborted:
                raise KeyboardInterrupt
            while True:
                print(f"\n[tag {tag_id} / top_left] 把工具尖端移动到该角点，")
                ans = input("按回车采集，或输入 s 跳过该 tag > ").strip().lower()
                if aborted:
                    raise KeyboardInterrupt
                if ans == "s":
                    results[tag_id] = None
                    stds[tag_id] = None
                    print("  已跳过。")
                    break

                tcp = wait_for_tcp_pose(robot)
                if tcp is None:
                    print("  未收到 TCP 反馈，检查 CAN 连接后重试。")
                    continue
                sample = capture_point(robot, args.samples_per_point, args.sample_interval)
                if sample is None:
                    print("  采样失败（无反馈），请重试。")
                    continue
                mean_xyz, std_xyz = sample
                std_mm = std_xyz * 1000.0
                print(f"  采到: x={mean_xyz[0]:.4f} y={mean_xyz[1]:.4f} z={mean_xyz[2]:.4f} (m)")
                print(f"  抖动: {std_mm[0]:.2f} / {std_mm[1]:.2f} / {std_mm[2]:.2f} mm (x/y/z)")
                if max(std_mm) > 2.0:
                    redo = input("  抖动偏大 (>2mm)，是否重新采集？[y/N] > ").strip().lower()
                    if redo == "y":
                        continue
                confirm = input("  接受该点？回车=接受，r=重新采集 > ").strip().lower()
                if confirm == "r":
                    continue
                results[tag_id] = mean_xyz.tolist()
                stds[tag_id] = std_xyz.tolist()
                break
    except KeyboardInterrupt:
        print("\n中止采集，正在安全退出 ...")
    finally:
        if args.drag_teach:
            robot.set_follower_mode()
            print("已恢复 follower 模式。")
        robot.disconnect()
        print("已断开连接。")

    skipped_tags = [tag_id for tag_id, v in results.items() if v is None]
    if skipped_tags:
        print(f"警告：以下 tag 被跳过，无法用于标定：{skipped_tags}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "units": "meters",
        "corner_captured": "top_left",
        "tag_size_m": args.tag_size_m,
        "tcp_offset_m": args.tcp_offset,
        "samples_per_point": args.samples_per_point,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "tags": {str(tag_id): v for tag_id, v in results.items() if v is not None},
        "capture_std_m": {str(tag_id): s for tag_id, s in stds.items() if s is not None},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    for tag_id, v in results.items():
        print(f"tag {tag_id}: {v}")


def main() -> int:
    args = parse_args()
    try:
        collect(args)
    except Exception as e:
        print(f"Failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
