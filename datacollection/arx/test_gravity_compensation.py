#!/usr/bin/env python3
"""Quick ARX gravity-compensation drift test.

This script only exercises ARX SDK control modes. It does not collect tags and
does not modify /home/arx/ARX5_beta. For custom payload compensation, pass a
copied URDF with a tuned link6 mass.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARX_SDK_ROOT = Path("/home/arx/ARX5_beta")
DEFAULT_CAN_PORTS = {"left": "can1", "right": "can3"}
ARM_TYPE_NAMES = {
    0: "X5-2023",
    2: "X5-2025",
    3: "A5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test ARX gravity compensation drift.")
    parser.add_argument("--arx-sdk-root", type=Path, default=DEFAULT_ARX_SDK_ROOT)
    parser.add_argument("--arm", choices=sorted(DEFAULT_CAN_PORTS), required=True)
    parser.add_argument("--can-port", default=None, help="Defaults to can1 for left, can3 for right.")
    parser.add_argument("--arm-type", type=int, default=2, choices=sorted(ARM_TYPE_NAMES))
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=None,
        help="Optional custom URDF path. Relative paths are resolved under this LifEgo checkout.",
    )
    parser.add_argument(
        "--gravity-scale",
        nargs=6,
        type=float,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="Optional per-joint gravity compensation scale.",
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Seconds to monitor drift.")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between pose prints.")
    parser.add_argument(
        "--protect-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call protect_mode() before closing (default: true).",
    )
    args = parser.parse_args()
    if args.can_port is None:
        args.can_port = DEFAULT_CAN_PORTS[args.arm]
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.interval <= 0:
        parser.error("--interval must be positive")
    if args.urdf_path is not None:
        args.urdf_path = args.urdf_path.expanduser()
        if not args.urdf_path.is_absolute():
            args.urdf_path = REPO_ROOT / args.urdf_path
        args.urdf_path = args.urdf_path.resolve()
        if not args.urdf_path.exists():
            parser.error(f"--urdf-path does not exist: {args.urdf_path}")
    return args


def import_arx_sdk(sdk_root: Path) -> Any:
    sdk_root = sdk_root.expanduser().resolve()
    if not sdk_root.exists():
        raise FileNotFoundError(f"ARX SDK root does not exist: {sdk_root}")
    if str(sdk_root) not in sys.path:
        sys.path.insert(0, str(sdk_root))
    try:
        from bimanual import SingleArm  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Could not import bimanual.SingleArm from {sdk_root}. "
            "Run this on the ARX host or pass --arx-sdk-root."
        ) from exc
    return SingleArm


def read_pose(arm: Any) -> np.ndarray:
    pose = np.asarray(arm.get_ee_pose_xyzrpy(), dtype=np.float64).reshape(-1)
    if pose.shape[0] < 6 or not np.isfinite(pose[:6]).all():
        raise RuntimeError(f"invalid ARX link6 pose: {pose}")
    return pose[:6].copy()


def main() -> int:
    args = parse_args()
    SingleArm = import_arx_sdk(args.arx_sdk_root)
    config: dict[str, Any] = {"can_port": args.can_port, "type": args.arm_type}
    if args.urdf_path is not None:
        config["urdf_path"] = str(args.urdf_path)
    if args.gravity_scale is not None:
        config["gravity_scale"] = args.gravity_scale

    print(f"Connecting {args.arm}: {config}")
    arm = SingleArm(config)
    interrupted = False

    def on_int(_sig, _frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, on_int)
    try:
        input("Hold the end effector, then press Enter to enter gravity compensation > ")
        ok = bool(arm.gravity_compensation())
        print(f"gravity_compensation -> {ok}")
        if not ok:
            return 1
        start_pose = read_pose(arm)
        start_time = time.monotonic()
        print("t_s, dx_mm, dy_mm, dz_mm, droll_deg, dpitch_deg, dyaw_deg, xyzrpy")
        while not interrupted:
            elapsed = time.monotonic() - start_time
            pose = read_pose(arm)
            delta = pose - start_pose
            print(
                f"{elapsed:.2f}, {delta[0] * 1000:+.2f}, {delta[1] * 1000:+.2f}, "
                f"{delta[2] * 1000:+.2f}, {np.degrees(delta[3]):+.2f}, "
                f"{np.degrees(delta[4]):+.2f}, {np.degrees(delta[5]):+.2f}, "
                + " ".join(f"{value:+.5f}" for value in pose),
                flush=True,
            )
            if elapsed >= args.duration:
                break
            time.sleep(args.interval)
    finally:
        if args.protect_on_exit:
            try:
                print(f"protect_mode -> {bool(arm.protect_mode())}")
            except Exception as exc:
                print(f"protect_mode failed: {exc}")
        try:
            arm.close()
        except Exception as exc:
            print(f"close failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
