#!/usr/bin/env python3
"""Collect AprilTag top-left corner positions in an ARX base frame.

This script is the ARX counterpart of datacollection/pyAgxArm/
collect_apriltag_corners.py.  It writes the same tag_corners_base.json shape
consumed by patches/calibrate_realsense_extrinsic_from_apriltags.py:

    {
      "units": "meters",
      "corner_captured": "top_left",
      "tag_size_m": 0.05,
      "tags": {"1": [x, y, z]}
    }

Run it from the LifEgo checkout on the ARX host, or on a machine that can
import the ARX SDK. It does not modify /home/arx/ARX5_beta; it only inserts
that directory into sys.path.

Important frame convention:
  - ARX SDK get_ee_pose_xyzrpy() returns the pose of ee_link == link6.
  - link6 is the arm end link/flange frame, not the gripper fingertip.
  - At the documented zero pose, link6 is aligned with the ARX reference/base
    frame. During motion, its local +X/+Y/+Z axes rotate with the wrist.
  - --tcp-offset X Y Z is a translation from link6 origin to the physical
    contact point, expressed in the moving link6 local frame, in meters.

Before running in direct CAN mode, stop the existing ARX control/data station
processes if they hold the same arm/camera:

    systemctl --user stop arx-data-station.service
    systemctl --user stop arx-button-control.service

Example:

    python datacollection/arx/collect_apriltag_corners.py \\
        --arm left --tag-ids 1 2 3 --tag-size-m 0.05 --tcp-offset 0 0 0

With single-arm hand-guided gravity compensation:

    python datacollection/arx/collect_apriltag_corners.py \\
        --arm left --tag-ids 1 2 3 --tag-size-m 0.05 --tcp-offset X Y Z \\
        --drag-teach
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARX_SDK_ROOT = Path("/home/arx/ARX5_beta")
DEFAULT_OUT_DIR = REPO_ROOT / "examples" / "calib"
DEFAULT_CAN_PORTS = {"left": "can1", "right": "can3"}
ARM_TYPE_NAMES = {
    0: "X5-2023",
    2: "X5-2025",
    3: "A5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect AprilTag top_left corner positions with an ARX arm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arx-sdk-root",
        type=Path,
        default=DEFAULT_ARX_SDK_ROOT,
        help=f"Path containing the bimanual ARX SDK package (default: {DEFAULT_ARX_SDK_ROOT})",
    )
    parser.add_argument(
        "--arm",
        choices=sorted(DEFAULT_CAN_PORTS),
        required=True,
        help="Which physical arm/base frame this calibration is for.",
    )
    parser.add_argument(
        "--can-port",
        default=None,
        help="ARX CAN port. Defaults to can1 for left, can3 for right.",
    )
    parser.add_argument(
        "--arm-type",
        type=int,
        default=2,
        choices=sorted(ARM_TYPE_NAMES),
        help="ARX model type: 0=X5-2023, 2=X5-2025, 3=A5 (default: 2).",
    )
    parser.add_argument(
        "-n",
        "--num-tags",
        type=int,
        default=None,
        help="Collect sequential tag ids 1..N. Ignored when --tag-ids is provided.",
    )
    parser.add_argument(
        "--tag-ids",
        nargs="+",
        type=int,
        default=None,
        help="AprilTag ids to collect, in order.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=0.05,
        help="Printed tag side length in meters (default: 0.05).",
    )
    parser.add_argument(
        "--tcp-offset",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help=(
            "Required translation from ARX link6 origin to the physical contact point, "
            "expressed in the link6 local frame, meters. Use '0 0 0' only if the link6 "
            "origin itself is the point touching the AprilTag corner."
        ),
    )
    parser.add_argument(
        "--drag-teach",
        action="store_true",
        help=(
            "Enter gravity_compensation() before collecting so the operator can hand-guide "
            "the arm. On exit, the script calls protect_mode(). Do not use while another "
            "process owns the same arm."
        ),
    )
    parser.add_argument(
        "--protect-on-exit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call protect_mode() during shutdown after --drag-teach (default: true).",
    )
    parser.add_argument("--samples-per-point", type=int, default=8, help="Pose readings averaged per touch.")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.02,
        help="Seconds between readings within a touch.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: examples/calib/tag_corners_base_arx_<arm>.json "
            "under this LifEgo checkout."
        ),
    )
    args = parser.parse_args()

    if args.tag_ids is not None:
        if args.num_tags is not None and args.num_tags != len(args.tag_ids):
            parser.error(
                f"--num-tags {args.num_tags} does not match --tag-ids length "
                f"{len(args.tag_ids)} ({args.tag_ids})"
            )
    else:
        n_tags = args.num_tags if args.num_tags is not None else 2
        if n_tags < 1:
            parser.error(f"--num-tags must be >= 1 (got {n_tags})")
        args.tag_ids = list(range(1, n_tags + 1))

    if args.tag_size_m <= 0:
        parser.error("--tag-size-m must be positive")
    if args.samples_per_point < 1:
        parser.error("--samples-per-point must be >= 1")
    if args.sample_interval < 0:
        parser.error("--sample-interval must be >= 0")
    if args.out is None:
        args.out = DEFAULT_OUT_DIR / f"tag_corners_base_arx_{args.arm}.json"
    if args.can_port is None:
        args.can_port = DEFAULT_CAN_PORTS[args.arm]
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


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return local-to-base rotation using the standard ROS/URDF RPY convention.

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). This matrix is used only to rotate the
    user-provided local TCP offset into the base frame.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def link6_pose_to_touch_xyz(link6_xyzrpy: np.ndarray, tcp_offset: np.ndarray) -> np.ndarray:
    pose = np.asarray(link6_xyzrpy, dtype=np.float64).reshape(-1)
    if pose.shape[0] < 6:
        raise ValueError(f"ARX link6 pose must have at least 6 values, got shape {pose.shape}")
    if not np.isfinite(pose[:6]).all():
        raise ValueError(f"ARX link6 pose contains non-finite values: {pose[:6]}")
    rotation = rpy_to_matrix(float(pose[3]), float(pose[4]), float(pose[5]))
    return pose[:3] + rotation @ tcp_offset


def read_link6_pose(arm: Any) -> np.ndarray | None:
    pose = arm.get_ee_pose_xyzrpy()
    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 6 or not np.isfinite(arr[:6]).all():
        return None
    return arr[:6].copy()


def wait_for_link6_pose(arm: Any, timeout: float = 5.0) -> np.ndarray | None:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        pose = read_link6_pose(arm)
        if pose is not None:
            return pose
        time.sleep(0.05)
    return None


def capture_point(
    arm: Any,
    tcp_offset: np.ndarray,
    samples: int,
    interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    touch_xyz_samples = []
    link6_pose_samples = []
    for _ in range(samples):
        pose = read_link6_pose(arm)
        if pose is not None:
            link6_pose_samples.append(pose)
            touch_xyz_samples.append(link6_pose_to_touch_xyz(pose, tcp_offset))
        time.sleep(interval)
    if not touch_xyz_samples:
        return None
    touch_arr = np.asarray(touch_xyz_samples, dtype=np.float64)
    pose_arr = np.asarray(link6_pose_samples, dtype=np.float64)
    return (
        touch_arr.mean(axis=0),
        touch_arr.std(axis=0),
        pose_arr.mean(axis=0),
        pose_arr.std(axis=0),
    )


def print_link6_frame_hint(tcp_offset: np.ndarray) -> None:
    print("ARX link6 frame:")
    print("  origin: SDK end-effector pose, ee_link == link6; not the gripper fingertip")
    print("  local axes: +X/+Y/+Z are link6 axes; at documented zero pose they align with the base/reference frame")
    print("  tcp offset: interpreted in link6 local frame and rotated by link6 RPY into base frame")
    print(f"  current --tcp-offset: {tcp_offset.tolist()} m")


def collect(args: argparse.Namespace) -> None:
    SingleArm = import_arx_sdk(args.arx_sdk_root)
    tcp_offset = np.asarray(args.tcp_offset, dtype=np.float64)
    print_link6_frame_hint(tcp_offset)
    print(
        f"Connecting ARX {args.arm} arm: can_port={args.can_port} "
        f"type={args.arm_type} ({ARM_TYPE_NAMES[args.arm_type]})"
    )

    arm = SingleArm({"can_port": args.can_port, "type": args.arm_type})
    aborted = False

    def on_int(_sig, _frame):
        nonlocal aborted
        aborted = True

    signal.signal(signal.SIGINT, on_int)

    if args.drag_teach:
        print("\nAbout to enter gravity_compensation mode. Hold the tool/end effector before continuing.")
        input("Press Enter to enter drag/teach mode, or Ctrl+C to abort > ")
        ok = bool(arm.gravity_compensation())
        if not ok:
            raise RuntimeError("ARX gravity_compensation() returned false")
        print("Entered gravity compensation. Move the physical contact point to each top_left corner.\n")
    else:
        print("\nDirect read mode: move the arm externally, then press Enter at each top_left corner.\n")

    results: dict[int, list[float] | None] = {}
    stds: dict[int, list[float] | None] = {}
    link6_means: dict[int, list[float]] = {}
    link6_stds: dict[int, list[float]] = {}

    try:
        for tag_id in args.tag_ids:
            if aborted:
                raise KeyboardInterrupt
            while True:
                print(f"\n[tag {tag_id} / top_left] Move the contact point to this corner.")
                answer = input("Press Enter to capture, or type s to skip this tag > ").strip().lower()
                if aborted:
                    raise KeyboardInterrupt
                if answer == "s":
                    results[tag_id] = None
                    stds[tag_id] = None
                    print("  skipped")
                    break

                pose = wait_for_link6_pose(arm)
                if pose is None:
                    print("  no fresh link6 pose; check CAN/power and try again")
                    continue
                sample = capture_point(arm, tcp_offset, args.samples_per_point, args.sample_interval)
                if sample is None:
                    print("  capture failed; no valid link6 pose samples")
                    continue

                mean_xyz, std_xyz, mean_link6, std_link6 = sample
                std_mm = std_xyz * 1000.0
                print(f"  touch xyz: x={mean_xyz[0]:+.5f} y={mean_xyz[1]:+.5f} z={mean_xyz[2]:+.5f} m")
                print(f"  touch jitter: {std_mm[0]:.2f} / {std_mm[1]:.2f} / {std_mm[2]:.2f} mm (x/y/z)")
                print(
                    "  link6 mean xyzrpy: "
                    + " ".join(f"{v:+.5f}" for v in mean_link6)
                    + "  (m/rad)"
                )
                if float(np.max(std_mm)) > 2.0:
                    redo = input("  jitter is >2mm; recapture? [y/N] > ").strip().lower()
                    if redo == "y":
                        continue
                confirm = input("  accept this point? Enter=accept, r=recapture > ").strip().lower()
                if confirm == "r":
                    continue
                results[tag_id] = mean_xyz.tolist()
                stds[tag_id] = std_xyz.tolist()
                link6_means[tag_id] = mean_link6.tolist()
                link6_stds[tag_id] = std_link6.tolist()
                break
    except KeyboardInterrupt:
        print("\nCollection aborted; writing any captured points before shutdown.")
    finally:
        if args.drag_teach and args.protect_on_exit:
            try:
                ok = bool(arm.protect_mode())
                print(f"protect_mode -> {ok}")
            except Exception as exc:
                print(f"protect_mode failed: {exc}")
        try:
            arm.close()
            print("ARX arm connection closed.")
        except Exception as exc:
            print(f"ARX arm close failed: {exc}")

    skipped_tags = [tag_id for tag_id, value in results.items() if value is None]
    if skipped_tags:
        print(f"Warning: skipped tags cannot be used for calibration: {skipped_tags}")

    out_path = args.out.expanduser()
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "units": "meters",
        "robot": "arx",
        "arm": args.arm,
        "can_port": args.can_port,
        "arm_type": args.arm_type,
        "arm_type_name": ARM_TYPE_NAMES[args.arm_type],
        "corner_captured": "top_left",
        "tag_size_m": args.tag_size_m,
        "tcp_offset_m": tcp_offset.tolist(),
        "tcp_offset_frame": "arx_link6_local",
        "pose_source": "bimanual.SingleArm.get_ee_pose_xyzrpy",
        "link6_pose_units": "meters/radians",
        "link6_rpy_convention_for_offset": "R_link6_in_base = Rz(yaw) @ Ry(pitch) @ Rx(roll)",
        "samples_per_point": args.samples_per_point,
        "sample_interval_s": args.sample_interval,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "tags": {str(tag_id): value for tag_id, value in results.items() if value is not None},
        "capture_std_m": {str(tag_id): std for tag_id, std in stds.items() if std is not None},
        "link6_pose_mean_xyzrpy": {str(tag_id): value for tag_id, value in link6_means.items()},
        "link6_pose_std_xyzrpy": {str(tag_id): value for tag_id, value in link6_stds.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    for tag_id, value in results.items():
        print(f"tag {tag_id}: {value}")


def main() -> int:
    args = parse_args()
    try:
        collect(args)
    except Exception as exc:
        print(f"Failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
