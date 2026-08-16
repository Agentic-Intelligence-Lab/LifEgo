#!/usr/bin/env python3
"""Replay an IK/EGO EEF trajectory on the real Nero arm with move_p only.

This script reads ``target_pos_m`` and ``target_quat_xyzw`` from a retarget/RL
``.npz`` file and commands the real robot in Cartesian pose mode for every
waypoint. It does not use ``joint_qpos`` and never calls ``move_j`` / ``move_js``.

Default mode is dry-run. Pass ``--execute`` and confirm at the prompt to send
commands to the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYAGX_ROOT = REPO_ROOT / "datacollection" / "pyAgxArm"
if str(PYAGX_ROOT) not in sys.path:
    sys.path.insert(0, str(PYAGX_ROOT))


DEFAULT_IK = "outputs/new_pipeline/ego_hand1/nero_eef_ik_flat_x0/nero_eef_ik.npz"


@dataclass
class EefTrajectory:
    pos_m: np.ndarray
    quat_xyzw: np.ndarray
    gripper_width_m: np.ndarray
    time_s: np.ndarray
    source_indices: np.ndarray

    @property
    def n(self) -> int:
        return int(self.pos_m.shape[0])


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def import_pyagx_runtime():
    try:
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
    except ImportError as exc:
        raise ImportError(
            "Failed to import pyAgxArm runtime dependencies. Install the robot "
            "control requirements first, e.g. python-can for CAN backends."
        ) from exc
    return {
        "AgxArmFactory": AgxArmFactory,
        "create_agx_arm_config": create_agx_arm_config,
    }


def normalize_quat_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(f"Invalid quaternion: {q.tolist()}")
    return q / norm


def quat_xyzw_to_rpy(quat: Sequence[float]) -> list[float]:
    x, y, z, w = normalize_quat_xyzw(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def load_eef_trajectory(
    path: Path,
    *,
    start: int,
    end: int | None,
    stride: int,
    fallback_fps: float,
    fallback_gripper_width_m: float,
) -> EefTrajectory:
    data = np.load(path, allow_pickle=True)
    required = ("target_pos_m", "target_quat_xyzw")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing required key(s): {missing}")

    pos = np.asarray(data["target_pos_m"], dtype=np.float64)
    quat = np.asarray(data["target_quat_xyzw"], dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected target_pos_m shape (T, 3), got {pos.shape}")
    if quat.shape != (pos.shape[0], 4):
        raise ValueError(f"Expected target_quat_xyzw shape ({pos.shape[0]}, 4), got {quat.shape}")
    if np.any(~np.isfinite(pos)) or np.any(~np.isfinite(quat)):
        raise ValueError("target_pos_m/target_quat_xyzw contains NaN/Inf")

    n_total = pos.shape[0]
    start = max(0, int(start))
    end_i = n_total if end is None else min(n_total, int(end))
    if start >= end_i:
        raise ValueError(f"Invalid frame range: start={start}, end={end_i}, n={n_total}")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    idx = np.arange(start, end_i, stride, dtype=np.int64)

    if "time_s" in data:
        time_s = np.asarray(data["time_s"], dtype=np.float64)
        if len(time_s) != n_total:
            raise ValueError("time_s length must match target_pos_m")
    else:
        time_s = np.arange(n_total, dtype=np.float64) / float(fallback_fps)

    if "gripper_width_m" in data:
        width = np.asarray(data["gripper_width_m"], dtype=np.float64)
        if len(width) != n_total:
            raise ValueError("gripper_width_m length must match target_pos_m")
    elif "grasp" in data:
        open_m = float(data["gripper_open_m"]) if "gripper_open_m" in data else 0.1
        closed_m = float(data["gripper_closed_m"]) if "gripper_closed_m" in data else 0.0
        width = np.where(np.asarray(data["grasp"], dtype=np.float64) > 0.5, closed_m, open_m)
    else:
        width = np.full(n_total, float(fallback_gripper_width_m), dtype=np.float64)

    return EefTrajectory(
        pos_m=pos[idx],
        quat_xyzw=np.asarray([normalize_quat_xyzw(q) for q in quat[idx]], dtype=np.float64),
        gripper_width_m=width[idx],
        time_s=time_s[idx] - float(time_s[idx][0]),
        source_indices=idx,
    )


def format_pose(pose: Sequence[float]) -> str:
    xyz = [float(v) for v in pose[:3]]
    rpy_deg = [math.degrees(float(v)) for v in pose[3:6]]
    return (
        f"xyz=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m  "
        f"rpy=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}] deg"
    )


def target_tcp_pose(traj: EefTrajectory, i: int, *, position_only_rpy: Sequence[float] | None = None) -> list[float]:
    if position_only_rpy is not None:
        rpy = list(position_only_rpy)
    else:
        rpy = quat_xyzw_to_rpy(traj.quat_xyzw[i])
    return [float(traj.pos_m[i, 0]), float(traj.pos_m[i, 1]), float(traj.pos_m[i, 2]), *rpy]


def read_tcp_pose(robot) -> list[float]:
    message = robot.get_tcp_pose()
    if message is None:
        raise RuntimeError("No NERO TCP feedback; check flange feedback and TCP offset")
    pose = list(message.msg)[:6]
    if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
        raise RuntimeError(f"Invalid NERO TCP feedback: {pose}")
    return pose


def pose_rotation_error_rad(actual: Sequence[float], target: Sequence[float]) -> float:
    delta = [float(a) - float(b) for a, b in zip(actual[3:6], target[3:6])]
    return max(abs(math.atan2(math.sin(value), math.cos(value))) for value in delta)


def wait_until_tcp_reached(
    robot,
    target_tcp_pose_: Sequence[float],
    *,
    timeout: float,
    pos_tolerance_m: float,
    rot_tolerance_rad: float,
    check_orientation: bool,
    min_wait_s: float,
) -> bool:
    start = time.monotonic()
    if min_wait_s > 0.0:
        time.sleep(float(min_wait_s))
    target_pos = np.asarray(target_tcp_pose_[:3], dtype=np.float64)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_tcp_pose(robot)
        pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
        rot_error = pose_rotation_error_rad(current, target_tcp_pose_)
        if pos_error <= pos_tolerance_m and (not check_orientation or rot_error <= rot_tolerance_rad):
            print(
                f"tcp reached: pos error {pos_error * 1000.0:.1f} mm, "
                f"rpy max error {math.degrees(rot_error):.2f} deg"
            )
            return True
        time.sleep(0.1)

    current = read_tcp_pose(robot)
    pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
    rot_error = pose_rotation_error_rad(current, target_tcp_pose_)
    print(
        f"move_p timeout: pos error {pos_error * 1000.0:.1f} mm, "
        f"rpy max error {math.degrees(rot_error):.2f} deg"
    )
    print(f"target tcp: {format_pose(target_tcp_pose_)}")
    print(f"final tcp:  {format_pose(current)}")
    print(f"elapsed before timeout check: {time.monotonic() - start:.2f}s")
    return False


def connect_and_enable_robot(rt: dict, args: argparse.Namespace):
    config = rt["create_agx_arm_config"](
        robot="nero",
        firmeware_version=args.firmware,
        interface=args.interface,
        channel=args.channel,
    )
    robot = rt["AgxArmFactory"].create_arm(config)
    robot.connect()

    deadline = time.monotonic() + args.enable_timeout
    while not robot.enable():
        if time.monotonic() >= deadline:
            robot.disconnect()
            raise TimeoutError(f"NERO enable timed out after {args.enable_timeout:.1f}s")
        time.sleep(0.01)
    return robot


def command_gripper(gripper, width_m: float, *, force: float) -> None:
    gripper.move_gripper_m(value=float(width_m), force=float(force))


def confirm_execute(args: argparse.Namespace, traj: EefTrajectory) -> None:
    if args.yes:
        return
    print()
    print("About to command the real Nero arm with move_p only.")
    print(f"  channel={args.channel} interface={args.interface} firmware={args.firmware}")
    print(f"  frames={traj.n} speed={args.speed_percent}% gripper={args.gripper}")
    text = input("Type EXECUTE to continue > ").strip()
    if text != "EXECUTE":
        raise RuntimeError("User cancelled execution")


def summarize(traj: EefTrajectory, args: argparse.Namespace) -> None:
    print("--- move_p EEF replay plan ---")
    print(f"frames: {traj.n}  source: {traj.source_indices[0]}..{traj.source_indices[-1]}")
    print(f"time span: {float(traj.time_s[-1]):.3f}s  speed_percent: {args.speed_percent}")
    print(f"execute: {args.execute}")
    print(f"first tcp: {format_pose(target_tcp_pose(traj, 0))}")
    print(f"last tcp:  {format_pose(target_tcp_pose(traj, traj.n - 1))}")
    print(
        "position min/max m:",
        np.round(np.min(traj.pos_m, axis=0), 4).tolist(),
        np.round(np.max(traj.pos_m, axis=0), 4).tolist(),
    )
    print(
        "gripper width m min/max:",
        f"{float(np.min(traj.gripper_width_m)):.4f} / {float(np.max(traj.gripper_width_m)):.4f}",
    )


def replay_on_robot(args: argparse.Namespace, traj: EefTrajectory) -> int:
    rt = import_pyagx_runtime()
    robot = None
    gripper = None
    try:
        print("===== connect / enable =====")
        print(f"Connecting NERO: channel={args.channel}, interface={args.interface}, firmware={args.firmware}")
        robot = connect_and_enable_robot(rt, args)
        robot.set_tcp_offset(
            [
                float(args.tcp_offset_m[0]),
                float(args.tcp_offset_m[1]),
                float(args.tcp_offset_m[2]),
                0.0,
                0.0,
                0.0,
            ]
        )
        robot.set_speed_percent(max(1, min(100, args.speed_percent)))

        if args.gripper:
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            command_gripper(gripper, traj.gripper_width_m[0], force=args.gripper_force)

        current_tcp = read_tcp_pose(robot)
        position_only_rpy = current_tcp[3:6] if args.position_only else None
        print(f"current tcp: {format_pose(current_tcp)}")

        print("===== move_p replay =====")
        t_wall0 = time.monotonic()
        for i, width in enumerate(traj.gripper_width_m):
            tcp_pose = target_tcp_pose(traj, i, position_only_rpy=position_only_rpy)
            flange_pose = robot.get_tcp2flange_pose(tcp_pose)
            print(
                f"frame {i + 1:04d}/{traj.n} src={traj.source_indices[i]} "
                f"t={traj.time_s[i]:.3f}s width={width:.3f}m"
            )
            print(f"target tcp:    {format_pose(tcp_pose)}")
            print(f"flange command:{format_pose(flange_pose)}")
            robot.move_p(list(flange_pose))
            if args.gripper and gripper is not None:
                command_gripper(gripper, width, force=args.gripper_force)
            if args.wait:
                ok = wait_until_tcp_reached(
                    robot,
                    tcp_pose,
                    timeout=args.motion_timeout,
                    pos_tolerance_m=args.pos_tol_m,
                    rot_tolerance_rad=args.rot_tol_rad,
                    check_orientation=not args.position_only,
                    min_wait_s=args.command_settle_s,
                )
                if not ok:
                    return 1
            elif args.command_settle_s > 0.0:
                time.sleep(args.command_settle_s)

            if args.follow_time and i + 1 < traj.n:
                target_elapsed = float(traj.time_s[i + 1]) / max(args.time_scale, 1.0e-6)
                sleep_s = target_elapsed - (time.monotonic() - t_wall0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        final_tcp = read_tcp_pose(robot)
        final_target = target_tcp_pose(traj, traj.n - 1, position_only_rpy=position_only_rpy)
        final_pos_err = float(
            np.linalg.norm(np.asarray(final_tcp[:3], dtype=np.float64) - np.asarray(final_target[:3], dtype=np.float64))
        )
        print("===== done =====")
        print(f"final tcp: {format_pose(final_tcp)}")
        print(f"final tcp position error: {final_pos_err * 1000.0:.1f} mm")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Use the physical emergency stop if the arm must stop immediately.")
        return 130
    finally:
        if robot is not None:
            try:
                # Do not call disable(): a raised arm can fall immediately when disabled.
                robot.disconnect()
            except Exception:
                pass


def write_plan(path: Path, traj: EefTrajectory, args: argparse.Namespace) -> None:
    payload = {
        "ik": str(as_abs(args.ik)),
        "frames": traj.n,
        "source_indices": traj.source_indices.tolist(),
        "time_s": traj.time_s.tolist(),
        "target_pos_m": traj.pos_m.tolist(),
        "target_quat_xyzw": traj.quat_xyzw.tolist(),
        "gripper_width_m": traj.gripper_width_m.tolist(),
        "first_target_tcp_pose": target_tcp_pose(traj, 0),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote plan: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik", default=DEFAULT_IK, help="IK/RL .npz with target_pos_m and target_quat_xyzw")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--fallback-gripper-width-m", type=float, default=0.1)
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the real robot")
    parser.add_argument("--yes", action="store_true", help="Skip the EXECUTE confirmation prompt")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can1")
    parser.add_argument("--firmware", choices=["default", "v111", "v112"], default="v112")
    parser.add_argument("--speed-percent", type=int, default=5)
    parser.add_argument("--enable-timeout", type=float, default=15.0)
    parser.add_argument("--motion-timeout", type=float, default=45.0)
    parser.add_argument("--pos-tol-m", type=float, default=0.04)
    parser.add_argument("--rot-tol-deg", type=float, default=10.0)
    parser.add_argument(
        "--command-settle-s",
        type=float,
        default=0.15,
        help="Minimum time to wait after each move_p command.",
    )
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for TCP feedback to reach each waypoint. Disable to stream commands with --command-settle-s only.",
    )
    parser.add_argument(
        "--position-only",
        action="store_true",
        help="Replay target xyz only while keeping the current TCP orientation.",
    )
    parser.add_argument(
        "--tcp-offset-m",
        nargs=3,
        type=float,
        default=(0.13, 0.0, 0.0),
        help="TCP offset in flange frame used to convert target TCP pose to move_p flange pose.",
    )
    parser.add_argument("--gripper", action="store_true", help="Also command AgxGripper width from gripper_width_m")
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--follow-time", action="store_true", help="Sleep between waypoints according to time_s")
    parser.add_argument("--time-scale", type=float, default=1.0, help=">1 speeds up --follow-time playback")
    parser.add_argument("--write-plan", default=None, help="Optional JSON dump of the exact move_p waypoints")
    args = parser.parse_args()

    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.speed_percent < 1 or args.speed_percent > 100:
        raise ValueError("--speed-percent must be in [1, 100]")
    if args.command_settle_s < 0.0:
        raise ValueError("--command-settle-s must be non-negative")
    if args.pos_tol_m <= 0.0:
        raise ValueError("--pos-tol-m must be positive")
    if args.motion_timeout <= 0.0:
        raise ValueError("--motion-timeout must be positive")
    args.rot_tol_rad = math.radians(args.rot_tol_deg)
    return args


def main() -> int:
    args = parse_args()
    ik_path = as_abs(args.ik)
    if not ik_path.is_file():
        raise FileNotFoundError(ik_path)

    traj = load_eef_trajectory(
        ik_path,
        start=args.start,
        end=args.end,
        stride=args.stride,
        fallback_fps=args.fallback_fps,
        fallback_gripper_width_m=args.fallback_gripper_width_m,
    )
    summarize(traj, args)
    if args.write_plan:
        write_plan(as_abs(args.write_plan), traj, args)
    if not args.execute:
        print("dry-run only. Add --execute to command the robot.")
        return 0

    confirm_execute(args, traj)
    return replay_on_robot(args, traj)


if __name__ == "__main__":
    sys.exit(main())
