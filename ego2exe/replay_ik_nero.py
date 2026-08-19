#!/usr/bin/env python3
"""Replay a Nero IK trajectory on the real robot.

This is a conservative real-robot runner for IK trajectories produced by
``retarget_with_mink.py`` / ``retarget_with_scipy.py`` / RL post-processing. It
uses pyAgxArm's planned joint-space command ``move_j`` for each waypoint.
Optionally, it can first use ``move_p`` to slowly move the robot TCP to the
first ego-video EEF target, wait for manual alignment confirmation, then switch
to ``move_j`` replay. It deliberately does not use ``move_js`` because the SDK
documents that mode as unsmoothed, instant-response, and high risk.

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


DEFAULT_IK = "outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz"
ARM_DOF = 7
NERO_JOINT_LIMITS = np.asarray(
    [
        [-2.705261, 2.705261],
        [-1.745330, 1.745330],
        [-2.757621, 2.757621],
        [-1.012291, 2.146755],
        [-2.757621, 2.757621],
        [-0.733039, 0.959932],
        [-1.570797, 1.570797],
    ],
    dtype=np.float64,
)


@dataclass
class IkTrajectory:
    q: np.ndarray
    gripper_width_m: np.ndarray
    time_s: np.ndarray
    source_indices: np.ndarray
    target_pos_m: np.ndarray | None = None
    target_quat_xyzw: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.q.shape[0])


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


def load_ik(
    path: Path,
    *,
    start: int,
    end: int | None,
    stride: int,
    fallback_gripper_width_m: float,
    fallback_fps: float,
) -> IkTrajectory:
    data = np.load(path, allow_pickle=True)
    if "joint_qpos" not in data:
        raise ValueError(f"{path} is missing required key: joint_qpos")

    q = np.asarray(data["joint_qpos"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != ARM_DOF:
        raise ValueError(f"Expected joint_qpos shape (T, {ARM_DOF}), got {q.shape}")

    n_total = q.shape[0]
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
            raise ValueError("time_s length must match joint_qpos")
    else:
        time_s = np.arange(n_total, dtype=np.float64) / float(fallback_fps)

    if "gripper_width_m" in data:
        width = np.asarray(data["gripper_width_m"], dtype=np.float64)
        if len(width) != n_total:
            raise ValueError("gripper_width_m length must match joint_qpos")
    elif "grasp" in data:
        open_m = float(data["gripper_open_m"]) if "gripper_open_m" in data else 0.1
        closed_m = float(data["gripper_closed_m"]) if "gripper_closed_m" in data else 0.0
        width = np.where(np.asarray(data["grasp"], dtype=np.float64) > 0.5, closed_m, open_m)
    else:
        width = np.full(n_total, float(fallback_gripper_width_m), dtype=np.float64)

    target_pos_m = None
    if "target_pos_m" in data:
        target_pos_m = np.asarray(data["target_pos_m"], dtype=np.float64)
        if target_pos_m.shape != (n_total, 3):
            raise ValueError(f"target_pos_m must have shape ({n_total}, 3), got {target_pos_m.shape}")

    target_quat_xyzw = None
    if "target_quat_xyzw" in data:
        target_quat_xyzw = np.asarray(data["target_quat_xyzw"], dtype=np.float64)
        if target_quat_xyzw.shape != (n_total, 4):
            raise ValueError(
                f"target_quat_xyzw must have shape ({n_total}, 4), got {target_quat_xyzw.shape}"
            )

    return IkTrajectory(
        q=q[idx],
        gripper_width_m=width[idx],
        time_s=time_s[idx] - float(time_s[idx][0]),
        source_indices=idx,
        target_pos_m=None if target_pos_m is None else target_pos_m[idx],
        target_quat_xyzw=None if target_quat_xyzw is None else target_quat_xyzw[idx],
    )


def load_joint_limits(margin_rad: float) -> tuple[np.ndarray, np.ndarray]:
    return NERO_JOINT_LIMITS[:, 0] + margin_rad, NERO_JOINT_LIMITS[:, 1] - margin_rad


def validate_trajectory(
    traj: IkTrajectory,
    *,
    joint_lower: np.ndarray,
    joint_upper: np.ndarray,
    gripper_range_m: tuple[float, float],
    max_step_rad: float,
    allow_large_steps: bool,
) -> None:
    if np.any(~np.isfinite(traj.q)):
        raise ValueError("joint_qpos contains NaN/Inf")
    if np.any(~np.isfinite(traj.gripper_width_m)):
        raise ValueError("gripper_width_m contains NaN/Inf")

    below = traj.q < joint_lower[None, :]
    above = traj.q > joint_upper[None, :]
    if np.any(below | above):
        frame, joint = np.argwhere(below | above)[0]
        val = traj.q[frame, joint]
        raise ValueError(
            f"Joint limit violation at replay frame {frame} "
            f"(source {traj.source_indices[frame]}), joint{joint + 1}: {val:.4f} rad "
            f"not in [{joint_lower[joint]:.4f}, {joint_upper[joint]:.4f}]"
        )

    gmin, gmax = gripper_range_m
    if np.any((traj.gripper_width_m < gmin) | (traj.gripper_width_m > gmax)):
        wmin = float(np.min(traj.gripper_width_m))
        wmax = float(np.max(traj.gripper_width_m))
        raise ValueError(f"Gripper width outside [{gmin}, {gmax}] m: {wmin}..{wmax}")

    if traj.n > 1:
        dq = np.abs(np.diff(traj.q, axis=0))
        worst = float(np.max(dq))
        if worst > max_step_rad:
            loc = np.unravel_index(int(np.argmax(dq)), dq.shape)
            msg = (
                f"Large adjacent waypoint jump: {worst:.3f} rad between replay frames "
                f"{loc[0]}->{loc[0] + 1}, joint{loc[1] + 1}. "
                f"Use a smaller --stride, smooth the IK, or pass --allow-large-steps."
            )
            if not allow_large_steps:
                raise ValueError(msg)
            print(f"[warn] {msg}")


def summarize(traj: IkTrajectory, *, speed_percent: int, execute: bool, command_settle_s: float) -> None:
    q_deg = np.rad2deg(traj.q)
    duration = float(traj.time_s[-1]) if traj.n else 0.0
    print("--- IK replay plan ---")
    print(f"frames: {traj.n}  source: {traj.source_indices[0]}..{traj.source_indices[-1]}")
    print(f"time span: {duration:.3f}s  speed_percent: {speed_percent}")
    print(f"command_settle_s: {command_settle_s:.3f}")
    print(f"execute: {execute}")
    print("joint deg min:", np.round(np.min(q_deg, axis=0), 2).tolist())
    print("joint deg max:", np.round(np.max(q_deg, axis=0), 2).tolist())
    if traj.n > 1:
        dq = np.linalg.norm(np.diff(traj.q, axis=0), axis=1)
        print(f"|dq| mean/max: {float(np.mean(dq)):.4f} / {float(np.max(dq)):.4f} rad")
    print(
        "gripper width m min/max:",
        f"{float(np.min(traj.gripper_width_m)):.4f} / {float(np.max(traj.gripper_width_m)):.4f}",
    )


def interpolate_to_start(current: Sequence[float], target: np.ndarray, max_step_rad: float) -> list[np.ndarray]:
    current_arr = np.asarray(current, dtype=np.float64)[:ARM_DOF]
    delta = target - current_arr
    steps = max(1, int(math.ceil(float(np.max(np.abs(delta))) / max_step_rad)))
    return [current_arr + delta * (i / steps) for i in range(1, steps + 1)]


def format_degrees(joints_rad: Sequence[float]) -> str:
    return "[" + ", ".join(f"{math.degrees(float(value)):.3f}" for value in joints_rad) + "] deg"


def format_pose(pose: Sequence[float]) -> str:
    xyz = [float(v) for v in pose[:3]]
    rpy_deg = [math.degrees(float(v)) for v in pose[3:6]]
    return (
        f"xyz=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m  "
        f"rpy=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}] deg"
    )


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


def first_tcp_pose_from_ik(traj: IkTrajectory) -> list[float]:
    if traj.target_pos_m is None or traj.target_quat_xyzw is None:
        raise RuntimeError(
            "The IK file does not contain target_pos_m/target_quat_xyzw. "
            "--prealign-move-p needs an IK npz generated by the current retarget/RL pipeline."
        )
    pos = np.asarray(traj.target_pos_m[0], dtype=np.float64)
    quat = np.asarray(traj.target_quat_xyzw[0], dtype=np.float64)
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(quat)):
        raise RuntimeError("First IK target TCP pose contains NaN/Inf")
    return [float(pos[0]), float(pos[1]), float(pos[2]), *quat_xyzw_to_rpy(quat)]


def read_joints(robot) -> list[float]:
    message = robot.get_joint_angles()
    if message is None:
        raise RuntimeError("No NERO joint feedback; check power, CAN wiring and channel")
    joints = list(message.msg)[:ARM_DOF]
    if len(joints) != ARM_DOF or not all(math.isfinite(value) for value in joints):
        raise RuntimeError(f"Invalid NERO joint feedback: {joints}")
    return joints


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
    target_tcp_pose: Sequence[float],
    *,
    timeout: float,
    pos_tolerance_m: float,
    rot_tolerance_rad: float,
    check_orientation: bool,
) -> bool:
    target_pos = np.asarray(target_tcp_pose[:3], dtype=np.float64)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_tcp_pose(robot)
        pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
        rot_error = pose_rotation_error_rad(current, target_tcp_pose)
        if pos_error <= pos_tolerance_m and (not check_orientation or rot_error <= rot_tolerance_rad):
            print(
                f"tcp reached: pos error {pos_error * 1000.0:.1f} mm, "
                f"rpy max error {math.degrees(rot_error):.2f} deg"
            )
            return True
        time.sleep(0.1)

    current = read_tcp_pose(robot)
    pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
    rot_error = pose_rotation_error_rad(current, target_tcp_pose)
    print(
        f"move_p timeout: pos error {pos_error * 1000.0:.1f} mm, "
        f"rpy max error {math.degrees(rot_error):.2f} deg"
    )
    print(f"final tcp pose: {format_pose(current)}")
    return False


def wait_until_reached(
    robot,
    target: Sequence[float],
    *,
    timeout: float,
    tolerance_rad: float,
    min_wait_s: float = 0.0,
) -> bool:
    start = time.monotonic()
    if min_wait_s > 0.0:
        time.sleep(float(min_wait_s))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_joints(robot)
        max_error = max(abs(actual - desired) for actual, desired in zip(current, target))
        if max_error <= tolerance_rad and time.monotonic() - start >= min_wait_s:
            print(f"reached: max joint error {math.degrees(max_error):.3f} deg")
            return True
        time.sleep(0.1)

    current = read_joints(robot)
    errors = [actual - desired for actual, desired in zip(current, target)]
    abs_errors = [abs(value) for value in errors]
    worst = int(np.argmax(abs_errors))
    max_error = abs_errors[worst]
    print(
        f"motion timeout: max joint error {math.degrees(max_error):.3f} deg "
        f"on joint{worst + 1}"
    )
    print(f"target joints: {format_degrees(target)}")
    print(f"final joints: {format_degrees(current)}")
    print(
        "joint errors:",
        "[" + ", ".join(f"{math.degrees(float(value)):+.3f}" for value in errors) + "] deg",
    )
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


def confirm_execute(args: argparse.Namespace, traj: IkTrajectory) -> None:
    if args.yes:
        return
    print()
    print("About to command the real Nero arm.")
    print(f"  channel={args.channel} interface={args.interface} firmware={args.firmware}")
    print(f"  frames={traj.n} speed={args.speed_percent}% gripper={args.gripper}")
    text = input("Type EXECUTE to continue > ").strip()
    if text != "EXECUTE":
        raise RuntimeError("User cancelled execution")


def confirm_prealign_continue(args: argparse.Namespace) -> None:
    if args.yes or not args.prealign_confirm:
        return
    text = input("Check the TCP/EEF alignment, then type READY to start move_j replay > ").strip()
    if text != "READY":
        raise RuntimeError("User cancelled after move_p prealignment")


def command_gripper(gripper, width_m: float, *, force: float) -> None:
    gripper.move_gripper_m(value=float(width_m), force=float(force))


def prealign_with_move_p(robot, args: argparse.Namespace, traj: IkTrajectory) -> None:
    target_tcp = first_tcp_pose_from_ik(traj)
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
    current_tcp = read_tcp_pose(robot)
    if args.prealign_position_only:
        target_tcp = target_tcp[:3] + current_tcp[3:6]

    target_flange = robot.get_tcp2flange_pose(target_tcp)
    print("===== move_p prealignment =====")
    print(f"current tcp:       {format_pose(current_tcp)}")
    print(f"target tcp:        {format_pose(target_tcp)}")
    print(f"target flange cmd: {format_pose(target_flange)}")
    print(f"move_p speed: {args.prealign_speed_percent}%")
    robot.set_speed_percent(max(1, min(100, args.prealign_speed_percent)))
    robot.move_p(list(target_flange))
    reached = wait_until_tcp_reached(
        robot,
        target_tcp,
        timeout=args.prealign_timeout,
        pos_tolerance_m=args.prealign_pos_tol_m,
        rot_tolerance_rad=args.prealign_rot_tol_rad,
        check_orientation=not args.prealign_position_only,
    )
    if not reached:
        raise RuntimeError("move_p prealignment did not reach the first TCP target")
    confirm_prealign_continue(args)
    robot.set_speed_percent(max(1, min(100, args.speed_percent)))


def replay_on_robot(args: argparse.Namespace, traj: IkTrajectory) -> int:
    rt = import_pyagx_runtime()
    robot = None
    gripper = None
    try:
        print("===== connect / enable =====")
        print(f"Connecting NERO: channel={args.channel}, interface={args.interface}, firmware={args.firmware}")
        robot = connect_and_enable_robot(rt, args)
        robot.set_speed_percent(max(1, min(100, args.speed_percent)))

        if args.gripper:
            gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            command_gripper(gripper, traj.gripper_width_m[0], force=args.gripper_force)

        if args.prealign_move_p:
            prealign_with_move_p(robot, args, traj)

        current = read_joints(robot)
        print(f"current joints: {format_degrees(current)}")
        print(f"first IK joints: {format_degrees(traj.q[0])}")
        initial_jump = float(np.max(np.abs(traj.q[0] - np.asarray(current))))
        print(f"current -> first max joint delta: {math.degrees(initial_jump):.2f} deg")
        if initial_jump > args.max_initial_jump_rad and not args.approach:
            raise RuntimeError(
                f"Initial jump is {initial_jump:.3f} rad. Re-run with --approach "
                "to insert intermediate waypoints, or move the robot closer first."
            )

        if args.approach:
            approach = interpolate_to_start(current, traj.q[0], args.approach_step_rad)
            print(f"approach waypoints: {len(approach)}")
            for i, q in enumerate(approach):
                print(f"approach {i + 1}/{len(approach)}")
                robot.move_j(q.tolist())
                if not wait_until_reached(
                    robot,
                    q.tolist(),
                    timeout=args.motion_timeout,
                    tolerance_rad=args.joint_tol_rad,
                    min_wait_s=args.command_settle_s,
                ):
                    return 1

        print("===== replay =====")
        t_wall0 = time.monotonic()
        for i, (q, width) in enumerate(zip(traj.q, traj.gripper_width_m)):
            print(
                f"frame {i + 1:04d}/{traj.n} src={traj.source_indices[i]} "
                f"t={traj.time_s[i]:.3f}s width={width:.3f}m"
            )
            robot.move_j(q.tolist())
            if args.gripper and gripper is not None:
                command_gripper(gripper, width, force=args.gripper_force)
            ok = wait_until_reached(
                robot,
                q.tolist(),
                timeout=args.motion_timeout,
                tolerance_rad=args.joint_tol_rad,
                min_wait_s=args.command_settle_s,
            )
            if not ok:
                return 1
            if args.follow_time and i + 1 < traj.n:
                target_elapsed = float(traj.time_s[i + 1]) / max(args.time_scale, 1.0e-6)
                sleep_s = target_elapsed - (time.monotonic() - t_wall0)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        final = read_joints(robot)
        err = max(abs(actual - desired) for actual, desired in zip(final, traj.q[-1]))
        print("===== done =====")
        print(f"final max joint error: {math.degrees(err):.2f} deg")
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


def write_plan(path: Path, traj: IkTrajectory, args: argparse.Namespace) -> None:
    payload = {
        "ik": str(as_abs(args.ik)),
        "frames": traj.n,
        "source_indices": traj.source_indices.tolist(),
        "time_s": traj.time_s.tolist(),
        "joint_qpos": traj.q.tolist(),
        "gripper_width_m": traj.gripper_width_m.tolist(),
        "first_target_tcp_pose": (
            first_tcp_pose_from_ik(traj)
            if traj.target_pos_m is not None and traj.target_quat_xyzw is not None
            else None
        ),
        "args": vars(args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote plan: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ik", default=DEFAULT_IK, help="IK .npz with joint_qpos and optional gripper_width_m")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--fallback-gripper-width-m", type=float, default=0.1)
    parser.add_argument("--gripper-range-m", nargs=2, type=float, default=(0.0, 0.1))
    parser.add_argument("--joint-limit-margin-deg", type=float, default=0.0)
    parser.add_argument("--max-step-deg", type=float, default=12.0)
    parser.add_argument("--allow-large-steps", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually send commands to the real robot")
    parser.add_argument("--yes", action="store_true", help="Skip the EXECUTE confirmation prompt")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can1")
    parser.add_argument("--firmware", choices=["default", "v111", "v112"], default="v112")
    parser.add_argument("--speed-percent", type=int, default=10)
    parser.add_argument("--enable-timeout", type=float, default=15.0)
    parser.add_argument("--motion-timeout", type=float, default=30.0)
    parser.add_argument("--joint-tol-deg", type=float, default=2.0)
    parser.add_argument(
        "--command-settle-s",
        type=float,
        default=0.0,
        help="Minimum time to wait after each move_j before checking joint tolerance.",
    )
    parser.add_argument("--max-initial-jump-deg", type=float, default=20.0)
    parser.add_argument("--approach", action="store_true", help="Interpolate current joints to the first IK waypoint")
    parser.add_argument("--approach-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--prealign-move-p",
        action="store_true",
        help="Before move_j replay, slowly move TCP to the first IK target with move_p and wait for confirmation",
    )
    parser.add_argument(
        "--prealign-position-only",
        action="store_true",
        help="For move_p prealignment, use the first target xyz but keep current TCP orientation",
    )
    parser.add_argument("--prealign-speed-percent", type=int, default=5)
    parser.add_argument("--prealign-timeout", type=float, default=45.0)
    parser.add_argument("--prealign-pos-tol-m", type=float, default=0.04)
    parser.add_argument("--prealign-rot-tol-deg", type=float, default=10.0)
    parser.add_argument(
        "--no-prealign-confirm",
        dest="prealign_confirm",
        action="store_false",
        help="Do not pause for READY after move_p prealignment",
    )
    parser.set_defaults(prealign_confirm=True)
    parser.add_argument(
        "--tcp-offset-m",
        nargs=3,
        type=float,
        default=(0.18, 0.0, 0.0),
        help="TCP offset in flange frame used by move_p prealignment",
    )
    parser.add_argument("--gripper", action="store_true", help="Also command AgxGripper width from gripper_width_m")
    parser.add_argument("--gripper-force", type=float, default=1.0)
    parser.add_argument("--follow-time", action="store_true", help="Sleep between waypoints according to time_s")
    parser.add_argument("--time-scale", type=float, default=1.0, help=">1 speeds up --follow-time playback")
    parser.add_argument("--write-plan", default=None, help="Optional JSON dump of the exact replay waypoints")
    args = parser.parse_args()

    args.joint_limit_margin_rad = math.radians(args.joint_limit_margin_deg)
    args.max_step_rad = math.radians(args.max_step_deg)
    args.joint_tol_rad = math.radians(args.joint_tol_deg)
    args.max_initial_jump_rad = math.radians(args.max_initial_jump_deg)
    args.approach_step_rad = math.radians(args.approach_step_deg)
    args.prealign_rot_tol_rad = math.radians(args.prealign_rot_tol_deg)
    args.gripper_range_m = (float(args.gripper_range_m[0]), float(args.gripper_range_m[1]))
    return args


def main() -> int:
    args = parse_args()
    ik_path = as_abs(args.ik)
    if not ik_path.is_file():
        raise FileNotFoundError(ik_path)

    lower, upper = load_joint_limits(args.joint_limit_margin_rad)
    traj = load_ik(
        ik_path,
        start=args.start,
        end=args.end,
        stride=args.stride,
        fallback_gripper_width_m=args.fallback_gripper_width_m,
        fallback_fps=args.fallback_fps,
    )
    validate_trajectory(
        traj,
        joint_lower=lower,
        joint_upper=upper,
        gripper_range_m=args.gripper_range_m,
        max_step_rad=args.max_step_rad,
        allow_large_steps=args.allow_large_steps,
    )
    if args.command_settle_s < 0.0:
        raise ValueError("--command-settle-s must be non-negative")
    summarize(
        traj,
        speed_percent=args.speed_percent,
        execute=args.execute,
        command_settle_s=args.command_settle_s,
    )
    if args.prealign_move_p:
        print(f"first target tcp: {format_pose(first_tcp_pose_from_ik(traj))}")
    if args.write_plan:
        write_plan(as_abs(args.write_plan), traj, args)
    if not args.execute:
        print("dry-run only. Add --execute to command the robot.")
        return 0

    confirm_execute(args, traj)
    return replay_on_robot(args, traj)


if __name__ == "__main__":
    sys.exit(main())
