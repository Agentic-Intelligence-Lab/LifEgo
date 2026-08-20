#!/usr/bin/env python3
"""Shared deployment utilities for Nero EEF policy inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EGO2EXE_ROOT = REPO_ROOT / "ego2exe"
PYAGX_ROOT = REPO_ROOT / "datacollection" / "pyAgxArm"
for path in (REPO_ROOT, EGO2EXE_ROOT, PYAGX_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_STANDBY_JSONL = (
    "DATA/20260816_nero_stack_object_horizontal/stack_object_20260816_163253_653146.jsonl"
)
DEFAULT_TASK_PROMPT = "Place the black pillar in the plate."
DEFAULT_INTERFACE = "socketcan"
DEFAULT_CHANNEL = "can1"
DEFAULT_FIRMWARE = "v112"
DEFAULT_CAMERA_WIDTH = 1280
DEFAULT_CAMERA_HEIGHT = 720
DEFAULT_CAMERA_FPS = 30.0
DEFAULT_REALSENSE_WIDTH = 640
DEFAULT_REALSENSE_HEIGHT = 480
DEFAULT_REALSENSE_FPS = 30.0
DEFAULT_IMAGE_WIDTH = 224
DEFAULT_IMAGE_HEIGHT = 224
DEFAULT_STANDBY_SPEED_PERCENT = 5
DEFAULT_SPEED_PERCENT = 5
DEFAULT_ENABLE_TIMEOUT = 15.0
DEFAULT_STANDBY_TIMEOUT = 45.0
DEFAULT_MOTION_TIMEOUT = 10.0
DEFAULT_CONTROL_HZ = 5.0
DEFAULT_EXECUTE_CHUNK_STEPS = 1
DEFAULT_MAX_STEP_M = 0.03
DEFAULT_POS_TOL_M = 0.04
DEFAULT_ROT_TOL_DEG = 10.0
DEFAULT_COMMAND_SETTLE_S = 0.15
DEFAULT_TCP_OFFSET_M = (0.13, 0.0, 0.0)
DEFAULT_XYZ_MIN = (-0.45, -0.65, 0.05)
DEFAULT_XYZ_MAX = (-0.15, -0.15, 0.35)
DEFAULT_GRIPPER_FORCE = 1.0
DEFAULT_GRIPPER_OPEN_M = 0.1
DEFAULT_GRIPPER_CLOSED_M = 0.0
DEFAULT_GRIPPER_THRESHOLD = 0.5
DEFAULT_DEFAULT_GRASP = 0.0


def as_abs(path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def normalize_quat_xyzw(quat: Sequence[float]) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError(f"invalid quaternion: {q.tolist()}")
    return q / norm


def rpy_to_quat_xyzw(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return normalize_quat_xyzw(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def quat_xyzw_to_rpy(quat: Sequence[float]) -> list[float]:
    x, y, z, w = normalize_quat_xyzw(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return [roll, pitch, yaw]


def pose_to_state(tcp_pose: Sequence[float], gripper_value: float) -> np.ndarray:
    quat = rpy_to_quat_xyzw(tcp_pose[3:6])
    return np.concatenate(
        [
            np.asarray(tcp_pose[:3], dtype=np.float64),
            quat,
            np.asarray([float(gripper_value)], dtype=np.float64),
        ]
    ).astype(np.float32)


def action_to_tcp_pose(action: Sequence[float]) -> list[float]:
    action = np.asarray(action, dtype=np.float64)
    quat = normalize_quat_xyzw(action[3:7])
    return [float(action[0]), float(action[1]), float(action[2]), *quat_xyzw_to_rpy(quat)]


def read_pose_message(message, *, name: str) -> list[float]:
    if message is None:
        raise RuntimeError(f"no {name} feedback")
    pose = list(message.msg)[:6]
    if len(pose) != 6 or not all(math.isfinite(float(v)) for v in pose):
        raise RuntimeError(f"invalid {name} feedback: {pose}")
    return [float(v) for v in pose]


def read_flange_pose(robot) -> list[float]:
    return read_pose_message(robot.get_flange_pose(), name="flange pose")


def pose_rotation_error_rad(actual: Sequence[float], target: Sequence[float]) -> float:
    delta = [float(a) - float(b) for a, b in zip(actual[3:6], target[3:6], strict=True)]
    return max(abs(math.atan2(math.sin(value), math.cos(value))) for value in delta)


def wait_until_flange_reached(
    robot,
    target_pose: Sequence[float],
    *,
    timeout: float,
    pos_tolerance_m: float,
    rot_tolerance_rad: float,
    min_wait_s: float,
) -> bool:
    if min_wait_s > 0.0:
        time.sleep(float(min_wait_s))
    target_pos = np.asarray(target_pose[:3], dtype=np.float64)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_flange_pose(robot)
        pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
        rot_error = pose_rotation_error_rad(current, target_pose)
        if pos_error <= pos_tolerance_m and rot_error <= rot_tolerance_rad:
            print(
                f"flange reached: pos error {pos_error * 1000.0:.1f} mm, "
                f"rpy max error {math.degrees(rot_error):.2f} deg"
            )
            return True
        time.sleep(0.1)
    current = read_flange_pose(robot)
    pos_error = float(np.linalg.norm(np.asarray(current[:3], dtype=np.float64) - target_pos))
    rot_error = pose_rotation_error_rad(current, target_pose)
    print(
        f"standby move_p timeout: pos error {pos_error * 1000.0:.1f} mm, "
        f"rpy max error {math.degrees(rot_error):.2f} deg"
    )
    print(f"target flange: {format_pose(target_pose)}")
    print(f"final flange:  {format_pose(current)}")
    return False


def load_standby_from_jsonl(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    fallback = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            poses = rec.get("poses") or {}
            if "flange_pose" not in poses:
                continue
            if fallback is None:
                fallback = rec
            if rec.get("alignment", {}).get("valid", True):
                return {
                    "seq": int(rec.get("seq", 0)),
                    "flange_pose": [float(v) for v in poses["flange_pose"][:6]],
                    "tcp_pose": [float(v) for v in poses.get("tcp_pose", [])[:6]],
                    "gripper": rec.get("gripper") or {},
                }
    if fallback is None:
        raise RuntimeError(f"no sample with poses.flange_pose in {path}")
    poses = fallback.get("poses") or {}
    return {
        "seq": int(fallback.get("seq", 0)),
        "flange_pose": [float(v) for v in poses["flange_pose"][:6]],
        "tcp_pose": [float(v) for v in poses.get("tcp_pose", [])[:6]],
        "gripper": fallback.get("gripper") or {},
    }


def resolve_standby_pose(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "standby_pose", None) is not None:
        pose = [float(v) for v in args.standby_pose]
        return {"seq": None, "flange_pose": pose, "tcp_pose": [], "gripper": {}}
    return load_standby_from_jsonl(as_abs(args.standby_jsonl))


def print_standby(standby: dict[str, Any], args: argparse.Namespace) -> None:
    print("===== standby =====")
    if standby["seq"] is not None:
        print(f"source: {as_abs(args.standby_jsonl)} seq={standby['seq']}")
    print(f"flange standby: {format_pose(standby['flange_pose'])}")
    if standby.get("tcp_pose"):
        print(f"tcp at standby:  {format_pose(standby['tcp_pose'])}")
    gripper = standby.get("gripper") or {}
    if gripper:
        print(f"standby gripper state: {gripper}")


def load_rgb_image(path: Path, *, image_size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != image_size:
            image = image.resize(image_size, resample=Image.Resampling.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


class FrameSource:
    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass


class StaticImageSource(FrameSource):
    def __init__(self, path: Path, image_size: tuple[int, int]):
        self.image = load_rgb_image(path, image_size=image_size)

    def read(self) -> np.ndarray:
        return self.image.copy()


class OpenCvSource(FrameSource):
    def __init__(
        self,
        *,
        camera: str | int,
        width: int | None,
        height: int | None,
        fps: float | None,
        image_size: tuple[int, int],
    ):
        import cv2

        self.cv2 = cv2
        self.image_size = image_size
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera/video source: {camera}")
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, float(fps))

    def read(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError("failed to read RGB frame from camera/video source")
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.image_size[0] or frame_rgb.shape[0] != self.image_size[1]:
            frame_rgb = self.cv2.resize(frame_rgb, self.image_size, interpolation=self.cv2.INTER_AREA)
        return np.asarray(frame_rgb, dtype=np.uint8)

    def close(self) -> None:
        self.cap.release()


class RealSenseSource(FrameSource):
    def __init__(
        self,
        *,
        serial: str | None,
        width: int,
        height: int,
        fps: float,
        image_size: tuple[int, int],
    ):
        import cv2
        import pyrealsense2 as rs

        self.cv2 = cv2
        self.image_size = image_size
        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
        self.pipeline.start(config)

    def read(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("failed to read RealSense color frame")
        frame_bgr = np.asanyarray(color_frame.get_data())
        frame_rgb = self.cv2.cvtColor(frame_bgr, self.cv2.COLOR_BGR2RGB)
        if frame_rgb.shape[1] != self.image_size[0] or frame_rgb.shape[0] != self.image_size[1]:
            frame_rgb = self.cv2.resize(frame_rgb, self.image_size, interpolation=self.cv2.INTER_AREA)
        return np.asarray(frame_rgb, dtype=np.uint8)

    def close(self) -> None:
        self.pipeline.stop()


def make_frame_source(args: argparse.Namespace) -> FrameSource:
    image_size = (int(args.image_width), int(args.image_height))
    if getattr(args, "image", None):
        return StaticImageSource(as_abs(args.image), image_size=image_size)
    if getattr(args, "video", None):
        return OpenCvSource(camera=str(as_abs(args.video)), width=None, height=None, fps=None, image_size=image_size)
    if getattr(args, "camera_backend", "opencv") == "realsense":
        return RealSenseSource(
            serial=getattr(args, "realsense_serial", None),
            width=int(getattr(args, "realsense_width", DEFAULT_REALSENSE_WIDTH)),
            height=int(getattr(args, "realsense_height", DEFAULT_REALSENSE_HEIGHT)),
            fps=float(getattr(args, "realsense_fps", DEFAULT_REALSENSE_FPS)),
            image_size=image_size,
        )
    return OpenCvSource(
        camera=int(args.camera_index),
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
        image_size=image_size,
    )


def width_to_grasp(width_m: float, *, open_m: float, closed_m: float) -> float:
    denom = max(open_m - closed_m, 1.0e-6)
    return float(np.clip((open_m - width_m) / denom, 0.0, 1.0))


def grasp_to_width(grasp: float, *, open_m: float, closed_m: float, threshold: float, mode: str) -> float:
    grasp = float(np.clip(grasp, 0.0, 1.0))
    if mode == "threshold":
        return closed_m if grasp >= threshold else open_m
    return float(open_m + grasp * (closed_m - open_m))


def read_gripper_state(gripper, fallback: float) -> float:
    if gripper is None:
        return float(fallback)
    try:
        feedback = gripper.get_gripper_status()
    except Exception:
        return float(fallback)
    if feedback is None:
        return float(fallback)
    msg = getattr(feedback, "msg", None)
    width = getattr(msg, "value", None)
    if width is None:
        width = getattr(feedback, "value", None)
    if width is None:
        return float(fallback)
    return width_to_grasp(float(width), open_m=DEFAULT_GRIPPER_OPEN_M, closed_m=DEFAULT_GRIPPER_CLOSED_M)


def clip_target_action(action: np.ndarray, current_state: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = np.asarray(action, dtype=np.float64).copy()
    out[:3] = np.clip(out[:3], np.asarray(args.xyz_min, dtype=np.float64), np.asarray(args.xyz_max, dtype=np.float64))
    delta = out[:3] - np.asarray(current_state[:3], dtype=np.float64)
    norm = float(np.linalg.norm(delta))
    if args.max_step_m > 0.0 and norm > args.max_step_m:
        out[:3] = np.asarray(current_state[:3], dtype=np.float64) + delta / norm * float(args.max_step_m)
    out[3:7] = normalize_quat_xyzw(out[3:7])
    out[7] = float(np.clip(out[7], 0.0, 1.0))
    return out.astype(np.float64)


def format_pose(pose: Sequence[float]) -> str:
    xyz = [float(v) for v in pose[:3]]
    rpy_deg = [math.degrees(float(v)) for v in pose[3:6]]
    return (
        f"xyz=[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}] m  "
        f"rpy=[{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}] deg"
    )


def import_robot_helpers():
    from replay_ik_nero_move_p import (
        command_gripper,
        connect_and_enable_robot,
        import_pyagx_runtime,
        read_tcp_pose,
        wait_until_tcp_reached,
    )

    return {
        "command_gripper": command_gripper,
        "connect_and_enable_robot": connect_and_enable_robot,
        "import_pyagx_runtime": import_pyagx_runtime,
        "read_tcp_pose": read_tcp_pose,
        "wait_until_tcp_reached": wait_until_tcp_reached,
    }


def move_to_standby(robot, standby_pose: Sequence[float], args: argparse.Namespace) -> bool:
    print("===== move to standby =====")
    robot.set_speed_percent(max(1, min(100, args.standby_speed_percent)))
    print(f"target standby flange: {format_pose(standby_pose)}")
    robot.move_p([float(v) for v in standby_pose])
    if args.wait_standby:
        return wait_until_flange_reached(
            robot,
            standby_pose,
            timeout=args.standby_timeout,
            pos_tolerance_m=args.pos_tol_m,
            rot_tolerance_rad=args.rot_tol_rad,
            min_wait_s=args.command_settle_s,
        )
    time.sleep(args.command_settle_s)
    return True
