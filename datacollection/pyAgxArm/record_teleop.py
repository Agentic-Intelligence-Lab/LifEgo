#!/usr/bin/env python3
"""Run Nero/Revo2 teleoperation data collection on one aligned timeline.

The arm keeps using its native CAN leader/follower linkage.  By default this
process maps the leader teach pendant to the Revo2 hand at 20 Hz while storing
aligned arm/hand state and action snapshots as JSONL.  It never sends arm
motion commands.

Each line is a complete JSON object, so a power loss can at most damage the
last line instead of the whole recording.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))
from teleop_mapping import (  # noqa: E402
    DEFAULT_MAX_ANGLE_DEG,
    DEFAULT_MAX_POSITION,
    DEFAULT_MAX_RANGE_M,
    DEFAULT_OPEN_POSE,
    GripperToGraspMapper,
    parse_hand_pose,
)
from teleop_quality import (  # noqa: E402
    TCP_JUMP_WARN_DEG,
    TCP_JUMP_WARN_MM,
    build_tcp_quality,
    validate_task,
)


SCHEMA_VERSION = 9

# Nero AgxGripper TCP：工具中心相对法兰的偏置（法兰局部坐标系，单位 m / rad）。
# 「沿法兰轴心向前伸出工具」= 法兰局部 +X（法兰指向夹爪尖端方向），不是 +Z。
# 修正记录：此前误用 +Z（[0,0,0.13,...]），根据 LifEgo 侧用 URDF 重新核对法兰/
# 夹爪几何后确认应为 +X；数值 0.13 本身没错，是塞进了错的分量。
# 现场实测：夹爪中心沿该轴伸出 13 cm → x=0.13；y/z/姿态暂为 0（轴心无侧偏、无额外旋转）。
# 该值仅保存在 SDK 实例内，不会下发到控制器；get_tcp_pose() / flange↔tcp 转换会用到它。
NERO_GRIPPER_TCP_OFFSET_M = [0.13, 0.0, 0.0, 0.0, 0.0, 0.0]

POSE_LABELS = ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")


def as_pose6(values: Optional[Sequence[float]]) -> Optional[list[float]]:
    if values is None:
        return None
    if len(values) != 6:
        raise ValueError(f"pose must have 6 elements, got {len(values)}")
    return [float(v) for v in values]


def gripper_grasp_from_records(
    mapper: GripperToGraspMapper,
    gripper_ctrl: Optional[dict[str, Any]],
    gripper_feedback: Optional[dict[str, Any]],
) -> tuple[Optional[float], Optional[float]]:
    """Return (state_grasp, action_grasp) in [0,1] from AgxGripper CAN records."""
    action_grasp = None
    if gripper_ctrl is not None:
        action_grasp = float(
            mapper.grasp(gripper_ctrl["mode"], float(gripper_ctrl["value"]))
        )
    state_grasp = None
    if gripper_feedback is not None:
        state_grasp = float(
            mapper.grasp(gripper_feedback["mode"], float(gripper_feedback["value"]))
        )
    elif action_grasp is not None:
        # No feedback yet: fall back to commanded opening for completeness.
        state_grasp = action_grasp
    return state_grasp, action_grasp


def hand_pose_arg(value: str) -> tuple[int, int, int, int, int, int]:
    try:
        return parse_hand_pose(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def default_can() -> tuple[str, str]:
    from platform import system

    plat = system()
    if plat == "Windows":
        return "gs_usb", "0"
    if plat == "Darwin":
        return "gs_usb", "candlelight"
    return "socketcan", "can1"


def parse_args() -> argparse.Namespace:
    default_interface, default_channel = default_can()
    parser = argparse.ArgumentParser(
        description="Nero + Revo2 teleoperation data collector (JSONL)"
    )
    parser.add_argument(
        "-c",
        "--channel",
        default=default_channel,
        help=f"CAN channel (default: {default_channel})",
    )
    parser.add_argument(
        "--interface",
        default=default_interface,
        help=f"CAN interface (default: {default_interface})",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="Snapshot rate")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .jsonl path (default: recordings/nero_teleop_<time>.jsonl)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 means until Ctrl+C",
    )
    parser.add_argument(
        "--task",
        required=True,
        help=(
            "Required episode-level English task instruction for LeRobot/VLA export "
            "(placeholders like nero_teleop are rejected)"
        ),
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for initial leader/follower data",
    )
    parser.add_argument(
        "--fsync-seconds",
        type=float,
        default=1.0,
        help="Force buffered data to disk at this interval; 0 disables periodic fsync",
    )
    parser.add_argument(
        "--open-pose",
        type=hand_pose_arg,
        default=DEFAULT_OPEN_POSE,
        help="Mapped Revo2 fully-open pose (default: 0,800,0,0,0,0)",
    )
    parser.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE_M)
    parser.add_argument("--max-angle", type=float, default=DEFAULT_MAX_ANGLE_DEG)
    parser.add_argument("--max-position", type=int, default=DEFAULT_MAX_POSITION)
    parser.add_argument(
        "--hand-feedback",
        action="store_true",
        help="Also record read-only Revo2 motor feedback",
    )
    parser.add_argument(
        "--hand-port",
        default="auto",
        help="Revo2 serial port, or 'auto' to scan ttyUSB devices",
    )
    parser.add_argument(
        "--hand-slave-id",
        type=lambda value: int(value, 0),
        default=0x7F,
    )
    parser.add_argument(
        "--hand-hz",
        type=float,
        default=20.0,
        help="Revo2 feedback polling rate (independent from CAN snapshot rate)",
    )
    hand_control = parser.add_mutually_exclusive_group()
    hand_control.add_argument(
        "--execute-hand",
        dest="execute_hand",
        action="store_true",
        help="Drive Revo2 from the teach pendant while recording (default)",
    )
    hand_control.add_argument(
        "--no-execute-hand",
        dest="execute_hand",
        action="store_false",
        help="Do not command Revo2; record/preview only",
    )
    parser.set_defaults(execute_hand=False)
    parser.add_argument("--hand-deadband", type=int, default=5)
    parser.add_argument("--hand-motion-ms", type=int, default=80)
    parser.add_argument(
        "--sync-control-file",
        type=Path,
        default=None,
        help="Optional file containing 1/0 to enable/disable Revo2 control at runtime",
    )
    camera_control = parser.add_mutually_exclusive_group()
    camera_control.add_argument(
        "--camera",
        dest="camera",
        action="store_true",
        help="Record aligned RGB video (default)",
    )
    camera_control.add_argument(
        "--no-camera",
        dest="camera",
        action="store_false",
        help="Disable RGB camera recording",
    )
    parser.set_defaults(camera=True)
    parser.add_argument(
        "--camera-backend",
        choices=["opencv", "realsense", "external"],
        default="opencv",
        help=(
            "RGB capture backend (default: opencv / Orbbec Dabai UVC). "
            "'external' talks to a persistent camera_idle_preview.py service "
            "instead of opening the device itself -- see --camera-control-path."
        ),
    )
    parser.add_argument(
        "--camera-control-path",
        type=Path,
        default=None,
        help="backend=external: control file shared with the persistent camera service",
    )
    parser.add_argument(
        "--camera-status-path",
        type=Path,
        default=None,
        help="backend=external: status file shared with the persistent camera service",
    )
    parser.add_argument(
        "--camera-meta-path",
        type=Path,
        default=None,
        help="backend=external: per-frame metadata file shared with the persistent camera service",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=1,
        help="OpenCV camera index for Dabai (default: 1; 0 is often the laptop webcam)",
    )
    parser.add_argument("--camera-serial", default="auto", help="RealSense serial if backend=realsense")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument(
        "--camera-rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=180,
        help="Rotate RGB after capture (degrees CW). Default 180 for inverted Dabai mount.",
    )
    parser.add_argument(
        "--camera-video",
        type=Path,
        default=None,
        help="RGB video path (default: output stem + .rgb.mp4)",
    )
    parser.add_argument(
        "--camera-preview-path",
        type=Path,
        default=None,
        help="Optional atomically-updated PNG for the GUI preview",
    )
    parser.add_argument(
        "--keep-unaligned",
        action="store_true",
        help="Keep rows failing alignment age checks (default: drop them)",
    )
    args = parser.parse_args()
    if args.execute_hand:
        args.hand_feedback = True
    return args


def default_output_path() -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return Path("recordings") / f"nero_teleop_{stamp}.jsonl"


def read_sync_control(path: Optional[Path], fallback: bool) -> bool:
    """Read a GUI-owned 1/0 switch without interrupting recording."""
    if path is None:
        return fallback
    try:
        value = path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return fallback
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no"}:
        return False
    return fallback


class Revo2FeedbackReader:
    """Own the Revo2 serial port for aligned feedback and optional control."""

    def __init__(
        self,
        port: str,
        slave_id: int,
        hz: float,
        execute: bool = False,
        deadband: int = 5,
        motion_ms: int = 80,
        control_capable: bool = False,
    ) -> None:
        self.port = port
        self.slave_id = slave_id
        self.hz = hz
        self.execute = execute
        self.deadband = deadband
        self.motion_ms = motion_ms
        self.control_capable = control_capable or execute
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._detected_port: Optional[str] = None
        self._target: Optional[list[int]] = None
        self._last_sent_target: Optional[list[int]] = None
        self._last_command: Optional[dict[str, Any]] = None
        self._thread = threading.Thread(
            target=self._thread_main,
            name="revo2-feedback",
            daemon=True,
        )

    @property
    def detected_port(self) -> Optional[str]:
        return self._detected_port

    def start(self, timeout: float = 8.0) -> None:
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("Revo2 feedback reader startup timed out")
        if self._error is not None:
            raise RuntimeError(f"Revo2 feedback reader failed: {self._error}")

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            if self._latest is None:
                return None
            return {
                key: list(value) if isinstance(value, list) else value
                for key, value in self._latest.items()
            }

    def set_target(self, positions: list[int]) -> None:
        with self._lock:
            self._target = list(positions)

    def set_execute(self, enabled: bool) -> None:
        with self._lock:
            self.execute = bool(enabled)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    async def _run(self) -> None:
        sdk_path = WORKSPACE / "brainco-hand-sdk" / "python"
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from common_imports import libstark  # type: ignore

        ports = (
            sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyHAND*"))
            if self.port == "auto"
            else [self.port]
        )
        detected = None
        for port in ports:
            devices = await libstark.auto_detect(False, port, "Modbus")
            detected = next(
                (device for device in devices if device.slave_id == self.slave_id),
                None,
            )
            if detected is not None:
                self._detected_port = port
                break
        if detected is None:
            raise RuntimeError(f"slave 0x{self.slave_id:02X} not found on {ports}")

        client = await libstark.init_from_detected(detected)
        try:
            unit_mode = await client.get_finger_unit_mode(self.slave_id)
            if self.control_capable and unit_mode != libstark.FingerUnitMode.Normalized:
                await client.set_finger_unit_mode(
                    self.slave_id, libstark.FingerUnitMode.Normalized
                )
                unit_mode = await client.get_finger_unit_mode(self.slave_id)
            if unit_mode != libstark.FingerUnitMode.Normalized:
                raise RuntimeError(f"Revo2 is not in normalized mode: {unit_mode}")
            interval = 1.0 / self.hz
            next_tick = time.monotonic()
            first_feedback = True
            while not self._stop.is_set():
                with self._lock:
                    target = None if self._target is None else list(self._target)
                    execute = self.execute
                should_send = (
                    execute
                    and target is not None
                    and (
                        self._last_sent_target is None
                        or max(
                            abs(new - old)
                            for new, old in zip(target, self._last_sent_target)
                        ) >= self.deadband
                    )
                )
                if should_send and target is not None:
                    command_start_ns = time.monotonic_ns()
                    await client.set_finger_positions_and_durations(
                        self.slave_id,
                        target,
                        [self.motion_ms] * 6,
                    )
                    command_end_ns = time.monotonic_ns()
                    self._last_sent_target = target
                    self._last_command = {
                        "position_normalized": target,
                        "sent_time_monotonic_ns": command_end_ns,
                        "send_duration_ms": (command_end_ns - command_start_ns) * 1e-6,
                    }
                request_start_mono_ns = time.monotonic_ns()
                request_start_unix_ns = time.time_ns()
                status = await client.get_motor_status(self.slave_id)
                response_mono_ns = time.monotonic_ns()
                round_trip_ns = response_mono_ns - request_start_mono_ns
                midpoint_mono_ns = request_start_mono_ns + round_trip_ns // 2
                midpoint_unix_ns = request_start_unix_ns + round_trip_ns // 2
                record = {
                    "position_normalized": [int(value) for value in status.positions],
                    "speed": [int(value) for value in status.speeds],
                    "current": [int(value) for value in status.currents],
                    "state": [str(value) for value in status.states],
                    "estimated_time_monotonic_ns": midpoint_mono_ns,
                    "estimated_time_unix_ns": midpoint_unix_ns,
                    "round_trip_ms": round_trip_ns * 1e-6,
                    "time_uncertainty_ms": round_trip_ns * 0.5e-6,
                    "last_command": self._last_command,
                }
                with self._lock:
                    self._latest = record
                if first_feedback:
                    self._ready.set()
                    first_feedback = False
                next_tick += interval
                delay = next_tick - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -interval * 5:
                    next_tick = time.monotonic() + interval
        finally:
            await libstark.modbus_close(client)


def _ffmpeg_raw_bgr_encoder(video_path: Path, width: int, height: int, fps: int) -> subprocess.Popen[bytes]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "bgr24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def camera_output_size(width: int, height: int, rotate_deg: int) -> tuple[int, int]:
    if int(rotate_deg) in (90, 270):
        return int(height), int(width)
    return int(width), int(height)


def orient_bgr_image(cv2: Any, image: Any, rotate_deg: int) -> Any:
    """Apply clockwise rotation after capture (0/90/180/270)."""
    deg = int(rotate_deg) % 360
    if deg == 0:
        return image
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported camera rotate: {rotate_deg}")


class OpenCVRGBRecorder:
    """Capture Orbbec Dabai (UVC) RGB continuously and expose the latest frame index.

    Capture, preview, and encode run on separate threads so a slow OpenCV ``mp4v``
    writer (common when ffmpeg is missing on Windows) cannot stall freshness.
    DirectShow can also freeze ``cap.read()``; a watchdog reopens the device.
    """

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        fps: int,
        video_path: Path,
        preview_path: Optional[Path] = None,
        rotate_deg: int = 180,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_deg = int(rotate_deg) % 360
        self.out_width, self.out_height = camera_output_size(
            width, height, self.rotate_deg
        )
        self.video_path = video_path
        self.preview_path = preview_path
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._detected_serial: Optional[str] = None
        self._frame_count = 0
        self._last_frame_monotonic_ns = 0
        self._encode_queue: queue.Queue[Optional[Any]] = queue.Queue(maxsize=2)
        self._preview_queue: queue.Queue[Optional[tuple[Any, int]]] = queue.Queue(
            maxsize=1
        )
        self._thread = threading.Thread(
            target=self._thread_main,
            name="opencv-rgb",
            daemon=True,
        )
        self._encode_thread: Optional[threading.Thread] = None
        self._preview_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._reopen_requested = threading.Event()
        self._stall_limit_s = 2.0
        # Shared with grab loop; set while reopen is in progress so watchdog waits.
        self._cap_holder: list[Any] = [None]
        self._cap_holder_lock = threading.Lock()

    @property
    def detected_serial(self) -> Optional[str]:
        return self._detected_serial

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, timeout: float = 20.0) -> None:
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        if self.preview_path is not None:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            self._stop.set()
            raise TimeoutError("OpenCV RGB startup timed out")
        if self._error is not None:
            raise RuntimeError(f"OpenCV RGB failed: {self._error}")

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        self._reopen_requested.set()
        self._force_release_cap()
        self._signal_queue(self._encode_queue)
        self._signal_queue(self._preview_queue)
        for worker in (
            self._encode_thread,
            self._preview_thread,
            self._watchdog_thread,
            self._thread,
        ):
            if worker is not None and worker.is_alive():
                worker.join(timeout=8.0)

    def _force_release_cap(self) -> None:
        """Best-effort release so a hung ``cap.read()`` can unblock on Windows."""
        with self._cap_holder_lock:
            cap = self._cap_holder[0]
            self._cap_holder[0] = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass

    @staticmethod
    def _signal_queue(target: queue.Queue[Any]) -> None:
        try:
            target.put_nowait(None)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(None)
            except queue.Full:
                pass

    def _open_capture(self, cv2: Any, backend: int) -> Any:
        """Open camera with retries — Windows DirectShow often needs a cool-down."""
        last_err: Optional[BaseException] = None
        for attempt in range(1, 6):
            if self._stop.is_set():
                raise RuntimeError("camera open cancelled")
            print(
                f"Opening OpenCV camera index={self.camera_index} "
                f"(attempt {attempt}/5) ...",
                flush=True,
            )
            cap = cv2.VideoCapture(self.camera_index, backend)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                ok, _frame = cap.read()
                if ok:
                    return cap
                cap.release()
                last_err = RuntimeError("opened but first read failed")
            else:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                last_err = RuntimeError(
                    f"Failed to open camera index {self.camera_index}"
                )
            time.sleep(0.6 * attempt)
        raise RuntimeError(str(last_err) if last_err else "camera open failed")

    def _enqueue_drop_old(self, target: queue.Queue[Any], item: Any) -> None:
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(item)
            except queue.Full:
                pass

    def _preview_loop(self, cv2: Any) -> None:
        last_write_s = 0.0
        while True:
            item = self._preview_queue.get()
            if item is None:
                break
            image, frame_index = item
            if self.preview_path is None:
                continue
            now_s = time.monotonic()
            # ~8 Hz GUI preview; skip surplus frames.
            if now_s - last_write_s < 0.12:
                continue
            temporary = self.preview_path.with_suffix(".tmp.png")
            try:
                if cv2.imwrite(str(temporary), image):
                    os.replace(temporary, self.preview_path)
                    # Sidecar tick avoids stuffing GUI stdout pipes (deadlock risk).
                    tick = self.preview_path.with_suffix(".tick")
                    tick.write_text(str(frame_index), encoding="utf-8")
                    last_write_s = now_s
            except OSError:
                pass

    def _encode_loop(
        self,
        *,
        use_ffmpeg: bool,
        encoder: Optional[subprocess.Popen[bytes]],
        writer: Any,
    ) -> None:
        while True:
            item = self._encode_queue.get()
            if item is None:
                break
            try:
                if use_ffmpeg:
                    assert encoder is not None and encoder.stdin is not None
                    encoder.stdin.write(item.tobytes())
                else:
                    assert writer is not None
                    writer.write(item)
            except Exception as exc:
                self._error = exc
                self._stop.set()
                break

    def _watchdog_loop(self) -> None:
        last_warn_s = 0.0
        while not self._stop.wait(0.5):
            if self._reopen_requested.is_set():
                continue
            last = self._last_frame_monotonic_ns
            if last == 0:
                continue
            age_s = (time.monotonic_ns() - last) * 1e-9
            if age_s < self._stall_limit_s:
                continue
            now_s = time.monotonic()
            if now_s - last_warn_s >= 2.0:
                print(
                    f"warning: camera stalled ({age_s:.1f}s), forcing reopen …",
                    flush=True,
                )
                last_warn_s = now_s
            self._reopen_requested.set()
            # Releasing the capture often unblocks a hung DirectShow read().
            self._force_release_cap()

    def _publish_frame(self, cv2: Any, image: Any, cap: Any) -> None:
        arrival_monotonic_ns = time.monotonic_ns()
        arrival_unix_ns = time.time_ns()
        frame_index = self._frame_count
        self._frame_count += 1
        record = {
            "video_frame_index": frame_index,
            "device_frame_number": frame_index,
            "device_timestamp_ms": float(cap.get(cv2.CAP_PROP_POS_MSEC)),
            "timestamp_domain": "host",
            "estimated_time_monotonic_ns": arrival_monotonic_ns,
            "estimated_time_unix_ns": arrival_unix_ns,
        }
        with self._lock:
            self._latest = record
            self._last_frame_monotonic_ns = arrival_monotonic_ns
        self._enqueue_drop_old(self._encode_queue, image.copy())
        if self.preview_path is not None:
            self._enqueue_drop_old(self._preview_queue, (image, frame_index))
        if frame_index == 0:
            self._ready.set()

    def _grab_until_reopen(self, cv2: Any, cap: Any) -> None:
        fail_streak = 0
        with self._cap_holder_lock:
            self._cap_holder[0] = cap
        try:
            while not self._stop.is_set() and not self._reopen_requested.is_set():
                try:
                    ok, image = cap.read()
                except Exception:
                    self._reopen_requested.set()
                    break
                if not ok or image is None:
                    fail_streak += 1
                    if fail_streak >= 30:
                        print(
                            "warning: camera read failed repeatedly, reopening …",
                            flush=True,
                        )
                        self._reopen_requested.set()
                        break
                    time.sleep(0.01)
                    continue
                fail_streak = 0
                if image.shape[1] != self.width or image.shape[0] != self.height:
                    image = cv2.resize(image, (self.width, self.height))
                image = orient_bgr_image(cv2, image, self.rotate_deg)
                self._publish_frame(cv2, image, cap)
        finally:
            with self._cap_holder_lock:
                if self._cap_holder[0] is cap:
                    self._cap_holder[0] = None
            try:
                cap.release()
            except Exception:
                pass

    def _thread_main(self) -> None:
        from platform import system

        encoder: Optional[subprocess.Popen[bytes]] = None
        writer = None
        use_ffmpeg = True
        try:
            import cv2  # type: ignore

            backend = cv2.CAP_DSHOW if system() == "Windows" else cv2.CAP_ANY

            try:
                encoder = _ffmpeg_raw_bgr_encoder(
                    self.video_path, self.out_width, self.out_height, self.fps
                )
                assert encoder.stdin is not None
            except FileNotFoundError:
                use_ffmpeg = False
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(self.video_path),
                    fourcc,
                    float(self.fps),
                    (self.out_width, self.out_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        "ffmpeg not found and OpenCV VideoWriter failed to open "
                        f"{self.video_path}"
                    )
                print(
                    "warning: ffmpeg not found; falling back to OpenCV mp4v writer",
                    flush=True,
                )

            self._encode_thread = threading.Thread(
                target=self._encode_loop,
                kwargs={
                    "use_ffmpeg": use_ffmpeg,
                    "encoder": encoder,
                    "writer": writer,
                },
                name="opencv-rgb-encode",
                daemon=True,
            )
            self._encode_thread.start()
            if self.preview_path is not None:
                self._preview_thread = threading.Thread(
                    target=self._preview_loop,
                    args=(cv2,),
                    name="opencv-rgb-preview",
                    daemon=True,
                )
                self._preview_thread.start()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop,
                name="opencv-rgb-watchdog",
                daemon=True,
            )
            self._watchdog_thread.start()

            first_open = True
            while not self._stop.is_set():
                self._reopen_requested.clear()
                cap = self._open_capture(cv2, backend)
                self._detected_serial = f"opencv:{self.camera_index}"
                if first_open:
                    print(f"OpenCV camera opened: {self._detected_serial}", flush=True)
                    first_open = False
                else:
                    print(
                        f"OpenCV camera reopened: {self._detected_serial}",
                        flush=True,
                    )
                for _ in range(max(5, self.fps // 2)):
                    if self._stop.is_set() or self._reopen_requested.is_set():
                        break
                    cap.read()
                self._grab_until_reopen(cv2, cap)
                if self._stop.is_set():
                    break
                # Brief cool-down before DirectShow reopen.
                time.sleep(0.4)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._stop.set()
            self._force_release_cap()
            self._signal_queue(self._encode_queue)
            self._signal_queue(self._preview_queue)
            if self._encode_thread is not None and self._encode_thread.is_alive():
                self._encode_thread.join(timeout=8.0)
            if self._preview_thread is not None and self._preview_thread.is_alive():
                self._preview_thread.join(timeout=3.0)
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            if encoder is not None:
                try:
                    if encoder.stdin is not None:
                        encoder.stdin.close()
                    encoder.wait(timeout=10.0)
                except Exception:
                    encoder.kill()
                    encoder.wait()

class RealSenseRGBRecorder:
    """Capture D435i RGB continuously and expose the latest encoded frame index."""

    def __init__(
        self,
        serial: str,
        width: int,
        height: int,
        fps: int,
        video_path: Path,
        preview_path: Optional[Path] = None,
        rotate_deg: int = 180,
    ) -> None:
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.rotate_deg = int(rotate_deg) % 360
        self.out_width, self.out_height = camera_output_size(
            width, height, self.rotate_deg
        )
        self.video_path = video_path
        self.preview_path = preview_path
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict[str, Any]] = None
        self._error: Optional[BaseException] = None
        self._detected_serial: Optional[str] = None
        self._frame_count = 0
        self._thread = threading.Thread(
            target=self._thread_main,
            name="realsense-rgb",
            daemon=True,
        )

    @property
    def detected_serial(self) -> Optional[str]:
        return self._detected_serial

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def start(self, timeout: float = 12.0) -> None:
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        if self.preview_path is not None:
            self.preview_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("RealSense RGB startup timed out")
        if self._error is not None:
            raise RuntimeError(f"RealSense RGB failed: {self._error}")

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=12.0)

    def _thread_main(self) -> None:
        pipeline = None
        encoder: Optional[subprocess.Popen[bytes]] = None
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
            import pyrealsense2 as rs  # type: ignore

            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial != "auto":
                config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            profile = pipeline.start(config)
            device = profile.get_device()
            self._detected_serial = device.get_info(rs.camera_info.serial_number)

            for _ in range(max(5, self.fps // 2)):
                if self._stop.is_set():
                    return
                pipeline.wait_for_frames(3000)

            encoder = _ffmpeg_raw_bgr_encoder(
                self.video_path, self.out_width, self.out_height, self.fps
            )
            assert encoder.stdin is not None

            while not self._stop.is_set():
                frames = pipeline.wait_for_frames(3000)
                color = frames.get_color_frame()
                if not color:
                    continue
                arrival_monotonic_ns = time.monotonic_ns()
                arrival_unix_ns = time.time_ns()
                image = orient_bgr_image(cv2, np.asanyarray(color.get_data()), self.rotate_deg)
                encoder.stdin.write(image.tobytes())
                frame_index = self._frame_count
                self._frame_count += 1
                record = {
                    "video_frame_index": frame_index,
                    "device_frame_number": int(color.get_frame_number()),
                    "device_timestamp_ms": float(color.get_timestamp()),
                    "timestamp_domain": str(color.get_frame_timestamp_domain()),
                    "estimated_time_monotonic_ns": arrival_monotonic_ns,
                    "estimated_time_unix_ns": arrival_unix_ns,
                }
                with self._lock:
                    self._latest = record
                if self.preview_path is not None and frame_index % 3 == 0:
                    temporary = self.preview_path.with_suffix(".tmp.png")
                    if cv2.imwrite(str(temporary), image):
                        os.replace(temporary, self.preview_path)
                if frame_index == 0:
                    self._ready.set()
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            if encoder is not None:
                try:
                    if encoder.stdin is not None:
                        encoder.stdin.close()
                    encoder.wait(timeout=10.0)
                except Exception:
                    encoder.kill()
                    encoder.wait()


class ExternalCameraFeed:
    """Talks to a persistent ``camera_idle_preview.py`` service instead of
    opening the camera itself.

    The GUI keeps that service running (and the camera open) for its whole
    session; ``start()``/``stop()`` here just tell it to begin/finalize one
    MP4 segment via a control file, so the device is never reopened around a
    recording. Exposes the same ``latest()``/``detected_serial``/``stop()``
    surface as ``OpenCVRGBRecorder``/``RealSenseRGBRecorder`` so ``main()``
    does not need to know which one it has.
    """

    def __init__(
        self,
        control_path: Path,
        status_path: Path,
        meta_path: Path,
        video_path: Path,
        fps: int,
    ) -> None:
        self.control_path = control_path
        self.status_path = status_path
        self.meta_path = meta_path
        self.video_path = video_path
        self.fps = fps
        self._detected_serial: Optional[str] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[dict[str, Any]] = None
        self._meta_mtime_ns = 0
        self._thread: Optional[threading.Thread] = None

    @property
    def detected_serial(self) -> Optional[str]:
        return self._detected_serial

    def _read_status(self) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self.status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _wait_for_state(self, state: str, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: Optional[dict[str, Any]] = None
        while time.monotonic() < deadline:
            status = self._read_status()
            if status is not None:
                last = status
                if status.get("error"):
                    raise RuntimeError(f"camera service error: {status['error']}")
                if status.get("state") == state:
                    return status
            time.sleep(0.02)
        raise TimeoutError(
            f"camera service did not reach state={state!r} within {timeout:.1f}s "
            f"(last status: {last})"
        )

    def start(self, timeout: float = 20.0) -> None:
        if not self.control_path.parent.exists():
            self.control_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cmd": "start", "video_path": str(self.video_path), "fps": self.fps}
        temporary = self.control_path.with_suffix(self.control_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, self.control_path)
        status = self._wait_for_state("recording", timeout)
        self._detected_serial = status.get("serial")
        self._thread = threading.Thread(target=self._tail_meta_loop, daemon=True)
        self._thread.start()

    def _tail_meta_loop(self) -> None:
        while not self._stop.is_set():
            try:
                mtime_ns = self.meta_path.stat().st_mtime_ns
            except OSError:
                time.sleep(0.005)
                continue
            if mtime_ns != self._meta_mtime_ns:
                self._meta_mtime_ns = mtime_ns
                try:
                    record = json.loads(self.meta_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    record = None
                if record is not None:
                    with self._lock:
                        self._latest = record
            time.sleep(0.005)

    def latest(self) -> Optional[dict[str, Any]]:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def stop(self, timeout: float = 20.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        payload = {"cmd": "stop"}
        try:
            temporary = self.control_path.with_suffix(self.control_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(temporary, self.control_path)
            self._wait_for_state("idle", timeout)
        except Exception as exc:
            print(f"warning: camera segment finalize wait failed: {exc}", flush=True)


def _is_unix_epoch_seconds(timestamp_s: float) -> bool:
    """True if python-can timestamp looks like wall-clock Unix seconds.

    SocketCAN on Linux often provides epoch seconds.  gs_usb on Windows often
    provides a relative bus/boot timer (hundreds of seconds), which must NOT be
    compared against ``time.time_ns()``.
    """
    return 1_500_000_000.0 <= float(timestamp_s) <= 2_500_000_000.0


def message_meta(message: Any, sample_unix_ns: Optional[int]) -> dict[str, Any]:
    timestamp_s = float(message.timestamp)
    can_timestamp_ns = int(round(timestamp_s * 1_000_000_000.0))
    unixish = _is_unix_epoch_seconds(timestamp_s)
    if sample_unix_ns is None:
        age_ms: Optional[float] = None
    elif unixish:
        age_ms = (sample_unix_ns - can_timestamp_ns) * 1e-6
    else:
        # Relative CAN clock: absolute age vs wall clock is meaningless.
        age_ms = None
    return {
        "can_timestamp_s": timestamp_s,
        "can_timestamp_ns": can_timestamp_ns,
        "can_timestamp_domain": "unix_epoch" if unixish else "relative_or_hw",
        "age_at_sample_ms": age_ms,
        "source_hz": float(message.hz),
    }


def joint_record(
    message: Any, sample_unix_ns: Optional[int]
) -> Optional[dict[str, Any]]:
    if message is None:
        return None
    return {
        "position_rad": [float(value) for value in message.msg],
        **message_meta(message, sample_unix_ns),
    }


def gripper_ctrl_record(
    message: Any, sample_unix_ns: Optional[int]
) -> Optional[dict[str, Any]]:
    """Preserve 0x159 data and derive a physical value from its status code.

    The current SDK always decodes the first four bytes with a 1e-6 scale.
    Reconstructing the integer payload keeps the recording reversible.  Status
    codes 0x04..0x07 indicate angle mode, whose payload unit is millidegrees;
    0x00..0x03 indicate width mode, whose payload unit is micrometres.
    """
    if message is None:
        return None

    msg = message.msg
    status_code = int(msg.status_code)
    raw_value = int(round(float(msg.value) * 1_000_000.0))
    mode = "angle" if status_code & 0x04 else "width"
    if mode == "angle":
        physical_value = raw_value * 1e-3
        physical_unit = "deg"
    else:
        physical_value = raw_value * 1e-6
        physical_unit = "m"

    return {
        "can_id": "0x159",
        "raw_value": raw_value,
        "mode": mode,
        "value": physical_value,
        "value_unit": physical_unit,
        "force_n": float(msg.force),
        "status_code": status_code,
        "set_zero": int(msg.set_zero),
        **message_meta(message, sample_unix_ns),
    }


def gripper_feedback_record(
    message: Any, sample_unix_ns: Optional[int]
) -> Optional[dict[str, Any]]:
    if message is None:
        return None

    msg = message.msg
    mode = str(msg.mode)
    return {
        "can_id": "0x2A8",
        "mode": mode,
        "value": float(msg.value),
        "value_unit": "deg" if mode == "angle" else "m",
        "force_n": float(msg.force),
        "status_code": int(msg.status_code),
        **message_meta(message, sample_unix_ns),
    }


def mapped_hand_record(
    gripper_ctrl: Optional[dict[str, Any]], mapper: GripperToGraspMapper
) -> Optional[dict[str, Any]]:
    if gripper_ctrl is None:
        return None
    grasp = mapper.grasp(gripper_ctrl["mode"], float(gripper_ctrl["value"]))
    return {
        "grasp": grasp,
        "position_normalized": mapper.positions(grasp),
        "derived_from_can_timestamp_ns": gripper_ctrl["can_timestamp_ns"],
    }


def build_training_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    follower = snapshot["follower"]
    leader = snapshot["leader"]
    hand_feedback = snapshot["revo2_feedback"]
    hand_target = snapshot["revo2_target"]
    poses = snapshot.get("poses") or {}
    gripper = snapshot.get("gripper") or {}

    state_arm = None if follower is None else list(follower["position_rad"])
    action_arm = None if leader is None else list(leader["position_rad"])

    # Prefer AgxGripper 0..1 grasp; keep Revo2 fields for optional legacy use.
    state_grasp = gripper.get("state_grasp")
    action_grasp = gripper.get("action_grasp")
    if state_grasp is None and hand_feedback is not None:
        state_grasp = float(hand_feedback["grasp"])
    if action_grasp is None and hand_target is not None:
        action_grasp = float(hand_target["grasp"])

    state_vector = (
        None if state_arm is None or state_grasp is None else state_arm + [state_grasp]
    )
    action_vector = (
        None
        if action_arm is None or action_grasp is None
        else action_arm + [action_grasp]
    )

    tcp = poses.get("tcp_pose")
    tcp_vector = None if tcp is None or state_grasp is None else list(tcp) + [state_grasp]
    tcp_action_vector = (
        None if tcp is None or action_grasp is None else list(tcp) + [action_grasp]
    )

    return {
        "state": {
            "arm_joint_position_rad": state_arm,
            "gripper_grasp": state_grasp,
            "hand_grasp": state_grasp,  # alias; Agx path no longer requires Revo2
            "hand_position_normalized": (
                None
                if hand_feedback is None
                else list(hand_feedback["position_normalized"])
            ),
            "flange_pose": poses.get("flange_pose"),
            "tcp_pose": poses.get("tcp_pose"),
            "fk_pose": poses.get("fk_pose"),
            "vector": state_vector,  # joints[7] + gripper[1]
            "vector_tcp_gripper": tcp_vector,  # tcp[6] + gripper[1]
        },
        "action_target": {
            "arm_joint_position_rad": action_arm,
            "gripper_grasp": action_grasp,
            "hand_grasp": action_grasp,
            "hand_position_normalized": (
                None
                if hand_target is None
                else list(hand_target["position_normalized"])
            ),
            "vector": action_vector,
            "vector_tcp_gripper": tcp_action_vector,
        },
        # Filled when the next sample arrives.  This is realized motion, not
        # the command sent by the human operator.
        "action_delta_state": None,
    }


def attach_next_state_delta(
    current: dict[str, Any], following: dict[str, Any]
) -> None:
    current_state = current["training"]["state"]
    following_state = following["training"]["state"]
    arm_current = current_state["arm_joint_position_rad"]
    arm_following = following_state["arm_joint_position_rad"]
    hand_current = current_state["hand_position_normalized"]
    hand_following = following_state["hand_position_normalized"]
    grasp_current = current_state["hand_grasp"]
    grasp_following = following_state["hand_grasp"]
    arm_delta = (
        None
        if arm_current is None or arm_following is None
        else [new - old for old, new in zip(arm_current, arm_following)]
    )
    grasp_delta = (
        None
        if grasp_current is None or grasp_following is None
        else grasp_following - grasp_current
    )
    current["training"]["action_delta_state"] = {
        "dt_s": (following["time_monotonic_ns"] - current["time_monotonic_ns"])
        * 1e-9,
        "arm_joint_position_rad": arm_delta,
        "hand_grasp": grasp_delta,
        "hand_position_normalized": (
            None
            if hand_current is None or hand_following is None
            else [new - old for old, new in zip(hand_current, hand_following)]
        ),
        "vector": (
            None if arm_delta is None or grasp_delta is None else arm_delta + [grasp_delta]
        ),
    }


def read_snapshot(
    robot: Any,
    gripper: Any,
    mapper: GripperToGraspMapper,
    hand_reader: Optional[Revo2FeedbackReader],
    camera_reader: Optional[Any],
    recording_start_ns: int,
    sequence: int,
    revo2_sync_enabled: bool,
) -> dict[str, Any]:
    capture_start_monotonic_ns = time.monotonic_ns()
    capture_start_unix_ns = time.time_ns()
    # Copy every mutable SDK message immediately.  The parser background
    # thread may update its cached objects while this function is running.
    leader = joint_record(robot.get_leader_joint_angles(), None)
    follower = joint_record(robot.get_joint_angles(), None)
    ctrl = gripper_ctrl_record(gripper.get_gripper_ctrl_states(), None)
    gripper_feedback = gripper_feedback_record(gripper.get_gripper_status(), None)
    flange_msg = robot.get_flange_pose()
    tcp_msg = robot.get_tcp_pose()

    fk_pose = None
    if follower is not None:
        try:
            fk_pose = as_pose6(robot.fk(list(follower["position_rad"])))
        except Exception:
            fk_pose = None
    flange_pose = None if flange_msg is None else as_pose6(flange_msg.msg)
    tcp_pose = None if tcp_msg is None else as_pose6(tcp_msg.msg)
    # If controller TCP is missing but flange exists, apply SDK offset locally.
    if tcp_pose is None and flange_pose is not None:
        try:
            tcp_pose = as_pose6(robot.get_flange2tcp_pose(list(flange_pose)))
        except Exception:
            tcp_pose = None

    state_grasp, action_grasp = gripper_grasp_from_records(
        mapper, ctrl, gripper_feedback
    )

    hand_feedback = None if hand_reader is None else hand_reader.latest()
    if hand_feedback is not None:
        hand_feedback["grasp"] = mapper.grasp_from_positions(
            hand_feedback["position_normalized"]
        )
    camera_rgb = None if camera_reader is None else camera_reader.latest()
    capture_end_monotonic_ns = time.monotonic_ns()
    # The aligned row time is capture end.  All values below were known by
    # this instant, so latest-value/zero-order-hold never leaks future data.
    sample_monotonic_ns = capture_end_monotonic_ns
    sample_unix_ns = capture_start_unix_ns + (
        sample_monotonic_ns - capture_start_monotonic_ns
    )

    can_records = [
        record
        for record in (leader, follower, ctrl, gripper_feedback)
        if record is not None
    ]
    for record in can_records:
        if record.get("can_timestamp_domain") == "unix_epoch":
            record["age_at_sample_ms"] = (
                sample_unix_ns - record["can_timestamp_ns"]
            ) * 1e-6
        else:
            # Relative/hardware CAN clock cannot be compared to wall time.
            record["age_at_sample_ms"] = None
    if hand_feedback is not None:
        hand_feedback["age_at_sample_ms"] = (
            sample_monotonic_ns - hand_feedback["estimated_time_monotonic_ns"]
        ) * 1e-6
    if camera_rgb is not None:
        camera_rgb["age_at_sample_ms"] = (
            sample_monotonic_ns - camera_rgb["estimated_time_monotonic_ns"]
        ) * 1e-6

    can_timestamps = [record["can_timestamp_ns"] for record in can_records]
    can_ages = [
        record["age_at_sample_ms"]
        for record in can_records
        if record.get("age_at_sample_ms") is not None
    ]
    can_skew_ms = (
        0.0
        if len(can_timestamps) < 2
        else (max(can_timestamps) - min(can_timestamps)) * 1e-6
    )
    relative_can_clock = any(
        record.get("can_timestamp_domain") == "relative_or_hw" for record in can_records
    )
    if not can_records:
        can_valid = False
    elif relative_can_clock or not can_ages:
        # Windows gs_usb: accept if sources exist and are mutually fresh (<50ms skew).
        can_valid = can_skew_ms <= 50.0
    else:
        can_valid = all(0.0 <= age <= 10.0 for age in can_ages)
    hand_valid = (
        hand_feedback is None
        if hand_reader is None
        else hand_feedback is not None
        and 0.0 <= hand_feedback["age_at_sample_ms"] <= 100.0
    )
    camera_valid = (
        camera_rgb is None
        if camera_reader is None
        else camera_rgb is not None
        # Windows OpenCV (esp. without ffmpeg) can lag; 500ms is still usable
        # for zero-order-hold alignment and keeps the GUI honest.
        and 0.0 <= camera_rgb["age_at_sample_ms"] <= 500.0
    )
    snapshot = {
        "kind": "sample",
        "seq": sequence,
        "time_unix_ns": sample_unix_ns,
        "time_monotonic_ns": sample_monotonic_ns,
        "time_iso": datetime.fromtimestamp(sample_unix_ns * 1e-9).astimezone().isoformat(
            timespec="milliseconds"
        ),
        "elapsed_s": (sample_monotonic_ns - recording_start_ns) * 1e-9,
        "capture_window_us": (
            capture_end_monotonic_ns - capture_start_monotonic_ns
        ) * 1e-3,
        "alignment": {
            "method": "latest value at or before aligned row time (zero-order hold)",
            "can_source_skew_ms": can_skew_ms,
            "can_valid_10ms": can_valid,
            "hand_valid_100ms": hand_valid,
            # camera_valid_100ms kept for older GUI; threshold is actually 500 ms.
            "camera_valid_100ms": camera_valid,
            "camera_valid_500ms": camera_valid,
            "valid": can_valid and hand_valid and camera_valid,
        },
        "leader": leader,
        "follower": follower,
        "gripper_ctrl": ctrl,
        "gripper_feedback": gripper_feedback,
        "gripper": {
            "state_grasp": state_grasp,
            "action_grasp": action_grasp,
            "unit": "normalized_0_closed_1_open_or_mapped",
        },
        "poses": {
            "frame": "base",
            "convention": "xyz + rpy(ZYX), meters/radians",
            "labels": list(POSE_LABELS),
            "flange_pose": flange_pose,
            "tcp_pose": tcp_pose,
            "fk_pose": fk_pose,
            "tcp_offset_flange_frame": list(NERO_GRIPPER_TCP_OFFSET_M),
            "sources": {
                "flange_pose": "robot.get_flange_pose()",
                "tcp_pose": "robot.get_tcp_pose() or flange2tcp(flange)",
                "fk_pose": "robot.fk(follower.joint_angles)",
            },
        },
        "revo2_target": mapped_hand_record(ctrl, mapper),
        "revo2_feedback": hand_feedback,
        "camera_rgb": camera_rgb,
        "revo2_sync_enabled": revo2_sync_enabled,
    }
    snapshot["training"] = build_training_fields(snapshot)
    snapshot["training"]["hand_sync_enabled"] = revo2_sync_enabled
    return snapshot


def write_json_line(file_obj: Any, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def wait_for_joint_data(robot: Any, timeout: float) -> tuple[bool, bool]:
    deadline = time.monotonic() + max(0.0, timeout)
    have_leader = False
    have_follower = False
    while time.monotonic() < deadline and not (have_leader and have_follower):
        have_leader = robot.get_leader_joint_angles() is not None
        have_follower = robot.get_joint_angles() is not None
        if not (have_leader and have_follower):
            time.sleep(0.01)
    return have_leader, have_follower


def main() -> int:
    args = parse_args()
    try:
        args.task = validate_task(args.task)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.hz <= 0:
        print("--hz must be greater than 0", file=sys.stderr)
        return 2
    if args.duration < 0:
        print("--duration cannot be negative", file=sys.stderr)
        return 2
    if args.max_range <= 0 or args.max_angle <= 0:
        print("mapping ranges must be greater than zero", file=sys.stderr)
        return 2
    if not 0 <= args.max_position <= 1000:
        print("--max-position must be in 0..1000", file=sys.stderr)
        return 2
    if args.hand_hz <= 0:
        print("--hand-hz must be greater than zero", file=sys.stderr)
        return 2
    if args.hand_deadband < 0 or args.hand_motion_ms <= 0:
        print("hand deadband/motion settings are invalid", file=sys.stderr)
        return 2
    if args.camera and (
        args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0
    ):
        print("camera width/height/fps must be greater than zero", file=sys.stderr)
        return 2
    if args.camera and args.camera_backend == "external" and (
        args.camera_control_path is None
        or args.camera_status_path is None
        or args.camera_meta_path is None
    ):
        print(
            "--camera-backend external requires --camera-control-path, "
            "--camera-status-path and --camera-meta-path",
            file=sys.stderr,
        )
        return 2

    output = args.output or default_output_path()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    camera_video = (
        (args.camera_video.expanduser().resolve() if args.camera_video else output.with_suffix(".rgb.mp4"))
        if args.camera
        else None
    )

    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V112,
        interface=args.interface,
        channel=args.channel,
    )
    if args.interface == "gs_usb":
        from safe_arm import _install_gs_usb_adapter

        _install_gs_usb_adapter()
    robot = AgxArmFactory.create_arm(config)
    gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
    mapper = GripperToGraspMapper(
        max_range_m=args.max_range,
        max_angle_deg=args.max_angle,
        max_position=args.max_position,
        open_pose=args.open_pose,
    )

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Windows GUI stops the child with CTRL_BREAK_EVENT → SIGBREAK.
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)

    print(f"Connecting to Nero on {args.interface}:{args.channel} (read only) ...")
    try:
        robot.connect()
    except Exception as exc:
        print(f"Connect failed: {exc}", file=sys.stderr)
        return 1

    # 见文件顶部 NERO_GRIPPER_TCP_OFFSET_M 注释：法兰局部 +X 方向 13 cm 到夹爪工具中心。
    robot.set_tcp_offset(list(NERO_GRIPPER_TCP_OFFSET_M))
    print(f"TCP offset (flange frame, m/rad): {NERO_GRIPPER_TCP_OFFSET_M}")

    samples = 0
    missing_leader = 0
    missing_follower = 0
    skipped_unaligned = 0
    jump_warns = 0
    start_ns = 0
    hand_reader: Optional[Revo2FeedbackReader] = None
    camera_reader: Optional[Any] = None
    sync_enabled = read_sync_control(args.sync_control_file, args.execute_hand)

    try:
        have_leader, have_follower = wait_for_joint_data(robot, args.wait_timeout)
        if not have_leader:
            print("Warning: leader joint frames are not available yet.")
        if not have_follower:
            print("Warning: follower joint frames are not available yet.")
        try:
            teach_param = gripper.get_gripper_teaching_pendant_param(
                timeout=1.0, min_interval=0.0
            )
        except Exception as exc:
            teach_param = None
            print(f"Warning: teach-pendant parameter query failed: {exc}")
        if teach_param is not None:
            mapper.update_max_range(teach_param.msg.max_range_config)
        if args.hand_feedback:
            hand_reader = Revo2FeedbackReader(
                port=args.hand_port,
                slave_id=args.hand_slave_id,
                hz=args.hand_hz,
                execute=sync_enabled,
                deadband=args.hand_deadband,
                motion_ms=args.hand_motion_ms,
                control_capable=args.sync_control_file is not None,
            )
            hand_mode = "CONTROL+RECORD" if sync_enabled else "READ ONLY"
            print(f"Connecting Revo2 ({hand_mode}) ...")
            hand_reader.start()
            print(
                f"Revo2 feedback: {hand_reader.detected_port} "
                f"slave=0x{args.hand_slave_id:02X} at {args.hand_hz:g} Hz"
            )
        if args.camera:
            assert camera_video is not None
            try:
                if args.camera_backend == "external":
                    camera_reader = ExternalCameraFeed(
                        control_path=args.camera_control_path.expanduser().resolve(),
                        status_path=args.camera_status_path.expanduser().resolve(),
                        meta_path=args.camera_meta_path.expanduser().resolve(),
                        video_path=camera_video,
                        fps=args.camera_fps,
                    )
                    print(
                        f"Connecting to persistent camera service via "
                        f"{args.camera_control_path} ..."
                    )
                elif args.camera_backend == "opencv":
                    camera_reader = OpenCVRGBRecorder(
                        camera_index=args.camera_index,
                        width=args.camera_width,
                        height=args.camera_height,
                        fps=args.camera_fps,
                        video_path=camera_video,
                        preview_path=(
                            None
                            if args.camera_preview_path is None
                            else args.camera_preview_path.expanduser().resolve()
                        ),
                        rotate_deg=args.camera_rotate,
                    )
                    print(
                        f"Connecting OpenCV RGB index={args.camera_index} "
                        f"{args.camera_width}x{args.camera_height}@{args.camera_fps} "
                        f"rotate={args.camera_rotate} ..."
                    )
                else:
                    camera_reader = RealSenseRGBRecorder(
                        serial=args.camera_serial,
                        width=args.camera_width,
                        height=args.camera_height,
                        fps=args.camera_fps,
                        video_path=camera_video,
                        preview_path=(
                            None
                            if args.camera_preview_path is None
                            else args.camera_preview_path.expanduser().resolve()
                        ),
                        rotate_deg=args.camera_rotate,
                    )
                    print(
                        f"Connecting RealSense RGB {args.camera_width}x{args.camera_height}"
                        f"@{args.camera_fps} ..."
                    )
                camera_reader.start()
                print(
                    f"RGB camera: id={camera_reader.detected_serial} "
                    f"video={camera_video}"
                )
            except Exception as camera_exc:
                print(f"Warning: camera start failed: {camera_exc}", flush=True)
                print(
                    "Continuing WITHOUT camera so arm/gripper/TCP can still be recorded.",
                    flush=True,
                )
                if camera_reader is not None:
                    try:
                        camera_reader.stop()
                    except Exception:
                        pass
                camera_reader = None
                args.camera = False

        metadata = {
            "kind": "metadata",
            "schema": "nero_teleop_jsonl",
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "task": args.task,
            "arm_model": "nero",
            "firmware_driver": "v112",
            "can_interface": args.interface,
            "can_channel": args.channel,
            "tcp_offset_flange_frame": list(NERO_GRIPPER_TCP_OFFSET_M),
            "sample_hz": args.hz,
            "joint_unit": "rad",
            "revo2_mapping": {
                "open_pose": list(mapper.open_pose),
                "closed_pose": [mapper.max_position] * 6,
                "max_range_m": mapper.max_range_m,
                "max_angle_deg": mapper.max_angle_deg,
            },
            "time_alignment": {
                "row_time": "capture-end host CLOCK_MONOTONIC timestamp",
                "sample_clock": "host CLOCK_MONOTONIC",
                "wall_clock": "host CLOCK_REALTIME",
                "can_clock": "SocketCAN kernel receive timestamp",
                "age_field": "sample_time - CAN_receive_time in milliseconds",
                "policy": "latest value at or before each uniform row time (zero-order hold)",
                    "valid_thresholds": {
                        "can_ms": 10.0,
                        "revo2_ms": 100.0,
                        "camera_ms": 500.0,
                    },
                    "valid_field_notes": {
                        "can_valid_10ms": "leader/follower ages <= 10ms (or skew<=50ms fallback)",
                        "hand_valid_100ms": "gripper feedback age <= 100ms",
                        "camera_valid_500ms": "RGB age <= 500ms (legacy alias: camera_valid_100ms)",
                    },
            },
            "collection": {
                "outcome": None,
                "outcome_note": (
                    "GUI writes success/failure/abort/unreviewed after stop; "
                    "converter trains on success (and legacy null)."
                ),
                "tcp_jump_warn_mm": TCP_JUMP_WARN_MM,
                "tcp_jump_warn_deg": TCP_JUMP_WARN_DEG,
            },
            "camera_rgb": {
                "enabled": args.camera,
                "backend": args.camera_backend,
                "camera_index": args.camera_index if args.camera_backend == "opencv" else None,
                "serial": (
                    None if camera_reader is None else camera_reader.detected_serial
                ),
                "capture_width": args.camera_width,
                "capture_height": args.camera_height,
                "rotate_deg_cw": args.camera_rotate,
                "width": camera_output_size(
                    args.camera_width, args.camera_height, args.camera_rotate
                )[0],
                "height": camera_output_size(
                    args.camera_width, args.camera_height, args.camera_rotate
                )[1],
                "capture_fps": args.camera_fps,
                "storage": "mp4_video_not_image_sequence",
                "video_path": None if camera_video is None else str(camera_video),
                "codec": "h264_or_opencv_mp4v_fallback",
                "alignment_reference": "camera_rgb.video_frame_index",
                "note": (
                    "RGB is stored as a continuous MP4 after camera-rotate. JSONL frames "
                    "point into it via camera_rgb.video_frame_index."
                ),
            },
            "poses": {
                "flange_pose": "robot.get_flange_pose() [x,y,z,r,p,y]",
                "tcp_pose": "robot.get_tcp_pose() with tcp_offset_flange_frame",
                "fk_pose": "robot.fk(follower joints) [x,y,z,r,p,y]",
                "units": "meters / radians, base frame, RPY ZYX",
            },
            "gripper": {
                "type": "agx_gripper",
                "state_grasp": "0..1 from gripper_feedback (fallback: ctrl)",
                "action_grasp": "0..1 from gripper_ctrl / teach pendant",
            },
            "revo2_feedback": {
                "enabled": args.hand_feedback,
                "execute_initial": sync_enabled,
                "runtime_switch": args.sync_control_file is not None,
                "port": None if hand_reader is None else hand_reader.detected_port,
                "slave_id": args.hand_slave_id,
                "poll_hz": args.hand_hz,
                "timestamp": "host midpoint of Modbus request/response",
                "uncertainty": "half of measured Modbus round-trip time",
            },
            "training_fields": {
                "state.vector": "8D [follower joints q1..q7 rad, gripper grasp 0..1]",
                "state.vector_tcp_gripper": "7D [tcp xyzrpy, gripper grasp 0..1]",
                "action_target.vector": "8D [leader joints q1..q7 rad, gripper grasp 0..1]",
                "poses": "flange_pose / tcp_pose / fk_pose all retained on each sample",
                "action_delta_state": "next actual state minus current actual state",
            },
            "notes": (
                "Native CAN leader/follower linkage remains in control of the arm. "
                "Raw JSONL keeps image(via mp4 index)+joints+flange+tcp+fk+gripper; "
                "export to LeRobot v3 later chooses which vector becomes observation.state/action."
            ),
        }

        start_ns = time.monotonic_ns()
        interval = 1.0 / args.hz
        started = time.monotonic()
        next_tick = started
        next_status = started + 1.0
        next_fsync = (
            started + args.fsync_seconds if args.fsync_seconds > 0 else float("inf")
        )

        with output.open("w", encoding="utf-8") as file_obj:
            write_json_line(file_obj, metadata)
            file_obj.flush()
            print(f"Recording to {output}")
            print(f"task: {args.task}")
            print("Press Ctrl+C to stop.")
            pending_snapshot: Optional[dict[str, Any]] = None
            prev_tcp: Optional[list[float]] = None

            while not stop_requested:
                now = time.monotonic()
                if args.duration > 0 and now - started >= args.duration:
                    break
                if now < next_tick:
                    time.sleep(next_tick - now)
                    continue

                sync_enabled = read_sync_control(
                    args.sync_control_file, sync_enabled
                )
                if hand_reader is not None:
                    hand_reader.set_execute(sync_enabled)
                snapshot = read_snapshot(
                    robot,
                    gripper,
                    mapper,
                    hand_reader,
                    camera_reader,
                    start_ns,
                    samples,
                    sync_enabled,
                )
                tcp_pose = (snapshot.get("poses") or {}).get("tcp_pose")
                quality = build_tcp_quality(prev_tcp, tcp_pose)
                snapshot["quality"] = quality
                if quality.get("tcp_jump_warn"):
                    jump_warns += 1
                    if jump_warns <= 5 or jump_warns % 20 == 0:
                        print(
                            f"\nWARN tcp jump #{jump_warns}: "
                            f"{quality.get('tcp_jump_mm'):.1f} mm / "
                            f"{quality.get('tcp_jump_deg'):.1f} deg "
                            f"(seq={snapshot.get('seq')})",
                            flush=True,
                        )
                if tcp_pose is not None:
                    prev_tcp = [float(v) for v in tcp_pose[:6]]
                if hand_reader is not None and snapshot["revo2_target"] is not None:
                    hand_reader.set_target(
                        snapshot["revo2_target"]["position_normalized"]
                    )
                if not snapshot["alignment"]["valid"] and not args.keep_unaligned:
                    skipped_unaligned += 1
                    next_tick += interval
                    continue
                if pending_snapshot is not None:
                    attach_next_state_delta(pending_snapshot, snapshot)
                    write_json_line(file_obj, pending_snapshot)
                pending_snapshot = snapshot
                samples += 1
                missing_leader += snapshot["leader"] is None
                missing_follower += snapshot["follower"] is None

                now = time.monotonic()
                if now >= next_fsync:
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                    next_fsync = now + args.fsync_seconds
                if now >= next_status:
                    gc = snapshot["gripper_ctrl"]
                    grip_text = "--" if gc is None else f"{gc['value']:.4f}{gc['value_unit']}"
                    target = snapshot["revo2_target"]
                    target_text = "--" if target is None else str(target["position_normalized"])
                    actual = snapshot["revo2_feedback"]
                    actual_text = "--" if actual is None else str(actual["position_normalized"])
                    camera = snapshot["camera_rgb"]
                    camera_text = (
                        "--"
                        if camera is None
                        else f"#{camera['video_frame_index']} {camera['age_at_sample_ms']:.1f}ms"
                    )
                    jump_mm = quality.get("tcp_jump_mm")
                    jump_text = (
                        "JUMP"
                        if quality.get("tcp_jump_warn")
                        else (
                            f"{jump_mm:.0f}mm"
                            if jump_mm is not None
                            else "--"
                        )
                    )
                    print(
                        f"\r{samples} samples | leader={'OK' if snapshot['leader'] else '--'} "
                        f"follower={'OK' if snapshot['follower'] else '--'} "
                        f"teach_gripper={grip_text} target={target_text} "
                        f"actual={actual_text} "
                        f"camera={camera_text} "
                        f"tcp={jump_text} "
                        f"hand_sync={'ON' if sync_enabled else 'OFF'} "
                        f"aligned={'Y' if snapshot['alignment']['valid'] else 'N'}   ",
                        end="",
                        flush=True,
                    )
                    next_status = now + 1.0

                next_tick += interval
                if now - next_tick > interval * 5:
                    next_tick = now + interval

            if pending_snapshot is not None:
                write_json_line(file_obj, pending_snapshot)
            file_obj.flush()
            os.fsync(file_obj.fileno())
    finally:
        if camera_reader is not None:
            camera_reader.stop()
        if hand_reader is not None:
            hand_reader.stop()
        try:
            robot.disconnect()
        except Exception:
            pass

    print()
    print(
        f"Saved {samples} samples to {output} "
        f"(missing leader={missing_leader}, follower={missing_follower}, "
        f"dropped unaligned={skipped_unaligned}, tcp_jump_warns={jump_warns})"
    )
    return 0 if samples else 1


if __name__ == "__main__":
    raise SystemExit(main())
