#!/usr/bin/env python3
"""Persistent RGB camera service (Orbbec Dabai / D435i UVC / RealSense).

Opens the camera once and keeps it open for the GUI's entire lifetime:
always writes a ~10 Hz preview PNG, and on request (via ``--control-path``)
also encodes the *same* already-open capture into an MP4 for one recording
segment at a time. The device is never closed/reopened between segments,
which is the whole point -- reopening a UVC/RealSense sensor forces its
auto-exposure/auto-white-balance to re-converge, ruining the first second
or so of every recording.

Control protocol (all files are optional; omit a path to disable that bit):
  --control-path: GUI/recorder writes one JSON object here to start or stop
      an encoding segment: {"cmd": "start", "video_path": "...", "fps": N}
      or {"cmd": "stop"} / {"cmd": "force_stop"}. Detected by mtime change,
      so re-reading the same file twice is a no-op.
  --status-path: this process writes {"state": "idle"|"recording",
      "serial": "...", "error": "..."} here after every transition.
  --meta-path: while recording, this process writes one JSON object per
      captured frame here (video_frame_index/device_timestamp_ms/...),
      mirroring the in-process camera readers' ``latest()`` shape so a
      recorder subprocess can align CAN samples against frames it never
      captured itself.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time
from pathlib import Path
from platform import system
from typing import Any, Optional


def orient_bgr(cv2, image, rotate_deg: int):
    deg = int(rotate_deg) % 360
    if deg == 0:
        return image
    if deg == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"unsupported rotate: {rotate_deg}")


def output_size(width: int, height: int, rotate_deg: int) -> tuple[int, int]:
    if int(rotate_deg) % 360 in (90, 270):
        return int(height), int(width)
    return int(width), int(height)


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        pass


def _ffmpeg_raw_bgr_encoder(video_path: Path, width: int, height: int, fps: int):
    import subprocess

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


class SegmentRecorder:
    """Owns one ffmpeg encoder for one recording segment.

    Encoding runs on a background thread with a drop-old queue so a slow
    encoder can never stall the capture loop feeding it.
    """

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=2)
        self._thread: Optional[threading.Thread] = None
        self._encoder = None

    def begin(self, video_path: Path) -> None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        self._encoder = _ffmpeg_raw_bgr_encoder(video_path, self.width, self.height, self.fps)
        if self._encoder.stdin is None:
            raise RuntimeError("failed to open ffmpeg encoder")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, image) -> None:
        try:
            self._queue.put_nowait(image)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(image)
            except queue.Full:
                pass

    def _run(self) -> None:
        encoder = self._encoder
        assert encoder is not None and encoder.stdin is not None
        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                encoder.stdin.write(item.tobytes())
            except Exception:
                break

    def end(self) -> None:
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=10.0)
            self._thread = None
        encoder = self._encoder
        self._encoder = None
        if encoder is not None:
            try:
                if encoder.stdin is not None:
                    encoder.stdin.close()
                encoder.wait(timeout=10.0)
            except Exception:
                encoder.kill()
                encoder.wait()


class ControlChannel:
    """Polls the control file and drives SegmentRecorder start/stop."""

    def __init__(
        self,
        control_path: Optional[Path],
        status_path: Optional[Path],
        meta_path: Optional[Path],
        serial: str,
        out_width: int,
        out_height: int,
        fps: int,
    ) -> None:
        self.control_path = control_path
        self.status_path = status_path
        self.meta_path = meta_path
        self.serial = serial
        self.recorder = SegmentRecorder(out_width, out_height, fps) if control_path else None
        self._control_mtime_ns = 0
        self._segment_frame_index = 0
        self.recording = False
        self._write_status()

    def _write_status(self, error: Optional[str] = None) -> None:
        if self.status_path is None:
            return
        _atomic_write_json(
            self.status_path,
            {
                "state": "recording" if self.recording else "idle",
                "serial": self.serial,
                "error": error,
            },
        )

    def poll(self) -> None:
        if self.control_path is None:
            return
        try:
            mtime_ns = self.control_path.stat().st_mtime_ns
        except OSError:
            return
        if mtime_ns == self._control_mtime_ns:
            return
        self._control_mtime_ns = mtime_ns
        try:
            payload = json.loads(self.control_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        cmd = payload.get("cmd")
        if cmd == "start":
            self._begin_segment(payload)
        elif cmd in ("stop", "force_stop"):
            self._end_segment()

    def _begin_segment(self, payload: dict) -> None:
        if self.recording:
            self._end_segment()
        assert self.recorder is not None
        self._segment_frame_index = 0
        try:
            self.recorder.begin(Path(payload["video_path"]))
        except Exception as exc:
            self._write_status(error=str(exc))
            return
        self.recording = True
        self._write_status()

    def _end_segment(self) -> None:
        if self.recording:
            assert self.recorder is not None
            self.recorder.end()
            self.recording = False
        self._write_status()

    def offer_frame(
        self,
        image,
        *,
        device_timestamp_ms: Optional[float] = None,
        device_frame_number: Optional[int] = None,
        timestamp_domain: str = "host",
    ) -> None:
        if not self.recording:
            return
        assert self.recorder is not None
        now_mono_ns = time.monotonic_ns()
        now_unix_ns = time.time_ns()
        self.recorder.push(image)
        frame_index = self._segment_frame_index
        self._segment_frame_index += 1
        if self.meta_path is not None:
            _atomic_write_json(
                self.meta_path,
                {
                    "video_frame_index": frame_index,
                    "device_frame_number": (
                        frame_index if device_frame_number is None else device_frame_number
                    ),
                    "device_timestamp_ms": device_timestamp_ms,
                    "timestamp_domain": timestamp_domain,
                    "estimated_time_monotonic_ns": now_mono_ns,
                    "estimated_time_unix_ns": now_unix_ns,
                },
            )

    def shutdown(self) -> None:
        self._end_segment()


def _install_stop_handler():
    stop = {"requested": False}

    def request_stop(_signum: int, _frame: object) -> None:
        stop["requested"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_stop)
    return stop


def run_opencv(args, stop: dict) -> int:
    import cv2  # type: ignore

    backend = cv2.CAP_DSHOW if system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera_index, backend)
    if not cap.isOpened():
        print(
            f"CAMERA_ERROR failed to open camera index {args.camera_index}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    out_w, out_h = output_size(args.width, args.height, args.rotate)
    control = ControlChannel(
        args.control_path,
        args.status_path,
        args.meta_path,
        f"opencv:{args.camera_index}",
        out_w,
        out_h,
        args.fps,
    )

    try:
        print(
            f"CAMERA_READY opencv:{args.camera_index} rotate={args.rotate}",
            flush=True,
        )
        frame_index = 0
        last_write = 0.0
        while not stop["requested"]:
            control.poll()
            ok, image = cap.read()
            if not ok or image is None:
                time.sleep(0.01)
                continue
            image = orient_bgr(cv2, image, args.rotate)
            control.offer_frame(
                image, device_timestamp_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC))
            )
            now = time.monotonic()
            # ~10 Hz preview updates.
            if now - last_write >= 0.1:
                temporary = args.output.with_suffix(".tmp.png")
                if cv2.imwrite(str(temporary), image):
                    os.replace(temporary, args.output)
                    last_write = now
                    print(f"CAMERA_FRAME {frame_index}", flush=True)
            frame_index += 1
    except Exception as exc:
        print(f"CAMERA_ERROR {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        control.shutdown()
        try:
            cap.release()
        except Exception:
            pass
        if system() == "Windows":
            time.sleep(0.3)
    return 0


def run_realsense(args, stop: dict) -> int:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import pyrealsense2 as rs  # type: ignore

    pipeline = rs.pipeline()
    config = rs.config()
    if args.camera_serial != "auto":
        config.enable_device(args.camera_serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )

    try:
        profile = pipeline.start(config)
    except Exception as exc:
        print(f"CAMERA_ERROR failed to start RealSense: {exc}", file=sys.stderr, flush=True)
        return 1

    out_w, out_h = output_size(args.width, args.height, args.rotate)
    control = None
    try:
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        control = ControlChannel(
            args.control_path,
            args.status_path,
            args.meta_path,
            f"realsense:{serial}",
            out_w,
            out_h,
            args.fps,
        )
        print(f"CAMERA_READY realsense:{serial} rotate={args.rotate}", flush=True)
        frame_index = 0
        last_write = 0.0
        while not stop["requested"]:
            control.poll()
            frames = pipeline.wait_for_frames(3000)
            color = frames.get_color_frame()
            if not color:
                continue
            image = orient_bgr(cv2, np.asanyarray(color.get_data()), args.rotate)
            control.offer_frame(
                image,
                device_timestamp_ms=float(color.get_timestamp()),
                device_frame_number=int(color.get_frame_number()),
                timestamp_domain=str(color.get_frame_timestamp_domain()),
            )
            now = time.monotonic()
            # ~10 Hz preview updates.
            if now - last_write >= 0.1:
                temporary = args.output.with_suffix(".tmp.png")
                if cv2.imwrite(str(temporary), image):
                    os.replace(temporary, args.output)
                    last_write = now
                    print(f"CAMERA_FRAME {frame_index}", flush=True)
            frame_index += 1
    except Exception as exc:
        print(f"CAMERA_ERROR {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if control is not None:
            control.shutdown()
        try:
            pipeline.stop()
        except Exception:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--backend",
        choices=["opencv", "realsense"],
        default="opencv",
        help="RGB capture backend (default: opencv / Orbbec Dabai UVC)",
    )
    parser.add_argument("--camera-index", type=int, default=1, help="OpenCV camera index")
    parser.add_argument("--camera-serial", default="auto", help="RealSense serial if backend=realsense")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--rotate",
        type=int,
        choices=(0, 90, 180, 270),
        default=180,
        help="Clockwise rotation after capture (default 180 for inverted mount)",
    )
    parser.add_argument(
        "--control-path",
        type=Path,
        default=None,
        help="Optional JSON control file to start/stop recording segments without reopening the camera",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help="Optional JSON status file this process updates after every control transition",
    )
    parser.add_argument(
        "--meta-path",
        type=Path,
        default=None,
        help="Optional per-frame JSON metadata file, written while a segment is recording",
    )
    args = parser.parse_args()

    stop = _install_stop_handler()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "realsense":
        return run_realsense(args, stop)
    return run_opencv(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
