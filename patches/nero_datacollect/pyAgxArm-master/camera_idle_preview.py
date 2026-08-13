#!/usr/bin/env python3
"""OpenCV RGB preview producer (Orbbec Dabai / UVC) used while the recorder is idle."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from platform import system


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

    try:
        print(
            f"CAMERA_READY opencv:{args.camera_index} rotate={args.rotate}",
            flush=True,
        )
        frame_index = 0
        last_write = 0.0
        while not stop["requested"]:
            ok, image = cap.read()
            if not ok or image is None:
                time.sleep(0.01)
                continue
            image = orient_bgr(cv2, image, args.rotate)
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

    try:
        serial = profile.get_device().get_info(rs.camera_info.serial_number)
        print(f"CAMERA_READY realsense:{serial} rotate={args.rotate}", flush=True)
        frame_index = 0
        last_write = 0.0
        while not stop["requested"]:
            frames = pipeline.wait_for_frames(3000)
            color = frames.get_color_frame()
            if not color:
                continue
            image = orient_bgr(cv2, np.asanyarray(color.get_data()), args.rotate)
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
    args = parser.parse_args()

    stop = _install_stop_handler()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "realsense":
        return run_realsense(args, stop)
    return run_opencv(args, stop)


if __name__ == "__main__":
    raise SystemExit(main())
