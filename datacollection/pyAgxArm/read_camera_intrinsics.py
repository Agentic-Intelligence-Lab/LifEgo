#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read RealSense D435i factory color-camera intrinsics and save them to JSON.

Run this on a machine with the official ``pyrealsense2`` wheel (Windows or
Linux x86_64 -- macOS has no official wheel, see docs/HANDOFF.md). The output
JSON is a static config other machines (e.g. the macOS dev box, which reads
the D435i as a plain UVC camera and cannot query intrinsics live) can load
without ever needing pyrealsense2 installed.

Usage:
  pip install pyrealsense2
  python read_camera_intrinsics.py
  python read_camera_intrinsics.py --serial 123622270356 --output configs/d435i_intrinsics.json
  python read_camera_intrinsics.py --resolutions 640x480 1280x720 1920x1080
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read RealSense D435i color intrinsics")
    p.add_argument(
        "--serial",
        default="auto",
        help="RealSense serial number, or 'auto' for the first device found",
    )
    p.add_argument(
        "--resolutions",
        nargs="+",
        default=["640x480", "1280x720", "1920x1080"],
        help="Color stream resolutions to query, e.g. 640x480 1280x720",
    )
    p.add_argument("--fps", type=int, default=30)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("configs/d435i_intrinsics.json"),
        help="Output JSON path (default: configs/d435i_intrinsics.json)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import pyrealsense2 as rs  # type: ignore

    result: dict = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "note": (
            "Factory color-sensor intrinsics read via official pyrealsense2. "
            "fx/fy/ppx/ppy scale with resolution; distortion coefficients are "
            "sensor-level and stay constant across resolutions for D400 series."
        ),
        "by_resolution": {},
    }

    for res in args.resolutions:
        width_str, height_str = res.lower().split("x")
        width, height = int(width_str), int(height_str)

        pipeline = rs.pipeline()
        config = rs.config()
        if args.serial != "auto":
            config.enable_device(args.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, args.fps)

        print(f"Starting pipeline at {width}x{height}@{args.fps} ...")
        profile = pipeline.start(config)
        try:
            device = profile.get_device()
            if "serial" not in result:
                result["serial"] = device.get_info(rs.camera_info.serial_number)
                result["name"] = device.get_info(rs.camera_info.name)
                result["firmware_version"] = device.get_info(rs.camera_info.firmware_version)

            color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_profile.get_intrinsics()
            result["by_resolution"][res] = {
                "width": intr.width,
                "height": intr.height,
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,
                "ppy": intr.ppy,
                "distortion_model": str(intr.model),
                "coeffs": list(intr.coeffs),
            }
            print(f"  fx={intr.fx:.3f} fy={intr.fy:.3f} ppx={intr.ppx:.3f} ppy={intr.ppy:.3f}")
        finally:
            pipeline.stop()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved intrinsics for {len(result['by_resolution'])} resolution(s) to {args.output}")
    print(f"Device: {result.get('name')} serial={result.get('serial')} fw={result.get('firmware_version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
