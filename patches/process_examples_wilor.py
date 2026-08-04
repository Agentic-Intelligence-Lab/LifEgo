#!/usr/bin/env python3
"""Run HumanEgo's standalone WiLoR hand visualization on example MP4 videos.

This script does not modify HumanEgo source files. It converts ordinary videos
into the minimal `mps_path/preprocess/all_data/<idx>/` layout expected by
`preprocess.WiLoRHands.run_wilor_hands`, then invokes that existing pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / "outputs" / ".hf_cache"))

LOCAL_DILL_PARENT = Path("/home/ymq/.cache/uv/archive-v0/91AFM0_OqHAx5ohh")
if LOCAL_DILL_PARENT.is_dir() and str(LOCAL_DILL_PARENT) not in sys.path:
    sys.path.insert(0, str(LOCAL_DILL_PARENT))

from preprocess import WiLoRHands as wilor_hands_module


DEFAULT_WILOR_CACHE = REPO_ROOT / ".cache" / "wilor_mini"


def resolve_wilor_pretrained_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value)

    return DEFAULT_WILOR_CACHE


def configure_wilor_pretrained_dir(pretrained_dir: Path) -> None:
    base_cls = wilor_hands_module.WiLorHandPose3dEstimationPipeline

    class CachedWiLorHandPose3dEstimationPipeline(base_cls):
        def __init__(self, **kwargs):
            kwargs.setdefault("wilor_pretrained_dir", str(pretrained_dir))
            super().__init__(**kwargs)

    wilor_hands_module.WiLorHandPose3dEstimationPipeline = CachedWiLorHandPose3dEstimationPipeline


def estimate_intrinsics(width: int, height: int, vfov_deg: float) -> list[list[float]]:
    fy = (height * 0.5) / math.tan(math.radians(vfov_deg) * 0.5)
    fx = fy
    return [
        [fx, 0.0, width * 0.5],
        [0.0, fy, height * 0.5],
        [0.0, 0.0, 1.0],
    ]


def prepare_video(video_path: Path, out_root: Path, vfov_deg: float) -> Path:
    session_dir = out_root / video_path.stem
    all_data_dir = session_dir / "preprocess" / "all_data"
    vis_dir = session_dir / "preprocess" / "vis"
    all_data_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    k = estimate_intrinsics(width, height, vfov_deg)
    c2w = np.eye(4).tolist()
    d = [0.0] * 8

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_dir = all_data_dir / f"{idx:05d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_dir / "rgb.png"), frame)

        ts = int(idx * 1_000_000_000 / fps)
        cam_json = {
            "idx": idx,
            "ts": ts,
            "fov": vfov_deg,
            "h": height,
            "w": width,
            "k": k,
            "d": d,
            "c2w": c2w,
            "c2d": c2w,
            "d2w": c2w,
            "rgb_path": os.path.join("preprocess", "all_data", f"{idx:05d}", "rgb.png"),
            "fps": fps,
        }
        with (frame_dir / "aria_cam_rgb.json").open("w", encoding="utf-8") as f:
            json.dump(cam_json, f, indent=2)
        idx += 1

    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")

    summary = {
        "total_frames": idx,
        "fps": fps,
        "first_ts": 0,
        "h": height,
        "w": width,
        "k": k,
        "d": d,
        "c2d": c2w,
        "source_video": str(video_path),
    }
    with (session_dir / "preprocess" / "aria_cam_rgb_config.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return session_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--videos",
        nargs="+",
        default=[
            "examples/ego_nero_easy.mp4",
            "examples/ego_nero_h.mp4",
            "examples/ego_nero_v.mp4",
        ],
    )
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--cfg", default="cfg/preprocess/base/AriaHands.yaml")
    parser.add_argument("--wilor-pretrained-dir", default=None)
    parser.add_argument("--vfov-deg", type=float, default=70.0)
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()

    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    pretrained_dir = resolve_wilor_pretrained_dir(args.wilor_pretrained_dir)
    configure_wilor_pretrained_dir(pretrained_dir)
    print(f"[process_examples_wilor] WiLoR pretrained dir: {pretrained_dir}")

    outputs: list[Path] = []
    for video in args.videos:
        video_path = Path(video)
        print(f"[process_examples_wilor] Preparing {video_path}")
        session_dir = prepare_video(video_path, out_root, args.vfov_deg)
        print(f"[process_examples_wilor] Running WiLoR on {session_dir}")
        wilor_hands_module.run_wilor_hands(
            mps_path=str(session_dir),
            cfg_path=args.cfg,
            export_video=True,
            export_gif=args.gif,
        )
        outputs.append(session_dir / "preprocess" / "vis" / "wilor_hands_vis.mp4")

    print("[process_examples_wilor] Outputs:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
