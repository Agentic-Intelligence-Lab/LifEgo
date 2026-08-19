#!/usr/bin/env python3
"""Overlay the real Nero arm's actual TCP pose (robot base frame) on its own video.

Companion to `visualize_hand2gripper.py`: that script projects the *desired*
gripper target (from WiLoR + hand2gripper) onto the ego video; this one
projects what the real robot *actually did* onto the teleop/execution video,
using the exact same camera calibration and drawing style so the two are
visually comparable.

Nero teleop capture (`nero_teleop_jsonl` schema) writes one MP4 (`camera_rgb`)
and one JSONL per run. Each sample carries `poses.flange_pose` (base frame,
`link7` origin) and `camera_rgb.video_frame_index`, the authoritative
sample<->video-frame alignment (per the JSONL metadata's
`camera_rgb.alignment_reference`) -- this script uses that instead of
assuming a fixed sample-rate/fps relationship, since sample_hz (typically 20)
and the video's capture fps (typically 30) don't line up 1:1.

UPDATED 2026-08-18: this script now computes the TCP point itself as
`p_flange + R_flange @ tcp_offset_m` (`--tcp-offset-m`, default 0.18 m along
flange +X, same rotation as flange), instead of reading the JSONL's own
`poses.tcp_pose`. The controller's `tcp_offset_flange_frame` baked into
`poses.tcp_pose` uses a different point definition than hand2gripper.py's
target (see assets.py `TcpOffset` notes) -- this way the drawn "real TCP"
point matches the same `site:tcp` / hand2gripper convention used everywhere
else in this repo (and by `replay_traj_compare_mujoco.py`'s real-robot
track), rather than whatever the controller itself considers its TCP.

For each video frame this draws:
  - The TCP origin + axes (X=red, Y=green, Z=blue), projected with the same
    camera intrinsics/extrinsics `visualize_hand2gripper.py` uses.
  - The flange origin (`poses.flange_pose`) as a yellow marker, plus a yellow
    line connecting flange -> TCP. That segment *is* the applied
    `--tcp-offset-m` translation, drawn to scale -- it should look parallel to
    the TCP's red (X) axis arrow (same rotation, `R_tcp == R_flange`, the two
    poses only differ by that one translation), and to the real
    flange-to-gripper-tip geometry visible in the frame.
  - A text block: elapsed time, TCP and flange pose in the base frame
    (xyz + rpy), their distance (will read exactly `--tcp-offset-m`'s norm,
    e.g. 180 mm, by construction -- this is now a readout of the applied
    offset rather than an independent cross-check against the controller),
    gripper width, and how far (in frames) the matched JSONL sample is from
    this exact video frame (0 = exact `video_frame_index` match).
  - The same fixed-corner robot-base-axes compass as `visualize_hand2gripper.py`.
"""

from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS
from visualize_hand2gripper import (
    AXIS_COLORS,
    CLOSED_COLOR,
    OPEN_COLOR,
    draw_base_axes_gizmo,
    draw_gripper_axes,
    draw_info_block,
    load_camera_to_base,
    project,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FLANGE_COLOR = (0, 220, 255)  # yellow, BGR


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def scaled_camera_intrinsics(width: int, height: int) -> np.ndarray:
    camera = DEFAULT_ASSETS.camera()
    if camera is None or not camera.intrinsics.is_filled():
        raise ValueError("DEFAULT_ASSETS.camera().intrinsics must be filled")
    intr = camera.intrinsics
    sx = width / intr.width if intr.width else 1.0
    sy = height / intr.height if intr.height else 1.0
    return np.array(
        [
            [intr.fx * sx, 0.0, intr.cx * sx],
            [0.0, intr.fy * sy, intr.cy * sy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_jsonl(path: Path) -> tuple[dict, list[dict]]:
    metadata: dict = {}
    samples: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("kind") == "metadata":
                metadata = rec
            elif rec.get("kind") == "sample":
                samples.append(rec)
    if not samples:
        raise RuntimeError(f"No 'sample' records found in {path}")
    return metadata, samples


def build_frame_index(samples: list[dict]) -> dict[int, dict]:
    """video_frame_index -> latest sample referencing that frame (zero-order hold)."""
    frame_map: dict[int, dict] = {}
    for sample in samples:
        cam = sample.get("camera_rgb")
        if cam is None or cam.get("video_frame_index") is None:
            continue
        frame_map[int(cam["video_frame_index"])] = sample
    if not frame_map:
        raise RuntimeError("No sample has a camera_rgb.video_frame_index; cannot align video<->jsonl")
    return frame_map


def nearest_sample(frame_map: dict[int, dict], sorted_idxs: list[int], frame_idx: int) -> tuple[dict, int]:
    exact = frame_map.get(frame_idx)
    if exact is not None:
        return exact, 0
    pos = bisect.bisect_left(sorted_idxs, frame_idx)
    candidates = [i for i in (pos - 1, pos) if 0 <= i < len(sorted_idxs)]
    best = min(candidates, key=lambda i: abs(sorted_idxs[i] - frame_idx))
    matched_idx = sorted_idxs[best]
    return frame_map[matched_idx], frame_idx - matched_idx


def _pose6_to_matrix(pose6: list[float]) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose6
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    T[:3, 3] = [x, y, z]
    return T


def tcp_pose_in_base(sample: dict, tcp_offset_flange: np.ndarray) -> np.ndarray | None:
    """p_flange + R_flange @ tcp_offset_flange; same rotation as flange.

    Deliberately does not read the JSONL's own `poses.tcp_pose` -- see the
    module docstring for why (different point definition than hand2gripper.py).
    """
    T_flange = flange_pose_in_base(sample)
    if T_flange is None:
        return None
    T_tcp = T_flange.copy()
    T_tcp[:3, 3] = T_flange[:3, 3] + T_flange[:3, :3] @ tcp_offset_flange
    return T_tcp


def flange_pose_in_base(sample: dict) -> np.ndarray | None:
    poses = sample.get("poses")
    if poses is None or poses.get("flange_pose") is None:
        return None
    return _pose6_to_matrix(poses["flange_pose"])


def draw_point_marker(frame: np.ndarray, K: np.ndarray, p_cam: np.ndarray, color: tuple[int, int, int], radius: int = 6) -> tuple[int, int] | None:
    p_2d = project(K, p_cam)
    if p_2d is None:
        return None
    cv2.circle(frame, p_2d, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, p_2d, radius, (0, 0, 0), 1, cv2.LINE_AA)
    return p_2d


def gripper_state(sample: dict) -> tuple[float | None, float | None]:
    width_m = None
    feedback = sample.get("gripper_feedback")
    if feedback is not None and feedback.get("value") is not None:
        width_m = float(feedback["value"])
    elif sample.get("gripper_ctrl", {}).get("value") is not None:
        width_m = float(sample["gripper_ctrl"]["value"])
    grasp_raw = sample.get("gripper", {}).get("state_grasp")
    return width_m, (float(grasp_raw) if grasp_raw is not None else None)


def render_frame(
    frame: np.ndarray,
    K: np.ndarray,
    T_cam_in_base: np.ndarray,
    T_base_in_cam: np.ndarray,
    sample: dict,
    frame_delta: int,
    axis_length_m: float,
    closed_width_m: float,
    tcp_offset_flange: np.ndarray,
) -> bool:
    T_tcp_in_base = tcp_pose_in_base(sample, tcp_offset_flange)
    if T_tcp_in_base is None:
        draw_info_block(frame, ["real TCP: no poses.flange_pose in this sample"], (12, 40), (0, 0, 220))
        return False

    T_tcp_in_cam = T_base_in_cam @ T_tcp_in_base
    tcp_2d = draw_gripper_axes(frame, K, T_tcp_in_cam, axis_length_m)

    T_flange_in_base = flange_pose_in_base(sample)
    offset_mm_txt = "n/a"
    if T_flange_in_base is not None:
        T_flange_in_cam = T_base_in_cam @ T_flange_in_base
        flange_2d = draw_point_marker(frame, K, T_flange_in_cam[:3, 3], FLANGE_COLOR)
        if flange_2d is not None and tcp_2d is not None:
            cv2.line(frame, flange_2d, tcp_2d, FLANGE_COLOR, 1, cv2.LINE_AA)
        offset_m = float(np.linalg.norm(T_tcp_in_base[:3, 3] - T_flange_in_base[:3, 3]))
        offset_mm_txt = f"{offset_m * 1000:.1f}mm"

    width_m, grasp_raw = gripper_state(sample)
    is_closed = width_m is not None and width_m < closed_width_m
    grasp_label = "CLOSED" if is_closed else ("OPEN" if width_m is not None else "n/a")
    width_txt = f"{width_m * 1000:.1f}mm" if width_m is not None else "n/a"
    grasp_raw_txt = f"{grasp_raw:.2f}" if grasp_raw is not None else "n/a"

    x, y, z = T_tcp_in_base[:3, 3]
    roll, pitch, yaw = R.from_matrix(T_tcp_in_base[:3, :3]).as_euler("xyz")
    lines = [
        f"real TCP(white)  grasp={grasp_label}  width={width_txt}  raw={grasp_raw_txt}",
        f"tcp_base(m)=({x:.3f},{y:.3f},{z:.3f})",
        f"rpy_base(rad)=({roll:.2f},{pitch:.2f},{yaw:.2f})",
    ]
    if T_flange_in_base is not None:
        fx, fy, fz = T_flange_in_base[:3, 3]
        lines.append(f"flange_base(yellow,m)=({fx:.3f},{fy:.3f},{fz:.3f})  |tcp-flange|={offset_mm_txt}")
    lines.append(f"t={sample.get('elapsed_s', float('nan')):.2f}s seq={sample.get('seq')} match_dframe={frame_delta:+d}")
    text_color = CLOSED_COLOR if is_closed else OPEN_COLOR
    draw_info_block(frame, lines, (12, 40), text_color)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Path to the *.rgb.mp4 teleop capture")
    parser.add_argument("--jsonl", required=True, help="Path to the matching *.jsonl teleop log")
    parser.add_argument("--out", default="", help="Output MP4 path. Default: outputs/realbot_vis/<video_stem>_tcp_vis.mp4")
    parser.add_argument("--axis-length-m", type=float, default=0.05, help="TCP axis arrow length in meters")
    parser.add_argument("--closed-width-m", type=float, default=0.02, help="gripper width below this counts as CLOSED for the OPEN/CLOSED label")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth video frame")
    parser.add_argument(
        "--tcp-offset-m",
        nargs=3,
        type=float,
        default=(0.18, 0.0, 0.0),
        help="Flange-frame offset applied to poses.flange_pose to get the real-robot TCP "
        "point (same convention as site:tcp / hand2gripper.py target; independent of the "
        "controller's own tcp_offset_flange_frame baked into poses.tcp_pose).",
    )
    args = parser.parse_args()
    tcp_offset_flange = np.asarray(args.tcp_offset_m, dtype=np.float64)

    video_path = as_abs(args.video)
    jsonl_path = as_abs(args.jsonl)
    _, samples = load_jsonl(jsonl_path)
    frame_map = build_frame_index(samples)
    sorted_idxs = sorted(frame_map.keys())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    K = scaled_camera_intrinsics(width, height)
    T_cam_in_base, T_base_in_cam, _ = load_camera_to_base(apply_axis_correction=False)

    out_path = as_abs(args.out) if args.out else REPO_ROOT / "outputs" / "realbot_vis" / f"{video_path.stem}_tcp_vis.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / max(args.stride, 1), (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer: {out_path}")

    frame_idx = 0
    n_written = 0
    n_exact = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % max(args.stride, 1) == 0:
                sample, delta = nearest_sample(frame_map, sorted_idxs, frame_idx)
                n_exact += int(delta == 0)
                render_frame(frame, K, T_cam_in_base, T_base_in_cam, sample, delta, args.axis_length_m, args.closed_width_m, tcp_offset_flange)
                draw_base_axes_gizmo(frame, T_base_in_cam[:3, :3], (70, height - 90))
                cv2.putText(frame, f"{frame_idx:05d}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(frame)
                n_written += 1
            frame_idx += 1
    finally:
        writer.release()
        cap.release()

    print(f"[visualize_realbot_tcp] frames written: {n_written} (exact video_frame_index match: {n_exact})")
    print(f"[visualize_realbot_tcp] output: {out_path}")


if __name__ == "__main__":
    main()
