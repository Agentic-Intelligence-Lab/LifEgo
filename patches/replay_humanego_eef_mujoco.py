#!/usr/bin/env python3
"""Replay exported HumanEgo/WiLoR EEF poses as a MuJoCo mocap marker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_eef(path: Path) -> tuple[np.ndarray, R, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    ts = []
    pos = []
    rot = []
    for i, rec in enumerate(data.get("records", [])):
        if not rec.get("valid"):
            continue
        T = np.array(rec["T_ee_in_base"]["T"], dtype=np.float64)
        stamp = rec.get("ts")
        ts.append(float(stamp) * 1e-9 if stamp is not None else i / 30.0)
        pos.append(T[:3, 3])
        rot.append(R.from_matrix(T[:3, :3]))
    if not pos:
        raise RuntimeError(f"No valid EEF records found in {path}")
    t = np.array(ts, dtype=np.float64)
    t -= t[0]
    return t, np.array(pos, dtype=np.float64), R.concatenate(rot)


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.32, -0.28, 0.28]
    cam.distance = 0.95
    cam.azimuth = 132.0
    cam.elevation = -30.0
    return cam


def draw_hud(frame_rgb: np.ndarray, frame_idx: int, total: int, t: float, pos: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"HumanEgo/WiLoR EEF replay  frame {frame_idx + 1}/{total}  t={t:.2f}s",
        "Base: -x front, -y left; RGB triads show sampled flange/EEF orientations",
        f"pos(m): x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}",
        f"rpy xyz(rad): r={rpy[0]:+.2f} p={rpy[1]:+.2f} y={rpy[2]:+.2f}",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def render_replay(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    eef_path = as_abs(args.eef)
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times, pos, rot = load_eef(eef_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    marker_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    mocap_id = int(model.body_mocapid[marker_body])
    if mocap_id < 0:
        raise RuntimeError("humanego_eef_marker exists but is not a mocap body")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    try:
        for i, (t, p, q_xyzw) in enumerate(zip(times, pos, rot.as_quat())):
            data.mocap_pos[mocap_id] = p
            data.mocap_quat[mocap_id] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            frame = draw_hud(rgb, i, len(pos), t, p, R.from_quat(q_xyzw).as_euler("xyz"))
            writer.write(frame)
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(scene_path),
        "eef": str(eef_path),
        "video": str(out_path),
        "frames": int(len(pos)),
        "fps": args.fps,
        "duration_s": float(times[-1]),
        "position_start_m": pos[0].tolist(),
        "position_end_m": pos[-1].tolist(),
        "position_min_m": np.min(pos, axis=0).tolist(),
        "position_max_m": np.max(pos, axis=0).tolist(),
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--eef", default="outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json")
    parser.add_argument("--out", default="outputs/mujoco_nero_scene/replays/ego_nero_easy_humanego_eef_scene_camera.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render_replay(args)


if __name__ == "__main__":
    main()
