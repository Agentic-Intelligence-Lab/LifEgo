#!/usr/bin/env python3
"""Replay Nero real-bot follower joint angles in the generated MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_follower_joints(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    times = []
    qpos = []
    gripper_width = []
    seqs = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        times.append(float(rec["elapsed_s"]))
        qpos.append(rec["follower"]["position_rad"])
        width = rec.get("gripper_feedback", {}).get("value")
        if width is None:
            width = rec.get("gripper_ctrl", {}).get("value", 0.1)
        gripper_width.append(float(width))
        seqs.append(int(rec["seq"]))

    if not qpos:
        raise RuntimeError(f"No sample records found in {jsonl_path}")

    t = np.array(times, dtype=np.float64)
    t = t - t[0]
    return t, np.array(qpos, dtype=np.float64), np.array(gripper_width, dtype=np.float64), seqs


def resample_series(times: np.ndarray, values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1])
    n = int(round(duration * fps)) + 1
    out_t = np.arange(n, dtype=np.float64) / fps
    out_t[-1] = min(out_t[-1], duration)
    if values.ndim == 1:
        return out_t, np.interp(out_t, times, values)
    out = np.zeros((len(out_t), values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(out_t, times, values[:, j])
    return out_t, out


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.28, -0.13, 0.25]
    cam.distance = 1.15
    cam.azimuth = 132.0
    cam.elevation = -27.0
    return cam


def draw_hud(frame_rgb: np.ndarray, frame_idx: int, total: int, t: float, q: np.ndarray, width_m: float) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"Nero real-bot follower joint replay  frame {frame_idx + 1}/{total}  t={t:.2f}s",
        "Base: -x front, -y left; RGB triads show sampled flange/EEF orientations",
        f"jaw-parallel gripper width={width_m * 1000.0:.1f} mm",
        "q(rad): " + " ".join(f"{v:+.2f}" for v in q),
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def render_replay(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    jsonl_path = as_abs(args.realbot)
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    times, qpos, gripper_width, _seqs = load_follower_joints(jsonl_path)
    render_t, render_q = resample_series(times, qpos, args.fps)
    _, render_gripper_width = resample_series(times, gripper_width, args.fps)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()
    joint_qpos_addr = {
        name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in [f"joint{i}" for i in range(1, 8)] + ["gripper_joint1", "gripper_joint2"]
    }
    actuator_id = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in [f"joint{i}_pos" for i in range(1, 8)] + ["gripper_joint1_pos", "gripper_joint2_pos"]
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    try:
        for i, (t, q, width_m) in enumerate(zip(render_t, render_q, render_gripper_width)):
            for j, val in enumerate(q, start=1):
                data.qpos[joint_qpos_addr[f"joint{j}"]] = val
                data.ctrl[actuator_id[f"joint{j}_pos"]] = val
            width_m = float(np.clip(width_m, 0.0, 0.1))
            finger = 0.5 * width_m
            data.qpos[joint_qpos_addr["gripper_joint1"]] = finger
            data.qpos[joint_qpos_addr["gripper_joint2"]] = -finger
            data.ctrl[actuator_id["gripper_joint1_pos"]] = finger
            data.ctrl[actuator_id["gripper_joint2_pos"]] = -finger
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            frame = draw_hud(rgb, i, len(render_q), t, q, width_m)
            writer.write(frame)
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(scene_path),
        "realbot": str(jsonl_path),
        "video": str(out_path),
        "source_samples": int(len(qpos)),
        "render_frames": int(len(render_q)),
        "fps": args.fps,
        "duration_s": float(render_t[-1]),
        "joint_order": [f"joint{i}" for i in range(1, 8)],
        "gripper_source": "gripper_feedback.value meters, split half to gripper_joint1 and gripper_joint2",
        "gripper_width_start_m": float(gripper_width[0]),
        "gripper_width_end_m": float(gripper_width[-1]),
        "qpos_start": qpos[0].tolist(),
        "qpos_end": qpos[-1].tolist(),
    }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument("--out", default="outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_follower_joints.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    render_replay(args)


if __name__ == "__main__":
    main()
