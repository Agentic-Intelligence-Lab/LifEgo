#!/usr/bin/env python3
"""Replay exported EEF targets as a MuJoCo mocap marker."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import MUJOCO_NERO_SCENE
from replay_utils_mujoco import as_abs, draw_marker_path, load_runtime, quat_xyzw_to_wxyz, require_runtime


def load_eef(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    times, pos, quat = [], [], []
    for i, rec in enumerate(data.get("records", [])):
        if not rec.get("valid"):
            continue
        ee = rec["T_ee_in_base"]
        stamp = rec.get("ts")
        times.append(float(stamp) * 1e-9 if stamp is not None else i / 30.0)
        pos.append(ee["translation_m"])
        quat.append(ee["quat_xyzw"])
    if not pos:
        raise RuntimeError(f"No valid EEF records in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return t, np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)


def marker_mocap_id(model) -> int:
    _, mujoco = require_runtime()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid < 0:
        raise RuntimeError("fixed scene is missing mocap body: humanego_eef_marker")
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError("humanego_eef_marker exists but is not mocap")
    return mid


def make_camera():
    _, mujoco = require_runtime()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.30, -0.22, 0.24]
    cam.distance = 1.0
    cam.azimuth = 132.0
    cam.elevation = -28.0
    return cam


def set_marker(data, mid: int, t: float, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    data.time = float(t)
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat_xyzw_to_wxyz(quat_xyzw)


def draw_hud(frame_rgb: np.ndarray, i: int, n: int, t: float, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    cv2, _ = require_runtime()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    rpy = R.from_quat(quat).as_euler("xyz")
    lines = [
        f"EEF marker replay  frame {i + 1}/{n}  t={t:.2f}s",
        "cyan marker = exported EEF target; robot is not driven",
        f"pos(m): {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}",
        f"rpy xyz(rad): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def launch_viewer(args: argparse.Namespace) -> None:
    _, mujoco = require_runtime()
    times, pos, quat = load_eef(as_abs(args.eef))
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    mid = marker_mocap_id(model)
    period = 1.0 / max(args.fps, 1e-9)

    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = make_camera()
        viewer.cam.type = cam.type
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation
        i = 0
        while viewer.is_running():
            with viewer.lock():
                set_marker(data, mid, times[i], pos[i], quat[i])
                mujoco.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                if args.show_path:
                    draw_marker_path(viewer.user_scn, pos, stride=args.path_stride)
            viewer.set_texts((None, None, f"EEF marker replay\n{i + 1}/{len(pos)}", f"pos {pos[i]}"))
            viewer.sync()
            if i < len(pos) - 1:
                i += 1
            elif not args.once:
                i = 0
            time.sleep(period)


def render_mp4(args: argparse.Namespace) -> None:
    cv2, mujoco = require_runtime()
    times, pos, quat = load_eef(as_abs(args.eef))
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    mid = marker_mocap_id(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = make_camera()
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out}")
    loops = 1 if args.once else max(int(args.loops), 1)
    try:
        for _ in range(loops):
            for i in range(len(pos)):
                set_marker(data, mid, times[i], pos[i], quat[i])
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                if args.show_path and hasattr(renderer, "scene"):
                    draw_marker_path(renderer.scene, pos, stride=args.path_stride)
                writer.write(draw_hud(renderer.render(), i, len(pos), times[i], pos[i], quat[i]))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    parser.add_argument("--eef", default="outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json")
    parser.add_argument("--out", default="outputs/new_pipeline/replays/eef_marker.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--show-path", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--path-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gl-backend", choices=["auto", "glfw", "egl", "osmesa"], default="auto")
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_mp4(args)


if __name__ == "__main__":
    main()
