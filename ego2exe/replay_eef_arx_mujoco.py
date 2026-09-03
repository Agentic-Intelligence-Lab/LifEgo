#!/usr/bin/env python3
"""Replay exported EEF targets as a MuJoCo mocap marker in the ARX scene."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from assets import MUJOCO_ARX_SCENE

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARX_SCENE = MUJOCO_ARX_SCENE
DEFAULT_EEF = REPO_ROOT / "outputs" / "new_pipeline" / "ego_hand1" / "robot_eef_scene_camera_axis_corrected" / "robot_eef_trajectory.json"
DEFAULT_OUT = REPO_ROOT / "outputs" / "new_pipeline" / "arx_replays" / "eef_marker.mp4"

cv2 = None
mujoco = None


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_runtime(viewer: bool, gl_backend: str) -> None:
    global cv2, mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    elif viewer:
        os.environ.setdefault("MUJOCO_GL", "glfw")

    import cv2 as cv2_module
    import mujoco as mujoco_module

    if not hasattr(mujoco_module, "Renderer"):
        from mujoco.rendering.classic.renderer import Renderer

        mujoco_module.Renderer = Renderer
    if viewer:
        import mujoco.viewer  # noqa: F401

    cv2 = cv2_module
    mujoco = mujoco_module


def require_runtime():
    if cv2 is None or mujoco is None:
        raise RuntimeError("call load_runtime() before using MuJoCo replay helpers")
    return cv2, mujoco


def quat_xyzw_to_wxyz(q) -> list[float]:
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


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
    _, mujoco_module = require_runtime()
    bid = mujoco_module.mj_name2id(model, mujoco_module.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid < 0:
        raise RuntimeError("ARX scene is missing mocap body: humanego_eef_marker")
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError("humanego_eef_marker exists but is not mocap")
    return mid


def make_camera():
    _, mujoco_module = require_runtime()
    cam = mujoco_module.MjvCamera()
    cam.type = mujoco_module.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.24, 0.0, 0.22]
    cam.distance = 1.15
    cam.azimuth = 145.0
    cam.elevation = -28.0
    return cam


def set_marker(data, mid: int, t: float, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    data.time = float(t)
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat_xyzw_to_wxyz(quat_xyzw)


def _add_capsule(scn, mujoco_module, p0, p1, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco_module.mjtCatBit.mjCAT_DECOR
    mujoco_module.mjv_connector(
        geom,
        mujoco_module.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
    )
    scn.ngeom += 1


def _add_sphere(scn, mujoco_module, pos, rgba, radius: float) -> None:
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco_module.mjv_initGeom(
        geom,
        mujoco_module.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.zeros(9, dtype=np.float64),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = mujoco_module.mjtCatBit.mjCAT_DECOR
    scn.ngeom += 1


def draw_marker_path(
    scn,
    points,
    *,
    rgba=(0.05, 0.75, 1.0, 0.72),
    radius: float = 0.004,
    stride: int = 1,
    start_rgba=(0.0, 1.0, 1.0, 1.0),
    end_rgba=(0.25, 0.1, 1.0, 1.0),
) -> None:
    """Draw a lightweight trajectory overlay in a MuJoCo user scene."""
    _, mujoco_module = require_runtime()
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        return
    step = max(int(stride), 1)
    sampled = pts[::step]
    if len(sampled) == 0 or not np.allclose(sampled[-1], pts[-1]):
        sampled = np.vstack([sampled, pts[-1]])
    for p0, p1 in zip(sampled[:-1], sampled[1:]):
        _add_capsule(scn, mujoco_module, p0, p1, rgba, radius)
    _add_sphere(scn, mujoco_module, pts[0], start_rgba, radius * 3.0)
    _add_sphere(scn, mujoco_module, pts[-1], end_rgba, radius * 3.0)


def draw_hud(frame_rgb: np.ndarray, i: int, n: int, t: float, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    cv2_module, _ = require_runtime()
    frame = cv2_module.cvtColor(frame_rgb, cv2_module.COLOR_RGB2BGR)
    rpy = R.from_quat(quat).as_euler("xyz")
    lines = [
        f"ARX EEF marker replay  frame {i + 1}/{n}  t={t:.2f}s",
        "cyan marker = exported EEF target; ARX robot is not driven",
        f"pos(m): {pos[0]:+.3f} {pos[1]:+.3f} {pos[2]:+.3f}",
        f"rpy xyz(rad): {rpy[0]:+.2f} {rpy[1]:+.2f} {rpy[2]:+.2f}",
    ]
    x, y = 18, 28
    for line in lines:
        cv2_module.putText(frame, line, (x, y), cv2_module.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2_module.LINE_AA)
        cv2_module.putText(frame, line, (x, y), cv2_module.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2_module.LINE_AA)
        y += 26
    return frame


def launch_viewer(args: argparse.Namespace) -> None:
    _, mujoco_module = require_runtime()
    times, pos, quat = load_eef(as_abs(args.eef))
    model = mujoco_module.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco_module.MjData(model)
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
                mujoco_module.mj_forward(model, data)
                viewer.user_scn.ngeom = 0
                if args.show_path:
                    draw_marker_path(viewer.user_scn, pos, stride=args.path_stride)
            viewer.set_texts((None, None, f"ARX EEF marker replay\n{i + 1}/{len(pos)}", f"pos {pos[i]}"))
            viewer.sync()
            if i < len(pos) - 1:
                i += 1
            elif not args.once:
                i = 0
            time.sleep(period)


def render_mp4(args: argparse.Namespace) -> None:
    cv2_module, mujoco_module = require_runtime()
    times, pos, quat = load_eef(as_abs(args.eef))
    model = mujoco_module.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco_module.MjData(model)
    mid = marker_mocap_id(model)
    renderer = mujoco_module.Renderer(model, height=args.height, width=args.width)
    camera = make_camera()
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2_module.VideoWriter(str(out), cv2_module.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out}")
    loops = 1 if args.once else max(int(args.loops), 1)
    try:
        for _ in range(loops):
            for i in range(len(pos)):
                set_marker(data, mid, times[i], pos[i], quat[i])
                mujoco_module.mj_forward(model, data)
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
    parser.add_argument("--scene", default=str(DEFAULT_ARX_SCENE))
    parser.add_argument("--eef", default=str(DEFAULT_EEF))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
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

    if not as_abs(args.scene).is_file():
        raise FileNotFoundError(f"scene file does not exist: {as_abs(args.scene)}")
    if not as_abs(args.eef).is_file():
        raise FileNotFoundError(f"EEF trajectory file does not exist: {as_abs(args.eef)}")

    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_mp4(args)


if __name__ == "__main__":
    main()
