#!/usr/bin/env python3
"""Temporary viewer/mp4 for comparing two EEF trajectory JSON files.

This script does not load the robot. It draws old/new EEF paths and advances two
current-frame markers so trajectory differences are easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass
class EefTrajectory:
    name: str
    idx: np.ndarray
    time_s: np.ndarray
    pos: np.ndarray
    quat_xyzw: np.ndarray


def load_eef(path: Path, name: str) -> EefTrajectory:
    data = json.loads(path.read_text(encoding="utf-8"))
    idx, times, pos, quat = [], [], [], []
    for i, rec in enumerate(data.get("records", [])):
        if not rec.get("valid"):
            continue
        ee = rec["T_ee_in_base"]
        idx.append(int(rec.get("idx", i)))
        stamp = rec.get("ts")
        times.append(float(stamp) * 1e-9 if stamp is not None else i / 30.0)
        pos.append(ee["translation_m"])
        quat.append(ee["quat_xyzw"])
    if not pos:
        raise RuntimeError(f"No valid EEF records in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return EefTrajectory(
        name=name,
        idx=np.asarray(idx, dtype=np.int32),
        time_s=t,
        pos=np.asarray(pos, dtype=np.float64),
        quat_xyzw=np.asarray(quat, dtype=np.float64),
    )


def quat_xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    q = R.from_matrix(rot).as_quat()
    return quat_xyzw_to_wxyz(q)


def make_marker_body(name: str, rgba: str) -> str:
    return f"""
    <body name="{name}" mocap="true" pos="0 0 0">
      <geom name="{name}_origin" type="sphere" size="0.018" rgba="{rgba}" contype="0" conaffinity="0"/>
      <geom name="{name}_x" type="capsule" fromto="0 0 0 0.080 0 0" size="0.005" rgba="1 0.05 0.05 1" contype="0" conaffinity="0"/>
      <geom name="{name}_y" type="capsule" fromto="0 0 0 0 0.080 0" size="0.005" rgba="0.05 0.8 0.1 1" contype="0" conaffinity="0"/>
      <geom name="{name}_z" type="capsule" fromto="0 0 0 0 0 0.080" size="0.005" rgba="0.1 0.3 1 1" contype="0" conaffinity="0"/>
    </body>
    """


def make_path_sites(name: str, traj: EefTrajectory, rgba: str, stride: int) -> str:
    lines = []
    for n, p in enumerate(traj.pos[::stride]):
        lines.append(
            f'<site name="{name}_path_{n:04d}" type="sphere" pos="{p[0]:.9g} {p[1]:.9g} {p[2]:.9g}" '
            f'size="0.008" rgba="{rgba}"/>'
        )
    start = traj.pos[0]
    end = traj.pos[-1]
    lines.append(
        f'<site name="{name}_start" type="sphere" pos="{start[0]:.9g} {start[1]:.9g} {start[2]:.9g}" '
        'size="0.024" rgba="0.2 1 0.2 1"/>'
    )
    lines.append(
        f'<site name="{name}_end" type="sphere" pos="{end[0]:.9g} {end[1]:.9g} {end[2]:.9g}" '
        'size="0.024" rgba="1 0.1 0.1 1"/>'
    )
    return "\n".join(lines)


def build_xml(old: EefTrajectory, new: EefTrajectory, stride: int) -> str:
    all_pos = np.vstack([old.pos, new.pos])
    center = all_pos.mean(axis=0)
    return f"""
<mujoco model="eef_compare">
  <option timestep="0.01"/>
  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <rgba haze="0.92 0.94 0.98 1"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="table" type="plane" pos="0 0 0" size="1.2 1.2 0.02" rgba="0.82 0.82 0.78 1"/>
    <site name="origin" type="sphere" pos="0 0 0" size="0.015" rgba="0 0 0 1"/>
    <body name="view_center" pos="{center[0]:.9g} {center[1]:.9g} {center[2]:.9g}">
      <camera name="track" mode="fixed" pos="0.55 -0.85 0.45" xyaxes="0.84 0.54 0 -0.23 0.36 0.90"/>
    </body>
    {make_path_sites("old", old, "1 0.82 0.05 0.85", stride)}
    {make_path_sites("new", new, "0.05 0.75 1 0.85", stride)}
    {make_marker_body("old_marker", "1 0.82 0.05 1")}
    {make_marker_body("new_marker", "0.05 0.75 1 1")}
  </worldbody>
</mujoco>
"""


def mocap_id(model, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"Missing mocap body: {name}")
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError(f"Body is not mocap: {name}")
    return mid


def set_marker(data, mid: int, traj: EefTrajectory, i: int) -> None:
    data.mocap_pos[mid] = traj.pos[i]
    data.mocap_quat[mid] = quat_xyzw_to_wxyz(traj.quat_xyzw[i])


def draw_hud(frame_rgb: np.ndarray, old: EefTrajectory, new: EefTrajectory, i: int, fps: float) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    diff = float(np.linalg.norm(new.pos[i] - old.pos[i]))
    lines = [
        f"EEF compare frame {i + 1}/{min(len(old.pos), len(new.pos))}  t={i / fps:.2f}s",
        "yellow=old outputs/ego_nero_easy   cyan=new outputs/new_pipeline/ego_nero_easy",
        f"position diff: {diff * 1000.0:.1f} mm",
        f"old p: {old.pos[i,0]:+.3f} {old.pos[i,1]:+.3f} {old.pos[i,2]:+.3f}",
        f"new p: {new.pos[i,0]:+.3f} {new.pos[i,1]:+.3f} {new.pos[i,2]:+.3f}",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def make_free_camera(old: EefTrajectory, new: EefTrajectory):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    all_pos = np.vstack([old.pos, new.pos])
    cam.lookat[:] = all_pos.mean(axis=0)
    cam.distance = 0.85
    cam.azimuth = 135.0
    cam.elevation = -28.0
    return cam


def render_mp4(args, model, data, old, new, old_mid, new_mid) -> None:
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = make_free_camera(old, new)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out}")
    n = min(len(old.pos), len(new.pos))
    try:
        for i in range(n):
            set_marker(data, old_mid, old, i)
            set_marker(data, new_mid, new, i)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.write(draw_hud(renderer.render(), old, new, i, args.fps))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def launch_viewer(args, model, data, old, new, old_mid, new_mid) -> None:
    import mujoco.viewer

    n = min(len(old.pos), len(new.pos))
    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = make_free_camera(old, new)
        viewer.cam.type = cam.type
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance = cam.distance
        viewer.cam.azimuth = cam.azimuth
        viewer.cam.elevation = cam.elevation
        t0 = time.time()
        while viewer.is_running():
            i = int((time.time() - t0) * args.fps) % n
            with viewer.lock():
                set_marker(data, old_mid, old, i)
                set_marker(data, new_mid, new, i)
                mujoco.mj_forward(model, data)
            diff = np.linalg.norm(new.pos[i] - old.pos[i]) * 1000.0
            viewer.set_texts(
                (
                    None,
                    None,
                    f"EEF compare\\nframe {i + 1}/{n}\\nyellow=old  cyan=new",
                    f"pos diff {diff:.1f} mm\\nold {old.pos[i]}\\nnew {new.pos[i]}",
                )
            )
            viewer.sync()
            time.sleep(1.0 / max(args.fps, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old",
        default="outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json",
    )
    parser.add_argument(
        "--new",
        default="outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json",
    )
    parser.add_argument("--out", default="outputs/new_pipeline/eef_compare_old_vs_new.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--path-stride", type=int, default=3)
    parser.add_argument(
        "--gl-backend",
        default="auto",
        choices=["auto", "glfw", "egl", "osmesa"],
        help="MuJoCo GL backend. Use glfw for desktop viewer.",
    )
    args = parser.parse_args()

    load_runtime(args.viewer, args.gl_backend)
    old = load_eef(as_abs(args.old), "old")
    new = load_eef(as_abs(args.new), "new")
    xml = build_xml(old, new, max(1, args.path_stride))
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml)
        xml_path = f.name
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    finally:
        Path(xml_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)
    old_mid = mocap_id(model, "old_marker")
    new_mid = mocap_id(model, "new_marker")
    if args.viewer:
        launch_viewer(args, model, data, old, new, old_mid, new_mid)
    else:
        render_mp4(args, model, data, old, new, old_mid, new_mid)


if __name__ == "__main__":
    main()
