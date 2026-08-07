#!/usr/bin/env python3
"""Open MuJoCo passive viewer at ZERO joint pose with flange / TCP / tip markers.

Legend:
  green   = flange   (site:recorded_flange)
  magenta = TCP      (site:tcp, controller definition)
  orange  = tip      (site:gripper_tip)

Usage (from repo root):
  $PY patches/view_nero_zero_pose_frames.py
  $PY patches/view_nero_zero_pose_frames.py --scene outputs/mujoco_nero_scene/scene.xml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
mujoco = None

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
MARKER_FLANGE = "vis_flange_marker"
MARKER_TCP = "vis_tcp_marker"
MARKER_TIP = "vis_tip_marker"


def load_runtime(gl_backend: str) -> None:
    global mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    import mujoco as mujoco_module
    import mujoco.viewer  # noqa: F401

    mujoco = mujoco_module


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site: {name}")
    return int(sid)


def mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(
            f"Missing mocap body '{body_name}'. Use the updated scene.xml that includes "
            "vis_flange_marker / vis_tcp_marker / vis_tip_marker."
        )
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError(f"Body '{body_name}' is not mocap")
    return mid


def set_zero_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint: {name}")
        data.qpos[model.jnt_qposadr[jid]] = 0.0
        # clear ctrl if actuator exists
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if aid >= 0:
            data.ctrl[aid] = 0.0


def set_mocap(data: mujoco.MjData, mid: int, pos: np.ndarray, rot: np.ndarray) -> None:
    q = R.from_matrix(rot).as_quat()  # xyzw
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = [q[3], q[0], q[1], q[2]]


def apply_camera(viewer: mujoco.viewer.Handle) -> None:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # zero pose: flange near (0, -0.02, 0.72); look at wrist / tool
    cam.lookat[:] = [0.0, -0.05, 0.72]
    cam.distance = 1.1
    cam.azimuth = 125.0
    cam.elevation = -20.0
    viewer.cam.type = cam.type
    viewer.cam.lookat[:] = cam.lookat
    viewer.cam.distance = cam.distance
    viewer.cam.azimuth = cam.azimuth
    viewer.cam.elevation = cam.elevation


def park_extra_mocaps(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    # Hide HumanEgo marker far away so only the 3-color legend is visible.
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid >= 0 and int(model.body_mocapid[bid]) >= 0:
        data.mocap_pos[int(model.body_mocapid[bid])] = [10.0, 10.0, 10.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument(
        "--gl-backend",
        default="glfw",
        choices=["auto", "glfw", "egl", "osmesa"],
        help="Viewer needs a display backend; default glfw.",
    )
    args = parser.parse_args()
    load_runtime(args.gl_backend)

    scene_path = as_abs(args.scene)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)

    sid_fl = site_id(model, "recorded_flange")
    sid_tcp = site_id(model, "tcp")
    sid_tip = site_id(model, "gripper_tip")
    mid_fl = mocap_id(model, MARKER_FLANGE)
    mid_tcp = mocap_id(model, MARKER_TCP)
    mid_tip = mocap_id(model, MARKER_TIP)

    set_zero_qpos(model, data)
    mujoco.mj_forward(model, data)
    park_extra_mocaps(model, data)

    fl_p = data.site_xpos[sid_fl].copy()
    fl_R = data.site_xmat[sid_fl].reshape(3, 3).copy()
    tcp_p = data.site_xpos[sid_tcp].copy()
    tcp_R = data.site_xmat[sid_tcp].reshape(3, 3).copy()
    tip_p = data.site_xpos[sid_tip].copy()
    tip_R = data.site_xmat[sid_tip].reshape(3, 3).copy()

    set_mocap(data, mid_fl, fl_p, fl_R)
    set_mocap(data, mid_tcp, tcp_p, tcp_R)
    set_mocap(data, mid_tip, tip_p, tip_R)
    mujoco.mj_forward(model, data)

    off_tcp = fl_R.T @ (tcp_p - fl_p)
    off_tip = fl_R.T @ (tip_p - fl_p)

    print("--- ZERO pose frame check ---")
    print("Legend: GREEN=flange  MAGENTA=TCP  ORANGE=tip")
    print(f"flange world: {fl_p}")
    print(f"tcp    world: {tcp_p}")
    print(f"tip    world: {tip_p}")
    print(f"tcp offset in flange frame: {off_tcp}  (expect ~[0.13,0,0] tool-centric)")
    print(f"tip offset in flange frame: {off_tip}  (expect ~[+X, 0, -0.02])")
    print(f"|tcp-fl|={np.linalg.norm(tcp_p-fl_p)*1000:.1f}mm  "
          f"|tip-fl|={np.linalg.norm(tip_p-fl_p)*1000:.1f}mm  "
          f"|tip-tcp|={np.linalg.norm(tip_p-tcp_p)*1000:.1f}mm")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_camera(viewer)
        while viewer.is_running():
            with viewer.lock():
                # keep zero pose fixed
                set_zero_qpos(model, data)
                park_extra_mocaps(model, data)
                mujoco.mj_forward(model, data)
                set_mocap(data, mid_fl, data.site_xpos[sid_fl], data.site_xmat[sid_fl].reshape(3, 3))
                set_mocap(data, mid_tcp, data.site_xpos[sid_tcp], data.site_xmat[sid_tcp].reshape(3, 3))
                set_mocap(data, mid_tip, data.site_xpos[sid_tip], data.site_xmat[sid_tip].reshape(3, 3))
                mujoco.mj_forward(model, data)
            left = "ZERO pose\njoint1..7 = 0"
            right = (
                "GREEN flange\n"
                "MAGENTA TCP (link7 +0.13 X, toward tip)\n"
                "ORANGE tip (gripper_base +0.175 Z)\n"
                f"tcp in flange {off_tcp[0]:+.3f} {off_tcp[1]:+.3f} {off_tcp[2]:+.3f}\n"
                f"tip in flange {off_tip[0]:+.3f} {off_tip[1]:+.3f} {off_tip[2]:+.3f}"
            )
            viewer.set_texts((None, None, left, right))
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
