#!/usr/bin/env python3
"""Replay a compact Nero EEF IK trajectory in MuJoCo."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
cv2 = None
mujoco = None

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")


def load_runtime(viewer: bool, gl_backend: str) -> None:
    global cv2, mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    elif viewer:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    import cv2 as cv2_module
    import mujoco as mujoco_module

    if viewer:
        import mujoco.viewer  # noqa: F401

    cv2 = cv2_module
    mujoco = mujoco_module


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_ik(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    required = ("joint_qpos", "target_pos_m", "target_quat_xyzw", "pos_err_m", "ang_err_deg")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"IK file {path} missing keys: {missing}")
    return out


def joint_qpos_addrs(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    names = tuple(f"joint{i}_pos" for i in range(1, 8)) + ("gripper_joint1_pos", "gripper_joint2_pos")
    out = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out[name] = int(aid)
    return out


def marker_mocap_id(model: mujoco.MjModel) -> int | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid < 0:
        return None
    mocap_id = int(model.body_mocapid[bid])
    return mocap_id if mocap_id >= 0 else None


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.28, -0.13, 0.25]
    cam.distance = 1.15
    cam.azimuth = 132.0
    cam.elevation = -27.0
    return cam


def apply_frame(
    data: mujoco.MjData,
    arm_addrs: np.ndarray,
    gripper_addrs: np.ndarray,
    act_ids: dict[str, int],
    q_arm: np.ndarray,
    gripper_width_m: float,
    t: float,
) -> float:
    data.time = float(t)
    data.qpos[arm_addrs] = np.asarray(q_arm, dtype=np.float64)
    for i, val in enumerate(q_arm, start=1):
        aid = act_ids.get(f"joint{i}_pos")
        if aid is not None:
            data.ctrl[aid] = float(val)
    gripper_width_m = float(np.clip(gripper_width_m, 0.0, 0.1))
    finger = 0.5 * gripper_width_m
    if len(gripper_addrs) == 2:
        data.qpos[gripper_addrs[0]] = finger
        data.qpos[gripper_addrs[1]] = -finger
    if "gripper_joint1_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint1_pos"]] = finger
    if "gripper_joint2_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint2_pos"]] = -finger
    return gripper_width_m


def apply_marker(data: mujoco.MjData, mocap_id: int | None, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    if mocap_id is None:
        return
    data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64)
    data.mocap_quat[mocap_id] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


def draw_hud(
    frame_rgb: np.ndarray,
    frame_idx: int,
    total: int,
    t: float,
    q: np.ndarray,
    pos_err_m: float,
    ang_err_deg: float,
) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"Nero IK replay  frame {frame_idx + 1}/{total}  t={t:.2f}s",
        "Robot follows IK joint_qpos; cyan marker shows target EEF",
        f"IK error: pos={pos_err_m * 1000.0:.1f} mm  ang={ang_err_deg:.2f} deg",
        "q(rad): " + " ".join(f"{v:+.2f}" for v in q),
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def viewer_text(frame_idx: int, total: int, t: float, pos_err_m: float, ang_err_deg: float) -> tuple[None, None, str, str]:
    left = f"Nero IK replay\nframe {frame_idx + 1}/{total}  t={t:.2f}s"
    right = f"pos err {pos_err_m * 1000.0:.1f} mm\nang err {ang_err_deg:.2f} deg"
    return (None, None, left, right)


def apply_viewer_camera(viewer: mujoco.viewer.Handle) -> None:
    cam = make_camera()
    viewer.cam.type = cam.type
    viewer.cam.lookat[:] = cam.lookat
    viewer.cam.distance = cam.distance
    viewer.cam.azimuth = cam.azimuth
    viewer.cam.elevation = cam.elevation


def render_replay(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    ik_path = as_abs(args.ik)
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ik = load_ik(ik_path)
    q = np.asarray(ik["joint_qpos"], dtype=np.float64)
    target_pos = np.asarray(ik["target_pos_m"], dtype=np.float64)
    target_quat = np.asarray(ik["target_quat_xyzw"], dtype=np.float64)
    pos_err = np.asarray(ik["pos_err_m"], dtype=np.float64)
    ang_err = np.asarray(ik["ang_err_deg"], dtype=np.float64)
    times = np.asarray(ik["time_s"], dtype=np.float64) if "time_s" in ik else np.arange(len(q), dtype=np.float64) / args.fps
    gripper_width = (
        np.asarray(ik["gripper_width_m"], dtype=np.float64)
        if "gripper_width_m" in ik
        else np.full(len(q), args.gripper_width_m, dtype=np.float64)
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    arm_addrs = joint_qpos_addrs(model, ARM_JOINTS)
    gripper_addrs = joint_qpos_addrs(model, GRIPPER_JOINTS)
    act_ids = actuator_ids(model)
    mocap_id = marker_mocap_id(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    try:
        for i in range(len(q)):
            apply_frame(data, arm_addrs, gripper_addrs, act_ids, q[i], gripper_width[i], times[i])
            apply_marker(data, mocap_id, target_pos[i], target_quat[i])
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            writer.write(draw_hud(rgb, i, len(q), times[i], q[i], pos_err[i], ang_err[i]))
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(scene_path),
        "ik": str(ik_path),
        "video": str(out_path),
        "frames": int(len(q)),
        "fps": float(args.fps),
        "pos_err_mean_mm": float(np.mean(pos_err) * 1000.0),
        "pos_err_max_mm": float(np.max(pos_err) * 1000.0),
        "ang_err_mean_deg": float(np.mean(ang_err)),
        "ang_err_max_deg": float(np.max(ang_err)),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")


def launch_viewer(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    ik_path = as_abs(args.ik)
    ik = load_ik(ik_path)
    q = np.asarray(ik["joint_qpos"], dtype=np.float64)
    target_pos = np.asarray(ik["target_pos_m"], dtype=np.float64)
    target_quat = np.asarray(ik["target_quat_xyzw"], dtype=np.float64)
    pos_err = np.asarray(ik["pos_err_m"], dtype=np.float64)
    ang_err = np.asarray(ik["ang_err_deg"], dtype=np.float64)
    times = np.asarray(ik["time_s"], dtype=np.float64) if "time_s" in ik else np.arange(len(q), dtype=np.float64) / args.fps
    gripper_width = (
        np.asarray(ik["gripper_width_m"], dtype=np.float64)
        if "gripper_width_m" in ik
        else np.full(len(q), args.gripper_width_m, dtype=np.float64)
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    arm_addrs = joint_qpos_addrs(model, ARM_JOINTS)
    gripper_addrs = joint_qpos_addrs(model, GRIPPER_JOINTS)
    act_ids = actuator_ids(model)
    mocap_id = marker_mocap_id(model)
    frame_period = 1.0 / max(args.fps, 1e-9)

    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_viewer_camera(viewer)
        frame_idx = 0
        while viewer.is_running():
            with viewer.lock():
                apply_frame(data, arm_addrs, gripper_addrs, act_ids, q[frame_idx], gripper_width[frame_idx], times[frame_idx])
                apply_marker(data, mocap_id, target_pos[frame_idx], target_quat[frame_idx])
                mujoco.mj_forward(model, data)
            viewer.set_texts(viewer_text(frame_idx, len(q), times[frame_idx], pos_err[frame_idx], ang_err[frame_idx]))
            viewer.sync()
            if frame_idx < len(q) - 1:
                frame_idx += 1
                time.sleep(frame_period)
            else:
                time.sleep(0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--ik", default="outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz")
    parser.add_argument("--out", default="outputs/mujoco_nero_scene/replays/ego_nero_easy_nero_eef_ik.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gripper-width-m", type=float, default=0.08)
    parser.add_argument("--viewer", action="store_true", help="Open an interactive MuJoCo viewer instead of writing mp4")
    parser.add_argument(
        "--gl-backend",
        choices=["auto", "glfw", "egl", "osmesa"],
        default="auto",
        help="MuJoCo GL backend. auto uses glfw for --viewer and leaves video rendering at MuJoCo default.",
    )
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_replay(args)


if __name__ == "__main__":
    main()
