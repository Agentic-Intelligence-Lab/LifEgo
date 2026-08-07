#!/usr/bin/env python3
"""Replay real-bot TCP (or flange) poses on the arm driven by the same JSONL joints.

By default this *synchronizes* follower joints + recorded TCP so you can see whether
the logged frame sits on the controller TCP (site:tcp) of the moving arm.

Earlier pose-only mode left the arm at the model rest pose, so the cyan marker
looked detached / "folded" relative to the mesh even when tcp_pose was correct.
"""

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

POSE_KEYS = ("tcp_pose", "flange_pose", "fk_pose")
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


def load_realbot_samples(path: Path, pose_key: str) -> dict:
    times = []
    positions = []
    rotations = []
    joints = []
    gripper_width = []
    seqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        poses = rec.get("poses") or {}
        if pose_key not in poses:
            raise RuntimeError(f"Missing poses.{pose_key} in sample seq={rec.get('seq')}")
        x, y, z, roll, pitch, yaw = poses[pose_key]
        times.append(float(rec["elapsed_s"]))
        positions.append([float(x), float(y), float(z)])
        rotations.append(R.from_euler("xyz", [float(roll), float(pitch), float(yaw)]))
        joints.append(rec["follower"]["position_rad"])
        width = rec.get("gripper_feedback", {}).get("value")
        if width is None:
            width = rec.get("gripper_ctrl", {}).get("value", 0.1)
        gripper_width.append(float(width))
        seqs.append(int(rec["seq"]))

    if not positions:
        raise RuntimeError(f"No sample records with poses.{pose_key} found in {path}")

    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "pos": np.asarray(positions, dtype=np.float64),
        "rot": R.concatenate(rotations),
        "joint_qpos": np.asarray(joints, dtype=np.float64),
        "gripper_width_m": np.asarray(gripper_width, dtype=np.float64),
        "seqs": seqs,
    }


def make_camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.28, -0.13, 0.25]
    cam.distance = 1.15
    cam.azimuth = 132.0
    cam.elevation = -27.0
    return cam


def joint_maps(model: mujoco.MjModel) -> tuple[dict[str, int], dict[str, int]]:
    joint_qpos_addr = {}
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        joint_qpos_addr[name] = int(model.jnt_qposadr[jid])
    actuator_id = {}
    for name in [f"joint{i}_pos" for i in range(1, 8)] + ["gripper_joint1_pos", "gripper_joint2_pos"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            actuator_id[name] = aid
    return joint_qpos_addr, actuator_id


def apply_joints(
    data: mujoco.MjData,
    joint_qpos_addr: dict[str, int],
    actuator_id: dict[str, int],
    q: np.ndarray,
    width_m: float,
) -> float:
    for j, val in enumerate(q, start=1):
        data.qpos[joint_qpos_addr[f"joint{j}"]] = float(val)
        aid = actuator_id.get(f"joint{j}_pos")
        if aid is not None:
            data.ctrl[aid] = float(val)
    width_m = float(np.clip(width_m, 0.0, 0.1))
    finger = 0.5 * width_m
    data.qpos[joint_qpos_addr["gripper_joint1"]] = finger
    data.qpos[joint_qpos_addr["gripper_joint2"]] = -finger
    if "gripper_joint1_pos" in actuator_id:
        data.ctrl[actuator_id["gripper_joint1_pos"]] = finger
    if "gripper_joint2_pos" in actuator_id:
        data.ctrl[actuator_id["gripper_joint2_pos"]] = -finger
    return width_m


def marker_mocap_id(model: mujoco.MjModel) -> int:
    marker_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if marker_body < 0:
        raise RuntimeError("Missing humanego_eef_marker body in scene")
    mocap_id = int(model.body_mocapid[marker_body])
    if mocap_id < 0:
        raise RuntimeError("humanego_eef_marker exists but is not a mocap body")
    return mocap_id


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site in scene: {name}")
    return int(sid)


def apply_marker(data: mujoco.MjData, mocap_id: int, p: np.ndarray, q_xyzw: np.ndarray) -> None:
    data.mocap_pos[mocap_id] = np.asarray(p, dtype=np.float64)
    data.mocap_quat[mocap_id] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def ang_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.linalg.norm(R.from_matrix(np.asarray(Ra).T @ np.asarray(Rb)).as_rotvec()) * 180.0 / np.pi)


def draw_hud(
    frame_rgb: np.ndarray,
    frame_idx: int,
    total: int,
    t: float,
    pos: np.ndarray,
    rpy: np.ndarray,
    pose_key: str,
    drive_joints: bool,
    pos_err_mm: float | None,
    ang_err: float | None,
    axes_line: str,
) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"Nero real-bot {pose_key}  frame {frame_idx + 1}/{total}  t={t:.2f}s",
        f"Arm joints: {'ON (synced with sample)' if drive_joints else 'OFF (pose-only; looks detached)'}",
        "Magenta site:tcp = FK; cyan mocap = logged pose (should overlap when joints ON)",
        "Controller TCP: +X~left  +Y~down  +Z~rear (offset), NOT jaw tip direction",
        axes_line,
        f"pos(m): x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}  rpy={rpy[0]:+.2f}/{rpy[1]:+.2f}/{rpy[2]:+.2f}",
    ]
    if pos_err_mm is not None and ang_err is not None:
        lines.append(f"logged vs site:tcp  pos_err={pos_err_mm:.1f} mm  ang_err={ang_err:.2f} deg")
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        y += 24
    return frame


def apply_viewer_camera(viewer: mujoco.viewer.Handle) -> None:
    cam = make_camera()
    viewer.cam.type = cam.type
    viewer.cam.lookat[:] = cam.lookat
    viewer.cam.distance = cam.distance
    viewer.cam.azimuth = cam.azimuth
    viewer.cam.elevation = cam.elevation


def marker_pose_for_frame(
    args: argparse.Namespace,
    data: mujoco.MjData,
    sid_tcp: int,
    logged_p: np.ndarray,
    logged_q_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, str]:
    """Return (marker_pos, marker_quat_xyzw, pos_err_mm, ang_err_deg, axes_line)."""
    logged_R = R.from_quat(logged_q_xyzw).as_matrix()
    if args.marker_source == "fk":
        p = data.site_xpos[sid_tcp].copy()
        Rm = data.site_xmat[sid_tcp].reshape(3, 3).copy()
        q = R.from_matrix(Rm).as_quat()
        axes = f"FK site:tcp axes  x·left={np.dot(Rm[:,0],[0,-1,0]):+.2f} y·down={np.dot(Rm[:,1],[0,0,-1]):+.2f} z·rear={np.dot(Rm[:,2],[1,0,0]):+.2f}"
        return p, q, 0.0, 0.0, axes

    # logged pose as marker; errors vs FK site when joints driven
    Rm = data.site_xmat[sid_tcp].reshape(3, 3)
    rp = data.site_xpos[sid_tcp]
    pos_err = float(np.linalg.norm(rp - logged_p) * 1000.0) if args.drive_joints else None
    ang_err = ang_err_deg(Rm, logged_R) if args.drive_joints else None
    axes = (
        f"logged axes  x·left={np.dot(logged_R[:,0],[0,-1,0]):+.2f} "
        f"y·down={np.dot(logged_R[:,1],[0,0,-1]):+.2f} "
        f"z·+X={np.dot(logged_R[:,2],[1,0,0]):+.2f}"
    )
    return logged_p, logged_q_xyzw, pos_err, ang_err, axes


def launch_viewer(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    realbot_path = as_abs(args.realbot)
    series = load_realbot_samples(realbot_path, args.pose_key)

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mocap_id = marker_mocap_id(model)
    sid_tcp = site_id(model, "tcp")
    joint_qpos_addr, actuator_id = joint_maps(model)
    quats = series["rot"].as_quat()
    frame_period = 1.0 / max(args.fps, 1e-9)

    print("Launching MuJoCo viewer. Close the viewer window to stop.")
    print(f"drive_joints={args.drive_joints}  marker_source={args.marker_source}  pose_key={args.pose_key}")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_viewer_camera(viewer)
        frame_idx = 0
        while viewer.is_running():
            t = series["time_s"][frame_idx]
            p_log = series["pos"][frame_idx]
            q_xyzw = quats[frame_idx]
            with viewer.lock():
                data.time = float(t)
                if args.drive_joints:
                    apply_joints(
                        data,
                        joint_qpos_addr,
                        actuator_id,
                        series["joint_qpos"][frame_idx],
                        series["gripper_width_m"][frame_idx],
                    )
                mujoco.mj_forward(model, data)
                p_m, q_m, pos_err, ang_err, axes = marker_pose_for_frame(args, data, sid_tcp, p_log, q_xyzw)
                apply_marker(data, mocap_id, p_m, q_m)
                mujoco.mj_forward(model, data)
            err = ""
            if pos_err is not None and ang_err is not None:
                err = f"\nerr vs site:tcp {pos_err:.1f}mm {ang_err:.2f}deg"
            left = f"Nero {args.pose_key}\nframe {frame_idx + 1}/{len(series['pos'])}  t={t:.2f}s\njoints={'ON' if args.drive_joints else 'OFF'}"
            right = f"{axes}{err}"
            viewer.set_texts((None, None, left, right))
            viewer.sync()

            if frame_idx < len(series["pos"]) - 1:
                frame_idx += 1
                time.sleep(frame_period)
            else:
                time.sleep(0.05)


def render_replay(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    realbot_path = as_abs(args.realbot)
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    series = load_realbot_samples(realbot_path, args.pose_key)
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()
    mocap_id = marker_mocap_id(model)
    sid_tcp = site_id(model, "tcp")
    joint_qpos_addr, actuator_id = joint_maps(model)
    quats = series["rot"].as_quat()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    pos_errs = []
    ang_errs = []
    try:
        for i in range(len(series["pos"])):
            t = series["time_s"][i]
            p_log = series["pos"][i]
            q_xyzw = quats[i]
            data.time = float(t)
            if args.drive_joints:
                apply_joints(
                    data,
                    joint_qpos_addr,
                    actuator_id,
                    series["joint_qpos"][i],
                    series["gripper_width_m"][i],
                )
            mujoco.mj_forward(model, data)
            p_m, q_m, pos_err, ang_err, axes = marker_pose_for_frame(args, data, sid_tcp, p_log, q_xyzw)
            apply_marker(data, mocap_id, p_m, q_m)
            mujoco.mj_forward(model, data)
            if pos_err is not None:
                pos_errs.append(pos_err)
            if ang_err is not None:
                ang_errs.append(ang_err)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            writer.write(
                draw_hud(
                    rgb,
                    i,
                    len(series["pos"]),
                    t,
                    p_m,
                    R.from_quat(q_m).as_euler("xyz"),
                    args.pose_key,
                    args.drive_joints,
                    pos_err,
                    ang_err,
                    axes,
                )
            )
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(scene_path),
        "realbot": str(realbot_path),
        "pose_key": args.pose_key,
        "drive_joints": bool(args.drive_joints),
        "marker_source": args.marker_source,
        "video": str(out_path),
        "frames": int(len(series["pos"])),
        "fps": float(args.fps),
        "duration_s": float(series["time_s"][-1]),
        "pose_convention": "xyz_m + rpy_ZYX_rad (scipy euler 'xyz')",
        "logged_vs_site_tcp_pos_err_mean_mm": float(np.mean(pos_errs)) if pos_errs else None,
        "logged_vs_site_tcp_pos_err_max_mm": float(np.max(pos_errs)) if pos_errs else None,
        "logged_vs_site_tcp_ang_err_mean_deg": float(np.mean(ang_errs)) if ang_errs else None,
        "logged_vs_site_tcp_ang_err_max_deg": float(np.max(ang_errs)) if ang_errs else None,
        "note": (
            "With --drive-joints, cyan mocap (logged tcp) should overlap magenta site:tcp. "
            "Jaw mesh points along jaw+Z≈left; controller TCP +Z is rear offset."
        ),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")
    if pos_errs:
        print(
            f"logged vs site:tcp  pos mean/max {np.mean(pos_errs):.1f}/{np.max(pos_errs):.1f} mm  "
            f"ang mean/max {np.mean(ang_errs):.2f}/{np.max(ang_errs):.2f} deg"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument(
        "--pose-key",
        default="tcp_pose",
        choices=list(POSE_KEYS),
        help="Which poses.* field to show as the cyan mocap marker (default: tcp_pose).",
    )
    parser.add_argument(
        "--drive-joints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive follower joints with the same samples (default: on). Disable for old pose-only mode.",
    )
    parser.add_argument(
        "--marker-source",
        choices=["logged", "fk"],
        default="logged",
        help="logged=JSONL pose (check overlap with site:tcp); fk=put mocap on MuJoCo site:tcp.",
    )
    parser.add_argument("--out", default="outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_tcp.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--gl-backend",
        default="auto",
        choices=["auto", "glfw", "egl", "osmesa"],
    )
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_replay(args)


if __name__ == "__main__":
    main()
