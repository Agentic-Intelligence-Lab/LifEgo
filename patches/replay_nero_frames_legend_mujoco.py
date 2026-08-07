#!/usr/bin/env python3
"""Replay real-bot joints and mark flange / controller TCP / gripper tip in distinct colors.

Legend (sphere origins):
  green  = flange     (site:recorded_flange / poses.flange_pose)
  magenta = TCP        (site:tcp / poses.tcp_pose, controller definition)
  orange = gripper tip (finger mid / site:gripper_tip)

RGB sticks on each marker are local +X/+Y/+Z.
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

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")

# Mocap legend markers injected in scene.xml
MARKER_FLANGE = "vis_flange_marker"
MARKER_TCP = "vis_tcp_marker"
MARKER_TIP = "vis_tip_marker"


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


def load_realbot_samples(path: Path) -> dict:
    times = []
    joints = []
    gripper_width = []
    flange = []
    tcp = []
    seqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        times.append(float(rec["elapsed_s"]))
        joints.append(rec["follower"]["position_rad"])
        width = rec.get("gripper_feedback", {}).get("value")
        if width is None:
            width = rec.get("gripper_ctrl", {}).get("value", 0.1)
        gripper_width.append(float(width))
        poses = rec["poses"]
        flange.append(poses["flange_pose"])
        tcp.append(poses["tcp_pose"])
        seqs.append(int(rec["seq"]))

    if not joints:
        raise RuntimeError(f"No sample records found in {path}")

    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "joint_qpos": np.asarray(joints, dtype=np.float64),
        "gripper_width_m": np.asarray(gripper_width, dtype=np.float64),
        "flange_pose": np.asarray(flange, dtype=np.float64),
        "tcp_pose": np.asarray(tcp, dtype=np.float64),
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
) -> None:
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


def mocap_id_of(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(
            f"Missing mocap body '{body_name}' in scene — rebuild scene or use updated scene.xml"
        )
    mid = int(model.body_mocapid[bid])
    if mid < 0:
        raise RuntimeError(f"Body '{body_name}' is not a mocap body")
    return mid


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site: {name}")
    return int(sid)


def set_mocap(data: mujoco.MjData, mid: int, pos: np.ndarray, rot_mat: np.ndarray) -> None:
    q_xyzw = R.from_matrix(rot_mat).as_quat()
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]


def pose6_to_Rt(pose6: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pose6[:3], dtype=np.float64)
    Rm = R.from_euler("xyz", pose6[3:6]).as_matrix()
    return p, Rm


def ang_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.linalg.norm(R.from_matrix(Ra.T @ Rb).as_rotvec()) * 180.0 / np.pi)


def fk_frames(model: mujoco.MjModel, data: mujoco.MjData, ids: dict) -> dict:
    fl_p = data.site_xpos[ids["flange"]].copy()
    fl_R = data.site_xmat[ids["flange"]].reshape(3, 3).copy()
    tcp_p = data.site_xpos[ids["tcp"]].copy()
    tcp_R = data.site_xmat[ids["tcp"]].reshape(3, 3).copy()

    # Tip: prefer site gripper_tip; fallback to mid of finger bodies.
    if ids.get("tip_site") is not None:
        tip_p = data.site_xpos[ids["tip_site"]].copy()
        tip_R = data.site_xmat[ids["tip_site"]].reshape(3, 3).copy()
    else:
        b1 = data.xpos[ids["finger1"]].copy()
        b2 = data.xpos[ids["finger2"]].copy()
        tip_p = 0.5 * (b1 + b2)
        if ids.get("jaw") is not None:
            tip_R = data.site_xmat[ids["jaw"]].reshape(3, 3).copy()
        else:
            tip_R = fl_R.copy()

    return {
        "flange": (fl_p, fl_R),
        "tcp": (tcp_p, tcp_R),
        "tip": (tip_p, tip_R),
    }


def resolve_ids(model: mujoco.MjModel) -> dict:
    ids = {
        "flange": site_id(model, "recorded_flange"),
        "tcp": site_id(model, "tcp"),
        "mocap_flange": mocap_id_of(model, MARKER_FLANGE),
        "mocap_tcp": mocap_id_of(model, MARKER_TCP),
        "mocap_tip": mocap_id_of(model, MARKER_TIP),
    }
    tip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tip")
    ids["tip_site"] = int(tip_sid) if tip_sid >= 0 else None
    jaw_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "jaw_parallel_flange")
    ids["jaw"] = int(jaw_sid) if jaw_sid >= 0 else None
    for name, key in (("gripper_link1", "finger1"), ("gripper_link2", "finger2")):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Missing body {name}")
        ids[key] = int(bid)
    return ids


def draw_hud(
    frame_rgb: np.ndarray,
    frame_idx: int,
    total: int,
    t: float,
    fl_err_mm: float,
    tcp_err_mm: float,
    fl_ang: float,
    tcp_ang: float,
    tip_p: np.ndarray,
    tcp_p: np.ndarray,
    fl_p: np.ndarray,
) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    d_tip_tcp = float(np.linalg.norm(tip_p - tcp_p) * 1000.0)
    d_tip_fl = float(np.linalg.norm(tip_p - fl_p) * 1000.0)
    d_tcp_fl = float(np.linalg.norm(tcp_p - fl_p) * 1000.0)
    lines = [
        f"Frame legend  {frame_idx + 1}/{total}  t={t:.2f}s",
        "GREEN sphere  = flange (recorded_flange / flange_pose)",
        "MAGENTA sphere = tool TCP (site:tcp, flange +0.13 X toward tip)",
        "ORANGE sphere = gripper tip (finger mid / gripper_tip)",
        "RGB sticks = local +X / +Y / +Z of each frame",
        f"FK vs log: flange {fl_err_mm:.1f}mm/{fl_ang:.2f}deg  TCP {tcp_err_mm:.1f}mm/{tcp_ang:.2f}deg",
        f"|tip-tcp|={d_tip_tcp:.0f}mm  |tip-flange|={d_tip_fl:.0f}mm  |tcp-flange|={d_tcp_fl:.0f}mm",
    ]
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


def sync_markers_from_fk(data: mujoco.MjData, ids: dict, frames: dict) -> None:
    set_mocap(data, ids["mocap_flange"], frames["flange"][0], frames["flange"][1])
    set_mocap(data, ids["mocap_tcp"], frames["tcp"][0], frames["tcp"][1])
    set_mocap(data, ids["mocap_tip"], frames["tip"][0], frames["tip"][1])


def step_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict,
    joint_qpos_addr: dict[str, int],
    actuator_id: dict[str, int],
    series: dict,
    i: int,
    use_logged_poses_for_markers: bool,
) -> tuple[dict, float, float, float, float]:
    data.time = float(series["time_s"][i])
    apply_joints(
        data,
        joint_qpos_addr,
        actuator_id,
        series["joint_qpos"][i],
        series["gripper_width_m"][i],
    )
    mujoco.mj_forward(model, data)
    frames = fk_frames(model, data, ids)

    fl_log_p, fl_log_R = pose6_to_Rt(series["flange_pose"][i])
    tcp_log_p, tcp_log_R = pose6_to_Rt(series["tcp_pose"][i])
    fl_err = float(np.linalg.norm(frames["flange"][0] - fl_log_p) * 1000.0)
    tcp_err = float(np.linalg.norm(frames["tcp"][0] - tcp_log_p) * 1000.0)
    fl_ang = ang_err_deg(frames["flange"][1], fl_log_R)
    tcp_ang = ang_err_deg(frames["tcp"][1], tcp_log_R)

    if use_logged_poses_for_markers:
        # Still put tip from FK; flange/TCP markers can show logged if preferred.
        set_mocap(data, ids["mocap_flange"], fl_log_p, fl_log_R)
        set_mocap(data, ids["mocap_tcp"], tcp_log_p, tcp_log_R)
        set_mocap(data, ids["mocap_tip"], frames["tip"][0], frames["tip"][1])
    else:
        sync_markers_from_fk(data, ids, frames)

    mujoco.mj_forward(model, data)
    return frames, fl_err, tcp_err, fl_ang, tcp_ang


def launch_viewer(args: argparse.Namespace) -> None:
    series = load_realbot_samples(as_abs(args.realbot))
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    ids = resolve_ids(model)
    joint_qpos_addr, actuator_id = joint_maps(model)
    period = 1.0 / max(args.fps, 1e-9)

    print("Legend: GREEN=flange  MAGENTA=TCP  ORANGE=tip")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_viewer_camera(viewer)
        i = 0
        while viewer.is_running():
            with viewer.lock():
                frames, fl_err, tcp_err, fl_ang, tcp_ang = step_frame(
                    model,
                    data,
                    ids,
                    joint_qpos_addr,
                    actuator_id,
                    series,
                    i,
                    args.markers_from_logged,
                )
            left = f"flange/TCP/tip\n{i + 1}/{len(series['time_s'])} t={series['time_s'][i]:.2f}s"
            right = (
                f"GREEN flange  MAGENTA tcp  ORANGE tip\n"
                f"FK vs log flange {fl_err:.1f}mm {fl_ang:.2f}d  tcp {tcp_err:.1f}mm {tcp_ang:.2f}d"
            )
            viewer.set_texts((None, None, left, right))
            viewer.sync()
            if i < len(series["time_s"]) - 1:
                i += 1
                time.sleep(period)
            else:
                time.sleep(0.05)


def render_replay(args: argparse.Namespace) -> None:
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = load_realbot_samples(as_abs(args.realbot))
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    ids = resolve_ids(model)
    joint_qpos_addr, actuator_id = joint_maps(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    fl_errs, tcp_errs = [], []
    try:
        for i in range(len(series["time_s"])):
            frames, fl_err, tcp_err, fl_ang, tcp_ang = step_frame(
                model,
                data,
                ids,
                joint_qpos_addr,
                actuator_id,
                series,
                i,
                args.markers_from_logged,
            )
            fl_errs.append(fl_err)
            tcp_errs.append(tcp_err)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            writer.write(
                draw_hud(
                    rgb,
                    i,
                    len(series["time_s"]),
                    series["time_s"][i],
                    fl_err,
                    tcp_err,
                    fl_ang,
                    tcp_ang,
                    frames["tip"][0],
                    frames["tcp"][0],
                    frames["flange"][0],
                )
            )
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(as_abs(args.scene)),
        "realbot": str(as_abs(args.realbot)),
        "video": str(out_path),
        "legend": {
            "green": "flange (recorded_flange / flange_pose)",
            "magenta": "controller TCP (site:tcp / tcp_pose)",
            "orange": "gripper tip (gripper_tip / finger mid)",
        },
        "markers_from_logged": bool(args.markers_from_logged),
        "frames": int(len(series["time_s"])),
        "flange_fk_vs_log_pos_mean_mm": float(np.mean(fl_errs)),
        "tcp_fk_vs_log_pos_mean_mm": float(np.mean(tcp_errs)),
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")
    print(f"FK vs log flange mean {np.mean(fl_errs):.1f} mm  TCP mean {np.mean(tcp_errs):.1f} mm")
    print("Legend: GREEN=flange  MAGENTA=TCP  ORANGE=tip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument(
        "--out",
        default="outputs/mujoco_nero_scene/replays/ego_nero_easy_frames_legend.mp4",
    )
    parser.add_argument(
        "--markers-from-logged",
        action="store_true",
        help="Place green/magenta markers from JSONL flange/tcp poses instead of FK sites "
        "(tip marker stays FK). Default uses FK for all three so they attach to the arm.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--gl-backend", default="auto", choices=["auto", "glfw", "egl", "osmesa"])
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_replay(args)


if __name__ == "__main__":
    main()
