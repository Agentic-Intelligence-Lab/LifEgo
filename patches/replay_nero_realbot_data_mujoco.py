#!/usr/bin/env python3
"""Replay Nero real-bot follower joints with current tool TCP + tip frames.

- Robot motion: follower joint angles (+ gripper width) from JSONL.
- Each frame marks current poses only (no tip/TCP path trails):
    MAGENTA = site:tcp  (tool-centric FK after joints)
    ORANGE  = site:gripper_tip
  RGB sticks = local +X/+Y/+Z.

Replaces the previous three scripts:
  replay_nero_realbot_joints_mujoco.py
  replay_nero_realbot_tcp_mujoco.py
  replay_nero_frames_legend_mujoco.py
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
MARKER_TCP = "vis_tcp_marker"
MARKER_TIP = "vis_tip_marker"
MARKER_FLANGE = "vis_flange_marker"
MARKER_HUMANEGO = "humanego_eef_marker"


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


def load_realbot(path: Path) -> dict:
    times, qpos, gripper_width, seqs = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
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
        raise RuntimeError(f"No sample records in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "joint_qpos": np.asarray(qpos, dtype=np.float64),
        "gripper_width_m": np.asarray(gripper_width, dtype=np.float64),
        "seqs": seqs,
    }


def resample_series(times: np.ndarray, values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1])
    n = max(int(round(duration * fps)) + 1, 1)
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


def joint_maps(model: mujoco.MjModel) -> tuple[dict[str, int], dict[str, int]]:
    joint_qpos_addr = {}
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint: {name}")
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
    t: float,
    q: np.ndarray,
    width_m: float,
) -> float:
    data.time = float(t)
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


def site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Missing site: {name}")
    return int(sid)


def mocap_id(model: mujoco.MjModel, body_name: str) -> int | None:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])
    return mid if mid >= 0 else None


def set_mocap(data: mujoco.MjData, mid: int, pos: np.ndarray, rot: np.ndarray) -> None:
    q = R.from_matrix(rot).as_quat()
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = [q[3], q[0], q[1], q[2]]


def park_mocap(data: mujoco.MjData, mid: int | None) -> None:
    if mid is None:
        return
    data.mocap_pos[mid] = [10.0, 10.0, 10.0]


def resolve_ids(model: mujoco.MjModel) -> dict:
    ids = {
        "tcp_site": site_id(model, "tcp"),
        "tip_site": site_id(model, "gripper_tip"),
        "mocap_tcp": mocap_id(model, MARKER_TCP),
        "mocap_tip": mocap_id(model, MARKER_TIP),
        "mocap_flange": mocap_id(model, MARKER_FLANGE),
        "mocap_humanego": mocap_id(model, MARKER_HUMANEGO),
    }
    if ids["mocap_tcp"] is None or ids["mocap_tip"] is None:
        raise RuntimeError(
            "Scene missing vis_tcp_marker / vis_tip_marker mocap bodies. "
            "Rebuild with patches/build_nero_mujoco_scene.py or use updated scene.xml."
        )
    return ids


def update_tool_markers(data: mujoco.MjData, ids: dict) -> tuple[np.ndarray, np.ndarray]:
    """Place TCP/tip markers at current FK only (no trajectory history)."""
    park_mocap(data, ids["mocap_flange"])
    park_mocap(data, ids["mocap_humanego"])

    tcp_p = data.site_xpos[ids["tcp_site"]].copy()
    tcp_R = data.site_xmat[ids["tcp_site"]].reshape(3, 3).copy()
    tip_p = data.site_xpos[ids["tip_site"]].copy()
    tip_R = data.site_xmat[ids["tip_site"]].reshape(3, 3).copy()
    set_mocap(data, ids["mocap_tcp"], tcp_p, tcp_R)
    set_mocap(data, ids["mocap_tip"], tip_p, tip_R)
    return tcp_p, tip_p


def draw_hud(
    frame_rgb: np.ndarray,
    frame_idx: int,
    total: int,
    t: float,
    q: np.ndarray,
    width_m: float,
    tcp_p: np.ndarray,
    tip_p: np.ndarray,
) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    dist_mm = float(np.linalg.norm(tip_p - tcp_p) * 1000.0)
    lines = [
        f"Nero real-bot joints + tool frames  {frame_idx + 1}/{total}  t={t:.2f}s",
        "MAGENTA=tcp (FK)  ORANGE=tip (FK)  current pose only",
        f"gripper width={width_m * 1000.0:.1f} mm  |tip-tcp|={dist_mm:.0f} mm",
        "q(rad): " + " ".join(f"{v:+.2f}" for v in q),
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def apply_viewer_camera(viewer: mujoco.viewer.Handle) -> None:
    cam = make_camera()
    viewer.cam.type = cam.type
    viewer.cam.lookat[:] = cam.lookat
    viewer.cam.distance = cam.distance
    viewer.cam.azimuth = cam.azimuth
    viewer.cam.elevation = cam.elevation


def prepare_series(realbot: Path, fps: float) -> dict:
    raw = load_realbot(realbot)
    t, q = resample_series(raw["time_s"], raw["joint_qpos"], fps)
    _, w = resample_series(raw["time_s"], raw["gripper_width_m"], fps)
    return {
        "time_s": t,
        "joint_qpos": q,
        "gripper_width_m": w,
        "source_samples": len(raw["joint_qpos"]),
        "seqs": raw["seqs"],
    }


def launch_viewer(args: argparse.Namespace) -> None:
    series = prepare_series(as_abs(args.realbot), args.fps)
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    joint_qpos_addr, actuator_id = joint_maps(model)
    ids = resolve_ids(model)
    period = 1.0 / max(args.fps, 1e-9)

    print("MAGENTA=tcp  ORANGE=tip  (current FK only). Close window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_viewer_camera(viewer)
        i = 0
        n = len(series["joint_qpos"])
        while viewer.is_running():
            with viewer.lock():
                width_m = apply_joints(
                    data,
                    joint_qpos_addr,
                    actuator_id,
                    series["time_s"][i],
                    series["joint_qpos"][i],
                    series["gripper_width_m"][i],
                )
                mujoco.mj_forward(model, data)
                tcp_p, tip_p = update_tool_markers(data, ids)
                mujoco.mj_forward(model, data)
            left = f"real-bot joints\n{i + 1}/{n} t={series['time_s'][i]:.2f}s"
            right = (
                f"MAGENTA tcp  ORANGE tip\n"
                f"gripper {width_m * 1000:.1f} mm\n"
                f"|tip-tcp| {np.linalg.norm(tip_p - tcp_p) * 1000:.0f} mm"
            )
            viewer.set_texts((None, None, left, right))
            viewer.sync()
            if i < n - 1:
                i += 1
                time.sleep(period)
            else:
                time.sleep(0.05)


def render_replay(args: argparse.Namespace) -> None:
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = prepare_series(as_abs(args.realbot), args.fps)
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    joint_qpos_addr, actuator_id = joint_maps(model)
    ids = resolve_ids(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    n = len(series["joint_qpos"])
    try:
        for i in range(n):
            width_m = apply_joints(
                data,
                joint_qpos_addr,
                actuator_id,
                series["time_s"][i],
                series["joint_qpos"][i],
                series["gripper_width_m"][i],
            )
            mujoco.mj_forward(model, data)
            tcp_p, tip_p = update_tool_markers(data, ids)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            rgb = renderer.render()
            writer.write(
                draw_hud(
                    rgb,
                    i,
                    n,
                    series["time_s"][i],
                    series["joint_qpos"][i],
                    width_m,
                    tcp_p,
                    tip_p,
                )
            )
    finally:
        writer.release()
        renderer.close()

    summary = {
        "scene": str(as_abs(args.scene)),
        "realbot": str(as_abs(args.realbot)),
        "video": str(out_path),
        "source_samples": int(series["source_samples"]),
        "render_frames": int(n),
        "fps": float(args.fps),
        "duration_s": float(series["time_s"][-1]),
        "markers": {
            "magenta": "site:tcp (tool-centric FK, current only)",
            "orange": "site:gripper_tip (current only)",
        },
        "joint_order": [f"joint{i}" for i in range(1, 8)],
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument(
        "--out",
        default="outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_data.mp4",
    )
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
