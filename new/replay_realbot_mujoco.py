#!/usr/bin/env python3
"""Replay real-bot JSONL on the left and optional IK npz on the right."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from assets import MUJOCO_NERO_DUAL_SCENE
from replay_ik_mujoco import load_ik
from replay_utils_mujoco import as_abs, load_runtime, quat_xyzw_to_wxyz, require_runtime

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
RIGHT_PREFIX = "r_"


def load_realbot(path: Path) -> dict:
    times, qpos, width = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        times.append(float(rec["elapsed_s"]))
        qpos.append(rec["follower"]["position_rad"])
        w = rec.get("gripper_feedback", {}).get("value")
        if w is None:
            w = rec.get("gripper_ctrl", {}).get("value", 0.1)
        width.append(float(w))
    if not qpos:
        raise RuntimeError(f"No sample records in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "joint_qpos": np.asarray(qpos, dtype=np.float64),
        "gripper_width_m": np.asarray(width, dtype=np.float64),
    }


def resample_series(src_t: np.ndarray, values: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if len(src_t) == 1:
        if values.ndim == 1:
            return np.full(len(dst_t), float(values[0]), dtype=np.float64)
        out = np.zeros((len(dst_t), values.shape[1]), dtype=np.float64)
        out[:] = values[0]
        return out
    src_u = (src_t - src_t[0]) / max(float(src_t[-1] - src_t[0]), 1e-12)
    dst_u = (dst_t - dst_t[0]) / max(float(dst_t[-1] - dst_t[0]), 1e-12) if len(dst_t) > 1 else np.zeros(len(dst_t))
    if values.ndim == 1:
        return np.interp(dst_u, src_u, values)
    out = np.zeros((len(dst_t), values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst_u, src_u, values[:, j])
    return out


def qpos_map(model, prefix: str) -> dict[str, int]:
    _, mujoco = require_runtime()
    out = {}
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name)
        if jid < 0:
            raise RuntimeError(f"missing joint in fixed dual scene: {prefix + name}")
        out[name] = int(model.jnt_qposadr[jid])
    return out


def actuator_map(model, prefix: str) -> dict[str, int]:
    _, mujoco = require_runtime()
    out = {}
    for name in tuple(f"joint{i}_pos" for i in range(1, 8)) + ("gripper_joint1_pos", "gripper_joint2_pos"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name)
        if aid >= 0:
            out[name] = int(aid)
    return out


def mocap_id(model, body_name: str) -> int | None:
    _, mujoco = require_runtime()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])
    return mid if mid >= 0 else None


def site_id(model, site_name: str) -> int | None:
    _, mujoco = require_runtime()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    return int(sid) if sid >= 0 else None


def apply_joints(data, qmap: dict[str, int], amap: dict[str, int], q: np.ndarray, width_m: float) -> None:
    for i, value in enumerate(q, start=1):
        data.qpos[qmap[f"joint{i}"]] = float(value)
        aid = amap.get(f"joint{i}_pos")
        if aid is not None:
            data.ctrl[aid] = float(value)
    width_m = float(np.clip(width_m, 0.0, 0.1))
    finger = 0.5 * width_m
    data.qpos[qmap["gripper_joint1"]] = finger
    data.qpos[qmap["gripper_joint2"]] = -finger
    if "gripper_joint1_pos" in amap:
        data.ctrl[amap["gripper_joint1_pos"]] = finger
    if "gripper_joint2_pos" in amap:
        data.ctrl[amap["gripper_joint2_pos"]] = -finger


def set_marker(data, mid: int | None, pos: np.ndarray | None, quat_xyzw: np.ndarray | None) -> None:
    if mid is None:
        return
    if pos is None or quat_xyzw is None:
        data.mocap_pos[mid] = [10.0, 10.0, 10.0]
        return
    data.mocap_pos[mid] = pos
    data.mocap_quat[mid] = quat_xyzw_to_wxyz(quat_xyzw)


def make_camera():
    _, mujoco = require_runtime()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.25, 0.45, 0.28]
    cam.distance = 1.6
    cam.azimuth = 145.0
    cam.elevation = -28.0
    return cam


def build_series(realbot_path: Path, ik_path: str | None, fps: float) -> dict:
    real = load_realbot(realbot_path)
    duration = float(real["time_s"][-1]) if len(real["time_s"]) > 1 else 0.0
    n = max(int(round(duration * fps)) + 1, 1)
    t = np.arange(n, dtype=np.float64) / fps
    if duration > 0:
        t[-1] = min(t[-1], duration)
    q_real = resample_series(real["time_s"], real["joint_qpos"], t)
    w_real = resample_series(real["time_s"], real["gripper_width_m"], t)

    q_ik = np.zeros((len(t), 7), dtype=np.float64)
    w_ik = np.full(len(t), 0.1, dtype=np.float64)
    target_pos = None
    target_quat = None
    has_ik = False
    if ik_path:
        ik = load_ik(as_abs(ik_path))
        has_ik = True
        q_ik = resample_series(np.asarray(ik["time_s"], dtype=np.float64), np.asarray(ik["joint_qpos"], dtype=np.float64), t)
        w_ik = resample_series(np.asarray(ik["time_s"], dtype=np.float64), np.asarray(ik["gripper_width_m"], dtype=np.float64), t)
        target_pos = resample_series(np.asarray(ik["time_s"], dtype=np.float64), np.asarray(ik["target_pos_m"], dtype=np.float64), t)
        src_t = np.asarray(ik["time_s"], dtype=np.float64)
        src_u = (src_t - src_t[0]) / max(float(src_t[-1] - src_t[0]), 1e-12)
        dst_u = (t - t[0]) / max(float(t[-1] - t[0]), 1e-12) if len(t) > 1 else np.zeros(len(t))
        quats = np.asarray(ik["target_quat_xyzw"], dtype=np.float64)
        nearest = np.clip(np.searchsorted(src_u, dst_u), 0, len(src_u) - 1)
        for i, u in enumerate(dst_u):
            j = int(nearest[i])
            if j > 0 and abs(src_u[j - 1] - u) < abs(src_u[j] - u):
                nearest[i] = j - 1
        target_quat = quats[nearest]
    return {
        "time_s": t,
        "q_real": q_real,
        "w_real": w_real,
        "q_ik": q_ik,
        "w_ik": w_ik,
        "target_pos": target_pos,
        "target_quat": target_quat,
        "has_ik": has_ik,
    }


def step(model, data, maps, series, i: int) -> None:
    _, mujoco = require_runtime()
    data.time = float(series["time_s"][i])
    apply_joints(data, maps["left_q"], maps["left_a"], series["q_real"][i], series["w_real"][i])
    apply_joints(data, maps["right_q"], maps["right_a"], series["q_ik"][i], series["w_ik"][i])
    if series["has_ik"]:
        set_marker(data, maps["left_he_mid"], None, None)
        right_pos = series["target_pos"][i].copy()
        right_pos[1] += 0.9
        set_marker(data, maps["right_he_mid"], right_pos, series["target_quat"][i])
    else:
        set_marker(data, maps["left_he_mid"], None, None)
        set_marker(data, maps["right_he_mid"], None, None)
    mujoco.mj_forward(model, data)


def resolve_maps(model) -> dict:
    return {
        "left_q": qpos_map(model, ""),
        "left_a": actuator_map(model, ""),
        "right_q": qpos_map(model, RIGHT_PREFIX),
        "right_a": actuator_map(model, RIGHT_PREFIX),
        "left_he_mid": mocap_id(model, "humanego_eef_marker"),
        "right_he_mid": mocap_id(model, "r_humanego_eef_marker"),
        "left_tcp": site_id(model, "tcp"),
        "right_tcp": site_id(model, "r_tcp"),
    }


def draw_hud(frame_rgb, series, i: int) -> np.ndarray:
    cv2, _ = require_runtime()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    lines = [
        f"LEFT real-bot JSONL | RIGHT {'IK npz' if series['has_ik'] else 'static zero'}  frame {i + 1}/{len(series['time_s'])}",
        f"left gripper={series['w_real'][i]*1000:.0f}mm  right gripper={series['w_ik'][i]*1000:.0f}mm",
        "cyan target marker appears on the right only when --ik is provided",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def setup(args):
    _, mujoco = require_runtime()
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.dual_scene)))
    data = mujoco.MjData(model)
    maps = resolve_maps(model)
    series = build_series(as_abs(args.realbot), args.ik if args.ik else None, args.fps)
    return model, data, maps, series


def launch_viewer(args) -> None:
    _, mujoco = require_runtime()
    model, data, maps, series = setup(args)
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
                step(model, data, maps, series, i)
            viewer.set_texts((None, None, f"LEFT real-bot\n{i + 1}/{len(series['time_s'])}", f"RIGHT {'IK' if series['has_ik'] else 'static zero'}"))
            viewer.sync()
            if i < len(series["time_s"]) - 1:
                i += 1
            elif not args.once:
                i = 0
            time.sleep(period)


def render_mp4(args) -> None:
    cv2, mujoco = require_runtime()
    model, data, maps, series = setup(args)
    out = as_abs(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = make_camera()
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {out}")
    loops = 1 if args.once else max(int(args.loops), 1)
    try:
        for _ in range(loops):
            for i in range(len(series["time_s"])):
                step(model, data, maps, series, i)
                renderer.update_scene(data, camera=camera)
                writer.write(draw_hud(renderer.render(), series, i))
    finally:
        writer.release()
        renderer.close()
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-scene", default=str(MUJOCO_NERO_DUAL_SCENE))
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument("--ik", default="", help="Optional IK npz. If omitted, right robot remains at zero pose.")
    parser.add_argument("--out", default="outputs/new_pipeline/replays/realbot_vs_ik.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1600)
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
