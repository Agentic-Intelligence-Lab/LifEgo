#!/usr/bin/env python3
"""Replay Nero IK results in the fixed MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from assets import MUJOCO_NERO_SCENE
from utils_replay import as_abs, load_runtime, quat_xyzw_to_wxyz, require_runtime

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")


def load_ik(path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    missing = [k for k in ("joint_qpos", "target_pos_m", "target_quat_xyzw") if k not in out]
    if missing:
        raise RuntimeError(f"IK file {path} missing keys: {missing}")
    n = len(out["joint_qpos"])
    if "time_s" not in out:
        out["time_s"] = np.arange(n, dtype=np.float64) / 30.0
    if "grasp" in out:
        out["grasp"] = np.asarray(out["grasp"], dtype=np.int32).reshape(-1)
    if "gripper_width_m" not in out:
        if "grasp" in out:
            open_m = float(out["gripper_open_m"]) if "gripper_open_m" in out else 0.1
            closed_m = float(out["gripper_closed_m"]) if "gripper_closed_m" in out else 0.0
            out["gripper_width_m"] = np.where(out["grasp"] > 0.5, closed_m, open_m).astype(np.float64)
        else:
            out["gripper_width_m"] = np.full(n, 0.1, dtype=np.float64)
    if "pos_err_m" not in out:
        out["pos_err_m"] = np.full(n, np.nan, dtype=np.float64)
    if "ang_err_deg" not in out:
        out["ang_err_deg"] = np.full(n, np.nan, dtype=np.float64)
    return out


def qpos_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    _, mujoco = require_runtime()
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"missing joint in fixed scene: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def actuator_ids(model) -> dict[str, int]:
    _, mujoco = require_runtime()
    out = {}
    for name in tuple(f"joint{i}_pos" for i in range(1, 8)) + ("gripper_joint1_pos", "gripper_joint2_pos"):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out[name] = int(aid)
    return out


def marker_mocap_id(model) -> int | None:
    _, mujoco = require_runtime()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "humanego_eef_marker")
    if bid < 0:
        return None
    mid = int(model.body_mocapid[bid])
    return mid if mid >= 0 else None


def make_camera():
    _, mujoco = require_runtime()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.28, -0.13, 0.25]
    cam.distance = 1.15
    cam.azimuth = 132.0
    cam.elevation = -27.0
    return cam


def apply_frame(data, arm_addrs, gripper_addrs, acts, q_arm, width_m, t, target_pos, target_quat, marker_mid) -> None:
    data.time = float(t)
    data.qpos[arm_addrs] = np.asarray(q_arm, dtype=np.float64)
    for i, val in enumerate(q_arm, start=1):
        aid = acts.get(f"joint{i}_pos")
        if aid is not None:
            data.ctrl[aid] = float(val)
    width_m = float(np.clip(width_m, 0.0, 0.1))
    finger = 0.5 * width_m
    if len(gripper_addrs) == 2:
        data.qpos[gripper_addrs[0]] = finger
        data.qpos[gripper_addrs[1]] = -finger
    if "gripper_joint1_pos" in acts:
        data.ctrl[acts["gripper_joint1_pos"]] = finger
    if "gripper_joint2_pos" in acts:
        data.ctrl[acts["gripper_joint2_pos"]] = -finger
    if marker_mid is not None:
        data.mocap_pos[marker_mid] = target_pos
        data.mocap_quat[marker_mid] = quat_xyzw_to_wxyz(target_quat)


def draw_hud(frame_rgb, i, n, t, pos_err, ang_err, width_m, grasp) -> np.ndarray:
    cv2, _ = require_runtime()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gtxt = f"grasp={'CLOSED' if grasp else 'OPEN'}" if grasp is not None else "grasp=n/a"
    lines = [
        f"Nero IK replay  frame {i + 1}/{n}  t={t:.2f}s",
        f"cyan=EEF target  gripper={width_m * 1000:.0f} mm  {gtxt}",
        f"IK error: pos={pos_err * 1000.0:.1f} mm  ang={ang_err:.2f} deg",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def setup(args):
    _, mujoco = require_runtime()
    ik = load_ik(as_abs(args.ik))
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    return ik, model, data, qpos_addrs(model, ARM_JOINTS), qpos_addrs(model, GRIPPER_JOINTS), actuator_ids(model), marker_mocap_id(model)


def launch_viewer(args) -> None:
    _, mujoco = require_runtime()
    ik, model, data, arm_addrs, gripper_addrs, acts, marker_mid = setup(args)
    q = np.asarray(ik["joint_qpos"], dtype=np.float64)
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
                apply_frame(
                    data,
                    arm_addrs,
                    gripper_addrs,
                    acts,
                    q[i],
                    ik["gripper_width_m"][i],
                    ik["time_s"][i],
                    ik["target_pos_m"][i],
                    ik["target_quat_xyzw"][i],
                    marker_mid,
                )
                mujoco.mj_forward(model, data)
            grasp = int(ik["grasp"][i]) if "grasp" in ik else None
            viewer.set_texts(
                (
                    None,
                    None,
                    f"Nero IK replay\n{i + 1}/{len(q)}",
                    f"pos err {ik['pos_err_m'][i] * 1000.0:.1f} mm\nang err {ik['ang_err_deg'][i]:.2f} deg\nw {ik['gripper_width_m'][i]*1000:.0f} mm",
                )
            )
            viewer.sync()
            if i < len(q) - 1:
                i += 1
            elif not args.once:
                i = 0
            time.sleep(period)


def render_mp4(args) -> None:
    cv2, mujoco = require_runtime()
    ik, model, data, arm_addrs, gripper_addrs, acts, marker_mid = setup(args)
    q = np.asarray(ik["joint_qpos"], dtype=np.float64)
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
            for i in range(len(q)):
                apply_frame(data, arm_addrs, gripper_addrs, acts, q[i], ik["gripper_width_m"][i], ik["time_s"][i], ik["target_pos_m"][i], ik["target_quat_xyzw"][i], marker_mid)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                grasp = int(ik["grasp"][i]) if "grasp" in ik else None
                writer.write(draw_hud(renderer.render(), i, len(q), ik["time_s"][i], ik["pos_err_m"][i], ik["ang_err_deg"][i], ik["gripper_width_m"][i], grasp))
    finally:
        writer.release()
        renderer.close()
    summary = {
        "scene": str(as_abs(args.scene)),
        "ik": str(as_abs(args.ik)),
        "video": str(out),
        "frames": int(len(q)),
        "pos_err_mean_mm": float(np.nanmean(ik["pos_err_m"]) * 1000.0),
        "pos_err_max_mm": float(np.nanmax(ik["pos_err_m"]) * 1000.0),
        "ang_err_mean_deg": float(np.nanmean(ik["ang_err_deg"])),
        "ang_err_max_deg": float(np.nanmax(ik["ang_err_deg"])),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    parser.add_argument("--ik", default="outputs/new_pipeline/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz")
    parser.add_argument("--out", default="outputs/new_pipeline/replays/ik.mp4")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loops", type=int, default=2)
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
