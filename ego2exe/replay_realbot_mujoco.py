#!/usr/bin/env python3
"""Replay ARX real-bot teleoperation parquet in the MuJoCo scene."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from assets import DEFAULT_ASSETS, MUJOCO_ARX_SCENE
from utils_replay import as_abs, load_runtime, require_runtime

LEFT_ARM_JOINTS = tuple(f"left_joint{i}" for i in range(1, 7))
LEFT_GRIPPER_JOINTS = ("left_joint7", "left_joint8")
RIGHT_ARM_JOINTS = tuple(f"right_joint{i}" for i in range(11, 17))
RIGHT_GRIPPER_JOINTS = ("right_joint17", "right_joint18")

DEFAULT_REALBOT = "DATA/arx_ego_dataset/data/chunk-000/file-000.parquet"
DEFAULT_OUT = "outputs/new_pipeline/arx_replays/realbot_file_000.mp4"


def load_parquet(path: Path, column: str) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required to replay LeRobot parquet data") from exc

    table = pq.read_table(path)
    names = None
    metadata = table.schema.metadata or {}
    hf_meta = metadata.get(b"huggingface")
    if hf_meta:
        try:
            info = json.loads(hf_meta.decode("utf-8")).get("info", {})
            names = info.get("features", {}).get(column, {}).get("names")
        except Exception:
            names = None
    if names is None:
        names = [
            "left_joint_1",
            "left_joint_2",
            "left_joint_3",
            "left_joint_4",
            "left_joint_5",
            "left_joint_6",
            "left_gripper",
            "right_joint_1",
            "right_joint_2",
            "right_joint_3",
            "right_joint_4",
            "right_joint_5",
            "right_joint_6",
            "right_gripper",
        ]

    if column not in table.column_names:
        raise RuntimeError(f"Column {column!r} missing from {path}; columns={table.column_names}")
    state = np.asarray(table[column].to_pylist(), dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 14:
        raise RuntimeError(f"Expected {column} shape [N,14], got {state.shape}")
    timestamp = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    timestamp -= timestamp[0]

    return {
        "time_s": timestamp,
        "left_qpos": state[:, :6],
        "left_gripper_raw": state[:, 6],
        "right_qpos": state[:, 7:13],
        "right_gripper_raw": state[:, 13],
        "frame_index": np.asarray(table["frame_index"].to_pylist(), dtype=np.int64),
        "episode_index": np.asarray(table["episode_index"].to_pylist(), dtype=np.int64),
        "task_index": np.asarray(table["task_index"].to_pylist(), dtype=np.int64),
        "feature_names": names,
        "source_column": column,
    }


def resample_series(src_t: np.ndarray, values: np.ndarray, dst_t: np.ndarray) -> np.ndarray:
    if len(src_t) == 1:
        if values.ndim == 1:
            return np.full(len(dst_t), float(values[0]), dtype=np.float64)
        out = np.zeros((len(dst_t), values.shape[1]), dtype=np.float64)
        out[:] = values[0]
        return out
    if values.ndim == 1:
        return np.interp(dst_t, src_t, values)
    out = np.zeros((len(dst_t), values.shape[1]), dtype=np.float64)
    for j in range(values.shape[1]):
        out[:, j] = np.interp(dst_t, src_t, values[:, j])
    return out


def gripper_raw_to_width(raw: np.ndarray, open_m: float, closed_m: float) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if np.all((raw >= 0.0) & (raw <= open_m * 1.5)):
        return np.clip(raw, closed_m, open_m)
    span = float(np.nanmax(raw) - np.nanmin(raw))
    if span < 1e-2:
        return np.full(len(raw), open_m, dtype=np.float64)
    # ARX teleop gripper logs are not meters in this dataset: larger raw value
    # corresponds to a more open gripper, lower raw value to a more closed one.
    norm = (raw - np.nanmin(raw)) / span
    return closed_m + norm * (open_m - closed_m)


def qpos_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    _, mujoco = require_runtime()
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"missing joint in ARX scene: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def actuator_ids(model, names: tuple[str, ...]) -> dict[str, int]:
    _, mujoco = require_runtime()
    out = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pos")
        if aid >= 0:
            out[name] = int(aid)
    return out


def apply_arm(data, addrs: np.ndarray, act_ids: dict[str, int], joint_names: tuple[str, ...], q: np.ndarray) -> None:
    data.qpos[addrs] = np.asarray(q, dtype=np.float64)
    for name, value in zip(joint_names, q):
        aid = act_ids.get(name)
        if aid is not None:
            data.ctrl[aid] = float(value)


def apply_gripper(data, addrs: np.ndarray, act_ids: dict[str, int], joint_names: tuple[str, ...], width_m: float) -> None:
    width_m = float(np.clip(width_m, 0.0, gripper_open_default()))
    finger = 0.5 * width_m
    data.qpos[addrs] = finger
    for name in joint_names:
        aid = act_ids.get(name)
        if aid is not None:
            data.ctrl[aid] = finger


def gripper_open_default() -> float:
    return float(DEFAULT_ASSETS.platform.gripper_open_m if DEFAULT_ASSETS.platform.gripper_open_m is not None else 0.088)


def gripper_closed_default() -> float:
    return float(DEFAULT_ASSETS.platform.gripper_closed_m if DEFAULT_ASSETS.platform.gripper_closed_m is not None else 0.0)


def make_camera():
    _, mujoco = require_runtime()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.25, 0.0, 0.18]
    cam.distance = 1.35
    cam.azimuth = 145.0
    cam.elevation = -24.0
    return cam


def build_series(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_parquet(as_abs(args.realbot), args.column)
    duration = float(raw["time_s"][-1]) if len(raw["time_s"]) > 1 else 0.0
    if args.native_timing:
        t = raw["time_s"]
        left_q = raw["left_qpos"]
        right_q = raw["right_qpos"]
        left_raw = raw["left_gripper_raw"]
        right_raw = raw["right_gripper_raw"]
    else:
        n = max(int(round(duration * args.fps)) + 1, 1)
        t = np.arange(n, dtype=np.float64) / args.fps
        if duration > 0:
            t[-1] = min(t[-1], duration)
        left_q = resample_series(raw["time_s"], raw["left_qpos"], t)
        right_q = resample_series(raw["time_s"], raw["right_qpos"], t)
        left_raw = resample_series(raw["time_s"], raw["left_gripper_raw"], t)
        right_raw = resample_series(raw["time_s"], raw["right_gripper_raw"], t)

    return {
        **raw,
        "time_s": t,
        "left_qpos": left_q,
        "right_qpos": right_q,
        "left_gripper_width_m": gripper_raw_to_width(left_raw, args.gripper_open_m, args.gripper_closed_m),
        "right_gripper_width_m": gripper_raw_to_width(right_raw, args.gripper_open_m, args.gripper_closed_m),
        "left_gripper_raw": left_raw,
        "right_gripper_raw": right_raw,
    }


def setup(args: argparse.Namespace):
    _, mujoco = require_runtime()
    model = mujoco.MjModel.from_xml_path(str(as_abs(args.scene)))
    data = mujoco.MjData(model)
    maps = {
        "left_arm_q": qpos_addrs(model, LEFT_ARM_JOINTS),
        "left_gripper_q": qpos_addrs(model, LEFT_GRIPPER_JOINTS),
        "right_arm_q": qpos_addrs(model, RIGHT_ARM_JOINTS),
        "right_gripper_q": qpos_addrs(model, RIGHT_GRIPPER_JOINTS),
        "left_a": actuator_ids(model, LEFT_ARM_JOINTS + LEFT_GRIPPER_JOINTS),
        "right_a": actuator_ids(model, RIGHT_ARM_JOINTS + RIGHT_GRIPPER_JOINTS),
    }
    series = build_series(args)
    return model, data, maps, series


def step(model, data, maps: dict[str, Any], series: dict[str, Any], i: int) -> None:
    _, mujoco = require_runtime()
    data.time = float(series["time_s"][i])
    apply_arm(data, maps["left_arm_q"], maps["left_a"], LEFT_ARM_JOINTS, series["left_qpos"][i])
    apply_gripper(data, maps["left_gripper_q"], maps["left_a"], LEFT_GRIPPER_JOINTS, series["left_gripper_width_m"][i])
    apply_arm(data, maps["right_arm_q"], maps["right_a"], RIGHT_ARM_JOINTS, series["right_qpos"][i])
    apply_gripper(data, maps["right_gripper_q"], maps["right_a"], RIGHT_GRIPPER_JOINTS, series["right_gripper_width_m"][i])
    mujoco.mj_forward(model, data)


def draw_hud(frame_rgb, series: dict[str, Any], i: int) -> np.ndarray:
    cv2, _ = require_runtime()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    episode = int(series["episode_index"][0]) if len(series["episode_index"]) else -1
    lines = [
        f"ARX real-bot parquet replay  frame {i + 1}/{len(series['time_s'])}  t={series['time_s'][i]:.2f}s",
        f"episode={episode}  source={series['source_column']}",
        f"left grip raw={series['left_gripper_raw'][i]:+.3f} width={series['left_gripper_width_m'][i]*1000:.0f}mm",
        f"right grip raw={series['right_gripper_raw'][i]:+.3f} width={series['right_gripper_width_m'][i]*1000:.0f}mm",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def launch_viewer(args: argparse.Namespace) -> None:
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
            viewer.set_texts(
                (
                    None,
                    None,
                    f"ARX real-bot\n{i + 1}/{len(series['time_s'])}",
                    f"right q {np.array2string(series['right_qpos'][i], precision=2)}",
                )
            )
            viewer.sync()
            if i < len(series["time_s"]) - 1:
                i += 1
            elif not args.once:
                i = 0
            time.sleep(period)


def render_mp4(args: argparse.Namespace) -> None:
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

    summary = {
        "realbot": str(as_abs(args.realbot)),
        "scene": str(as_abs(args.scene)),
        "video": str(out),
        "frames": int(len(series["time_s"])),
        "duration_s": float(series["time_s"][-1]) if len(series["time_s"]) else 0.0,
        "source_column": series["source_column"],
        "feature_names": series["feature_names"],
        "left_gripper_raw_minmax": [float(np.min(series["left_gripper_raw"])), float(np.max(series["left_gripper_raw"]))],
        "right_gripper_raw_minmax": [float(np.min(series["right_gripper_raw"])), float(np.max(series["right_gripper_raw"]))],
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=str(MUJOCO_ARX_SCENE))
    parser.add_argument("--realbot", default=DEFAULT_REALBOT)
    parser.add_argument("--column", choices=["observation.state", "action"], default="observation.state")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--native-timing", action="store_true", help="Use parquet timestamps directly instead of FPS resampling.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--gripper-open-m", type=float, default=gripper_open_default())
    parser.add_argument("--gripper-closed-m", type=float, default=gripper_closed_default())
    parser.add_argument("--gl-backend", choices=["auto", "glfw", "egl", "osmesa"], default="auto")
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)
    if args.viewer:
        launch_viewer(args)
    else:
        render_mp4(args)


if __name__ == "__main__":
    main()
