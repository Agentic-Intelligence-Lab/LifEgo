#!/usr/bin/env python3
"""Side-by-side MuJoCo replay: real-bot joints (left) vs IK solution (right).

Builds a dual-platform scene from the single Nero table scene: the original
platform stays on the left; an identical robot platform is copied to the right
(+Y / robot-right). Left replays JSONL follower joints; right replays a chosen
EEF-IK npz. Playback loops by default so real vs HumanEgo-reconstructed
motion can be compared continuously.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
cv2 = None
mujoco = None

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
RIGHT_PREFIX = "r_"

# Static path/frame clutter removed when building the dual scene.
STRIP_NAME_PREFIXES = (
    "real_flange_path_",
    "real_flange_frame_",
    "real_flange_start",
    "real_flange_end",
    "humanego_eef_path_",
    "humanego_eef_frame_",
    "humanego_eef_start",
    "humanego_eef_end",
)

# Bodies cloned for the right platform (siblings of world, not under base_link).
CLONE_WORLD_BODIES = (
    "vis_tcp_marker",
    "vis_tip_marker",
    "vis_flange_marker",
    "humanego_eef_marker",
)


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
    times, qpos, gripper_width = [], [], []
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
    if not qpos:
        raise RuntimeError(f"No sample records in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return {
        "time_s": t,
        "joint_qpos": np.asarray(qpos, dtype=np.float64),
        "gripper_width_m": np.asarray(gripper_width, dtype=np.float64),
    }


def load_ik(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    required = ("joint_qpos", "target_pos_m", "target_quat_xyzw")
    missing = [k for k in required if k not in out]
    if missing:
        raise RuntimeError(f"IK file {path} missing keys: {missing}")
    if "time_s" not in out:
        n = len(out["joint_qpos"])
        out["time_s"] = np.arange(n, dtype=np.float64)
    if "gripper_width_m" not in out:
        out["gripper_width_m"] = np.full(len(out["joint_qpos"]), 0.1, dtype=np.float64)
    t = np.asarray(out["time_s"], dtype=np.float64)
    out["time_s"] = t - t[0]
    return out


def resample_series(times: np.ndarray, values: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    duration = float(times[-1]) if len(times) > 1 else 0.0
    n = max(int(round(duration * fps)) + 1, 1)
    out_t = np.arange(n, dtype=np.float64) / fps
    if duration > 0:
        out_t[-1] = min(out_t[-1], duration)
    if values.ndim == 1:
        if len(times) == 1:
            return out_t, np.full(n, float(values[0]), dtype=np.float64)
        return out_t, np.interp(out_t, times, values)
    out = np.zeros((len(out_t), values.shape[1]), dtype=np.float64)
    if len(times) == 1:
        out[:] = values[0]
        return out_t, out
    for j in range(values.shape[1]):
        out[:, j] = np.interp(out_t, times, values[:, j])
    return out_t, out


def resample_to_timeline(
    src_t: np.ndarray, values: np.ndarray, dst_t: np.ndarray
) -> np.ndarray:
    """Map a series from src_t onto dst_t by normalized progress (0..1)."""
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


def prefix_element_names(elem: ET.Element, prefix: str) -> None:
    """Prefix MuJoCo object names; keep mesh/material refs shared."""
    for attr in ("name", "joint", "site", "body", "tendon", "actuator", "geom"):
        if attr in elem.attrib and elem.attrib[attr]:
            elem.attrib[attr] = prefix + elem.attrib[attr]
    for child in elem:
        prefix_element_names(child, prefix)


def should_strip(name: str | None) -> bool:
    if not name:
        return False
    return any(name.startswith(p) for p in STRIP_NAME_PREFIXES)


def strip_path_decoration(worldbody: ET.Element) -> None:
    for child in list(worldbody):
        name = child.attrib.get("name")
        if should_strip(name):
            worldbody.remove(child)


def offset_body_tree_positions(elem: ET.Element, dy: float) -> None:
    """Add dy to top-level absolute positions (world siblings only via caller)."""
    if "pos" in elem.attrib:
        parts = [float(x) for x in elem.attrib["pos"].split()]
        while len(parts) < 3:
            parts.append(0.0)
        parts[1] += dy
        elem.attrib["pos"] = " ".join(f"{v:.9g}" for v in parts)


def build_dual_scene(src_scene: Path, out_scene: Path, side_offset_y: float) -> Path:
    """Duplicate robot platform to the right (+Y) inside a dual MJCF file."""
    tree = ET.parse(src_scene)
    root = tree.getroot()
    root.set("model", "nero_dual_realbot_ik_compare")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(f"No worldbody in {src_scene}")

    strip_path_decoration(worldbody)

    # Widen floor / stats for two tables.
    for geom in worldbody.findall("geom"):
        if geom.attrib.get("name") == "table_top_z0":
            # Left table stays; add a second table for the right platform.
            right_table = copy.deepcopy(geom)
            right_table.set("name", "r_table_top_z0")
            parts = [float(x) for x in geom.attrib.get("pos", "0 0 0").split()]
            while len(parts) < 3:
                parts.append(0.0)
            parts[1] += side_offset_y
            right_table.set("pos", " ".join(f"{v:.9g}" for v in parts))
            worldbody.insert(list(worldbody).index(geom) + 1, right_table)
        if geom.attrib.get("name") == "left_front_workspace_patch":
            right_patch = copy.deepcopy(geom)
            right_patch.set("name", "r_left_front_workspace_patch")
            parts = [float(x) for x in geom.attrib.get("pos", "0 0 0").split()]
            while len(parts) < 3:
                parts.append(0.0)
            parts[1] += side_offset_y
            right_patch.set("pos", " ".join(f"{v:.9g}" for v in parts))
            worldbody.insert(list(worldbody).index(geom) + 1, right_patch)

    base = worldbody.find("body[@name='base_link']")
    if base is None:
        raise RuntimeError("Scene missing body base_link")

    right_base = copy.deepcopy(base)
    prefix_element_names(right_base, RIGHT_PREFIX)
    # Place whole right robot at +Y (robot-right of original platform).
    right_base.set("pos", f"0 {side_offset_y:.9g} 0")
    # Insert immediately after left base for stable joint order (L then R).
    idx = list(worldbody).index(base)
    worldbody.insert(idx + 1, right_base)

    for body_name in CLONE_WORLD_BODIES:
        body = worldbody.find(f"body[@name='{body_name}']")
        if body is None:
            continue
        right_body = copy.deepcopy(body)
        prefix_element_names(right_body, RIGHT_PREFIX)
        offset_body_tree_positions(right_body, side_offset_y)
        worldbody.append(right_body)

    actuator = root.find("actuator")
    if actuator is not None:
        for pos_el in list(actuator.findall("position")):
            clone = copy.deepcopy(pos_el)
            if "name" in clone.attrib:
                clone.set("name", RIGHT_PREFIX + clone.attrib["name"])
            if "joint" in clone.attrib:
                clone.set("joint", RIGHT_PREFIX + clone.attrib["joint"])
            actuator.append(clone)

    # Dual zero keyframe: left 9 + right 9 dofs.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            q = key.attrib.get("qpos", "").strip().split()
            c = key.attrib.get("ctrl", "").strip().split()
            if q:
                key.set("qpos", " ".join(q + q))
            if c:
                key.set("ctrl", " ".join(c + c))

    for statistic in root.findall("statistic"):
        statistic.set("center", f"-0.15 {0.5 * side_offset_y:.3g} 0.28")
        statistic.set("extent", f"{1.2 + 0.5 * abs(side_offset_y):.3g}")

    out_scene.parent.mkdir(parents=True, exist_ok=True)
    # Preserve meshdir relative to dual xml sibling of original meshes.
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        meshdir = Path(compiler.attrib["meshdir"])
        if not meshdir.is_absolute():
            abs_mesh = (src_scene.parent / meshdir).resolve()
            rel = os.path.relpath(abs_mesh, out_scene.parent)
            compiler.set("meshdir", rel)

    indent = getattr(ET, "indent", None)
    if indent is not None:
        indent(root)
    ET.ElementTree(root).write(out_scene, encoding="utf-8", xml_declaration=True)
    return out_scene


def joint_maps(model: mujoco.MjModel, prefix: str = "") -> tuple[dict[str, int], dict[str, int]]:
    joint_qpos_addr = {}
    for name in ARM_JOINTS + GRIPPER_JOINTS:
        full = prefix + name
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise RuntimeError(f"Missing joint: {full}")
        joint_qpos_addr[name] = int(model.jnt_qposadr[jid])
    actuator_id = {}
    for name in [f"joint{i}_pos" for i in range(1, 8)] + ["gripper_joint1_pos", "gripper_joint2_pos"]:
        full = prefix + name
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
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


def resolve_side_ids(model: mujoco.MjModel, prefix: str) -> dict:
    ids = {
        "tcp_site": site_id(model, prefix + "tcp"),
        "tip_site": site_id(model, prefix + "gripper_tip"),
        "mocap_tcp": mocap_id(model, prefix + "vis_tcp_marker"),
        "mocap_tip": mocap_id(model, prefix + "vis_tip_marker"),
        "mocap_flange": mocap_id(model, prefix + "vis_flange_marker"),
        "mocap_humanego": mocap_id(model, prefix + "humanego_eef_marker"),
    }
    if ids["mocap_tcp"] is None or ids["mocap_tip"] is None:
        raise RuntimeError(
            f"Missing {prefix}vis_tcp_marker / {prefix}vis_tip_marker. "
            "Rebuild scene with build_nero_mujoco_scene.py then re-run compare."
        )
    return ids


def update_tool_markers(
    data: mujoco.MjData,
    ids: dict,
    *,
    humanego_pos: np.ndarray | None = None,
    humanego_quat_xyzw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    park_mocap(data, ids["mocap_flange"])
    if humanego_pos is None or humanego_quat_xyzw is None:
        park_mocap(data, ids["mocap_humanego"])
    else:
        mid = ids["mocap_humanego"]
        if mid is not None:
            q = np.asarray(humanego_quat_xyzw, dtype=np.float64)
            # xyzw -> wxyz
            data.mocap_pos[mid] = humanego_pos
            data.mocap_quat[mid] = [q[3], q[0], q[1], q[2]]

    tcp_p = data.site_xpos[ids["tcp_site"]].copy()
    tcp_R = data.site_xmat[ids["tcp_site"]].reshape(3, 3).copy()
    tip_p = data.site_xpos[ids["tip_site"]].copy()
    tip_R = data.site_xmat[ids["tip_site"]].reshape(3, 3).copy()
    set_mocap(data, ids["mocap_tcp"], tcp_p, tcp_R)
    set_mocap(data, ids["mocap_tip"], tip_p, tip_R)
    return tcp_p, tip_p


def make_camera(side_offset_y: float) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [-0.25, 0.5 * side_offset_y, 0.28]
    cam.distance = 1.15 + 0.55 * abs(side_offset_y)
    cam.azimuth = 145.0
    cam.elevation = -28.0
    return cam


def prepare_series(realbot: Path, ik_path: Path, fps: float) -> dict:
    real = load_realbot(realbot)
    ik = load_ik(ik_path)
    t_real, q_real = resample_series(real["time_s"], real["joint_qpos"], fps)
    _, w_real = resample_series(real["time_s"], real["gripper_width_m"], fps)

    # Align IK to the same frame clock via normalized progress (same length as real).
    q_ik = resample_to_timeline(ik["time_s"], np.asarray(ik["joint_qpos"], dtype=np.float64), t_real)
    w_ik = resample_to_timeline(ik["time_s"], np.asarray(ik["gripper_width_m"], dtype=np.float64), t_real)
    tgt_pos = resample_to_timeline(ik["time_s"], np.asarray(ik["target_pos_m"], dtype=np.float64), t_real)
    # Quat: independent component interp is crude; use Slerp-like via R.
    src_u = (ik["time_s"] - ik["time_s"][0]) / max(float(ik["time_s"][-1] - ik["time_s"][0]), 1e-12)
    dst_u = (t_real - t_real[0]) / max(float(t_real[-1] - t_real[0]), 1e-12) if len(t_real) > 1 else np.zeros(len(t_real))
    # Nearest-neighbor quat (stable) then optional short slerp between neighbors
    quats = np.asarray(ik["target_quat_xyzw"], dtype=np.float64)
    idx = np.clip(np.searchsorted(src_u, dst_u), 0, len(src_u) - 1)
    # refine to nearest
    for i, u in enumerate(dst_u):
        j = int(idx[i])
        if j > 0 and abs(src_u[j - 1] - u) < abs(src_u[j] - u):
            idx[i] = j - 1
    tgt_quat = quats[idx]

    return {
        "time_s": t_real,
        "real_joint_qpos": q_real,
        "real_gripper_width_m": w_real,
        "ik_joint_qpos": q_ik,
        "ik_gripper_width_m": w_ik,
        "ik_target_pos_m": tgt_pos,
        "ik_target_quat_xyzw": tgt_quat,
        "n_real_src": len(real["joint_qpos"]),
        "n_ik_src": len(ik["joint_qpos"]),
    }


def apply_viewer_camera(viewer, side_offset_y: float) -> None:
    cam = make_camera(side_offset_y)
    viewer.cam.type = cam.type
    viewer.cam.lookat[:] = cam.lookat
    viewer.cam.distance = cam.distance
    viewer.cam.azimuth = cam.azimuth
    viewer.cam.elevation = cam.elevation


def step_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_maps: tuple,
    right_maps: tuple,
    left_ids: dict,
    right_ids: dict,
    series: dict,
    i: int,
    side_offset_y: float,
) -> dict:
    t = float(series["time_s"][i])
    data.time = t
    apply_joints(data, left_maps[0], left_maps[1], series["real_joint_qpos"][i], series["real_gripper_width_m"][i])
    apply_joints(data, right_maps[0], right_maps[1], series["ik_joint_qpos"][i], series["ik_gripper_width_m"][i])
    mujoco.mj_forward(model, data)

    left_tcp, left_tip = update_tool_markers(data, left_ids)
    # HumanEgo IK target on the right platform (+Y offset to that table frame).
    tgt_pos = series["ik_target_pos_m"][i].copy()
    tgt_pos[1] += side_offset_y
    right_tcp, right_tip = update_tool_markers(
        data,
        right_ids,
        humanego_pos=tgt_pos,
        humanego_quat_xyzw=series["ik_target_quat_xyzw"][i],
    )
    mujoco.mj_forward(model, data)
    return {
        "t": t,
        "left_tcp": left_tcp,
        "left_tip": left_tip,
        "right_tcp": right_tcp,
        "right_tip": right_tip,
        "tgt_pos": tgt_pos,
    }


def launch_viewer(args: argparse.Namespace, dual_scene: Path) -> None:
    series = prepare_series(as_abs(args.realbot), as_abs(args.ik), args.fps)
    model = mujoco.MjModel.from_xml_path(str(dual_scene))
    data = mujoco.MjData(model)
    left_maps = joint_maps(model, "")
    right_maps = joint_maps(model, RIGHT_PREFIX)
    left_ids = resolve_side_ids(model, "")
    right_ids = resolve_side_ids(model, RIGHT_PREFIX)
    period = 1.0 / max(args.fps, 1e-9)
    n = len(series["time_s"])

    print("LEFT=real-bot joints  RIGHT=IK joints  CYAN=HumanEgo EEF target (right only)")
    print("MAGENTA=tcp  ORANGE=tip | loops by default  --once to stop after one pass")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        apply_viewer_camera(viewer, args.side_offset_y)
        i = 0
        while viewer.is_running():
            with viewer.lock():
                info = step_frame(
                    model,
                    data,
                    left_maps,
                    right_maps,
                    left_ids,
                    right_ids,
                    series,
                    i,
                    args.side_offset_y,
                )
            err_mm = float(np.linalg.norm(info["right_tcp"] - info["tgt_pos"]) * 1000.0)
            left = f"LEFT real-bot\n{i + 1}/{n}  t={info['t']:.2f}s\nloop={'on' if not args.once else 'off'}"
            right = (
                f"RIGHT IK (HumanEgo)\n"
                f"CYAN target  |tcp-tgt|={err_mm:.1f} mm\n"
                f"MAGENTA tcp  ORANGE tip"
            )
            viewer.set_texts((None, None, left, right))
            viewer.sync()

            if i < n - 1:
                i += 1
                time.sleep(period)
            elif not args.once:
                i = 0
                time.sleep(period)
            else:
                time.sleep(0.05)


def draw_hud(frame_rgb: np.ndarray, i: int, n: int, info: dict, loop: bool) -> np.ndarray:
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    err_mm = float(np.linalg.norm(info["right_tcp"] - info["tgt_pos"]) * 1000.0)
    lines = [
        f"LEFT real-bot  |  RIGHT IK (HumanEgo)   {i + 1}/{n}  t={info['t']:.2f}s  loop={'on' if loop else 'off'}",
        f"MAGENTA=tcp  ORANGE=tip  CYAN=HumanEgo target (right)  |tcp-tgt|={err_mm:.1f} mm",
    ]
    x, y = 18, 28
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
        y += 26
    return frame


def render_replay(args: argparse.Namespace, dual_scene: Path) -> None:
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series = prepare_series(as_abs(args.realbot), as_abs(args.ik), args.fps)
    model = mujoco.MjModel.from_xml_path(str(dual_scene))
    data = mujoco.MjData(model)
    left_maps = joint_maps(model, "")
    right_maps = joint_maps(model, RIGHT_PREFIX)
    left_ids = resolve_side_ids(model, "")
    right_ids = resolve_side_ids(model, RIGHT_PREFIX)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = make_camera(args.side_offset_y)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    n = len(series["time_s"])
    loops = 1 if args.once else max(int(args.loops), 1)
    try:
        for _ in range(loops):
            for i in range(n):
                info = step_frame(
                    model,
                    data,
                    left_maps,
                    right_maps,
                    left_ids,
                    right_ids,
                    series,
                    i,
                    args.side_offset_y,
                )
                renderer.update_scene(data, camera=cam)
                rgb = renderer.render()
                writer.write(draw_hud(rgb, i, n, info, loop=not args.once))
    finally:
        writer.release()
        renderer.close()

    summary = {
        "dual_scene": str(dual_scene),
        "source_scene": str(as_abs(args.scene)),
        "realbot": str(as_abs(args.realbot)),
        "ik": str(as_abs(args.ik)),
        "video": str(out_path),
        "side_offset_y_m": float(args.side_offset_y),
        "frames_per_loop": int(n),
        "loops": int(loops),
        "fps": float(args.fps),
        "n_real_src": int(series["n_real_src"]),
        "n_ik_src": int(series["n_ik_src"]),
        "legend": {
            "left": "real-bot follower joints from JSONL",
            "right": "IK joint_qpos from npz",
            "cyan_right": "HumanEgo EEF IK target (world + right offset)",
            "magenta": "site:tcp FK",
            "orange": "site:gripper_tip FK",
        },
    }
    out_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml", help="Single-platform Nero scene")
    parser.add_argument(
        "--dual-scene",
        default="outputs/mujoco_nero_scene/scene_dual_compare.xml",
        help="Generated dual-platform MJCF (overwritten each run unless --reuse-dual-scene)",
    )
    parser.add_argument("--reuse-dual-scene", action="store_true")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument("--ik", default="outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz")
    parser.add_argument(
        "--out",
        default="outputs/mujoco_nero_scene/replays/ego_nero_easy_realbot_vs_ik.mp4",
    )
    parser.add_argument(
        "--side-offset-y",
        type=float,
        default=0.90,
        help="Right platform offset along base +Y (robot-right), meters",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play a single pass (default: loop forever in viewer; N loops in video)",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=2,
        help="Video only: number of full passes when not --once (default 2)",
    )
    parser.add_argument(
        "--gl-backend",
        default="auto",
        choices=["auto", "glfw", "egl", "osmesa"],
    )
    args = parser.parse_args()
    load_runtime(args.viewer, args.gl_backend)

    dual_path = as_abs(args.dual_scene)
    if not (args.reuse_dual_scene and dual_path.is_file()):
        dual_path = build_dual_scene(as_abs(args.scene), dual_path, args.side_offset_y)
        print(f"Built dual scene: {dual_path}")
    else:
        print(f"Reusing dual scene: {dual_path}")

    if args.viewer:
        launch_viewer(args, dual_path)
    else:
        render_replay(args, dual_path)


if __name__ == "__main__":
    main()
