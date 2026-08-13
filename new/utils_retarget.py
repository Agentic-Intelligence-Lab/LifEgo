#!/usr/bin/env python3
"""Shared utilities for Nero EEF retargeting scripts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS, MUJOCO_NERO_SCENE

REPO_ROOT = Path(__file__).resolve().parents[1]

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")
DEFAULT_EEF = "outputs/new_pipeline/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json"

mujoco = None


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_runtime(gl_backend: str) -> Any:
    global mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    import mujoco as mujoco_module

    mujoco = mujoco_module
    return mujoco_module


def gripper_open_default() -> float:
    return float(DEFAULT_ASSETS.platform.gripper_open_m if DEFAULT_ASSETS.platform.gripper_open_m is not None else 0.1)


def gripper_closed_default() -> float:
    return float(DEFAULT_ASSETS.platform.gripper_closed_m if DEFAULT_ASSETS.platform.gripper_closed_m is not None else 0.0)


def load_eef(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load valid robot-base EEF samples from preprocess_export_eef.py output."""
    data = json.loads(path.read_text(encoding="utf-8"))
    idxs, times, pos, quat, grasps = [], [], [], [], []
    for i, rec in enumerate(data.get("records", [])):
        if not rec.get("valid"):
            continue
        ee = rec["T_ee_in_base"]
        T = np.asarray(ee["T"], dtype=np.float64)
        stamp = rec.get("ts")
        idxs.append(int(rec.get("idx", i)))
        times.append(float(stamp) * 1e-9 if stamp is not None else i / 30.0)
        pos.append(T[:3, 3])
        quat.append(ee.get("quat_xyzw", R.from_matrix(T[:3, :3]).as_quat()))
        grasp = rec.get("grasp", 0)
        grasps.append(1 if float(grasp) > 0.5 else 0)
    if not pos:
        raise RuntimeError(f"No valid EEF records found in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return (
        np.asarray(idxs, dtype=np.int32),
        t,
        np.asarray(pos, dtype=np.float64),
        np.asarray(quat, dtype=np.float64),
        np.asarray(grasps, dtype=np.int32),
    )


def grasp_to_width_m(grasp: np.ndarray | float, open_m: float, closed_m: float) -> np.ndarray | float:
    open_m = float(np.clip(open_m, 0.0, 0.1))
    closed_m = float(np.clip(closed_m, 0.0, 0.1))
    if np.isscalar(grasp) or (isinstance(grasp, np.ndarray) and grasp.ndim == 0):
        return closed_m if float(grasp) > 0.5 else open_m
    g = np.asarray(grasp, dtype=np.float64)
    return np.where(g > 0.5, closed_m, open_m).astype(np.float64)


def joint_qpos_addrs(model, names: tuple[str, ...]) -> np.ndarray:
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def joint_dof_indices(model, names: tuple[str, ...]) -> set[int]:
    dofs = set()
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        dofs.add(int(model.jnt_dofadr[jid]))
    return dofs


def joint_bounds(model, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        if model.jnt_limited[jid]:
            jlo, jhi = model.jnt_range[jid]
        else:
            jlo, jhi = -np.pi, np.pi
        lo.append(float(jlo))
        hi.append(float(jhi))
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def actuator_ids(model, names: tuple[str, ...]) -> dict[str, int]:
    out = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out[name] = int(aid)
    return out


def target_frame_id(model, frame_type: str, frame_name: str) -> int:
    obj_type = mujoco.mjtObj.mjOBJ_SITE if frame_type == "site" else mujoco.mjtObj.mjOBJ_BODY
    frame_id = mujoco.mj_name2id(model, obj_type, frame_name)
    if frame_id < 0:
        raise RuntimeError(f"Missing {frame_type} in scene: {frame_name}")
    return int(frame_id)


def frame_pose(data, frame_type: str, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_type == "site":
        return data.site_xpos[frame_id].copy(), data.site_xmat[frame_id].reshape(3, 3).copy()
    return data.xpos[frame_id].copy(), data.xmat[frame_id].reshape(3, 3).copy()


def write_arm_qpos(data, arm_addrs: np.ndarray, q: np.ndarray) -> None:
    data.qpos[arm_addrs] = np.asarray(q, dtype=np.float64)


def set_gripper_width(data, gripper_addrs: np.ndarray, act_ids: dict[str, int], width_m: float) -> float:
    width_m = float(np.clip(width_m, 0.0, 0.1))
    finger = 0.5 * width_m
    if len(gripper_addrs) == 2:
        data.qpos[gripper_addrs[0]] = finger
        data.qpos[gripper_addrs[1]] = -finger
    if "gripper_joint1_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint1_pos"]] = finger
    if "gripper_joint2_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint2_pos"]] = -finger
    return width_m


def set_actuator_targets(data, act_ids: dict[str, int], q_arm: np.ndarray, gripper_width_m: float) -> None:
    for i, val in enumerate(q_arm, start=1):
        aid = act_ids.get(f"joint{i}_pos")
        if aid is not None:
            data.ctrl[aid] = float(val)
    if "gripper_joint1_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint1_pos"]] = 0.5 * gripper_width_m
    if "gripper_joint2_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint2_pos"]] = -0.5 * gripper_width_m


def orientation_error_rotvec(actual_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    return R.from_matrix(np.asarray(target_rot, dtype=np.float64) @ np.asarray(actual_rot, dtype=np.float64).T).as_rotvec()


def seed_candidates(
    q_warm: np.ndarray,
    q_zero: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    n_random: int,
    rng: np.random.Generator,
) -> list[tuple[str, np.ndarray]]:
    seeds = [("warm", q_warm.copy())]
    if not np.allclose(q_warm, q_zero):
        seeds.append(("zero", q_zero.copy()))
    lo, hi = bounds
    for i in range(max(0, int(n_random))):
        seeds.append((f"random{i}", rng.uniform(lo, hi)))
    return seeds


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    eef_path = as_abs(args.eef)
    frame_idx, times, target_pos, target_quat, grasp = load_eef(eef_path)
    stop = min(args.end, len(frame_idx)) if args.end >= 0 else len(frame_idx)
    start = max(args.start, 0)
    if start >= stop:
        raise ValueError(f"empty frame range: start={start}, end={stop}")
    frame_idx = frame_idx[start:stop]
    times = times[start:stop]
    target_pos = target_pos[start:stop]
    target_quat = target_quat[start:stop]
    grasp = grasp[start:stop]
    return {
        "scene_path": as_abs(args.scene),
        "eef_path": eef_path,
        "out_path": as_abs(args.out),
        "frame_idx": frame_idx,
        "time_s": times,
        "target_pos_m": target_pos,
        "target_quat_xyzw": target_quat,
        "target_rot": R.from_quat(target_quat).as_matrix(),
        "grasp": grasp,
        "gripper_width_m": grasp_to_width_m(grasp, args.gripper_open_m, args.gripper_closed_m),
    }


def save_result(args: argparse.Namespace, prepared: dict[str, Any], result: dict[str, Any]) -> None:
    out_path = prepared["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_idx = prepared["frame_idx"]
    pos_err = np.asarray(result["pos_err_m"], dtype=np.float64)
    ang_err = np.asarray(result["ang_err_deg"], dtype=np.float64)
    success = np.asarray(result["success"], dtype=bool)
    elapsed_s = float(result.get("elapsed_s", 0.0))

    np.savez(
        out_path,
        frame_idx=frame_idx,
        time_s=prepared["time_s"],
        joint_names=np.asarray(ARM_JOINTS),
        joint_qpos=result["joint_qpos"],
        full_qpos=result["full_qpos"],
        ctrl=result["ctrl"],
        gripper_width_m=prepared["gripper_width_m"],
        grasp=prepared["grasp"],
        target_type=np.asarray(args.target_type),
        target_name=np.asarray(args.target_name),
        target_pos_m=prepared["target_pos_m"],
        target_quat_xyzw=prepared["target_quat_xyzw"],
        achieved_pos_m=result["achieved_pos_m"],
        achieved_quat_xyzw=result["achieved_quat_xyzw"],
        pos_err_m=pos_err,
        ang_err_deg=ang_err,
        nfev=np.asarray(result["nfev"], dtype=np.int32),
        success=success,
        seed_label=np.asarray(result["seed_label"]),
        source_eef=np.asarray(str(prepared["eef_path"])),
        scene=np.asarray(str(prepared["scene_path"])),
        solver=np.asarray(str(result["solver"])),
        gripper_open_m=np.asarray(float(args.gripper_open_m)),
        gripper_closed_m=np.asarray(float(args.gripper_closed_m)),
    )

    meta = {
        "solver": str(result["solver"]),
        "scene": str(prepared["scene_path"]),
        "eef": str(prepared["eef_path"]),
        "out": str(out_path),
        "frames": int(len(frame_idx)),
        "frame_start": int(frame_idx[0]),
        "frame_end": int(frame_idx[-1]),
        "target_type": args.target_type,
        "target_name": args.target_name,
        "joint_names": list(ARM_JOINTS),
        "gripper_open_m": float(args.gripper_open_m),
        "gripper_closed_m": float(args.gripper_closed_m),
        "grasp_closed_frames": int(np.count_nonzero(prepared["grasp"])),
        "grasp_open_frames": int(len(frame_idx) - np.count_nonzero(prepared["grasp"])),
        "pos_weight": float(args.pos_weight),
        "ori_weight": float(args.ori_weight),
        "smooth_weight": float(args.smooth_weight),
        "pos_err_mean_mm": float(np.mean(pos_err) * 1000.0),
        "pos_err_max_mm": float(np.max(pos_err) * 1000.0),
        "ang_err_mean_deg": float(np.mean(ang_err)),
        "ang_err_max_deg": float(np.max(ang_err)),
        "success_frames": int(np.count_nonzero(success)),
        "elapsed_s": elapsed_s,
    }
    meta.update(result.get("meta", {}))
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("--- Retarget summary ---")
    print(f"solver: {result['solver']}")
    print(f"saved:  {out_path}")
    print(f"meta:   {out_path.with_suffix('.json')}")
    print(f"pos mean/max: {meta['pos_err_mean_mm']:.1f}/{meta['pos_err_max_mm']:.1f} mm")
    print(f"ang mean/max: {meta['ang_err_mean_deg']:.2f}/{meta['ang_err_max_deg']:.2f} deg")
    print(
        f"grasp open/closed frames: {meta['grasp_open_frames']}/{meta['grasp_closed_frames']}  "
        f"width open/closed: {args.gripper_open_m:.3f}/{args.gripper_closed_m:.3f} m"
    )


def add_common_args(parser: argparse.ArgumentParser, *, default_out: str) -> None:
    parser.add_argument("--scene", default=str(MUJOCO_NERO_SCENE))
    parser.add_argument("--eef", default=DEFAULT_EEF)
    parser.add_argument("--out", default=default_out)
    parser.add_argument("--target-type", choices=["site", "body"], default="site")
    parser.add_argument("--target-name", default="tcp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--ori-weight", type=float, default=4.0)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--ang-score-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-q", type=float, default=0.0)
    parser.add_argument("--gripper-open-m", type=float, default=gripper_open_default())
    parser.add_argument("--gripper-closed-m", type=float, default=gripper_closed_default())
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--gl-backend", choices=["auto", "glfw", "egl", "osmesa"], default="auto")


def allocate_result_arrays(count: int, model) -> dict[str, Any]:
    return {
        "joint_qpos": np.zeros((count, len(ARM_JOINTS)), dtype=np.float64),
        "full_qpos": np.zeros((count, model.nq), dtype=np.float64),
        "ctrl": np.zeros((count, model.nu), dtype=np.float64),
        "achieved_pos_m": np.zeros((count, 3), dtype=np.float64),
        "achieved_quat_xyzw": np.zeros((count, 4), dtype=np.float64),
        "pos_err_m": np.zeros(count, dtype=np.float64),
        "ang_err_deg": np.zeros(count, dtype=np.float64),
        "nfev": np.zeros(count, dtype=np.int32),
        "success": np.zeros(count, dtype=bool),
        "seed_label": [],
        "t0": time.time(),
    }

