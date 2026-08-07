#!/usr/bin/env python3
"""Solve a compact Nero arm IK trajectory from exported HumanEgo/WiLoR EEF poses."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
mujoco = None

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
GRIPPER_JOINTS = ("gripper_joint1", "gripper_joint2")


def load_runtime(gl_backend: str) -> None:
    global mujoco
    if gl_backend != "auto":
        os.environ["MUJOCO_GL"] = gl_backend
    import mujoco as mujoco_module

    mujoco = mujoco_module


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_eef(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    idxs = []
    times = []
    pos = []
    rot = []
    for i, rec in enumerate(data.get("records", [])):
        if not rec.get("valid"):
            continue
        T = np.asarray(rec["T_ee_in_base"]["T"], dtype=np.float64)
        stamp = rec.get("ts")
        idxs.append(int(rec.get("idx", i)))
        times.append(float(stamp) * 1e-9 if stamp is not None else i / 30.0)
        pos.append(T[:3, 3])
        rot.append(R.from_matrix(T[:3, :3]).as_quat())
    if not pos:
        raise RuntimeError(f"No valid EEF records found in {path}")
    t = np.asarray(times, dtype=np.float64)
    t -= t[0]
    return (
        np.asarray(idxs, dtype=np.int32),
        t,
        np.asarray(pos, dtype=np.float64),
        np.asarray(rot, dtype=np.float64),
    )


def joint_qpos_addrs(model: mujoco.MjModel, names: tuple[str, ...]) -> np.ndarray:
    addrs = []
    for name in names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint in scene: {name}")
        addrs.append(int(model.jnt_qposadr[jid]))
    return np.asarray(addrs, dtype=np.int32)


def joint_bounds(model: mujoco.MjModel, names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    lo = []
    hi = []
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


def actuator_ids(model: mujoco.MjModel, names: tuple[str, ...]) -> dict[str, int]:
    out = {}
    for name in names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            out[name] = int(aid)
    return out


def target_frame_id(model: mujoco.MjModel, frame_type: str, frame_name: str) -> int:
    obj_type = mujoco.mjtObj.mjOBJ_SITE if frame_type == "site" else mujoco.mjtObj.mjOBJ_BODY
    frame_id = mujoco.mj_name2id(model, obj_type, frame_name)
    if frame_id < 0:
        raise RuntimeError(f"Missing {frame_type} in scene: {frame_name}")
    return int(frame_id)


def frame_pose(data: mujoco.MjData, frame_type: str, frame_id: int) -> tuple[np.ndarray, np.ndarray]:
    if frame_type == "site":
        return data.site_xpos[frame_id].copy(), data.site_xmat[frame_id].reshape(3, 3).copy()
    return data.xpos[frame_id].copy(), data.xmat[frame_id].reshape(3, 3).copy()


def write_arm_qpos(data: mujoco.MjData, arm_addrs: np.ndarray, q: np.ndarray) -> None:
    data.qpos[arm_addrs] = np.asarray(q, dtype=np.float64)


def set_gripper_width(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    gripper_addrs: np.ndarray,
    act_ids: dict[str, int],
    width_m: float,
) -> float:
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


def set_actuator_targets(data: mujoco.MjData, act_ids: dict[str, int], q: np.ndarray, gripper_width_m: float) -> None:
    for i, val in enumerate(q, start=1):
        aid = act_ids.get(f"joint{i}_pos")
        if aid is not None:
            data.ctrl[aid] = float(val)
    if "gripper_joint1_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint1_pos"]] = 0.5 * gripper_width_m
    if "gripper_joint2_pos" in act_ids:
        data.ctrl[act_ids["gripper_joint2_pos"]] = -0.5 * gripper_width_m


def orientation_error_rotvec(actual_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    return R.from_matrix(np.asarray(target_rot, dtype=np.float64) @ np.asarray(actual_rot, dtype=np.float64).T).as_rotvec()


def solve_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    q_base: np.ndarray,
    q_seed: np.ndarray,
    q_ref: np.ndarray,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    arm_addrs: np.ndarray,
    gripper_addrs: np.ndarray,
    act_ids: dict[str, int],
    frame_type: str,
    frame_id: int,
    bounds: tuple[np.ndarray, np.ndarray],
    gripper_width_m: float,
    pos_weight: float,
    ori_weight: float,
    smooth_weight: float,
    max_nfev: int,
) -> tuple[np.ndarray, dict]:
    sqrt_pos = float(np.sqrt(max(pos_weight, 0.0)))
    sqrt_ori = float(np.sqrt(max(ori_weight, 0.0)))
    sqrt_smooth = float(np.sqrt(max(smooth_weight, 0.0)))
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_rot = np.asarray(target_rot, dtype=np.float64).reshape(3, 3)

    def residual(q_arm: np.ndarray) -> np.ndarray:
        data.qpos[:] = q_base
        write_arm_qpos(data, arm_addrs, q_arm)
        width = set_gripper_width(model, data, gripper_addrs, act_ids, gripper_width_m)
        set_actuator_targets(data, act_ids, q_arm, width)
        mujoco.mj_forward(model, data)
        actual_pos, actual_rot = frame_pose(data, frame_type, frame_id)
        return np.concatenate(
            [
                sqrt_pos * (actual_pos - target_pos),
                sqrt_ori * orientation_error_rotvec(actual_rot, target_rot),
                sqrt_smooth * (q_arm - q_ref),
            ]
        )

    res = least_squares(
        residual,
        np.asarray(q_seed, dtype=np.float64),
        bounds=bounds,
        max_nfev=max_nfev,
        xtol=1e-6,
        ftol=1e-6,
        gtol=1e-6,
    )
    data.qpos[:] = q_base
    write_arm_qpos(data, arm_addrs, res.x)
    width = set_gripper_width(model, data, gripper_addrs, act_ids, gripper_width_m)
    set_actuator_targets(data, act_ids, res.x, width)
    mujoco.mj_forward(model, data)
    actual_pos, actual_rot = frame_pose(data, frame_type, frame_id)
    metrics = {
        "success": bool(res.success),
        "status": int(res.status),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "pos_err_m": float(np.linalg.norm(actual_pos - target_pos)),
        "ang_err_deg": float(np.linalg.norm(orientation_error_rotvec(actual_rot, target_rot)) * 180.0 / np.pi),
        "achieved_pos_m": actual_pos,
        "achieved_quat_xyzw": R.from_matrix(actual_rot).as_quat(),
    }
    return res.x.copy(), metrics


def seed_candidates(q_warm: np.ndarray, q_zero: np.ndarray, bounds: tuple[np.ndarray, np.ndarray], n_random: int, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    seeds = [("warm", q_warm.copy())]
    if not np.allclose(q_warm, q_zero):
        seeds.append(("zero", q_zero.copy()))
    lo, hi = bounds
    for i in range(max(0, n_random)):
        seeds.append((f"random{i}", rng.uniform(lo, hi)))
    return seeds


def solve_trajectory(args: argparse.Namespace) -> None:
    scene_path = as_abs(args.scene)
    eef_path = as_abs(args.eef)
    out_path = as_abs(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_idx, times, target_pos, target_quat = load_eef(eef_path)
    if args.end >= 0:
        stop = min(args.end, len(frame_idx))
    else:
        stop = len(frame_idx)
    start = max(args.start, 0)
    if start >= stop:
        raise ValueError(f"empty frame range: start={start}, end={stop}")
    frame_idx = frame_idx[start:stop]
    times = times[start:stop]
    target_pos = target_pos[start:stop]
    target_quat = target_quat[start:stop]
    target_rot = R.from_quat(target_quat).as_matrix()

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    arm_addrs = joint_qpos_addrs(model, ARM_JOINTS)
    gripper_addrs = joint_qpos_addrs(model, GRIPPER_JOINTS)
    bounds = joint_bounds(model, ARM_JOINTS)
    frame_id = target_frame_id(model, args.target_type, args.target_name)
    act_ids = actuator_ids(model, tuple(f"joint{i}_pos" for i in range(1, 8)) + ("gripper_joint1_pos", "gripper_joint2_pos"))

    q_base = data.qpos.copy()
    q_zero = np.asarray([args.initial_q] * len(ARM_JOINTS), dtype=np.float64)
    q_zero = np.clip(q_zero, bounds[0], bounds[1])
    q_warm = q_zero.copy()
    rng = np.random.default_rng(args.seed)

    count = len(frame_idx)
    joint_q = np.zeros((count, len(ARM_JOINTS)), dtype=np.float64)
    full_qpos = np.zeros((count, model.nq), dtype=np.float64)
    ctrl = np.zeros((count, model.nu), dtype=np.float64)
    achieved_pos = np.zeros((count, 3), dtype=np.float64)
    achieved_quat = np.zeros((count, 4), dtype=np.float64)
    pos_err = np.zeros(count, dtype=np.float64)
    ang_err = np.zeros(count, dtype=np.float64)
    nfev = np.zeros(count, dtype=np.int32)
    success = np.zeros(count, dtype=bool)
    seed_label = []

    t0 = time.time()
    for i in range(count):
        best = None
        for label, q_seed in seed_candidates(q_warm, q_zero, bounds, args.random_seeds if i == 0 else 0, rng):
            q_sol, metrics = solve_frame(
                model,
                data,
                q_base=q_base,
                q_seed=q_seed,
                q_ref=q_warm,
                target_pos=target_pos[i],
                target_rot=target_rot[i],
                arm_addrs=arm_addrs,
                gripper_addrs=gripper_addrs,
                act_ids=act_ids,
                frame_type=args.target_type,
                frame_id=frame_id,
                bounds=bounds,
                gripper_width_m=args.gripper_width_m,
                pos_weight=args.pos_weight,
                ori_weight=args.ori_weight,
                smooth_weight=args.smooth_weight,
                max_nfev=args.max_nfev,
            )
            score = metrics["pos_err_m"] + args.ang_score_weight * np.radians(metrics["ang_err_deg"])
            if best is None or score < best[0]:
                best = (score, label, q_sol, metrics, data.qpos.copy(), data.ctrl.copy())
        assert best is not None
        _, label, q_sol, metrics, qpos_snapshot, ctrl_snapshot = best
        q_warm = q_sol.copy()
        joint_q[i] = q_sol
        full_qpos[i] = qpos_snapshot
        ctrl[i] = ctrl_snapshot
        achieved_pos[i] = metrics["achieved_pos_m"]
        achieved_quat[i] = metrics["achieved_quat_xyzw"]
        pos_err[i] = metrics["pos_err_m"]
        ang_err[i] = metrics["ang_err_deg"]
        nfev[i] = metrics["nfev"]
        success[i] = metrics["success"]
        seed_label.append(label)

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or i == count - 1):
            print(
                f"frame {int(frame_idx[i]):5d} ({i + 1}/{count}) "
                f"pos {pos_err[i] * 1000:6.1f}mm  ang {ang_err[i]:6.2f}deg  "
                f"seed {label}  {time.time() - t0:.1f}s"
            )

    np.savez(
        out_path,
        frame_idx=frame_idx,
        time_s=times,
        joint_names=np.asarray(ARM_JOINTS),
        joint_qpos=joint_q,
        full_qpos=full_qpos,
        ctrl=ctrl,
        gripper_width_m=np.full(count, args.gripper_width_m, dtype=np.float64),
        target_type=np.asarray(args.target_type),
        target_name=np.asarray(args.target_name),
        target_pos_m=target_pos,
        target_quat_xyzw=target_quat,
        achieved_pos_m=achieved_pos,
        achieved_quat_xyzw=achieved_quat,
        pos_err_m=pos_err,
        ang_err_deg=ang_err,
        nfev=nfev,
        success=success,
        seed_label=np.asarray(seed_label),
        source_eef=np.asarray(str(eef_path)),
        scene=np.asarray(str(scene_path)),
    )
    meta = {
        "scene": str(scene_path),
        "eef": str(eef_path),
        "out": str(out_path),
        "frames": int(count),
        "frame_start": int(frame_idx[0]),
        "frame_end": int(frame_idx[-1]),
        "target_type": args.target_type,
        "target_name": args.target_name,
        "joint_names": list(ARM_JOINTS),
        "gripper_width_m": float(args.gripper_width_m),
        "pos_weight": float(args.pos_weight),
        "ori_weight": float(args.ori_weight),
        "smooth_weight": float(args.smooth_weight),
        "pos_err_mean_mm": float(np.mean(pos_err) * 1000.0),
        "pos_err_max_mm": float(np.max(pos_err) * 1000.0),
        "ang_err_mean_deg": float(np.mean(ang_err)),
        "ang_err_max_deg": float(np.max(ang_err)),
        "success_frames": int(np.count_nonzero(success)),
        "elapsed_s": float(time.time() - t0),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("--- IK summary ---")
    print(f"saved: {out_path}")
    print(f"meta:  {out_path.with_suffix('.json')}")
    print(f"pos mean/max: {meta['pos_err_mean_mm']:.1f}/{meta['pos_err_max_mm']:.1f} mm")
    print(f"ang mean/max: {meta['ang_err_mean_deg']:.2f}/{meta['ang_err_max_deg']:.2f} deg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="outputs/mujoco_nero_scene/scene.xml")
    parser.add_argument("--eef", default="outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json")
    parser.add_argument("--out", default="outputs/ego_nero_easy/nero_eef_ik/nero_eef_ik.npz")
    parser.add_argument("--target-type", choices=["site", "body"], default="site")
    # Default: drive site:tcp to track HumanEgo/WiLoR EEF targets (not realbot poses).
    parser.add_argument("--target-name", default="tcp")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--pos-weight", type=float, default=100.0)
    parser.add_argument("--ori-weight", type=float, default=4.0)
    parser.add_argument("--smooth-weight", type=float, default=0.02)
    parser.add_argument("--ang-score-weight", type=float, default=0.02)
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--random-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--initial-q", type=float, default=0.0)
    parser.add_argument("--gripper-width-m", type=float, default=0.08)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--gl-backend", choices=["auto", "glfw", "egl", "osmesa"], default="auto")
    args = parser.parse_args()
    load_runtime(args.gl_backend)
    solve_trajectory(args)


if __name__ == "__main__":
    main()
