#!/usr/bin/env python3
"""Retarget exported EEF targets to Nero joint IK with SciPy least-squares."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from utils_retarget import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    actuator_ids,
    add_common_args,
    allocate_result_arrays,
    frame_pose,
    joint_bounds,
    joint_qpos_addrs,
    load_runtime,
    orientation_error_rotvec,
    prepare_inputs,
    save_result,
    seed_candidates,
    set_actuator_targets,
    set_gripper_width,
    target_frame_id,
    write_arm_qpos,
)

mujoco = None


def solve_frame(
    model,
    data,
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
        width = set_gripper_width(data, gripper_addrs, act_ids, gripper_width_m)
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
    width = set_gripper_width(data, gripper_addrs, act_ids, gripper_width_m)
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


def solve_trajectory(args: argparse.Namespace) -> None:
    global mujoco
    mujoco = load_runtime(args.gl_backend)
    prepared = prepare_inputs(args)
    model = mujoco.MjModel.from_xml_path(str(prepared["scene_path"]))
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

    count = len(prepared["frame_idx"])
    out = allocate_result_arrays(count, model)
    for i in range(count):
        best = None
        n_random = args.random_seeds if i == 0 else 0
        for label, q_seed in seed_candidates(q_warm, q_zero, bounds, n_random, rng):
            q_sol, metrics = solve_frame(
                model,
                data,
                q_base=q_base,
                q_seed=q_seed,
                q_ref=q_warm,
                target_pos=prepared["target_pos_m"][i],
                target_rot=prepared["target_rot"][i],
                arm_addrs=arm_addrs,
                gripper_addrs=gripper_addrs,
                act_ids=act_ids,
                frame_type=args.target_type,
                frame_id=frame_id,
                bounds=bounds,
                gripper_width_m=float(prepared["gripper_width_m"][i]),
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
        out["joint_qpos"][i] = q_sol
        out["full_qpos"][i] = qpos_snapshot
        out["ctrl"][i] = ctrl_snapshot
        out["achieved_pos_m"][i] = metrics["achieved_pos_m"]
        out["achieved_quat_xyzw"][i] = metrics["achieved_quat_xyzw"]
        out["pos_err_m"][i] = metrics["pos_err_m"]
        out["ang_err_deg"][i] = metrics["ang_err_deg"]
        out["nfev"][i] = metrics["nfev"]
        out["success"][i] = metrics["success"]
        out["seed_label"].append(label)

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or i == count - 1):
            print(
                f"frame {int(prepared['frame_idx'][i]):5d} ({i + 1}/{count}) "
                f"pos {out['pos_err_m'][i] * 1000:6.1f}mm  ang {out['ang_err_deg'][i]:6.2f}deg  "
                f"seed {label}  {time.time() - out['t0']:.1f}s"
            )

    out["solver"] = "scipy_least_squares"
    out["elapsed_s"] = time.time() - out["t0"]
    save_result(args, prepared, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_out="outputs/new_pipeline/ego_nero_easy/nero_eef_ik_scipy/nero_eef_ik.npz")
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--random-seeds", type=int, default=2)
    args = parser.parse_args()
    solve_trajectory(args)


if __name__ == "__main__":
    main()
