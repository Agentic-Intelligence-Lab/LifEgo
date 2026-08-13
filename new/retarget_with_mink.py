#!/usr/bin/env python3
"""Retarget exported EEF targets to Nero joint IK with mink."""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils_retarget import (
    ARM_JOINTS,
    GRIPPER_JOINTS,
    actuator_ids,
    add_common_args,
    allocate_result_arrays,
    frame_pose,
    joint_bounds,
    joint_dof_indices,
    joint_qpos_addrs,
    load_runtime,
    orientation_error_rotvec,
    prepare_inputs,
    save_result,
    set_actuator_targets,
    set_gripper_width,
    target_frame_id,
    write_arm_qpos,
)

mujoco = None


def import_mink():
    try:
        import mink
    except ImportError as exc:
        raise ImportError(
            "mink is not installed in this Python environment. Install mink before "
            "running retarget_with_mink.py, or use new/retarget_with_scipy.py."
        ) from exc
    return mink


def se3_from_pos_rot(mink, pos: np.ndarray, rot: np.ndarray):
    return mink.SE3.from_rotation_and_translation(
        rotation=mink.SO3.from_matrix(np.asarray(rot, dtype=np.float64).reshape(3, 3)),
        translation=np.asarray(pos, dtype=np.float64),
    )


def posture_task(mink, model, q0: np.ndarray, active_joints: tuple[str, ...], active_cost: float):
    active_dofs = joint_dof_indices(model, active_joints)
    cost = np.full(model.nv, 80.0, dtype=np.float64)
    for dof in active_dofs:
        cost[dof] = float(active_cost)
    task = mink.PostureTask(model, cost=cost)
    task.set_target(q0)
    return task


def solve_frame(
    mink,
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
    frame_name: str,
    frame_id: int,
    bounds: tuple[np.ndarray, np.ndarray],
    gripper_width_m: float,
    pos_weight: float,
    ori_weight: float,
    smooth_weight: float,
    n_iter: int,
    dt: float,
    solver: str,
    damping: float,
    lm_damping: float,
) -> tuple[np.ndarray, dict]:
    del bounds
    data.qpos[:] = q_base
    write_arm_qpos(data, arm_addrs, q_seed)
    width = set_gripper_width(data, gripper_addrs, act_ids, gripper_width_m)
    set_actuator_targets(data, act_ids, q_seed, width)
    mujoco.mj_forward(model, data)
    q0 = data.qpos.copy()

    configuration = mink.Configuration(model, q0)
    frame_task = mink.FrameTask(
        frame_name=frame_name,
        frame_type=frame_type,
        position_cost=float(pos_weight),
        orientation_cost=float(ori_weight),
        lm_damping=float(lm_damping),
    )
    frame_task.set_target(se3_from_pos_rot(mink, target_pos, target_rot))
    posture = posture_task(mink, model, q_ref_full(model, q0, arm_addrs, q_ref), ARM_JOINTS, active_cost=smooth_weight)
    limits = [mink.ConfigurationLimit(model)]

    active_solver = solver
    success = True
    status = active_solver
    nfev = 0
    for _ in range(int(n_iter)):
        try:
            vel = mink.solve_ik(
                configuration,
                [frame_task, posture],
                float(dt),
                active_solver,
                damping=float(damping),
                limits=limits,
            )
        except Exception as exc:  # noqa: BLE001
            if active_solver != "quadprog":
                active_solver = "quadprog"
                vel = mink.solve_ik(
                    configuration,
                    [frame_task, posture],
                    float(dt),
                    active_solver,
                    damping=float(damping),
                    limits=limits,
                )
                status = "fallback_quadprog"
            else:
                success = False
                status = f"failed:{type(exc).__name__}"
                break
        configuration.integrate_inplace(vel, float(dt))
        nfev += 1

    data.qpos[:] = q_base
    q_sol_full = configuration.q.copy()
    q_sol = np.asarray(q_sol_full[arm_addrs], dtype=np.float64)
    write_arm_qpos(data, arm_addrs, q_sol)
    width = set_gripper_width(data, gripper_addrs, act_ids, gripper_width_m)
    set_actuator_targets(data, act_ids, q_sol, width)
    mujoco.mj_forward(model, data)
    actual_pos, actual_rot = frame_pose(data, frame_type, frame_id)
    metrics = {
        "success": bool(success),
        "status": status,
        "cost": float(np.linalg.norm(actual_pos - target_pos)),
        "nfev": int(nfev),
        "pos_err_m": float(np.linalg.norm(actual_pos - target_pos)),
        "ang_err_deg": float(np.linalg.norm(orientation_error_rotvec(actual_rot, target_rot)) * 180.0 / np.pi),
        "achieved_pos_m": actual_pos,
        "achieved_quat_xyzw": R.from_matrix(actual_rot).as_quat(),
    }
    return q_sol.copy(), metrics


def q_ref_full(model, q0: np.ndarray, arm_addrs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    q = np.asarray(q0, dtype=np.float64).copy()
    q[arm_addrs] = np.asarray(q_ref, dtype=np.float64)
    return q


def solve_trajectory(args: argparse.Namespace) -> None:
    global mujoco
    mujoco = load_runtime(args.gl_backend)
    mink = import_mink()
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

    count = len(prepared["frame_idx"])
    out = allocate_result_arrays(count, model)
    status_labels = []
    for i in range(count):
        q_sol, metrics = solve_frame(
            mink,
            model,
            data,
            q_base=q_base,
            q_seed=q_warm,
            q_ref=q_warm,
            target_pos=prepared["target_pos_m"][i],
            target_rot=prepared["target_rot"][i],
            arm_addrs=arm_addrs,
            gripper_addrs=gripper_addrs,
            act_ids=act_ids,
            frame_type=args.target_type,
            frame_name=args.target_name,
            frame_id=frame_id,
            bounds=bounds,
            gripper_width_m=float(prepared["gripper_width_m"][i]),
            pos_weight=args.pos_weight,
            ori_weight=args.ori_weight,
            smooth_weight=args.smooth_weight,
            n_iter=args.n_iter,
            dt=args.dt,
            solver=args.solver,
            damping=args.damping,
            lm_damping=args.lm_damping,
        )
        q_warm = q_sol.copy()
        out["joint_qpos"][i] = q_sol
        out["full_qpos"][i] = data.qpos.copy()
        out["ctrl"][i] = data.ctrl.copy()
        out["achieved_pos_m"][i] = metrics["achieved_pos_m"]
        out["achieved_quat_xyzw"][i] = metrics["achieved_quat_xyzw"]
        out["pos_err_m"][i] = metrics["pos_err_m"]
        out["ang_err_deg"][i] = metrics["ang_err_deg"]
        out["nfev"][i] = metrics["nfev"]
        out["success"][i] = metrics["success"]
        out["seed_label"].append("warm")
        status_labels.append(str(metrics["status"]))

        if args.progress_every > 0 and ((i + 1) % args.progress_every == 0 or i == count - 1):
            print(
                f"frame {int(prepared['frame_idx'][i]):5d} ({i + 1}/{count}) "
                f"pos {out['pos_err_m'][i] * 1000:6.1f}mm  ang {out['ang_err_deg'][i]:6.2f}deg  "
                f"{metrics['status']}  {time.time() - out['t0']:.1f}s"
            )

    out["solver"] = "mink"
    out["elapsed_s"] = time.time() - out["t0"]
    out["meta"] = {
        "mink_solver": args.solver,
        "mink_status_labels": status_labels,
        "n_iter": int(args.n_iter),
        "dt": float(args.dt),
        "damping": float(args.damping),
        "lm_damping": float(args.lm_damping),
    }
    save_result(args, prepared, out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_out="outputs/new_pipeline/ego_nero_easy/nero_eef_ik_mink/nero_eef_ik.npz")
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--solver", default="daqp")
    parser.add_argument("--damping", type=float, default=1e-2)
    parser.add_argument("--lm-damping", type=float, default=1e-2)
    args = parser.parse_args()
    solve_trajectory(args)


if __name__ == "__main__":
    main()
