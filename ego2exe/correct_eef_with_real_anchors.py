#!/usr/bin/env python3
"""Task-specific EEF position correction from real-robot grasp anchors.

This is deliberately a standalone post-processing entry point.  It does not
change the WiLoR/export pipeline: it reads one exported
``robot_eef_trajectory.json`` and writes a corrected copy in a new directory.

The calibration target for every selected grasp event is sampled independently
from a regularized 3D Gaussian fitted to the same event in the real calibration
set.  EEF orientation is never changed.

Correction spaces:

``free_xyz``
    Match the sampled 3D anchor position directly.

``ray_depth``
    Move the EEF only along its original camera ray.  At an anchor, choose the
    depth whose point on that ray is closest to the sampled real position.

Propagation methods:

``linear``
    Piecewise-linear interpolation of anchor displacements.  Start/end are
    hard zero-displacement controls.

``min_bending``
    Hard start/end and anchor constraints; minimize the discrete squared
    second difference of the per-frame displacement.

``smooth_spline``
    Start/end and anchors are all weighted soft observations; minimize their
    residual plus bending and displacement-magnitude penalties.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

from assets import DEFAULT_ASSETS
from eval.traj_metrics import DEFAULT_TCP_OFFSET_M, grasp_events, load_real_jsonl


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT.parent / "DATA_new"

TASK_CONFIG = {
    # Anchor indices use the eval CLI convention: zero-based indices into the
    # grasp-toggle sequence.  For object tasks, events 1/2 are pickup close and
    # place open; bowl has only two events, so both are used.
    "stack_object_horizontal": {"anchors": [1, 2], "expected_events": 4},
    "stack_object_lean": {"anchors": [1, 2], "expected_events": 4},
    "stack_bowl": {"anchors": [0, 1], "expected_events": 2},
}


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def infer_real_dir(task: str, data_root: Path) -> Path:
    task_dir = data_root / task
    hits = sorted(p for p in task_dir.glob("*nero*") if p.is_dir())
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one real-robot directory matching '*nero*' under "
            f"{task_dir}, found {len(hits)}: {[str(p) for p in hits]}"
        )
    return hits[0]


def read_real_episodes(real_dir: Path, pose_key: str, tcp_offset_m: float) -> dict[str, Any]:
    episodes = {}
    for path in sorted(real_dir.glob("*.jsonl")):
        traj = load_real_jsonl(path, pose_key=pose_key, tcp_offset_m=tcp_offset_m)
        if traj is not None:
            episodes[path.name] = traj
    if not episodes:
        raise RuntimeError(f"No usable real-robot JSONL episodes found in {real_dir}")
    return episodes


def create_or_load_split(
    path: Path,
    *,
    task: str,
    real_dir: Path,
    episodes: dict[str, Any],
    calibration_size: int,
    eval_size: int,
    split_seed: int,
    expected_events: int,
    pose_key: str,
    tcp_offset_m: float,
) -> dict[str, Any]:
    """Create a stable 10/10 task split, prioritizing anchor-healthy calibration data.

    Some real sets contain episodes with fewer grasp toggles than the task mode
    (currently bowl has two such episodes).  Calibration must contain the full
    expected anchor tuple.  Such episodes may remain in eval/reserve, where the
    existing evaluator can report their event mismatch instead of allowing them
    to corrupt the fitted anchor distribution.
    """
    if path.is_file():
        split = json.loads(path.read_text(encoding="utf-8"))
        if split.get("task") != task:
            raise ValueError(f"Split manifest task is {split.get('task')!r}, expected {task!r}")
        groups = {name: list(split.get(name, [])) for name in ("calibration", "eval", "reserve")}
        known = set(episodes)
        missing = [name for values in groups.values() for name in values if name not in known]
        if missing:
            raise FileNotFoundError(f"Split manifest references missing real episodes: {missing}")
        assigned = [name for values in groups.values() for name in values]
        if len(assigned) != len(set(assigned)):
            raise ValueError(f"Split manifest contains duplicate/overlapping episodes: {path}")
        if set(assigned) != known:
            unassigned = sorted(known.difference(assigned))
            raise ValueError(f"Split manifest does not assign all real episodes: {unassigned}")
        if len(split.get("calibration", [])) != calibration_size or len(split.get("eval", [])) != eval_size:
            raise ValueError(
                f"Existing split has {len(split.get('calibration', []))}/"
                f"{len(split.get('eval', []))} calibration/eval episodes, requested "
                f"{calibration_size}/{eval_size}. Use a different --split-manifest."
            )
        return split

    healthy = sorted(name for name, traj in episodes.items() if len(traj.events) >= expected_events)
    unhealthy = sorted(name for name, traj in episodes.items() if len(traj.events) < expected_events)
    if len(healthy) < calibration_size:
        raise RuntimeError(
            f"Need {calibration_size} calibration episodes with >= {expected_events} grasp events, "
            f"but only {len(healthy)} are available in {real_dir}"
        )
    if len(episodes) < calibration_size + eval_size:
        raise RuntimeError(
            f"Need at least {calibration_size + eval_size} real episodes for the split, "
            f"but only {len(episodes)} are available"
        )

    rng = np.random.default_rng(split_seed)
    healthy = list(np.asarray(healthy, dtype=object)[rng.permutation(len(healthy))])
    unhealthy = list(np.asarray(unhealthy, dtype=object)[rng.permutation(len(unhealthy))]) if unhealthy else []
    calibration = healthy[:calibration_size]
    remaining_healthy = healthy[calibration_size:]
    # Maximize the number of anchor-healthy episodes in eval.  Bowl currently
    # has only 19 two-event episodes, so a strict 10/10 file split necessarily
    # puts one one-event episode in eval; its anchor metrics will be unavailable,
    # while it can still contribute to whole-trajectory metrics.
    evaluation = remaining_healthy[:eval_size]
    healthy_reserve = remaining_healthy[eval_size:]
    if len(evaluation) < eval_size:
        needed = eval_size - len(evaluation)
        evaluation.extend(unhealthy[:needed])
        unhealthy = unhealthy[needed:]
    evaluation = list(np.asarray(evaluation, dtype=object)[rng.permutation(len(evaluation))])
    reserve = healthy_reserve + unhealthy

    split = {
        "schema_version": 1,
        "task": task,
        "real_dir": str(real_dir.resolve()),
        "split_seed": int(split_seed),
        "calibration_size": int(calibration_size),
        "eval_size": int(eval_size),
        "expected_events": int(expected_events),
        "pose_key": pose_key,
        "tcp_offset_m": float(tcp_offset_m),
        "event_count_histogram": dict(sorted(Counter(len(t.events) for t in episodes.values()).items())),
        "calibration": calibration,
        "eval": evaluation,
        "reserve": reserve,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(split), indent=2), encoding="utf-8")
    return split


def stable_episode_seed(base_seed: int, task: str, episode_identity: str) -> int:
    """Derive a reproducible seed independent of output variant and invocation order."""
    payload = f"{base_seed}|{task}|{episode_identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def load_or_sample_targets(
    manifest_path: Path | None,
    *,
    target_key: str,
    task: str,
    split_path: Path,
    calibration: list[Any],
    anchor_indices: list[int],
    sample_seed: int,
    covariance_shrinkage: float,
    max_mahalanobis: float,
    pose_key: str,
    tcp_offset_m: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any], int]:
    """Load a frozen per-episode target tuple, or sample and persist it once.

    The manifest deliberately excludes correction space and interpolation
    method.  Consequently ray/XYZ and all three propagation methods receive
    exactly the same sampled real target for a given ego episode.
    """
    config = {
        "task": task,
        "split_manifest": str(split_path.resolve()),
        "split_manifest_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "anchor_indices_zero_based": list(anchor_indices),
        "sample_seed": int(sample_seed),
        "covariance_shrinkage": float(covariance_shrinkage),
        "max_mahalanobis": float(max_mahalanobis),
        "pose_key": pose_key,
        "tcp_offset_m": float(tcp_offset_m),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "config": config,
        "samples": {},
    }
    if manifest_path is not None and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported target manifest schema_version in {manifest_path}: "
                f"{manifest.get('schema_version')!r}"
            )
        if manifest.get("config") != config:
            raise ValueError(
                f"Target manifest configuration does not match this run: {manifest_path}\n"
                f"stored: {manifest.get('config')}\ncurrent: {config}\n"
                "Use a different --target-manifest for a different sampling configuration."
            )

    stored = manifest.setdefault("samples", {}).get(target_key)
    if stored is not None:
        raw_targets = stored.get("targets_m", {})
        missing = [k for k in anchor_indices if str(k) not in raw_targets]
        if missing:
            raise ValueError(
                f"Frozen target {target_key!r} in {manifest_path} is missing anchors {missing}"
            )
        targets = {k: np.asarray(raw_targets[str(k)], dtype=np.float64) for k in anchor_indices}
        return targets, stored.get("sampling_stats", {}), int(stored["derived_episode_seed"])

    episode_seed = stable_episode_seed(sample_seed, task, target_key)
    rng = np.random.default_rng(episode_seed)
    targets, sampling_stats = fit_and_sample_anchor_targets(
        calibration,
        anchor_indices,
        rng=rng,
        covariance_shrinkage=covariance_shrinkage,
        max_mahalanobis=max_mahalanobis,
    )
    if manifest_path is not None:
        manifest["samples"][target_key] = {
            "derived_episode_seed": int(episode_seed),
            "targets_m": {str(k): targets[k].tolist() for k in anchor_indices},
            "sampling_stats": jsonable(sampling_stats),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        tmp_path.write_text(json.dumps(jsonable(manifest), indent=2), encoding="utf-8")
        tmp_path.replace(manifest_path)
    return targets, sampling_stats, episode_seed


def regularized_gaussian(points: np.ndarray, shrinkage: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"Expected at least two 3D calibration points, got {points.shape}")
    mean = points.mean(axis=0)
    cov = np.asarray(np.cov(points, rowvar=False, ddof=1), dtype=np.float64)
    scale = max(float(np.trace(cov) / 3.0), 1e-10)
    cov = (1.0 - shrinkage) * cov + shrinkage * scale * np.eye(3)
    cov += 1e-10 * np.eye(3)
    return mean, cov


def sample_truncated_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    max_mahalanobis: float,
    max_attempts: int = 1000,
) -> tuple[np.ndarray, float, int]:
    inv = np.linalg.inv(cov)
    for attempt in range(1, max_attempts + 1):
        sample = rng.multivariate_normal(mean, cov)
        delta = sample - mean
        mahal = float(np.sqrt(max(0.0, delta @ inv @ delta)))
        if max_mahalanobis <= 0.0 or mahal <= max_mahalanobis:
            return sample, mahal, attempt
    raise RuntimeError(
        f"Failed to sample within Mahalanobis radius {max_mahalanobis} after {max_attempts} attempts"
    )


def fit_and_sample_anchor_targets(
    calibration: list[Any],
    anchor_indices: list[int],
    *,
    rng: np.random.Generator,
    covariance_shrinkage: float,
    max_mahalanobis: float,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    targets = {}
    stats = {}
    for k in anchor_indices:
        points = np.asarray([traj.pos[traj.events[k]] for traj in calibration], dtype=np.float64)
        mean, cov = regularized_gaussian(points, covariance_shrinkage)
        target, mahal, attempts = sample_truncated_gaussian(rng, mean, cov, max_mahalanobis)
        targets[k] = target
        stats[str(k)] = {
            "n": len(points),
            "mean_m": mean,
            "covariance_m2": cov,
            "sampled_target_m": target,
            "sample_mahalanobis": mahal,
            "sample_attempts": attempts,
        }
    return targets, stats


def load_ego_eef(path: Path) -> tuple[dict[str, Any], list[int], np.ndarray, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    record_indices, positions, grasps = [], [], []
    for ri, rec in enumerate(data.get("records", [])):
        if not rec.get("valid") or rec.get("T_ee_in_base") is None:
            continue
        T = np.asarray(rec["T_ee_in_base"]["T"], dtype=np.float64)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            continue
        grasp = rec.get("grasp")
        if grasp is None:
            continue
        record_indices.append(ri)
        positions.append(T[:3, 3])
        grasps.append(float(grasp))
    if not positions:
        raise RuntimeError(f"No valid EEF positions with grasp states found in {path}")
    return (
        data,
        record_indices,
        np.asarray(positions, dtype=np.float64),
        np.asarray(grasps, dtype=np.float64),
    )


def second_difference_matrix(n: int) -> csc_matrix:
    if n < 3:
        return csc_matrix((0, n), dtype=np.float64)
    one = np.ones(n - 2, dtype=np.float64)
    return diags((one, -2.0 * one, one), (0, 1, 2), shape=(n - 2, n), format="csc")


def merge_controls(indices: list[int], values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    merged: dict[int, np.ndarray] = {}
    for idx, value in zip(indices, values):
        value = np.asarray(value, dtype=np.float64)
        if idx in merged and not np.allclose(merged[idx], value, atol=1e-10):
            raise ValueError(f"Conflicting correction controls at trajectory index {idx}")
        merged[idx] = value
    ordered = np.asarray(sorted(merged), dtype=np.int32)
    return ordered, np.asarray([merged[int(i)] for i in ordered], dtype=np.float64)


def propagate_linear(n: int, control_idx: np.ndarray, control_values: np.ndarray) -> np.ndarray:
    grid = np.arange(n, dtype=np.float64)
    if control_values.ndim == 1:
        return np.interp(grid, control_idx, control_values)
    return np.column_stack(
        [np.interp(grid, control_idx, control_values[:, d]) for d in range(control_values.shape[1])]
    )


def propagate_min_bending(n: int, control_idx: np.ndarray, control_values: np.ndarray) -> np.ndarray:
    """Hard-constrained discrete minimum-bending displacement."""
    if n <= 2 or len(control_idx) >= n:
        return propagate_linear(n, control_idx, control_values)
    values_2d = control_values[:, None] if control_values.ndim == 1 else control_values
    L = second_difference_matrix(n)
    H = (L.T @ L).tocsc()
    fixed = np.zeros(n, dtype=bool)
    fixed[control_idx] = True
    free_idx = np.flatnonzero(~fixed)
    fixed_idx = np.flatnonzero(fixed)
    fixed_lookup = {int(idx): value for idx, value in zip(control_idx, values_2d)}
    fixed_values = np.asarray([fixed_lookup[int(i)] for i in fixed_idx], dtype=np.float64)
    Hff = H[free_idx][:, free_idx] + 1e-12 * eye(len(free_idx), format="csc")
    Hfc = H[free_idx][:, fixed_idx]
    out = np.zeros((n, values_2d.shape[1]), dtype=np.float64)
    out[fixed_idx] = fixed_values
    rhs = -(Hfc @ fixed_values)
    for d in range(values_2d.shape[1]):
        out[free_idx, d] = spsolve(Hff, rhs[:, d])
    return out[:, 0] if control_values.ndim == 1 else out


def propagate_smooth_spline(
    n: int,
    control_idx: np.ndarray,
    control_values: np.ndarray,
    *,
    endpoint_weight: float,
    anchor_weight: float,
    bend_weight: float,
    magnitude_weight: float,
) -> np.ndarray:
    """Weighted soft controls plus discrete spline bending regularization."""
    values_2d = control_values[:, None] if control_values.ndim == 1 else control_values
    weights = np.full(len(control_idx), float(anchor_weight), dtype=np.float64)
    weights[(control_idx == 0) | (control_idx == n - 1)] = float(endpoint_weight)
    obs_weight = np.zeros(n, dtype=np.float64)
    rhs = np.zeros((n, values_2d.shape[1]), dtype=np.float64)
    for idx, weight, value in zip(control_idx, weights, values_2d):
        obs_weight[int(idx)] += weight
        rhs[int(idx)] += weight * value
    L = second_difference_matrix(n)
    H = diags(obs_weight, 0, shape=(n, n), format="csc")
    H = H + float(bend_weight) * (L.T @ L)
    H = H + max(float(magnitude_weight), 1e-12) * eye(n, format="csc")
    out = np.column_stack([spsolve(H, rhs[:, d]) for d in range(values_2d.shape[1])])
    return out[:, 0] if control_values.ndim == 1 else out


def propagate(
    method: str,
    n: int,
    anchor_traj_idx: list[int],
    anchor_displacements: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = 1 if anchor_displacements.ndim == 1 else anchor_displacements.shape[1]
    zero = np.zeros(width, dtype=np.float64)
    anchor_values = anchor_displacements[:, None] if anchor_displacements.ndim == 1 else anchor_displacements
    control_idx, control_values = merge_controls(
        [0, *anchor_traj_idx, n - 1],
        np.vstack([zero, anchor_values, zero]),
    )
    if anchor_displacements.ndim == 1:
        control_values = control_values[:, 0]

    if method == "linear":
        correction = propagate_linear(n, control_idx, control_values)
    elif method == "min_bending":
        correction = propagate_min_bending(n, control_idx, control_values)
    elif method == "smooth_spline":
        correction = propagate_smooth_spline(
            n,
            control_idx,
            control_values,
            endpoint_weight=args.endpoint_weight,
            anchor_weight=args.anchor_weight,
            bend_weight=args.bend_weight,
            magnitude_weight=args.magnitude_weight,
        )
    else:  # argparse choices make this defensive only.
        raise ValueError(f"Unknown propagation method {method!r}")
    return correction, control_idx, control_values


def camera_transforms() -> tuple[np.ndarray, np.ndarray]:
    camera = DEFAULT_ASSETS.camera()
    if camera is None or camera.extrinsics.T_cam_in_base is None:
        raise RuntimeError("DEFAULT_ASSETS scene camera extrinsics are not available")
    T_cam_in_base = np.asarray(camera.extrinsics.T_cam_in_base, dtype=np.float64)
    return T_cam_in_base, np.linalg.inv(T_cam_in_base)


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def closest_depth_on_ray(
    point_base: np.ndarray,
    target_base: np.ndarray,
    T_cam_in_base: np.ndarray,
    T_base_in_cam: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    point_cam = transform_points(T_base_in_cam, point_base[None])[0]
    if abs(float(point_cam[2])) < 1e-8:
        raise ValueError("EEF anchor has near-zero camera depth")
    ray = point_cam / point_cam[2]
    A = T_cam_in_base[:3, :3] @ ray
    b = T_cam_in_base[:3, 3]
    depth = float(A @ (target_base - b) / max(float(A @ A), 1e-12))
    projected = b + A * depth
    residual = float(np.linalg.norm(projected - target_base))
    return depth, projected, residual


def compute_correction(
    positions: np.ndarray,
    events: np.ndarray,
    anchor_indices: list[int],
    targets: dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchor_traj_idx = [int(events[k]) for k in anchor_indices]
    anchor_meta = {}
    if args.space == "free_xyz":
        displacements = np.asarray(
            [targets[k] - positions[t] for k, t in zip(anchor_indices, anchor_traj_idx)],
            dtype=np.float64,
        )
        correction, controls, control_values = propagate(
            args.method, len(positions), anchor_traj_idx, displacements, args
        )
        corrected = positions + correction
        for row, (k, t) in enumerate(zip(anchor_indices, anchor_traj_idx)):
            anchor_meta[str(k)] = {
                "trajectory_index": t,
                "source_position_m": positions[t],
                "target_position_m": targets[k],
                "requested_displacement_base_m": displacements[row],
                "corrected_position_m": corrected[t],
                "target_residual_m": float(np.linalg.norm(corrected[t] - targets[k])),
            }
        depth_delta = None
    else:
        T_cam_in_base, T_base_in_cam = camera_transforms()
        source_cam = transform_points(T_base_in_cam, positions)
        source_depth = source_cam[:, 2]
        requested_depth_delta = []
        projected_targets = []
        projection_residuals = []
        for k, t in zip(anchor_indices, anchor_traj_idx):
            depth, projected, residual = closest_depth_on_ray(
                positions[t], targets[k], T_cam_in_base, T_base_in_cam
            )
            requested_depth_delta.append(depth - source_depth[t])
            projected_targets.append(projected)
            projection_residuals.append(residual)
        requested_depth_delta = np.asarray(requested_depth_delta, dtype=np.float64)
        depth_delta, controls, control_values = propagate(
            args.method, len(positions), anchor_traj_idx, requested_depth_delta, args
        )
        corrected_cam = source_cam.copy()
        for i in range(len(corrected_cam)):
            if abs(float(source_cam[i, 2])) < 1e-8:
                continue
            ray = source_cam[i] / source_cam[i, 2]
            corrected_cam[i] = ray * (source_cam[i, 2] + depth_delta[i])
        corrected = transform_points(T_cam_in_base, corrected_cam)
        correction = corrected - positions
        for row, (k, t) in enumerate(zip(anchor_indices, anchor_traj_idx)):
            anchor_meta[str(k)] = {
                "trajectory_index": t,
                "source_position_m": positions[t],
                "sampled_target_position_m": targets[k],
                "ray_projected_target_position_m": projected_targets[row],
                "sampled_target_to_ray_residual_m": projection_residuals[row],
                "requested_depth_delta_m": requested_depth_delta[row],
                "applied_depth_delta_m": depth_delta[t],
                "corrected_position_m": corrected[t],
                "sampled_target_residual_m": float(np.linalg.norm(corrected[t] - targets[k])),
            }

    details = {
        "anchor_details": anchor_meta,
        "control_trajectory_indices": controls,
        "control_values": control_values,
        "base_displacement_m": correction,
        "depth_displacement_m": depth_delta,
    }
    return corrected, details


def update_matrix_record(
    record: dict[str, Any], position: np.ndarray, rotation: np.ndarray | None = None
) -> None:
    T = np.asarray(record["T"], dtype=np.float64)
    T[:3, 3] = np.asarray(position, dtype=np.float64)
    if rotation is not None:
        T[:3, :3] = np.asarray(rotation, dtype=np.float64)
    record["T"] = T.tolist()
    record["translation_m"] = np.asarray(position, dtype=np.float64).tolist()
    if rotation is not None:
        rot = R.from_matrix(T[:3, :3])
        record["quat_xyzw"] = rot.as_quat().tolist()
        record["rotvec"] = rot.as_rotvec().tolist()


def write_outputs(
    out_dir: Path,
    source_data: dict[str, Any],
    record_indices: list[int],
    corrected_positions: np.ndarray,
    corrected_rotations: np.ndarray | None,
    details: dict[str, Any],
    metadata: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "robot_eef_trajectory.json"
    jsonl_path = out_dir / "robot_eef_trajectory.jsonl"
    csv_path = out_dir / "robot_eef_trajectory.csv"
    meta_path = out_dir / "anchor_correction_meta.json"
    existing = [p for p in (json_path, jsonl_path, csv_path, meta_path) if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output files already exist; pass --overwrite to replace them: {existing}")

    data = deepcopy(source_data)
    data.setdefault("metadata", {})["real_anchor_correction"] = jsonable(metadata)
    base_delta = np.asarray(details["base_displacement_m"], dtype=np.float64)
    depth_delta = details.get("depth_displacement_m")
    for local_i, (record_i, position) in enumerate(zip(record_indices, corrected_positions)):
        rec = data["records"][record_i]
        rec["T_ee_in_base_before_anchor_correction"] = deepcopy(rec["T_ee_in_base"])
        rotation = None if corrected_rotations is None else corrected_rotations[local_i]
        update_matrix_record(rec["T_ee_in_base"], position, rotation)
        rec["anchor_correction"] = {
            "base_displacement_m": base_delta[local_i].tolist(),
            "depth_displacement_m": None if depth_delta is None else float(depth_delta[local_i]),
        }

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in data.get("records", []):
            fh.write(json.dumps(rec) + "\n")

    fieldnames = [
        "idx", "ts", "valid", "hand_key", "confidence", "grasp",
        "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "rx", "ry", "rz",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in data.get("records", []):
            row = {key: None for key in fieldnames}
            row.update(
                idx=rec.get("idx"), ts=rec.get("ts"), valid=rec.get("valid", False),
                hand_key=rec.get("hand_key"), confidence=rec.get("confidence"), grasp=rec.get("grasp"),
            )
            if rec.get("valid") and rec.get("T_ee_in_base") is not None:
                ee = rec["T_ee_in_base"]
                pos = ee["translation_m"]
                quat = ee["quat_xyzw"]
                rotvec = ee["rotvec"]
                row.update(
                    x_m=pos[0], y_m=pos[1], z_m=pos[2],
                    qx=quat[0], qy=quat[1], qz=quat[2], qw=quat[3],
                    rx=rotvec[0], ry=rotvec[1], rz=rotvec[2],
                )
            writer.writerow(row)

    meta_path.write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=sorted(TASK_CONFIG))
    parser.add_argument("--eef", required=True, help="Source robot_eef_trajectory.json")
    parser.add_argument("--out", required=True, help="New output directory; source files are never modified")
    parser.add_argument("--real-dir", default=None, help="Real JSONL directory; inferred from DATA_new/<task> if omitted")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--anchors", type=int, nargs="+", default=None,
        help="Zero-based grasp-event indices. Defaults: bowl=0 1, horizontal/lean=1 2.",
    )
    parser.add_argument(
        "--position-correction", choices=("anchors", "none"), default="anchors",
        help="anchors applies real-anchor position correction; none preserves source positions",
    )
    parser.add_argument("--space", choices=("free_xyz", "ray_depth"), default="free_xyz")
    parser.add_argument("--method", choices=("linear", "min_bending", "smooth_spline"), default="min_bending")
    parser.add_argument("--rotation-correction", choices=("none", "global_so3"), default="none")
    parser.add_argument("--rotation-manifest", default=None,
                        help="Task-level manifest from fit_global_rotation_correction.py")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--calibration-size", type=int, default=10)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument(
        "--split-manifest", default=None,
        help="Default: outputs/anchor_calibration_splits/<task>.json",
    )
    parser.add_argument(
        "--target-manifest", default=None,
        help="Optional JSON file that freezes sampled targets across correction variants.",
    )
    parser.add_argument(
        "--target-key", default=None,
        help="Stable ego episode identity within --target-manifest. Default: source EEF path.",
    )
    parser.add_argument("--pose-key", default="tcp_tip_pose")
    parser.add_argument("--tcp-offset-m", type=float, default=DEFAULT_TCP_OFFSET_M)
    parser.add_argument("--covariance-shrinkage", type=float, default=0.1)
    parser.add_argument(
        "--max-mahalanobis", type=float, default=0.0,
        help="Optional Gaussian truncation radius; <=0 (default) uses the full Gaussian.",
    )
    parser.add_argument("--anchor-weight", type=float, default=1.0, help="smooth_spline only")
    parser.add_argument("--endpoint-weight", type=float, default=1.0, help="smooth_spline only")
    parser.add_argument("--bend-weight", type=float, default=1000.0, help="smooth_spline only")
    parser.add_argument("--magnitude-weight", type=float, default=1e-4, help="smooth_spline only")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.position_correction == "none" and args.rotation_correction == "none":
        raise ValueError(
            "--position-correction none with --rotation-correction none would make an unchanged copy"
        )
    task_cfg = TASK_CONFIG[args.task]
    anchor_indices = list(task_cfg["anchors"] if args.anchors is None else args.anchors)
    if not anchor_indices or any(k < 0 for k in anchor_indices) or len(set(anchor_indices)) != len(anchor_indices):
        raise ValueError(f"--anchors must contain unique non-negative indices, got {anchor_indices}")
    anchor_indices = sorted(anchor_indices)
    if not 0.0 <= args.covariance_shrinkage <= 1.0:
        raise ValueError("--covariance-shrinkage must be in [0, 1]")
    if min(args.anchor_weight, args.endpoint_weight, args.bend_weight, args.magnitude_weight) < 0.0:
        raise ValueError("smooth_spline weights must be non-negative")

    eef_path = as_abs(args.eef)
    out_dir = as_abs(args.out)
    if out_dir.resolve() == eef_path.parent.resolve():
        raise ValueError("--out must be a new directory, not the source EEF directory")
    data_root = as_abs(args.data_root)
    real_dir = as_abs(args.real_dir) if args.real_dir else infer_real_dir(args.task, data_root)
    split_path = (
        as_abs(args.split_manifest)
        if args.split_manifest
        else REPO_ROOT / "outputs" / "anchor_calibration_splits" / f"{args.task}.json"
    )
    target_manifest_path = as_abs(args.target_manifest) if args.target_manifest else None
    target_key = args.target_key or str(eef_path.resolve())

    episodes = read_real_episodes(real_dir, args.pose_key, args.tcp_offset_m)
    split = create_or_load_split(
        split_path,
        task=args.task,
        real_dir=real_dir,
        episodes=episodes,
        calibration_size=args.calibration_size,
        eval_size=args.eval_size,
        split_seed=args.split_seed,
        expected_events=int(task_cfg["expected_events"]),
        pose_key=args.pose_key,
        tcp_offset_m=args.tcp_offset_m,
    )
    source_data, record_indices, positions, grasps = load_ego_eef(eef_path)
    events = grasp_events(grasps)
    sampling_stats: dict[str, Any] = {}
    episode_seed: int | None = None
    if args.position_correction == "anchors":
        calibration = [episodes[name] for name in split["calibration"]]
        needed_events = max(anchor_indices) + 1
        bad_calibration = [traj.name for traj in calibration if len(traj.events) < needed_events]
        if bad_calibration:
            raise RuntimeError(
                f"Calibration episodes do not contain requested anchor {max(anchor_indices)}: {bad_calibration}. "
                "Use the task-default split/anchors or create a compatible split manifest."
            )
        if len(events) < needed_events:
            raise RuntimeError(
                f"Ego trajectory has {len(events)} grasp events but anchor {max(anchor_indices)} was requested"
            )
        targets, sampling_stats, episode_seed = load_or_sample_targets(
            target_manifest_path,
            target_key=target_key,
            task=args.task,
            split_path=split_path,
            calibration=calibration,
            anchor_indices=anchor_indices,
            sample_seed=args.sample_seed,
            covariance_shrinkage=args.covariance_shrinkage,
            max_mahalanobis=args.max_mahalanobis,
            pose_key=args.pose_key,
            tcp_offset_m=args.tcp_offset_m,
        )
        corrected_positions, details = compute_correction(
            positions, events, anchor_indices, targets, args
        )
    else:
        corrected_positions = positions.copy()
        details = {
            "anchor_details": {},
            "control_trajectory_indices": [],
            "control_values": [],
            "base_displacement_m": np.zeros_like(positions),
            "depth_displacement_m": None,
        }

    corrected_rotations = None
    rotation_metadata = {"method": "none", "orientation_modified": False}
    if args.rotation_correction == "global_so3":
        if not args.rotation_manifest:
            raise ValueError("--rotation-correction global_so3 requires --rotation-manifest")
        rotation_manifest_path = as_abs(args.rotation_manifest)
        rotation_manifest = json.loads(rotation_manifest_path.read_text(encoding="utf-8"))
        if rotation_manifest.get("task") != args.task:
            raise ValueError(
                f"rotation manifest task {rotation_manifest.get('task')!r} != {args.task!r}"
            )
        bias = R.from_matrix(np.asarray(rotation_manifest["R_bias_matrix"], dtype=np.float64))
        source_rotations = []
        for record_i in record_indices:
            T = np.asarray(source_data["records"][record_i]["T_ee_in_base"]["T"], dtype=np.float64)
            source_rotations.append(T[:3, :3])
        corrected_rotations = (
            R.from_matrix(np.asarray(source_rotations)) * bias.inv()
        ).as_matrix()
        rotation_metadata = {
            "method": "global_so3",
            "orientation_modified": True,
            "manifest": str(rotation_manifest_path.resolve()),
            "application": "R_corrected = R_source @ R_bias.inv()",
            "bias_angle_deg": rotation_manifest["angle_deg"],
            "bias_axis_local": rotation_manifest["axis_local"],
            "calibration_episode_names": rotation_manifest["real_calibration_episode_names"],
            "eval_episode_names_not_used": rotation_manifest["real_eval_episode_names_not_used"],
        }

    base_delta = np.asarray(details["base_displacement_m"], dtype=np.float64)
    metadata = {
        "schema_version": 1,
        "source_eef": str(eef_path.resolve()),
        "task": args.task,
        "position_correction": args.position_correction,
        "position_modified": args.position_correction != "none",
        "space": args.space,
        "method": args.method,
        "orientation_modified": bool(rotation_metadata["orientation_modified"]),
        "rotation_correction": rotation_metadata,
        "anchor_indices_zero_based": (
            anchor_indices if args.position_correction == "anchors" else []
        ),
        "ego_grasp_event_count": int(len(events)),
        "ego_anchor_trajectory_indices": (
            [int(events[k]) for k in anchor_indices]
            if args.position_correction == "anchors" else []
        ),
        "boundary_policy": {
            "start": "soft_zero" if args.method == "smooth_spline" else "hard_zero",
            "end": "soft_zero" if args.method == "smooth_spline" else "hard_zero",
        } if args.position_correction == "anchors" else None,
        "target_sampling": {
            "method": "independent_per_anchor_3d_gaussian",
            "sample_seed": int(args.sample_seed),
            "derived_episode_seed": int(episode_seed) if episode_seed is not None else None,
            "target_manifest": (
                str(target_manifest_path.resolve()) if target_manifest_path is not None else None
            ),
            "target_key": target_key,
            "covariance_shrinkage": float(args.covariance_shrinkage),
            "max_mahalanobis": float(args.max_mahalanobis),
            "anchors": sampling_stats,
        } if args.position_correction == "anchors" else None,
        "real_split": {
            "manifest": str(split_path.resolve()),
            "calibration": split["calibration"],
            "eval": split["eval"],
            "reserve": split["reserve"],
        },
        "smooth_spline_weights": {
            "anchor": float(args.anchor_weight),
            "endpoint": float(args.endpoint_weight),
            "bend": float(args.bend_weight),
            "magnitude": float(args.magnitude_weight),
        } if args.position_correction == "anchors" and args.method == "smooth_spline" else None,
        "anchor_results": details["anchor_details"],
        "correction_summary": {
            "mean_base_displacement_mm": float(1000.0 * np.linalg.norm(base_delta, axis=1).mean()),
            "max_base_displacement_mm": float(1000.0 * np.linalg.norm(base_delta, axis=1).max()),
            "start_base_displacement_mm": (1000.0 * base_delta[0]).tolist(),
            "end_base_displacement_mm": (1000.0 * base_delta[-1]).tolist(),
        },
    }
    write_outputs(
        out_dir,
        source_data,
        record_indices,
        corrected_positions,
        corrected_rotations,
        details,
        metadata,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
