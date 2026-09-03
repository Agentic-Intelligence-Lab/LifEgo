#!/usr/bin/env python3
"""Fit one task-level local SO(3) bias from calibration real trajectories.

The fitted correction is shared by every ego episode of a source pipeline.
Position-DTW supplies correspondences inside the task-critical grasp segment;
only names from the calibration block of the real split are loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent / "eval"
sys.path.insert(0, str(EVAL_DIR))
from traj_metrics import (  # noqa: E402
    DEFAULT_TCP_OFFSET_M,
    crop_to_segment,
    dtw_pairs,
    find_ego_csv,
    load_ego_csv,
    load_real_jsonl,
)

TASK_SEGMENTS = {
    "stack_bowl": (0, 1),
    "stack_object_horizontal": (1, 2),
    "stack_object_lean": (1, 2),
}


def as_abs(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=sorted(TASK_SEGMENTS))
    p.add_argument("--source-task-dir", required=True,
                   help="outputs/<source-pipeline>/<task>")
    p.add_argument("--real-dir", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ego-subdir", default="robot_eef_scene_camera_axis_corrected")
    p.add_argument("--ego-hz", type=float, default=30.0)
    p.add_argument("--pose-key", default="tcp_tip_pose")
    p.add_argument("--tcp-offset-m", type=float, default=DEFAULT_TCP_OFFSET_M)
    p.add_argument("--segment-start", type=int, default=None)
    p.add_argument("--segment-end", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    seg_start, seg_end = TASK_SEGMENTS[args.task]
    if args.segment_start is not None:
        seg_start = args.segment_start
    if args.segment_end is not None:
        seg_end = args.segment_end

    split_path = as_abs(args.split_manifest)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split.get("task") != args.task:
        raise ValueError(f"split task {split.get('task')!r} != {args.task!r}")

    real_dir = as_abs(args.real_dir)
    reals = []
    for name in split["calibration"]:
        traj = load_real_jsonl(real_dir / name, pose_key=args.pose_key,
                               tcp_offset_m=args.tcp_offset_m)
        cropped = None if traj is None else crop_to_segment(traj, seg_start, seg_end)
        if cropped is None:
            raise RuntimeError(f"calibration real episode lacks segment {seg_start}->{seg_end}: {name}")
        reals.append(cropped)

    source_task_dir = as_abs(args.source_task_dir)
    egos = []
    source_files = []
    for session in sorted(p for p in source_task_dir.glob("*/*") if p.is_dir()):
        csv_path = find_ego_csv(session, args.ego_subdir)
        if csv_path is None:
            continue
        traj = load_ego_csv(csv_path, hz=args.ego_hz)
        cropped = crop_to_segment(traj, seg_start, seg_end)
        if cropped is None:
            continue
        egos.append(cropped)
        source_files.append(str(csv_path.resolve()))
    if len(egos) < 2:
        raise RuntimeError(f"need at least 2 healthy ego trajectories, found {len(egos)}")

    per_ego = []
    for ego in egos:
        ep, er = ego.resampled_pos(), ego.resampled_rot()
        offsets = []
        for real in reals:
            rp, rr = real.resampled_pos(), real.resampled_rot()
            pe, pr = dtw_pairs(ep, rp)
            offsets.append((rr[pr].inv() * er[pe]).as_rotvec())
        per_ego.append(R.from_rotvec(np.concatenate(offsets)).mean())

    fitted = R.from_rotvec([rotation.as_rotvec() for rotation in per_ego]).mean()
    rotvec = fitted.as_rotvec()
    angle = float(np.degrees(np.linalg.norm(rotvec)))
    spread = float(np.mean([
        np.degrees((fitted.inv() * rotation).magnitude()) for rotation in per_ego
    ]))
    payload = {
        "schema_version": 1,
        "type": "task_global_local_so3",
        "task": args.task,
        "source_task_dir": str(source_task_dir.resolve()),
        "segment": {"start_event": seg_start, "end_event": seg_end},
        "fit_correspondence": "position_dtw",
        "application": "R_corrected = R_source @ R_bias.inv()",
        "R_bias_matrix": fitted.as_matrix().tolist(),
        "R_bias_quat_xyzw": fitted.as_quat().tolist(),
        "R_bias_rotvec": rotvec.tolist(),
        "angle_deg": angle,
        "axis_local": (rotvec / max(np.linalg.norm(rotvec), 1e-12)).tolist(),
        "per_ego_spread_deg": spread,
        "n_ego": len(egos),
        "n_real_calibration": len(reals),
        "ego_source_files": source_files,
        "real_split_manifest": str(split_path.resolve()),
        "real_calibration_episode_names": list(split["calibration"]),
        "real_eval_episode_names_not_used": list(split["eval"]),
        "pose_key": args.pose_key,
        "tcp_offset_m": args.tcp_offset_m,
    }

    out = as_abs(args.out)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"rotation manifest exists; pass --overwrite: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"global SO(3): {angle:.2f} deg, spread {spread:.2f} deg, "
          f"{len(egos)} ego x {len(reals)} calibration real")


if __name__ == "__main__":
    main()
