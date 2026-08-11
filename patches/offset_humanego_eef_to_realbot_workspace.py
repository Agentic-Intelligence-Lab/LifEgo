#!/usr/bin/env python3
"""Place HumanEgo EEF poses into the rough workspace of a real-bot recording."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def matrix_record(T: np.ndarray) -> dict:
    rot = R.from_matrix(T[:3, :3])
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": rot.as_quat().tolist(),
        "rotvec": rot.as_rotvec().tolist(),
    }


def bbox_center(points: np.ndarray) -> np.ndarray:
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def load_real_tcp_poses(path: Path) -> tuple[np.ndarray, R]:
    points = []
    rots = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        tcp_pose = rec["poses"]["tcp_pose"]
        points.append(tcp_pose[:3])
        # The real-bot log stores xyz + roll/pitch/yaw in a ZYX/RPY convention.
        rots.append(R.from_euler("xyz", tcp_pose[3:6]))
    if not points:
        raise RuntimeError(f"No sample TCP poses found in {path}")
    return np.asarray(points, dtype=np.float64), R.concatenate(rots)


def load_valid_eef_poses(data: dict) -> tuple[np.ndarray, R]:
    points = []
    rots = []
    for rec in data.get("records", []):
        if not rec.get("valid"):
            continue
        pose = rec["T_ee_in_base"]
        points.append(pose["translation_m"])
        rots.append(R.from_quat(pose["quat_xyzw"]))
    if not points:
        raise RuntimeError("No valid EEF records found")
    return np.asarray(points, dtype=np.float64), R.concatenate(rots)


def first_last(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return points[0].copy(), points[-1].copy()


def mean_rpy_xyz(rots: R) -> list[float]:
    return rots.mean().as_euler("xyz").tolist()


def write_csv(data: dict, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx",
            "ts",
            "valid",
            "grasp",
            "x_m",
            "y_m",
            "z_m",
            "qx",
            "qy",
            "qz",
            "qw",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in data.get("records", []):
            row = {
                "idx": rec.get("idx"),
                "ts": rec.get("ts"),
                "valid": rec.get("valid"),
                "grasp": rec.get("grasp"),
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "qx": None,
                "qy": None,
                "qz": None,
                "qw": None,
            }
            if rec.get("valid"):
                pose = rec["T_ee_in_base"]
                p = pose["translation_m"]
                q = pose["quat_xyzw"]
                row.update(
                    {
                        "x_m": p[0],
                        "y_m": p[1],
                        "z_m": p[2],
                        "qx": q[0],
                        "qy": q[1],
                        "qz": q[2],
                        "qw": q[3],
                    }
                )
            writer.writerow(row)


def run(args: argparse.Namespace) -> None:
    eef_path = as_abs(args.eef)
    realbot_path = as_abs(args.realbot)
    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(eef_path.read_text(encoding="utf-8"))
    eef_points, eef_rots = load_valid_eef_poses(data)
    real_points, real_rots = load_real_tcp_poses(realbot_path)

    eef_center = bbox_center(eef_points)
    real_center = bbox_center(real_points)
    base_offset = real_center - eef_center
    extra_offset = np.asarray(args.extra_offset, dtype=np.float64)
    shape_scale = np.asarray(args.shape_scale, dtype=np.float64)
    if np.any(~np.isfinite(shape_scale)) or np.any(np.isclose(shape_scale, 0.0)):
        raise ValueError("--shape-scale values must be finite and non-zero")

    shifted_points = eef_points + base_offset + extra_offset
    shifted_start, shifted_end = first_last(shifted_points)
    real_start, real_end = first_last(real_points)

    if args.anchor == "bbox-center":
        anchor_before = bbox_center(shifted_points)
        anchor_after = anchor_before
    elif args.anchor == "shifted-end":
        anchor_before = shifted_end
        anchor_after = shifted_end
    elif args.anchor == "real-end":
        anchor_before = shifted_end
        anchor_after = real_end + extra_offset
    elif args.anchor == "real-start":
        anchor_before = shifted_start
        anchor_after = real_start + extra_offset
    else:
        raise ValueError(f"Unsupported anchor: {args.anchor}")

    transformed_points = anchor_after + (shifted_points - anchor_before) * shape_scale

    rot_delta = R.identity()
    if args.match_real_mean_orientation:
        rot_delta = real_rots.mean() * eef_rots.mean().inv()
    if args.orientation_euler_xyz:
        rot_delta = R.from_euler("xyz", args.orientation_euler_xyz) * rot_delta

    valid_i = 0

    for rec in data.get("records", []):
        if not rec.get("valid"):
            continue
        T = np.asarray(rec["T_ee_in_base"]["T"], dtype=np.float64)
        T[:3, 3] = transformed_points[valid_i]
        if args.match_real_mean_orientation or args.orientation_euler_xyz:
            T[:3, :3] = (rot_delta * R.from_matrix(T[:3, :3])).as_matrix()
        rec["T_ee_in_base_before_workspace_offset"] = rec["T_ee_in_base"]
        rec["T_ee_in_base"] = matrix_record(T)
        valid_i += 1

    metadata = data.setdefault("metadata", {})
    metadata["realbot_workspace_offset"] = {
        "applied": True,
        "source_eef": str(eef_path),
        "realbot": str(realbot_path),
        "method": (
            "bbox-center workspace translation followed by optional per-axis shape "
            "scale around an anchor; optional mean-orientation match to realbot tcp_pose"
        ),
        "base_offset_m": base_offset.tolist(),
        "extra_offset_m": extra_offset.tolist(),
        "anchor": args.anchor,
        "anchor_before_m": anchor_before.tolist(),
        "anchor_after_m": anchor_after.tolist(),
        "shape_scale_xyz": shape_scale.tolist(),
        "match_real_mean_orientation": bool(args.match_real_mean_orientation),
        "orientation_euler_xyz_rad": (
            list(args.orientation_euler_xyz) if args.orientation_euler_xyz else None
        ),
        "orientation_delta_quat_xyzw": rot_delta.as_quat().tolist(),
        "eef_bbox_before": {
            "min_m": eef_points.min(axis=0).tolist(),
            "max_m": eef_points.max(axis=0).tolist(),
            "center_m": eef_center.tolist(),
            "start_m": eef_points[0].tolist(),
            "end_m": eef_points[-1].tolist(),
            "mean_rpy_xyz_rad": mean_rpy_xyz(eef_rots),
        },
        "eef_bbox_after": {
            "min_m": transformed_points.min(axis=0).tolist(),
            "max_m": transformed_points.max(axis=0).tolist(),
            "center_m": bbox_center(transformed_points).tolist(),
            "start_m": transformed_points[0].tolist(),
            "end_m": transformed_points[-1].tolist(),
            "mean_rpy_xyz_rad": mean_rpy_xyz(rot_delta * eef_rots),
        },
        "realbot_tcp_bbox": {
            "min_m": real_points.min(axis=0).tolist(),
            "max_m": real_points.max(axis=0).tolist(),
            "center_m": real_center.tolist(),
            "start_m": real_points[0].tolist(),
            "end_m": real_points[-1].tolist(),
            "mean_rpy_xyz_rad": mean_rpy_xyz(real_rots),
        },
    }

    json_path = out_dir / "robot_eef_trajectory.json"
    jsonl_path = out_dir / "robot_eef_trajectory.jsonl"
    csv_path = out_dir / "robot_eef_trajectory.csv"
    summary_path = out_dir / "workspace_offset_summary.json"

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in data.get("records", []):
            f.write(json.dumps(rec) + "\n")
    write_csv(data, csv_path)
    summary_path.write_text(
        json.dumps(metadata["realbot_workspace_offset"], indent=2), encoding="utf-8"
    )

    print(f"base_offset_m: {base_offset.tolist()}")
    print(f"anchor: {args.anchor}")
    print(f"shape_scale_xyz: {shape_scale.tolist()}")
    print(f"start_before_shifted_m: {shifted_start.tolist()}")
    print(f"start_after_m: {transformed_points[0].tolist()}")
    print(f"end_before_shifted_m: {shifted_end.tolist()}")
    print(f"end_after_m: {transformed_points[-1].tolist()}")
    print(f"orientation_delta_quat_xyzw: {rot_delta.as_quat().tolist()}")
    print(f"Wrote {json_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eef",
        default=(
            "outputs/nero_pick_place_human/robot_eef_scene_camera_axis_corrected/"
            "robot_eef_trajectory.json"
        ),
    )
    parser.add_argument("--realbot", default="examples/nero_pick_place_realbot.jsonl")
    parser.add_argument(
        "--out",
        default="outputs/nero_pick_place_human/robot_eef_realbot_workspace_offset",
    )
    parser.add_argument("--extra-offset", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument(
        "--anchor",
        choices=("bbox-center", "shifted-end", "real-end", "real-start"),
        default="bbox-center",
        help=(
            "Anchor used after the initial bbox-center workspace translation. "
            "shifted-end preserves the previous endpoint while changing trajectory shape."
        ),
    )
    parser.add_argument(
        "--shape-scale",
        nargs=3,
        type=float,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help="Per-axis scale around --anchor after bbox workspace translation.",
    )
    parser.add_argument(
        "--match-real-mean-orientation",
        action="store_true",
        help="Left-multiply all EEF rotations so their mean orientation matches realbot tcp_pose.",
    )
    parser.add_argument(
        "--orientation-euler-xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Additional world-frame Euler XYZ rotation, applied before the mean-orientation delta.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
