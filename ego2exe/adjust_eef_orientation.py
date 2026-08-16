#!/usr/bin/env python3
"""Manually adjust exported EEF orientation before IK retargeting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def matrix_record(T: np.ndarray) -> dict[str, Any]:
    rot = R.from_matrix(T[:3, :3])
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": rot.as_quat().tolist(),
        "rotvec": rot.as_rotvec().tolist(),
    }


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def local_z_angle_for_x_elevation(
    rotation: np.ndarray,
    *,
    target_elevation_deg: float,
) -> float:
    """Return the smallest local-Z rotation that sets local X z elevation."""
    target_z = math.sin(math.radians(float(target_elevation_deg)))
    x_z = float(rotation[2, 0])
    y_z = float(rotation[2, 1])
    radius = math.hypot(x_z, y_z)
    if radius < 1.0e-9:
        return 0.0

    target_z = float(np.clip(target_z, -radius, radius))
    phi = math.atan2(y_z, x_z)
    alpha = math.acos(float(np.clip(target_z / radius, -1.0, 1.0)))
    candidates = [wrap_pi(phi + alpha), wrap_pi(phi - alpha)]
    return min(candidates, key=abs)


def adjust_rotation(rotation: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if args.mode == "fixed-local-z":
        angle_rad = math.radians(args.fixed_local_z_deg)
    else:
        angle_rad = local_z_angle_for_x_elevation(
            rotation,
            target_elevation_deg=args.target_x_elevation_deg,
        )

    angle_rad *= float(args.blend)
    if args.max_rotation_deg is not None:
        limit = math.radians(float(args.max_rotation_deg))
        angle_rad = float(np.clip(angle_rad, -limit, limit))

    delta = R.from_euler("z", angle_rad).as_matrix()
    return rotation @ delta, angle_rad


def x_elevation_deg(rotation: np.ndarray) -> float:
    return math.degrees(math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0))))


def write_outputs(out_dir: Path, data: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "robot_eef_trajectory.json"
    jsonl_path = out_dir / "robot_eef_trajectory.jsonl"
    csv_path = out_dir / "robot_eef_trajectory.csv"

    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in data.get("records", []):
            f.write(json.dumps(rec) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "idx",
            "ts",
            "valid",
            "hand_key",
            "confidence",
            "grasp",
            "x_m",
            "y_m",
            "z_m",
            "qx",
            "qy",
            "qz",
            "qw",
            "rx",
            "ry",
            "rz",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in data.get("records", []):
            row = {
                "idx": rec.get("idx"),
                "ts": rec.get("ts"),
                "valid": rec.get("valid"),
                "hand_key": rec.get("hand_key"),
                "confidence": rec.get("confidence"),
                "grasp": rec.get("grasp"),
                "x_m": None,
                "y_m": None,
                "z_m": None,
                "qx": None,
                "qy": None,
                "qz": None,
                "qw": None,
                "rx": None,
                "ry": None,
                "rz": None,
            }
            ee = rec.get("T_ee_in_base")
            if rec.get("valid") and ee is not None:
                row.update(
                    {
                        "x_m": ee["translation_m"][0],
                        "y_m": ee["translation_m"][1],
                        "z_m": ee["translation_m"][2],
                        "qx": ee["quat_xyzw"][0],
                        "qy": ee["quat_xyzw"][1],
                        "qz": ee["quat_xyzw"][2],
                        "qw": ee["quat_xyzw"][3],
                        "rx": ee["rotvec"][0],
                        "ry": ee["rotvec"][1],
                        "rz": ee["rotvec"][2],
                    }
                )
            writer.writerow(row)

    print(f"Wrote {json_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {csv_path}")


def adjust_eef(args: argparse.Namespace) -> None:
    src = as_abs(args.input)
    out_dir = as_abs(args.out)
    data = json.loads(src.read_text(encoding="utf-8"))

    before, after, angles = [], [], []
    adjusted = 0
    for rec in data.get("records", []):
        if not rec.get("valid"):
            continue
        ee = rec.get(args.field)
        if ee is None:
            continue
        T = np.asarray(ee["T"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Record idx={rec.get('idx')} field {args.field} is not 4x4")

        T_new = T.copy()
        before.append(x_elevation_deg(T[:3, :3]))
        new_rot, angle_rad = adjust_rotation(T[:3, :3], args)
        T_new[:3, :3] = new_rot
        after.append(x_elevation_deg(T_new[:3, :3]))
        angles.append(math.degrees(angle_rad))

        rec["T_ee_in_base_before_orientation_adjust"] = rec.get("T_ee_in_base")
        rec["T_ee_in_base"] = matrix_record(T_new)
        adjusted += 1

    if adjusted == 0:
        raise RuntimeError(f"No valid records with field {args.field} in {src}")

    metadata = data.setdefault("metadata", {})
    metadata["manual_orientation_adjust"] = {
        "source": str(src),
        "field": args.field,
        "mode": args.mode,
        "composition": "R_adjusted = R_source @ Rz_local(angle)",
        "target_x_elevation_deg": float(args.target_x_elevation_deg),
        "fixed_local_z_deg": float(args.fixed_local_z_deg),
        "blend": float(args.blend),
        "max_rotation_deg": None if args.max_rotation_deg is None else float(args.max_rotation_deg),
        "adjusted_records": int(adjusted),
        "x_elevation_before_mean_deg": float(np.mean(before)),
        "x_elevation_before_min_deg": float(np.min(before)),
        "x_elevation_before_max_deg": float(np.max(before)),
        "x_elevation_after_mean_deg": float(np.mean(after)),
        "x_elevation_after_min_deg": float(np.min(after)),
        "x_elevation_after_max_deg": float(np.max(after)),
        "local_z_angle_mean_deg": float(np.mean(angles)),
        "local_z_angle_min_deg": float(np.min(angles)),
        "local_z_angle_max_deg": float(np.max(angles)),
    }

    write_outputs(out_dir, data)
    print("--- Orientation adjust summary ---")
    print(f"records: {adjusted}")
    print(
        "EEF local X elevation before mean/min/max deg: "
        f"{np.mean(before):.2f} / {np.min(before):.2f} / {np.max(before):.2f}"
    )
    print(
        "EEF local X elevation after  mean/min/max deg: "
        f"{np.mean(after):.2f} / {np.min(after):.2f} / {np.max(after):.2f}"
    )
    print(
        "local Z angle mean/min/max deg: "
        f"{np.mean(angles):.2f} / {np.min(angles):.2f} / {np.max(angles):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input robot_eef_trajectory.json")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--field",
        choices=["T_ee_in_base", "T_ee_in_base_uncorrected"],
        default="T_ee_in_base",
        help="Source pose field to adjust.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto-flatten-x", "fixed-local-z"],
        default="auto-flatten-x",
        help="auto-flatten-x rotates around local Z to set local X elevation; fixed-local-z applies a constant local Z rotation.",
    )
    parser.add_argument(
        "--target-x-elevation-deg",
        type=float,
        default=0.0,
        help="Target elevation of EEF local X relative to the robot-base horizontal plane.",
    )
    parser.add_argument(
        "--fixed-local-z-deg",
        type=float,
        default=0.0,
        help="Constant local-Z rotation used by --mode fixed-local-z.",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        help="Scale the computed/manual local-Z angle. 1.0 applies the full adjustment; 0.5 applies half.",
    )
    parser.add_argument(
        "--max-rotation-deg",
        type=float,
        default=None,
        help="Optional clamp on the local-Z adjustment magnitude.",
    )
    args = parser.parse_args()
    adjust_eef(args)


if __name__ == "__main__":
    main()
