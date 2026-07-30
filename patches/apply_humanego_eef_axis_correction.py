#!/usr/bin/env python3
"""Apply a fixed local-axis correction to exported HumanEgo/WiLoR EEF poses."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]

# Observed axis relation:
# real x = HumanEgo z, real y = -HumanEgo y, real z = HumanEgo x.
DEFAULT_C_REAL_FROM_HUMANEGO = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


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


def apply_axis_correction(args: argparse.Namespace) -> None:
    src_path = as_abs(args.input)
    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(src_path.read_text(encoding="utf-8"))
    correction = DEFAULT_C_REAL_FROM_HUMANEGO.copy()
    if args.correction_json:
        correction = np.array(json.loads(args.correction_json), dtype=np.float64)
    if correction.shape != (3, 3):
        raise ValueError("correction matrix must be 3x3")
    if not np.allclose(correction.T @ correction, np.eye(3), atol=1e-6):
        raise ValueError("correction matrix must be orthonormal")
    if np.linalg.det(correction) < 0.0:
        raise ValueError("correction matrix must be a proper rotation, det > 0")

    for rec in data.get("records", []):
        if not rec.get("valid"):
            continue
        T = np.array(rec["T_ee_in_base"]["T"], dtype=np.float64)
        T[:3, :3] = T[:3, :3] @ correction
        rec["T_ee_in_base_uncorrected"] = rec["T_ee_in_base"]
        rec["T_ee_in_base"] = matrix_record(T)

    metadata = data.setdefault("metadata", {})
    metadata["orientation_axis_correction"] = {
        "applied": True,
        "scope": "T_ee_in_base rotation only; translation is unchanged",
        "description": "real x = HumanEgo z, real y = -HumanEgo y, real z = HumanEgo x",
        "C_real_from_humanego_local": correction.tolist(),
        "composition": "R_corrected = R_humanego @ C_real_from_humanego_local",
    }

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
            if rec.get("valid"):
                ee = rec["T_ee_in_base"]
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
    print(f"Applied local-axis correction det={np.linalg.det(correction):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/ego_nero_easy/robot_eef_scene_camera/robot_eef_trajectory.json")
    parser.add_argument("--out", default="outputs/ego_nero_easy/robot_eef_scene_camera_axis_corrected")
    parser.add_argument("--correction-json", default=None)
    args = parser.parse_args()
    apply_axis_correction(args)


if __name__ == "__main__":
    main()
