#!/usr/bin/env python3
"""Export robot-base EEF targets from WiLoR hand reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from assets import DEFAULT_ASSETS
from hand2gripper import HumanEgoMode

REPO_ROOT = Path(__file__).resolve().parents[1]

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
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": quat_xyzw.tolist(),
        "rotvec": rotvec.tolist(),
    }


def load_camera_from_assets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = DEFAULT_ASSETS.camera()
    if camera is None:
        raise ValueError("DEFAULT_ASSETS has no scene_rgb camera asset")
    cam_extr = camera.extrinsics
    if not cam_extr.is_filled():
        raise ValueError(
            "patches/assets.py DEFAULT_ASSETS camera extrinsics are not filled; "
            "fill DEFAULT_ASSETS.camera().extrinsics.T_cam_in_base before exporting EEF trajectories."
        )
    T_cam_in_base = np.array(cam_extr.T_cam_in_base, dtype=np.float64)
    T_base_in_cam = np.linalg.inv(T_cam_in_base)
    p_cam_in_base = T_cam_in_base[:3, 3].copy()
    return T_base_in_cam, T_cam_in_base, p_cam_in_base


def make_hand2gripper_mode() -> HumanEgoMode:
    T_hand_to_ee = DEFAULT_ASSETS.extra_transforms.get("T_hand_to_ee")
    T_hand_from_eef = None if T_hand_to_ee is None else np.linalg.inv(np.array(T_hand_to_ee, dtype=np.float64))
    return HumanEgoMode(T_hand_from_eef=T_hand_from_eef)


def validate_axis_correction(correction: np.ndarray) -> np.ndarray:
    correction = np.asarray(correction, dtype=np.float64)
    if correction.shape != (3, 3):
        raise ValueError("axis correction matrix must be 3x3")
    if not np.allclose(correction.T @ correction, np.eye(3), atol=1e-6):
        raise ValueError("axis correction matrix must be orthonormal")
    if np.linalg.det(correction) < 0.0:
        raise ValueError("axis correction matrix must be a proper rotation, det > 0")
    return correction


def apply_axis_correction(T_ee_in_base: np.ndarray, correction: np.ndarray) -> np.ndarray:
    corrected = T_ee_in_base.copy()
    corrected[:3, :3] = corrected[:3, :3] @ correction
    return corrected


def export_eef(args: argparse.Namespace) -> None:
    session_dir = as_abs(args.session)
    all_data_dir = session_dir / "preprocess" / "all_data"
    if not all_data_dir.is_dir():
        raise FileNotFoundError(f"missing all_data directory: {all_data_dir}")

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_base_in_cam, T_cam_in_base, p_cam_in_base = load_camera_from_assets()
    hand2gripper = make_hand2gripper_mode()
    axis_correction = validate_axis_correction(DEFAULT_C_REAL_FROM_HUMANEGO)
    apply_correction = not args.no_axis_correction

    records = []
    frame_dirs = sorted(p for p in all_data_dir.iterdir() if p.is_dir() and p.name.isdigit())
    for frame_dir in frame_dirs:
        path = frame_dir / args.hands_file
        if not path.is_file() and args.hands_file != "wilor_hands.json":
            path = frame_dir / "wilor_hands.json"
        if not path.is_file():
            continue

        data = json.loads(path.read_text(encoding="utf-8"))
        hand = data.get(args.hand_key)
        rec = {
            "idx": int(data.get("idx", int(frame_dir.name))),
            "ts": data.get("ts"),
            "hand_key": args.hand_key,
            "valid": False,
            "confidence": None,
            "grasp": None,
            "T_gripper_hand_in_cam": None,
            "T_hand_in_cam": None,
            "T_ee_in_cam": None,
            "T_ee_in_base": None,
            "T_ee_in_base_uncorrected": None,
            "hand2gripper": None,
        }

        if hand is not None:
            target = hand2gripper.from_hand_record(hand)
            if target is not None:
                T_ee_in_base_uncorrected = T_cam_in_base @ target.T_eef_in_cam
                T_ee_in_base = (
                    apply_axis_correction(T_ee_in_base_uncorrected, axis_correction)
                    if apply_correction
                    else T_ee_in_base_uncorrected
                )
                rec.update(
                    {
                        "valid": True,
                        "confidence": target.confidence,
                        "grasp": target.grasp_state,
                        "T_gripper_hand_in_cam": matrix_record(target.T_hand_in_cam),
                        "T_hand_in_cam": matrix_record(target.T_hand_in_cam),
                        "T_ee_in_cam": matrix_record(target.T_eef_in_cam),
                        "T_ee_in_base": matrix_record(T_ee_in_base),
                        "T_ee_in_base_uncorrected": matrix_record(T_ee_in_base_uncorrected) if apply_correction else None,
                        "hand2gripper": target.to_json_dict(),
                    }
                )
        records.append(rec)

    metadata = {
        "session": str(session_dir),
        "source": "WiLoR kpts_3d -> hand2gripper.HumanEgoMode",
        "hand2gripper_mode": "HumanEgoMode",
        "hand_key": args.hand_key,
        "units": "meters",
        "base_frame": {
            "x": "robot base +x as provided",
            "y": "robot base +y as provided",
            "z": "up from table",
        },
        "camera_setup": {
            "camera_source": "assets",
            "camera_position_base_m": p_cam_in_base.tolist(),
            "assumption": "Uses DEFAULT_ASSETS.camera().extrinsics.T_cam_in_base from patches/assets.py.",
        },
        "T_hand_to_ee": DEFAULT_ASSETS.extra_transforms.get("T_hand_to_ee").tolist()
        if DEFAULT_ASSETS.extra_transforms.get("T_hand_to_ee") is not None
        else None,
        "T_base_in_cam": T_base_in_cam.tolist(),
        "T_cam_in_base": T_cam_in_base.tolist(),
        "orientation_axis_correction": {
            "applied": apply_correction,
            "scope": "T_ee_in_base rotation only; translation is unchanged",
            "description": "real x = HumanEgo z, real y = -HumanEgo y, real z = HumanEgo x",
            "C_real_from_humanego_local": axis_correction.tolist(),
            "composition": "R_corrected = R_humanego @ C_real_from_humanego_local",
            "disable_flag": "--no-axis-correction",
        },
        "valid_frames": sum(1 for r in records if r["valid"]),
        "total_records": len(records),
    }

    json_path = out_dir / "robot_eef_trajectory.json"
    jsonl_path = out_dir / "robot_eef_trajectory.jsonl"
    csv_path = out_dir / "robot_eef_trajectory.csv"

    json_path.write_text(
        json.dumps({"metadata": metadata, "records": records}, indent=2),
        encoding="utf-8",
    )
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in records:
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
        for rec in records:
            row = {
                "idx": rec["idx"],
                "ts": rec["ts"],
                "valid": rec["valid"],
                "hand_key": rec["hand_key"],
                "confidence": rec["confidence"],
                "grasp": rec["grasp"],
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
            if rec["valid"]:
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
    print(f"Valid frames: {metadata['valid_frames']} / {metadata['total_records']}")
    print(f"Camera position in base: {p_cam_in_base.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default="outputs/ego_nero_easy")
    parser.add_argument("--out", default="outputs/ego_nero_easy/robot_eef_scene_camera")
    parser.add_argument("--hand-key", default="hand_r", choices=["hand_r", "hand_l"])
    parser.add_argument(
        "--hands-file",
        default="wilor_hands_processed.json",
        help="Per-frame hand JSON filename under preprocess/all_data/<frame>/.",
    )
    parser.add_argument(
        "--no-axis-correction",
        action="store_true",
        help="Disable the default HumanEgo-to-real local EEF axis correction.",
    )
    args = parser.parse_args()
    export_eef(args)


if __name__ == "__main__":
    main()
