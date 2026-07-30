#!/usr/bin/env python3
"""Export robot-base EE targets from HumanEgo WiLoR hand reconstructions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_T_ALIGN = np.array(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_vec3(values: list[float] | tuple[float, ...], name: str) -> np.ndarray:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return np.array(values, dtype=np.float64)


def make_camera_transforms(
    camera_height_m: float,
    pitch_down_deg: float,
    optical_projection_base: np.ndarray,
    camera_target_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build T_base_in_cam and T_cam_in_base from the camera setup.

    Base frame convention used here: +z is upward from the table, and the camera
    optical frame follows OpenCV: +x image-right, +y image-down, +z forward.
    """
    proj = optical_projection_base.astype(np.float64).copy()
    proj[2] = 0.0
    norm = np.linalg.norm(proj)
    if norm < 1e-9:
        raise ValueError("optical projection must have a non-zero xy component")
    proj /= norm

    pitch = math.radians(pitch_down_deg)
    z_cam_base = np.array(
        [math.cos(pitch) * proj[0], math.cos(pitch) * proj[1], -math.sin(pitch)],
        dtype=np.float64,
    )
    z_cam_base /= np.linalg.norm(z_cam_base)

    x_cam_base = np.array([proj[1], -proj[0], 0.0], dtype=np.float64)
    x_cam_base /= np.linalg.norm(x_cam_base)

    y_cam_base = np.cross(z_cam_base, x_cam_base)
    y_cam_base /= np.linalg.norm(y_cam_base)

    # Re-orthogonalize x to suppress roundoff.
    x_cam_base = np.cross(y_cam_base, z_cam_base)
    x_cam_base /= np.linalg.norm(x_cam_base)

    R_cam_in_base = np.column_stack([x_cam_base, y_cam_base, z_cam_base])

    # If camera_target_base is the optical-axis intersection with the table,
    # camera position is target minus the forward ray scaled to the given height.
    ray_scale = camera_height_m / max(1e-9, -z_cam_base[2])
    p_cam_in_base = camera_target_base - ray_scale * z_cam_base
    p_cam_in_base[2] = camera_height_m

    T_cam_in_base = np.eye(4, dtype=np.float64)
    T_cam_in_base[:3, :3] = R_cam_in_base
    T_cam_in_base[:3, 3] = p_cam_in_base

    T_base_in_cam = np.linalg.inv(T_cam_in_base)
    return T_base_in_cam, T_cam_in_base, p_cam_in_base


def load_pose(hand: dict, key: str) -> np.ndarray | None:
    pose = hand.get(key)
    if pose is None:
        return None
    arr = np.array(pose, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    return arr


def matrix_record(T: np.ndarray) -> dict:
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    rotvec = R.from_matrix(T[:3, :3]).as_rotvec()
    return {
        "T": T.tolist(),
        "translation_m": T[:3, 3].tolist(),
        "quat_xyzw": quat_xyzw.tolist(),
        "rotvec": rotvec.tolist(),
    }


def export_eef(args: argparse.Namespace) -> None:
    session_dir = as_abs(args.session)
    all_data_dir = session_dir / "preprocess" / "all_data"
    if not all_data_dir.is_dir():
        raise FileNotFoundError(f"missing all_data directory: {all_data_dir}")

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_align = np.array(json.loads(args.t_align_json), dtype=np.float64) if args.t_align_json else DEFAULT_T_ALIGN
    T_hand_to_ee = np.linalg.inv(T_align)

    T_base_in_cam, T_cam_in_base, p_cam_in_base = make_camera_transforms(
        camera_height_m=args.camera_height_m,
        pitch_down_deg=args.pitch_down_deg,
        optical_projection_base=parse_vec3(args.optical_projection_base, "optical_projection_base"),
        camera_target_base=parse_vec3(args.camera_target_base, "camera_target_base"),
    )

    records = []
    frame_dirs = sorted(p for p in all_data_dir.iterdir() if p.is_dir() and p.name.isdigit())
    for frame_dir in frame_dirs:
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
            "T_hand_in_cam": None,
            "T_ee_in_cam": None,
            "T_ee_in_base": None,
        }

        if hand is not None:
            T_hand_in_cam = load_pose(hand, args.pose_key)
            if T_hand_in_cam is not None:
                T_ee_in_cam = T_hand_in_cam @ T_hand_to_ee
                T_ee_in_base = T_cam_in_base @ T_ee_in_cam
                rec.update(
                    {
                        "valid": True,
                        "confidence": hand.get("confidence"),
                        "grasp": hand.get("grasp_state"),
                        "T_hand_in_cam": matrix_record(T_hand_in_cam),
                        "T_ee_in_cam": matrix_record(T_ee_in_cam),
                        "T_ee_in_base": matrix_record(T_ee_in_base),
                    }
                )
        records.append(rec)

    metadata = {
        "session": str(session_dir),
        "source": "HumanEgo WiLoR midpoint_pose_opt_world",
        "pose_key": args.pose_key,
        "hand_key": args.hand_key,
        "units": "meters",
        "base_frame": {
            "x": "robot base +x as provided",
            "y": "robot base +y as provided",
            "z": "up from table",
        },
        "camera_setup": {
            "camera_height_m": args.camera_height_m,
            "pitch_down_deg": args.pitch_down_deg,
            "optical_projection_base": args.optical_projection_base,
            "camera_target_base_m": args.camera_target_base,
            "camera_position_base_m": p_cam_in_base.tolist(),
            "assumption": (
                "camera_target_base is the table point hit by the optical axis. "
                "Defaults match patches/build_nero_mujoco_scene.py."
            ),
        },
        "T_align_hand_from_ee": T_align.tolist(),
        "T_base_in_cam": T_base_in_cam.tolist(),
        "T_cam_in_base": T_cam_in_base.tolist(),
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
    parser.add_argument("--pose-key", default="midpoint_pose_opt_world")
    parser.add_argument("--camera-height-m", type=float, default=0.585)
    parser.add_argument("--pitch-down-deg", type=float, default=45.0)
    parser.add_argument("--optical-projection-base", nargs=3, type=float, default=[-1.0, 0.0, 0.0])
    parser.add_argument("--camera-target-base", nargs=3, type=float, default=[-0.585, -0.45, 0.0])
    parser.add_argument("--t-align-json", default=None)
    args = parser.parse_args()
    export_eef(args)


if __name__ == "__main__":
    main()
