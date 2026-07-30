#!/usr/bin/env python3
"""Compare real robot TCP poses with HumanEgo/WiLoR exported EE poses."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_humanego(path: Path) -> tuple[np.ndarray, R, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    positions = []
    rotations = []
    idxs = []
    for rec in data["records"]:
        if not rec.get("valid"):
            continue
        T = np.array(rec["T_ee_in_base"]["T"], dtype=np.float64)
        positions.append(T[:3, 3])
        rotations.append(R.from_matrix(T[:3, :3]))
        idxs.append(rec["idx"])
    return np.array(positions), R.concatenate(rotations), np.array(idxs)


def load_realbot(path: Path, pose_key: str) -> tuple[np.ndarray, R, np.ndarray]:
    positions = []
    rotations = []
    seqs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") != "sample":
            continue
        pose = rec["poses"][pose_key]
        x, y, z, roll, pitch, yaw = pose
        positions.append([x, y, z])
        # Metadata says "xyz + rpy(ZYX)": R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
        rotations.append(R.from_euler("xyz", [roll, pitch, yaw]))
        seqs.append(rec["seq"])
    return np.array(positions, dtype=np.float64), R.concatenate(rotations), np.array(seqs)


def resample_positions(pos: np.ndarray, n: int) -> np.ndarray:
    src = np.linspace(0.0, 1.0, len(pos))
    dst = np.linspace(0.0, 1.0, n)
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(3):
        out[:, i] = np.interp(dst, src, pos[:, i])
    return out


def resample_rotations(rot: R, n: int) -> R:
    src = np.linspace(0.0, 1.0, len(rot))
    dst = np.linspace(0.0, 1.0, n)
    return Slerp(src, rot)(dst)


def angle_deg(rot_a: R, rot_b: R) -> np.ndarray:
    return (rot_a.inv() * rot_b).magnitude() * 180.0 / np.pi


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "min": float(np.min(values)),
    }


def vec_stats(pos: np.ndarray) -> dict:
    return {
        "start": pos[0].tolist(),
        "end": pos[-1].tolist(),
        "delta": (pos[-1] - pos[0]).tolist(),
        "range_min": np.min(pos, axis=0).tolist(),
        "range_max": np.max(pos, axis=0).tolist(),
    }


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def plot_positions(out_path: Path, he: np.ndarray, rb: np.ndarray) -> None:
    t = np.linspace(0.0, 1.0, len(he))
    labels = ["x", "y", "z"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, he[:, i], label="HumanEgo/WiLoR EE")
        ax.plot(t, rb[:, i], label="real robot TCP")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("normalized trajectory time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_relative_positions(out_path: Path, he: np.ndarray, rb: np.ndarray) -> None:
    he_rel = he - he[0]
    rb_rel = rb - rb[0]
    t = np.linspace(0.0, 1.0, len(he))
    labels = ["dx", "dy", "dz"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, he_rel[:, i], label="HumanEgo/WiLoR EE relative")
        ax.plot(t, rb_rel[:, i], label="real robot TCP relative")
        ax.set_ylabel(f"{labels[i]} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    axes[-1].set_xlabel("normalized trajectory time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_orientation_errors(out_path: Path, abs_err: np.ndarray, offset_err: np.ndarray, delta_err: np.ndarray) -> None:
    t = np.linspace(0.0, 1.0, len(abs_err))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, abs_err, label="absolute orientation error")
    ax.plot(t, offset_err, label="after initial constant offset")
    ax.plot(t, delta_err, label="relative orientation-change error")
    ax.set_xlabel("normalized trajectory time")
    ax.set_ylabel("angle error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humanego", default="outputs/ego_nero_easy/robot_eef/robot_eef_trajectory.json")
    parser.add_argument("--realbot", default="examples/ego_nero_easy_real_bot.jsonl")
    parser.add_argument("--real-pose-key", default="tcp_pose", choices=["tcp_pose", "flange_pose", "fk_pose"])
    parser.add_argument("--out", default="outputs/ego_nero_easy/compare_realbot_humanego")
    parser.add_argument("--samples", type=int, default=120)
    args = parser.parse_args()

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    he_pos, he_rot, he_idxs = load_humanego(as_abs(args.humanego))
    rb_pos, rb_rot, rb_seqs = load_realbot(as_abs(args.realbot), args.real_pose_key)

    n = args.samples
    he_p = resample_positions(he_pos, n)
    rb_p = resample_positions(rb_pos, n)
    he_r = resample_rotations(he_rot, n)
    rb_r = resample_rotations(rb_rot, n)

    pos_abs_err = np.linalg.norm(he_p - rb_p, axis=1)
    pos_rel_err = np.linalg.norm((he_p - he_p[0]) - (rb_p - rb_p[0]), axis=1)
    he_delta = he_p[-1] - he_p[0]
    rb_delta = rb_p[-1] - rb_p[0]
    delta_cos = float(np.dot(unit(he_delta), unit(rb_delta)))
    delta_angle = float(np.degrees(np.arccos(np.clip(delta_cos, -1.0, 1.0))))

    abs_ori_err = angle_deg(he_r, rb_r)
    offset = rb_r[0] * he_r[0].inv()
    offset_ori_err = angle_deg(offset * he_r, rb_r)
    he_rel_r = he_r[0].inv() * he_r
    rb_rel_r = rb_r[0].inv() * rb_r
    delta_ori_err = angle_deg(he_rel_r, rb_rel_r)

    summary = {
        "inputs": {
            "humanego": str(as_abs(args.humanego)),
            "realbot": str(as_abs(args.realbot)),
            "real_pose_key": args.real_pose_key,
            "humanego_frames": int(len(he_pos)),
            "realbot_samples": int(len(rb_pos)),
            "comparison_samples": int(n),
        },
        "position": {
            "humanego": vec_stats(he_p),
            "realbot": vec_stats(rb_p),
            "absolute_error_m": stats(pos_abs_err),
            "relative_to_start_error_m": stats(pos_rel_err),
            "delta_direction_cosine": delta_cos,
            "delta_direction_angle_deg": delta_angle,
            "delta_norms_m": {
                "humanego": float(np.linalg.norm(he_delta)),
                "realbot": float(np.linalg.norm(rb_delta)),
            },
        },
        "orientation": {
            "absolute_angle_error_deg": stats(abs_ori_err),
            "after_initial_constant_offset_angle_error_deg": stats(offset_ori_err),
            "relative_change_angle_error_deg": stats(delta_ori_err),
            "initial_constant_offset_rotvec_rad": offset.as_rotvec().tolist(),
            "initial_constant_offset_angle_deg": float(offset.magnitude() * 180.0 / np.pi),
        },
        "interpretation": {
            "position_note": "Absolute position includes the current assumed camera-to-base extrinsic.",
            "orientation_note": (
                "If absolute orientation error is large but offset-corrected and relative-change errors are small, "
                "the motion trend is consistent but the EE/hand frame convention has a constant rotation offset."
            ),
        },
    }

    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (out_dir / "comparison_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "t_norm",
            "he_x",
            "he_y",
            "he_z",
            "rb_x",
            "rb_y",
            "rb_z",
            "pos_abs_err_m",
            "ori_abs_err_deg",
            "ori_offset_err_deg",
            "ori_delta_err_deg",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow(
                {
                    "t_norm": i / (n - 1),
                    "he_x": he_p[i, 0],
                    "he_y": he_p[i, 1],
                    "he_z": he_p[i, 2],
                    "rb_x": rb_p[i, 0],
                    "rb_y": rb_p[i, 1],
                    "rb_z": rb_p[i, 2],
                    "pos_abs_err_m": pos_abs_err[i],
                    "ori_abs_err_deg": abs_ori_err[i],
                    "ori_offset_err_deg": offset_ori_err[i],
                    "ori_delta_err_deg": delta_ori_err[i],
                }
            )

    plot_positions(out_dir / "positions_absolute.png", he_p, rb_p)
    plot_relative_positions(out_dir / "positions_relative_to_start.png", he_p, rb_p)
    plot_orientation_errors(out_dir / "orientation_errors.png", abs_ori_err, offset_ori_err, delta_ori_err)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
