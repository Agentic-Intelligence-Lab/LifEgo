#!/usr/bin/env python3
"""Compare reconstructed ego EEF trajectories against real robot teleop TCP."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / "outputs" / ".matplotlib"))


def as_abs(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def quat_continuous(quats: np.ndarray) -> np.ndarray:
    out = np.asarray(quats, dtype=np.float64).copy()
    for i in range(len(out)):
        norm = float(np.linalg.norm(out[i]))
        if norm > 1e-9:
            out[i] /= norm
        if i > 0 and float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def load_ego_eef(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pos, quat, grasp, idx = [], [], [], []
    for rec in data.get("records", []):
        if not rec.get("valid", False):
            continue
        ee = rec.get("T_ee_in_base")
        if not ee:
            continue
        pos.append(ee["translation_m"])
        quat.append(ee["quat_xyzw"])
        grasp.append(float(rec.get("grasp", 0.0)))
        idx.append(int(rec.get("idx", len(idx))))
    if not pos:
        raise ValueError(f"no valid EEF records: {path}")
    return {
        "name": path.parents[1].name,
        "path": str(path),
        "pos": np.asarray(pos, dtype=np.float64),
        "quat": quat_continuous(np.asarray(quat, dtype=np.float64)),
        "grasp": np.asarray(grasp, dtype=np.float64),
        "idx": np.asarray(idx, dtype=np.int64),
    }


def load_realbot_jsonl(path: Path, *, require_valid: bool = True) -> dict[str, Any]:
    pos, rpy, grasp, elapsed = [], [], [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("kind") != "sample":
                continue
            if require_valid and not rec.get("alignment", {}).get("valid", False):
                continue
            training = rec.get("training", {})
            state = training.get("state", {})
            tcp = state.get("tcp_pose") or rec.get("poses", {}).get("tcp_pose")
            if tcp is None or len(tcp) < 6:
                continue
            pos.append(tcp[:3])
            rpy.append(tcp[3:6])
            grasp.append(float(state.get("gripper_grasp", rec.get("gripper", {}).get("state_grasp", 0.0))))
            elapsed.append(float(rec.get("elapsed_s", len(elapsed))))
    if not pos:
        raise ValueError(f"no valid realbot samples: {path}")
    rot = R.from_euler("xyz", np.asarray(rpy, dtype=np.float64))
    return {
        "name": path.stem,
        "path": str(path),
        "pos": np.asarray(pos, dtype=np.float64),
        "quat": quat_continuous(rot.as_quat()),
        "grasp": np.asarray(grasp, dtype=np.float64),
        "time": np.asarray(elapsed, dtype=np.float64),
    }


def load_ego_roots(roots: list[Path], eef_rel: Path) -> list[dict[str, Any]]:
    episodes = []
    for root in roots:
        for path in sorted(root.glob(f"*/{eef_rel.as_posix()}")):
            episodes.append(load_ego_eef(path))
    if not episodes:
        raise ValueError("no ego EEF episodes found")
    return episodes


def load_real_root(root: Path, *, require_valid: bool) -> list[dict[str, Any]]:
    episodes = [load_realbot_jsonl(path, require_valid=require_valid) for path in sorted(root.glob("*.jsonl"))]
    if not episodes:
        raise ValueError("no realbot jsonl episodes found")
    return episodes


def path_length(pos: np.ndarray) -> float:
    if len(pos) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def x_elevation_deg(quat: np.ndarray) -> np.ndarray:
    rot = R.from_quat(quat).as_matrix()
    return np.degrees(np.arcsin(np.clip(rot[:, 2, 0], -1.0, 1.0)))


def resample_pos(pos: np.ndarray, n: int) -> np.ndarray:
    if len(pos) == n:
        return pos
    src = np.linspace(0.0, 1.0, len(pos))
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, pos[:, axis]) for axis in range(3)], axis=-1)


def stack_resampled(episodes: list[dict[str, Any]], n: int) -> np.ndarray:
    return np.stack([resample_pos(ep["pos"], n) for ep in episodes], axis=0)


def summarize_group(episodes: list[dict[str, Any]], name: str) -> dict[str, Any]:
    points = np.concatenate([ep["pos"] for ep in episodes], axis=0)
    starts = np.stack([ep["pos"][0] for ep in episodes])
    ends = np.stack([ep["pos"][-1] for ep in episodes])
    deltas = ends - starts
    lengths = np.asarray([path_length(ep["pos"]) for ep in episodes])
    straight = np.linalg.norm(deltas, axis=1)
    ratio = np.divide(straight, lengths, out=np.zeros_like(straight), where=lengths > 1e-9)
    elev = np.concatenate([x_elevation_deg(ep["quat"]) for ep in episodes])
    frames = np.asarray([len(ep["pos"]) for ep in episodes], dtype=np.int64)
    return {
        "name": name,
        "episodes": len(episodes),
        "frames_total": int(frames.sum()),
        "frames_mean": float(frames.mean()),
        "position_mean_m": points.mean(axis=0).tolist(),
        "position_std_m": points.std(axis=0).tolist(),
        "bbox_min_m": points.min(axis=0).tolist(),
        "bbox_max_m": points.max(axis=0).tolist(),
        "bbox_span_m": (points.max(axis=0) - points.min(axis=0)).tolist(),
        "start_mean_m": starts.mean(axis=0).tolist(),
        "start_std_m": starts.std(axis=0).tolist(),
        "end_mean_m": ends.mean(axis=0).tolist(),
        "end_std_m": ends.std(axis=0).tolist(),
        "delta_mean_m": deltas.mean(axis=0).tolist(),
        "delta_std_m": deltas.std(axis=0).tolist(),
        "path_length_mean_m": float(lengths.mean()),
        "path_length_std_m": float(lengths.std()),
        "straight_over_path_mean": float(ratio.mean()),
        "local_x_elevation_mean_deg": float(elev.mean()),
        "local_x_elevation_std_deg": float(elev.std()),
    }


def compare_trends(ego: list[dict[str, Any]], real: list[dict[str, Any]], n: int) -> dict[str, Any]:
    ego_res = stack_resampled(ego, n)
    real_res = stack_resampled(real, n)
    ego_mean = ego_res.mean(axis=0)
    real_mean = real_res.mean(axis=0)
    diff = ego_mean - real_mean
    ego_delta = ego_mean[-1] - ego_mean[0]
    real_delta = real_mean[-1] - real_mean[0]
    denom = float(np.linalg.norm(ego_delta) * np.linalg.norm(real_delta))
    cosine = float(np.dot(ego_delta, real_delta) / denom) if denom > 1e-9 else 0.0
    return {
        "samples": n,
        "mean_trend_offset_m": diff.mean(axis=0).tolist(),
        "rms_trend_error_m": np.sqrt((diff**2).mean(axis=0)).tolist(),
        "ego_mean_delta_m": ego_delta.tolist(),
        "real_mean_delta_m": real_delta.tolist(),
        "mean_delta_difference_m": (ego_delta - real_delta).tolist(),
        "delta_direction_cosine": cosine,
    }


def make_plots(ego: list[dict[str, Any]], real: list[dict[str, Any]], out_dir: Path, n: int) -> None:
    import matplotlib.pyplot as plt

    ego_res = stack_resampled(ego, n)
    real_res = stack_resampled(real, n)
    t = np.linspace(0.0, 1.0, n)
    labels = ["x", "y", "z"]

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for axis, ax in enumerate(axes):
        for arr, color, label in ((ego_res, "tab:blue", "ego"), (real_res, "tab:orange", "realbot")):
            mean = arr[:, :, axis].mean(axis=0)
            std = arr[:, :, axis].std(axis=0)
            ax.plot(t, mean, color=color, label=label)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)
        ax.set_ylabel(f"{labels[axis]} m")
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("normalized time")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_position_trends.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*ego_res.mean(axis=0).T, color="tab:blue", label="ego mean")
    ax.plot(*real_res.mean(axis=0).T, color="tab:orange", label="realbot mean")
    ax.set_xlabel("x m")
    ax.set_ylabel("y m")
    ax.set_zlabel("z m")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_trajectory_3d.png", dpi=160)
    plt.close(fig)


def fmt_vec(values: list[float]) -> str:
    return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    ego = summary["ego"]
    real = summary["realbot"]
    cmp = summary["comparison"]
    lines = [
        "# Ego EEF vs Realbot TCP Summary",
        "",
        f"Ego episodes: {ego['episodes']}, frames: {ego['frames_total']}",
        f"Realbot episodes: {real['episodes']}, frames: {real['frames_total']}",
        "",
        "## Workspace",
        "",
        f"- Ego bbox min/max/span m: {fmt_vec(ego['bbox_min_m'])} / {fmt_vec(ego['bbox_max_m'])} / {fmt_vec(ego['bbox_span_m'])}",
        f"- Real bbox min/max/span m: {fmt_vec(real['bbox_min_m'])} / {fmt_vec(real['bbox_max_m'])} / {fmt_vec(real['bbox_span_m'])}",
        f"- Mean workspace offset ego-real m: {fmt_vec(cmp['mean_trend_offset_m'])}",
        f"- RMS mean-trend error m: {fmt_vec(cmp['rms_trend_error_m'])}",
        "",
        "## Motion",
        "",
        f"- Ego mean start/end/delta m: {fmt_vec(ego['start_mean_m'])} / {fmt_vec(ego['end_mean_m'])} / {fmt_vec(ego['delta_mean_m'])}",
        f"- Real mean start/end/delta m: {fmt_vec(real['start_mean_m'])} / {fmt_vec(real['end_mean_m'])} / {fmt_vec(real['delta_mean_m'])}",
        f"- Mean delta difference ego-real m: {fmt_vec(cmp['mean_delta_difference_m'])}",
        f"- Delta direction cosine: {cmp['delta_direction_cosine']:.3f}",
        f"- Path length mean ego/real m: {ego['path_length_mean_m']:.4f} / {real['path_length_mean_m']:.4f}",
        f"- Straight-over-path mean ego/real: {ego['straight_over_path_mean']:.3f} / {real['straight_over_path_mean']:.3f}",
        "",
        "## Orientation",
        "",
        f"- Ego local-X elevation mean/std deg: {ego['local_x_elevation_mean_deg']:.3f} / {ego['local_x_elevation_std_deg']:.3f}",
        f"- Realbot local-X elevation mean/std deg: {real['local_x_elevation_mean_deg']:.3f} / {real['local_x_elevation_std_deg']:.3f}",
        "",
        "Plots:",
        "",
        "- mean_position_trends.png",
        "- mean_trajectory_3d.png",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ego-root", action="append", required=True, help="Ego output root. Repeatable.")
    parser.add_argument("--ego-eef-rel", default="robot_eef_scene_camera_axis_corrected_flat_x0/robot_eef_trajectory.json")
    parser.add_argument("--realbot-root", required=True)
    parser.add_argument("--out", default="outputs/analysis/ego_vs_realbot_stack_object_horizontal")
    parser.add_argument("--resample", type=int, default=100)
    parser.add_argument("--include-invalid-realbot", action="store_true")
    args = parser.parse_args()

    out_dir = as_abs(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ego = load_ego_roots([as_abs(path) for path in args.ego_root], Path(args.ego_eef_rel))
    real = load_real_root(as_abs(args.realbot_root), require_valid=not args.include_invalid_realbot)
    summary = {
        "ego": summarize_group(ego, "ego"),
        "realbot": summarize_group(real, "realbot"),
        "comparison": compare_trends(ego, real, int(args.resample)),
        "inputs": {
            "ego_roots": [str(as_abs(path)) for path in args.ego_root],
            "ego_eef_rel": args.ego_eef_rel,
            "realbot_root": str(as_abs(args.realbot_root)),
        },
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, out_dir / "summary.md")
    make_plots(ego, real, out_dir, int(args.resample))

    print(f"Wrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")
    print(f"Wrote {out_dir / 'mean_position_trends.png'}")
    print(f"Wrote {out_dir / 'mean_trajectory_3d.png'}")
    print()
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
