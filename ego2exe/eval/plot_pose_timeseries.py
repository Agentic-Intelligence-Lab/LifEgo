#!/usr/bin/env python3
"""Plot one ego episode's pose components over time, to inspect orientation jitter.

Four panels share a time axis:
    1. position x/y/z
    2. orientation as Euler angles (roll/pitch/yaw), unwrapped
    3. frame-to-frame angular step, converted to deg/s - the jitter panel
    4. grasp state, with the same anchor events marked on every panel above

Component 1 of position (x) and component 1 of orientation (roll) share a
color slot, and likewise for 2/3, so the eye can track "the same channel" up
and down the figure.

Example
-------
    python ego2exe/eval/plot_pose_timeseries.py \
        --ego outputs/experiments/h2g_pinch_index/stack_object_20260816_184834_540180.rgb \
        --real DATA/20260816_nero_stack_object_horizontal \
        --out outputs/eval/h2g_pinch_index/pose_timeseries.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))

from traj_metrics import find_ego_csv, load_ego_csv, load_real_dir  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]

# dataviz reference palette (references/palette.md): categorical slots 1/2/3,
# reused across the position and orientation panels for the "same channel"
# color mapping described above.
COLOR_1 = "#2a78d6"  # blue   - x / roll
COLOR_2 = "#eb6834"  # orange - y / pitch
COLOR_3 = "#1baf7a"  # aqua   - z / yaw
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def as_abs(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    cwd = Path.cwd() / p
    return cwd if cwd.exists() else REPO_ROOT / p


def unwrapped_euler_deg(rot, seq: str = "xyz") -> np.ndarray:
    """Euler angles with 2*pi jumps removed per channel, for a clean line plot."""
    euler = rot.as_euler(seq, degrees=False)
    return np.degrees(np.unwrap(euler, axis=0))


def frame_step_deg_s(rot, hz: float) -> np.ndarray:
    """Per-frame geodesic angle step (deg), scaled to deg/s. Length n-1."""
    step_rad = (rot[:-1].inv() * rot[1:]).magnitude()
    return np.degrees(step_rad) * hz


def ema_smooth_rot(rot, alpha: float):
    """HumanEgo-style basis EMA + Gram-Schmidt re-orthonormalisation."""
    from scipy.spatial.transform import Rotation as R

    m = rot.as_matrix()
    x_ema = y_ema = None
    out = []
    for k in range(len(m)):
        x, y = m[k, :, 0], m[k, :, 1]
        x_ema = x if x_ema is None else alpha * x + (1 - alpha) * x_ema
        y_ema = y if y_ema is None else alpha * y + (1 - alpha) * y_ema
        xs = x_ema / (np.linalg.norm(x_ema) + 1e-12)
        z = np.cross(xs, y_ema)
        z /= np.linalg.norm(z) + 1e-12
        ys = np.cross(z, xs)
        out.append(np.column_stack([xs, ys, z]))
    return R.from_matrix(np.array(out))


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ego", required=True, help="ego session directory")
    p.add_argument("--ego-subdir", default="robot_eef_scene_camera_axis_corrected")
    p.add_argument("--ego-hz", type=float, default=30.0)
    p.add_argument("--real", default=None, help="real-episode directory, for the jitter reference line")
    p.add_argument("--pose-key", default="tcp_tip_pose", choices=["tcp_tip_pose", "tcp_pose", "flange_pose", "fk_pose"])
    p.add_argument("--smooth-alpha", type=float, default=None,
                    help="overlay a dashed EMA-smoothed orientation at this alpha (e.g. 0.15), for comparison")
    p.add_argument("--euler-seq", default="xyz", help="scipy Euler sequence for the orientation panel")
    p.add_argument("--out", default=None, help="output PNG path (default: <ego>/pose_timeseries.png)")
    args = p.parse_args()

    session_dir = as_abs(args.ego)
    csv_path = find_ego_csv(session_dir, args.ego_subdir)
    if csv_path is None:
        raise SystemExit(f"no robot_eef_trajectory.csv under {session_dir}")
    traj = load_ego_csv(csv_path, args.ego_hz)
    valid = traj.valid if traj.valid is not None else np.ones(traj.n, dtype=bool)
    t = np.arange(traj.n) / traj.hz

    real_jit_mean = real_jit_std = None
    if args.real:
        reals = load_real_dir(as_abs(args.real), args.pose_key)
        if reals:
            steps = [frame_step_deg_s(r.rot, r.hz) for r in reals]
            per_episode_mean = np.array([s.mean() for s in steps])
            real_jit_mean, real_jit_std = float(per_episode_mean.mean()), float(per_episode_mean.std())

    pos_mm = 1000.0 * traj.pos
    euler = unwrapped_euler_deg(traj.rot, args.euler_seq)
    step_deg_s = frame_step_deg_s(traj.rot, traj.hz)

    smoothed = None
    if args.smooth_alpha is not None:
        sm_rot = ema_smooth_rot(traj.rot, args.smooth_alpha)
        smoothed = {
            "euler": unwrapped_euler_deg(sm_rot, args.euler_seq),
            "step_deg_s": frame_step_deg_s(sm_rot, traj.hz),
        }

    fig, axes = plt.subplots(
        4, 1, figsize=(12.5, 10.5), sharex=True,
        gridspec_kw={"height_ratios": [1.1, 1.1, 1.0, 0.55], "hspace": 0.12},
    )
    fig.subplots_adjust(right=0.84)
    fig.patch.set_facecolor(SURFACE)
    ax_pos, ax_rot, ax_jit, ax_grasp = axes
    LEGEND_KW = dict(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
                      fontsize=9, labelcolor=INK_SECONDARY, handlelength=1.6)

    # invalid-frame shading (a WiLoR miss, interpolated/carried over)
    if not valid.all():
        bad = np.where(~valid)[0]
        for ax in axes:
            ax.scatter(t[bad], np.full(bad.shape, ax.get_ylim()[0]), marker="|", s=1, color=INK_MUTED, zorder=1)

    # --- 1. position -------------------------------------------------------
    for comp, color, label in zip(pos_mm.T, (COLOR_1, COLOR_2, COLOR_3), ("x", "y", "z")):
        ax_pos.plot(t, comp, color=color, linewidth=1.6, label=label, zorder=3)
    style_axis(ax_pos)
    ax_pos.set_ylabel("position (mm)")
    ax_pos.legend(**LEGEND_KW)
    ax_pos.set_title(f"{traj.name}  —  pose components over time  ({traj.n} frames @ {traj.hz:g} Hz)",
                      color=INK_PRIMARY, fontsize=13, loc="left", pad=28)

    # --- 2. orientation ------------------------------------------------------
    for comp, color, label in zip(euler.T, (COLOR_1, COLOR_2, COLOR_3), ("roll", "pitch", "yaw")):
        ax_rot.plot(t, comp, color=color, linewidth=1.6, label=label, zorder=3)
    if smoothed is not None:
        for comp, color in zip(smoothed["euler"].T, (COLOR_1, COLOR_2, COLOR_3)):
            ax_rot.plot(t, comp, color=color, linewidth=1.2, linestyle=(0, (3, 2)), alpha=0.85, zorder=4)
        ax_rot.plot([], [], color=INK_SECONDARY, linestyle=(0, (3, 2)), linewidth=1.2,
                    label=f"EMA α={args.smooth_alpha:g}")
    style_axis(ax_rot)
    ax_rot.set_ylabel(f"orientation (deg, Euler {args.euler_seq})")
    ax_rot.legend(**LEGEND_KW)

    # --- 3. jitter: frame-to-frame angular step -----------------------------
    t_mid = 0.5 * (t[:-1] + t[1:])
    ego_mean = float(step_deg_s.mean())
    ax_jit.plot(t_mid, step_deg_s, color=COLOR_1, linewidth=1.1, zorder=3, label="frame step (raw)")
    ax_jit.fill_between(t_mid, 0, step_deg_s, color=COLOR_1, alpha=0.12, zorder=2)
    if smoothed is not None:
        ax_jit.plot(t_mid, smoothed["step_deg_s"], color=INK_SECONDARY, linewidth=1.2,
                    linestyle=(0, (3, 2)), alpha=0.9, zorder=4,
                    label=f"EMA α={args.smooth_alpha:g}")
    ax_jit.axhline(ego_mean, color=COLOR_1, linewidth=1.0, linestyle=":", alpha=0.7,
                    label=f"ego mean {ego_mean:.1f} deg/s")
    if real_jit_mean is not None:
        ax_jit.axhline(real_jit_mean, color=INK_MUTED, linewidth=1.2, linestyle="--", zorder=3,
                        label=f"real floor {real_jit_mean:.1f}±{real_jit_std:.1f} deg/s")
    style_axis(ax_jit)
    ax_jit.set_ylabel("frame-to-frame\nangular step (deg/s)")
    ax_jit.legend(**LEGEND_KW)

    # --- 4. grasp state ------------------------------------------------------
    ax_grasp.step(t, traj.grasp, where="post", color=INK_SECONDARY, linewidth=1.4, zorder=3)
    ax_grasp.fill_between(t, 0, traj.grasp, step="post", color=INK_SECONDARY, alpha=0.10, zorder=2)
    ax_grasp.set_ylim(-0.15, 1.15)
    ax_grasp.set_yticks([0, 1])
    ax_grasp.set_yticklabels(["open", "closed"])
    style_axis(ax_grasp)
    ax_grasp.set_xlabel("time (s)")

    # --- shared anchor markers across all panels ------------------------------
    labels = traj.event_labels
    for ax in axes:
        for idx, lab in zip(traj.events, labels):
            ax.axvline(t[idx], color=INK_MUTED, linewidth=0.9, linestyle="-", alpha=0.35, zorder=1)
    ymax = ax_pos.get_ylim()[1]
    for idx, lab in zip(traj.events, labels):
        ax_pos.text(t[idx], ymax, f" {lab}", color=INK_MUTED, fontsize=8, va="bottom", ha="left", rotation=0)

    for ax in axes:
        ax.set_xlim(t[0], t[-1])
    ax_grasp.xaxis.set_major_locator(MultipleLocator(1.0))

    out_path = as_abs(args.out) if args.out else session_dir / "pose_timeseries.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")
    print(f"  ego frame-to-frame angular step: mean {ego_mean:.1f} deg/s"
          + (f"   (real floor: {real_jit_mean:.1f} ± {real_jit_std:.1f} deg/s)" if real_jit_mean else ""))


if __name__ == "__main__":
    main()
