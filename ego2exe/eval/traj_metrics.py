#!/usr/bin/env python3
"""Trajectory containers, loaders and distance primitives for ego2exe evaluation.

Everything downstream of this module works on :class:`Traj` objects and on a
single pooled pairwise distance matrix.  Keeping the primitives here means the
metric layers (L0-L3) never touch file formats or DTW internals.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# Resampling length used for every trajectory comparison.  Both sides are put on
# this grid so that duration differences (human is ~2.5x faster than teleop) do
# not leak into the spatial numbers.
N_RESAMPLE = 200

try:  # numba is a large win here (~1800 DTW pairs for 30 ego + 31 real)
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - fallback keeps the script runnable
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def wrap(fn):
            return fn

        return wrap


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------


@dataclass
class Traj:
    """One demonstration, already in robot-base frame."""

    name: str
    kind: str  # "real" | "ego"
    pos: np.ndarray  # (N, 3) metres
    rot: R  # (N,) rotations, gripper frame in base
    grasp: np.ndarray  # (N,) 0..1
    hz: float
    events: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    valid: np.ndarray | None = None  # (N,) bool, None means all valid

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def event_dirs(self) -> np.ndarray:
        """+1 where the gripper closes at that event, -1 where it opens.

        Derived from the signal, never from the event index: on this dataset the
        episodes start *closed*, so the first toggle is an open, and assuming
        alternation from index parity labels every anchor backwards.
        """
        binary = (self.grasp > 0.5).astype(int)
        return np.array([1 if binary[i + 1] > binary[i] else -1 for i in self.events], dtype=int)

    @property
    def event_labels(self) -> list[str]:
        return [
            f"E{k + 1}_{'close' if d > 0 else 'open'}" for k, d in enumerate(self.event_dirs)
        ]

    @property
    def duration_s(self) -> float:
        return self.n / self.hz

    @property
    def path_len_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.pos, axis=0), axis=1).sum())

    def resampled_pos(self, n: int = N_RESAMPLE) -> np.ndarray:
        return resample(self.pos, n)

    def resampled_rot(self, n: int = N_RESAMPLE) -> R:
        idx = np.clip(np.round(np.linspace(0, self.n - 1, n)).astype(int), 0, self.n - 1)
        return self.rot[idx]


def resample(arr: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    """Linear resample an (N, D) array onto n points of normalised arclength-time."""
    t = np.linspace(0.0, 1.0, len(arr))
    tn = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(tn, t, arr[:, i]) for i in range(arr.shape[1])], axis=1)


def grasp_events(grasp: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Indices where the binarised gripper signal toggles."""
    binary = (np.asarray(grasp) > threshold).astype(int)
    return np.where(np.diff(binary) != 0)[0]


def crop_to_segment(t: Traj, start_event: int | None, end_event: int | None) -> Traj | None:
    """Restrict a trajectory to the span between two gripper-event boundaries.

    ``start_event``/``end_event`` are 0-based indices into ``t.events`` - the
    same convention as ``--anchors``. ``None`` means "episode start" /
    "episode end" respectively, so ``crop_to_segment(t, None, None)`` is a
    (cheap) full copy. Returns ``None`` if this trajectory does not have the
    requested event - the caller must drop it, not guess a boundary.

    Example: "from the 1st open to the 1st close to the 2nd open" is
    ``crop_to_segment(t, 0, 2)`` - it does not need to name the event in the
    middle, only the two that bound the span.
    """
    n = t.n
    if start_event is None:
        a = 0
    else:
        if not (0 <= start_event < len(t.events)):
            return None
        a = int(t.events[start_event])
    if end_event is None:
        b = n - 1
    else:
        if not (0 <= end_event < len(t.events)):
            return None
        b = int(t.events[end_event])
    if b <= a:
        return None
    sl = slice(a, b + 1)
    cropped_grasp = t.grasp[sl].copy()
    return Traj(
        name=t.name,
        kind=t.kind,
        pos=t.pos[sl].copy(),
        rot=t.rot[sl],
        grasp=cropped_grasp,
        hz=t.hz,
        events=grasp_events(cropped_grasp),
        valid=None if t.valid is None else t.valid[sl].copy(),
    )


# Default fingertip offset along the flange's local +X, metres. This is the
# NERO arm's ``site:tcp`` convention (ego2exe/README.md): R_tcp = R_flange,
# p_tcp = p_flange + R_flange @ [offset, 0, 0]. Robot-specific; pass a
# different value via --tcp-offset-m for another arm.
DEFAULT_TCP_OFFSET_M = 0.18


def flange_to_tip_pose(flange_xyzrpy: np.ndarray, offset_m: float = DEFAULT_TCP_OFFSET_M) -> np.ndarray:
    """Recompute a fingertip TCP pose from a flange pose, matching the ego side's convention.

    The robot's own ``poses.tcp_pose`` reports the GRIPPER-CENTER point: empirically
    verified against this dataset, it equals ``flange + R_flange @ [0.13, 0, 0]``
    with the flange's orientation unchanged (``tcp_offset_flange_frame`` in the
    jsonl metadata). The ego2exe pipeline's own notion of "TCP" - what mink IK
    solves for and what hand2gripper's EEF frame is meant to land on - is the
    FINGERTIP, ``flange + R_flange @ [0.18, 0, 0]`` (``site:tcp`` in
    ego2exe/README.md, corrected from 0.13 to 0.18 on 2026-08-18). Same axis,
    different magnitude: comparing the real robot's ``tcp_pose`` field directly
    against ego's reconstructed pose compares two points ~50 mm apart along the
    approach axis. This recomputes the real side from ``flange_pose`` instead,
    so both sides describe the same physical point.

    ``flange_xyzrpy``: (..., 6) array, ``[x, y, z, roll, pitch, yaw]``, base
    frame, RPY ZYX (same layout as the jsonl ``poses.*`` fields). Returns the
    same layout.
    """
    pos = flange_xyzrpy[..., :3]
    rot = R.from_euler("ZYX", flange_xyzrpy[..., 3:][..., ::-1])
    tip_pos = pos + rot.apply(np.array([offset_m, 0.0, 0.0]))
    rpy = flange_xyzrpy[..., 3:]  # R_tip == R_flange, orientation is unchanged
    return np.concatenate([tip_pos, rpy], axis=-1)


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


def load_real_jsonl(
    path: Path,
    pose_key: str = "tcp_tip_pose",
    min_samples: int = 20,
    tcp_offset_m: float = DEFAULT_TCP_OFFSET_M,
) -> Traj | None:
    """Load one nero_teleop_jsonl episode.

    The pose block is ``[x, y, z, roll, pitch, yaw]`` in base frame with RPY ZYX,
    so the euler angles are reversed before handing them to scipy.

    ``pose_key="tcp_tip_pose"`` (the default) recomputes the fingertip TCP from
    ``poses.flange_pose`` via :func:`flange_to_tip_pose` instead of reading the
    robot's own ``poses.tcp_pose`` field, which is the gripper-center point, not
    the fingertip ego assumes - see that function's docstring. Pass
    ``pose_key="tcp_pose"`` to fall back to the robot-reported (gripper-center)
    field, or ``"flange_pose"`` / ``"fk_pose"`` to read those directly.
    """
    samples = []
    meta: dict = {}
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "metadata":
                meta = rec
            elif rec.get("kind") == "sample":
                samples.append(rec)
    if len(samples) < min_samples:
        return None

    if pose_key == "tcp_tip_pose":
        flange = np.array([s["poses"]["flange_pose"] for s in samples], dtype=np.float64)
        pose = flange_to_tip_pose(flange, tcp_offset_m)
    else:
        pose = np.array([s["poses"][pose_key] for s in samples], dtype=np.float64)
    grasp = np.array([s["gripper"]["action_grasp"] for s in samples], dtype=np.float64)
    rot = R.from_euler("ZYX", pose[:, 3:][:, ::-1])
    hz = float(meta.get("sample_hz", 20.0))
    return Traj(
        name=path.stem,
        kind="real",
        pos=pose[:, :3],
        rot=rot,
        grasp=grasp,
        hz=hz,
        events=grasp_events(grasp),
    )


def load_ego_csv(path: Path, hz: float = 30.0) -> Traj:
    """Load a robot_eef_trajectory.csv produced by preprocess_export_eef.py."""
    rows = list(csv.DictReader(open(path)))
    pos = np.array([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])] for r in rows])
    quat = np.array([[float(r["qx"]), float(r["qy"]), float(r["qz"]), float(r["qw"])] for r in rows])
    grasp = np.array([float(r["grasp"]) for r in rows])
    valid = np.array([str(r.get("valid", "True")).lower() == "true" for r in rows])
    return Traj(
        name=path.parent.parent.name,
        kind="ego",
        pos=pos,
        rot=R.from_quat(quat),
        grasp=grasp,
        hz=hz,
        events=grasp_events(grasp),
        valid=valid,
    )


def find_ego_csv(session_dir: Path, subdir: str = "robot_eef_scene_camera_axis_corrected") -> Path | None:
    direct = session_dir / subdir / "robot_eef_trajectory.csv"
    if direct.is_file():
        return direct
    hits = sorted(session_dir.glob("*/robot_eef_trajectory.csv"))
    return hits[0] if hits else None


def load_real_dir(
    path: Path, pose_key: str = "tcp_tip_pose", tcp_offset_m: float = DEFAULT_TCP_OFFSET_M
) -> list[Traj]:
    """Load a real-robot reference set, refusing an ego directory by mistake.

    In the DATA_new layout an ego directory sits next to the real one under the
    same task and carries .jsonl files with the *identical* schema and filenames
    - the arm simply sat parked while the human demonstrated, so every pose is
    frozen. Globbing the wrong directory would silently load those as reference
    episodes and destroy the noise floor, so a motionless "demonstration" is
    treated as a hard error rather than a warning: no genuine teleop episode has
    a stationary TCP.
    """
    out = [load_real_jsonl(f, pose_key, tcp_offset_m=tcp_offset_m) for f in sorted(path.glob("*.jsonl"))]
    trajs = [t for t in out if t is not None]
    if trajs:
        spans = [float(np.linalg.norm(t.pos.max(axis=0) - t.pos.min(axis=0))) for t in trajs]
        if max(spans) < 1e-3:  # 1 mm across the whole set
            raise SystemExit(
                f"{path} looks like an ego directory, not a real-robot one: the arm never "
                f"moves in any of its {len(trajs)} episodes (max TCP span "
                f"{1000 * max(spans):.2f} mm). Point --real at the *nero* directory of the task."
            )
    return trajs


def load_ego_dirs(paths: list[Path], subdir: str, hz: float) -> list[Traj]:
    out = []
    for p in paths:
        csv_path = find_ego_csv(p, subdir)
        if csv_path is None:
            print(f"  [warn] no robot_eef_trajectory.csv under {p}")
            continue
        out.append(load_ego_csv(csv_path, hz))
    return out


# --------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------


@njit(cache=True)
def _dtw_cost(C):  # pragma: no cover - numba kernel
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = D[i - 1, j]
            b = D[i, j - 1]
            c = D[i - 1, j - 1]
            mn = a if a < b else b
            if c < mn:
                mn = c
            D[i, j] = C[i - 1, j - 1] + mn
    return D[n, m]


@njit(cache=True)
def _dtw_path(C):  # pragma: no cover - numba kernel
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a = D[i - 1, j]
            b = D[i, j - 1]
            c = D[i - 1, j - 1]
            mn = a if a < b else b
            if c < mn:
                mn = c
            D[i, j] = C[i - 1, j - 1] + mn
    pi = np.empty(n + m, np.int64)
    pj = np.empty(n + m, np.int64)
    k = 0
    i = n
    j = m
    while i > 0 and j > 0:
        pi[k] = i - 1
        pj[k] = j - 1
        k += 1
        a = D[i - 1, j]
        b = D[i, j - 1]
        c = D[i - 1, j - 1]
        if c <= a and c <= b:
            i -= 1
            j -= 1
        elif a <= b:
            i -= 1
        else:
            j -= 1
    return pi[:k][::-1].copy(), pj[:k][::-1].copy()


def _cost_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=2))


def dtw_dist(X: np.ndarray, Y: np.ndarray) -> float:
    """DTW distance normalised by the mean sequence length -> metres per step."""
    return float(_dtw_cost(_cost_matrix(X, Y)) / ((len(X) + len(Y)) / 2.0))


def dtw_pairs(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs of the optimal warping path."""
    return _dtw_path(_cost_matrix(X, Y))


def rot_dist(A: R, B: R) -> float:
    """Mean geodesic angle (deg) between two equal-length rotation sequences."""
    return float(np.degrees(np.linalg.norm((A.inv() * B).as_rotvec(), axis=1)).mean())


def rot_dist_dtw(a: Traj, b: Traj, n: int = N_RESAMPLE) -> float:
    """Mean geodesic angle along the position-DTW correspondence.

    Orientation is paired through the *position* warping path rather than a
    separate rotation DTW: we want "is the gripper oriented correctly at the
    place it is", not "does this orientation appear somewhere in the episode".
    """
    pa, pb = dtw_pairs(a.resampled_pos(n), b.resampled_pos(n))
    ra = a.resampled_rot(n)[pa]
    rb = b.resampled_rot(n)[pb]
    return rot_dist(ra, rb)


def geodesic_deg(a: R, b: R) -> float:
    return float(np.degrees(np.linalg.norm((a.inv() * b).as_rotvec())))


def mean_rotation(rots: R) -> R:
    """Chordal L2 mean on SO(3) (scipy uses the quaternion outer-product eigvec)."""
    return rots.mean()
