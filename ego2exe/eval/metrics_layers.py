#!/usr/bin/env python3
"""L0-L3 metric layers for ego -> robot-base EEF trajectory evaluation.

Design invariants
-----------------
* **Everything is reported as a ratio to a noise floor.**  ``rho = D(ego, real)
  / D(real, real)``.  Absolute millimetres are not comparable across tasks (the
  same pipeline scores 17 mm on one task and 57 mm on another purely because the
  reference sets differ); ``rho`` is.  ``rho -> 1`` means the ego trajectory sits
  as close to the real set as one real demo sits to the others.
* **The noise floor uses a leave-one-out convention.**  The quantity under test
  is "one ego vs N real", so the reference must be "one real vs the other N-1",
  not the all-pairs mean.  Same estimator, comparable spread, and it yields a
  z-score for free.
* **L0 never blocks.**  Gates are reported as pass/fail and everything
  downstream is computed regardless.
* **Nothing is fitted and scored on the same data.**  Any global correction is
  validated leave-one-out; the self-fitted residual is reported next to it only
  to show the size of the overfit.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from traj_metrics import (
    N_RESAMPLE,
    Traj,
    dtw_dist,
    dtw_pairs,
    geodesic_deg,
    grasp_events,
    rot_dist,
)


# ==========================================================================
# pooled distance matrices
# ==========================================================================


def _real_cache_path(cache_key: str, cache_dir: Path | None) -> Path:
    root = cache_dir or (Path(__file__).resolve().parents[2] / "outputs" / "_cache" / "real_dtw")
    return root / (hashlib.sha1(cache_key.encode()).hexdigest()[:16] + ".npz")


def _load_real_block(cache_key: str | None, cache_dir: Path | None, names: list[str]) -> dict | None:
    """Cached real-vs-real distances, or None when absent or stale.

    The episode names are stored alongside and re-checked: a cache keyed only by
    the directory path would survive an episode being added or removed and then
    silently return a block of the wrong shape or ordering.
    """
    if not cache_key or not names:
        return None
    path = _real_cache_path(cache_key, cache_dir)
    if not path.is_file():
        return None
    try:
        z = np.load(path, allow_pickle=False)
        if list(z["names"]) != names:
            return None
        return {f: z[f] for f in ("abs", "shape", "offset", "rot")}
    except Exception:
        return None  # a corrupt or older cache file just means "recompute"


def _save_real_block(cache_key: str, cache_dir: Path | None, names: list[str], blocks: dict) -> None:
    path = _real_cache_path(cache_key, cache_dir)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, names=np.array(names), **blocks)
    except OSError:
        pass  # caching is an optimisation; never fail the evaluation over it


class DistanceBook:
    """All pairwise distances over the pooled ego+real set, computed once.

    Three position variants are kept because their *difference* is the main
    diagnostic:

    ``abs``     raw base-frame DTW - what actually matters downstream.
    ``shape``   DTW after removing each trajectory's own centroid - the part of
                the error that a rigid re-placement cannot fix.
    ``offset``  distance between centroids - the part it can.
    """

    def __init__(self, trajs: list[Traj], n: int = N_RESAMPLE, verbose: bool = True,
                 cache_key: str | None = None, cache_dir: Path | None = None):
        self.trajs = trajs
        self.n = n
        self.names = [t.name for t in trajs]
        self.kinds = np.array([t.kind for t in trajs])
        self.is_ego = self.kinds == "ego"
        self.is_real = self.kinds == "real"

        self._pos = [t.resampled_pos(n) for t in trajs]
        self._cen = [p.mean(axis=0) for p in self._pos]
        self._posc = [p - c for p, c in zip(self._pos, self._cen)]
        self._rot = [t.resampled_rot(n) for t in trajs]

        m = len(trajs)
        self.abs = np.zeros((m, m))
        self.shape = np.zeros((m, m))
        self.offset = np.zeros((m, m))
        self.rot = np.zeros((m, m))

        # The real-vs-real block is ~1/3 of the work and identical on every run
        # against the same reference set, so it is cached on disk keyed by the
        # inputs that can change it.
        real_idx = np.where(self.is_real)[0]
        cached = _load_real_block(cache_key, cache_dir, [self.names[i] for i in real_idx])
        if cached is not None:
            for field, block in cached.items():
                M = getattr(self, field)
                M[np.ix_(real_idx, real_idx)] = block

        pairs = [(i, j) for i in range(m) for j in range(i + 1, m)
                 if cached is None or not (self.is_real[i] and self.is_real[j])]
        total = len(pairs)
        for done, (i, j) in enumerate(pairs, start=1):
            self.abs[i, j] = self.abs[j, i] = dtw_dist(self._pos[i], self._pos[j])
            self.shape[i, j] = self.shape[j, i] = dtw_dist(self._posc[i], self._posc[j])
            self.offset[i, j] = self.offset[j, i] = float(np.linalg.norm(self._cen[i] - self._cen[j]))
            pa, pb = dtw_pairs(self._pos[i], self._pos[j])
            self.rot[i, j] = self.rot[j, i] = rot_dist(self._rot[i][pa], self._rot[j][pb])
            if verbose and total and sys.stdout.isatty():
                print(f"\r  pairwise distances: {done}/{total} ({100.0 * done / total:.0f}%)",
                      end="", flush=True)
        if verbose:
            note = "" if cached is None else f", {len(real_idx) * (len(real_idx) - 1) // 2} real-real from cache"
            print(f"\r  pairwise distances: {total}/{total} (100%){note}")

        if cached is None and cache_key:
            _save_real_block(
                cache_key, cache_dir, [self.names[i] for i in real_idx],
                {f: getattr(self, f)[np.ix_(real_idx, real_idx)] for f in ("abs", "shape", "offset", "rot")},
            )

    def block(self, field: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return getattr(self, field)[np.ix_(rows, cols)]

    def loo_floor(self, field: str) -> tuple[float, float, np.ndarray]:
        """Leave-one-out noise floor over the real set: mean, std, per-episode."""
        idx = np.where(self.is_real)[0]
        per = np.array(
            [np.mean([getattr(self, field)[i, j] for j in idx if j != i]) for i in idx]
        )
        return float(per.mean()), float(per.std()), per

    def ego_to_real(self, field: str) -> np.ndarray:
        """(n_ego,) mean distance of each ego trajectory to the whole real set."""
        return self.block(field, np.where(self.is_ego)[0], np.where(self.is_real)[0]).mean(axis=1)

    def real_centroid(self) -> np.ndarray:
        return np.mean([self._cen[i] for i in np.where(self.is_real)[0]], axis=0)

    def offset_floor(self) -> tuple[float, float]:
        """Noise floor for the centroid offset, matching how ego is scored.

        Both sides measure *displacement from the centre of the reference set*,
        so the floor must be ``||c_i - mean(other c)||`` and not the mean pairwise
        centroid distance - by Jensen the latter is systematically larger and
        would deflate every ``rho_offset``.
        """
        idx = np.where(self.is_real)[0]
        cen = np.array([self._cen[i] for i in idx])
        per = np.array(
            [np.linalg.norm(cen[i] - np.delete(cen, i, axis=0).mean(axis=0)) for i in range(len(idx))]
        )
        return float(per.mean()), float(per.std())


# ==========================================================================
# L0 - gates (reported, never blocking)
# ==========================================================================


def l0_gates(
    ego: Traj,
    reals: list[Traj],
    bbox_margin_m: float = 0.02,
    joint_vel_ref: np.ndarray | None = None,
) -> dict:
    """Cheap sanity checks that need no alignment.  All are informational."""
    all_pos = np.vstack([t.pos for t in reals])
    lo = all_pos.min(axis=0) - bbox_margin_m
    hi = all_pos.max(axis=0) + bbox_margin_m
    inside = ((ego.pos >= lo) & (ego.pos <= hi)).all(axis=1)
    per_axis_out = [float(((ego.pos[:, k] < lo[k]) | (ego.pos[:, k] > hi[k])).mean()) for k in range(3)]

    z_min_real = float(all_pos[:, 2].min())
    below = float((ego.pos[:, 2] < z_min_real).mean())

    real_events = [len(t.events) for t in reals]
    mode_events = int(np.bincount(real_events).argmax()) if real_events else 0
    ego_events = len(ego.events)

    valid_rate = 1.0 if ego.valid is None else float(ego.valid.mean())

    step_mm = 1000.0 * np.linalg.norm(np.diff(ego.pos, axis=0), axis=1)
    real_step = np.concatenate(
        [1000.0 * np.linalg.norm(np.diff(t.pos, axis=0), axis=1) for t in reals]
    )
    # Speeds are compared per second, not per frame: ego is 30 Hz, real 20 Hz.
    ego_speed = step_mm * ego.hz
    real_speed = real_step * reals[0].hz
    speed_p99 = float(np.percentile(real_speed, 99))
    speed_viol = float((ego_speed > speed_p99).mean())

    gates = {
        "grasp_events_match": {
            "pass": ego_events == mode_events,
            # Fail-rate (0/1), matching the other gates' [0, 1] scale so a table of
            # per-gate means stays comparable. The raw magnitude of the mismatch is
            # `abs(ego - real_mode)`, printed separately per episode as "ego X vs real mode Y".
            "value": float(ego_events != mode_events),
            "ego": ego_events,
            "real_mode": mode_events,
            "event_count_diff": int(abs(ego_events - mode_events)),
            "real_counts": {str(k): int(v) for k, v in zip(*np.unique(real_events, return_counts=True))},
        },
        "workspace_entry_rate": {
            "pass": float(inside.mean()) >= 0.99,
            "value": float(inside.mean()),
            "per_axis_outside": per_axis_out,
            "bbox_lo": lo.tolist(),
            "bbox_hi": hi.tolist(),
        },
        "table_penetration_rate": {
            "pass": below <= 0.01,
            "value": below,
            "real_z_min_m": z_min_real,
            "ego_z_min_m": float(ego.pos[:, 2].min()),
        },
        "valid_frame_rate": {"pass": valid_rate >= 0.95, "value": valid_rate},
        "speed_feasibility": {
            "pass": speed_viol <= 0.02,
            "value": speed_viol,
            "real_p99_mm_s": speed_p99,
            "ego_max_mm_s": float(ego_speed.max()),
        },
    }
    if joint_vel_ref is not None:
        gates["joint_velocity"] = {"pass": None, "note": "supply IK output to enable"}
    gates["_n_failed"] = int(sum(1 for k, v in gates.items() if not k.startswith("_") and v.get("pass") is False))
    return gates


# ==========================================================================
# L1 - headline ratios
# ==========================================================================


def l1_ratios(book: DistanceBook) -> dict:
    """rho_pos, rho_rot and their geometric mean, per ego episode."""
    floors = {f: book.loo_floor(f) for f in ("abs", "shape", "offset", "rot")}
    ego_idx = np.where(book.is_ego)[0]
    out = {"floors": {f: {"mean": m, "std": s} for f, (m, s, _) in floors.items()}, "per_ego": {}}
    for row, gi in enumerate(ego_idx):
        d_pos = book.ego_to_real("abs")[row]
        d_rot = book.ego_to_real("rot")[row]
        rho_pos = d_pos / floors["abs"][0]
        rho_rot = d_rot / floors["rot"][0]
        _, _, per_real = floors["abs"]
        out["per_ego"][book.names[gi]] = {
            "D_pos_mm": 1000.0 * d_pos,
            "D_rot_deg": d_rot,
            "rho_pos": rho_pos,
            "rho_rot": rho_rot,
            "rho_se3": float(np.sqrt(rho_pos * rho_rot)),
            "z_pos": float((d_pos - per_real.mean()) / max(per_real.std(), 1e-12)),
        }
    return out


# ==========================================================================
# L2 - diagnostic decomposition
# ==========================================================================


def l2_offset_shape(book: DistanceBook) -> dict:
    """Split the position error into a placement part and a shape part."""
    f_abs, _, _ = book.loo_floor("abs")
    f_shp, _, _ = book.loo_floor("shape")
    f_off, _ = book.offset_floor()
    real_centroid = book.real_centroid()

    out = {
        "floors_mm": {"abs": 1000 * f_abs, "shape": 1000 * f_shp, "offset": 1000 * f_off},
        "per_ego": {},
    }
    for row, gi in enumerate(np.where(book.is_ego)[0]):
        d_abs = book.ego_to_real("abs")[row]
        d_shp = book.ego_to_real("shape")[row]
        d_off = float(np.linalg.norm(book._cen[gi] - real_centroid))
        out["per_ego"][book.names[gi]] = {
            "D_abs_mm": 1000 * d_abs,
            "D_shape_mm": 1000 * d_shp,
            "D_offset_mm": 1000 * d_off,
            "rho_abs": d_abs / f_abs,
            "rho_shape": d_shp / f_shp,
            "rho_offset": d_off / f_off,
            "centroid_offset_mm": (1000 * (book._cen[gi] - real_centroid)).tolist(),
        }
    return out


def anchor_reference(reals: list[Traj], select: list[int] | None = None) -> dict:
    """Per-anchor real-side statistics: the reference every ego is scored against.

    Computed once for the whole run rather than per ego episode. These values
    depend only on the real set, so recomputing them per episode both duplicated
    them ~30x in the JSON and dominated the runtime - the geodesic median rotation
    alone is O(n^2) over the real anchor cloud.
    """
    counts = [len(t.events) for t in reals]
    mode = int(np.bincount(counts).argmax()) if counts else 0
    usable = [t for t in reals if len(t.events) == mode]
    if not usable:
        return {"available": False, "n_anchors": mode, "anchors": {}}

    keep = list(range(mode)) if select is None else [k for k in select if 0 <= k < mode]
    real_dirs = np.array([t.event_dirs for t in usable])
    out = {}
    for k in keep:
        real_pos = np.array([t.pos[t.events[k]] for t in usable])
        real_rot = R.concatenate([t.rot[t.events[k]] for t in usable])
        d_real = int(np.sign(real_dirs[:, k].sum()))
        real_g = np.array([[t.grasp[t.events[k]], t.grasp[t.events[k] + 1]] for t in usable])
        med_pos = np.median(real_pos, axis=0)
        med_rot = _geodesic_median_rotation(real_rot)
        phases = [100.0 * t.events[k] / t.n for t in usable]
        out[k] = {
            "index": k,
            "label": f"E{k + 1}_{'close' if d_real > 0 else 'open'}",
            "gripper_transition": "close" if d_real > 0 else "open",
            "dir": d_real,
            "dir_unanimous": bool(np.all(real_dirs[:, k] == d_real)),
            "grasp_before_after": [float(real_g[:, 0].mean()), float(real_g[:, 1].mean())],
            "median_pos_m": med_pos.tolist(),
            "pos_std_mm": (1000 * real_pos.std(axis=0)).tolist(),
            # noise floor: spread of the real anchor cloud, same estimator as the error
            "floor_pos_mm": 1000 * float(np.linalg.norm(real_pos - med_pos, axis=1).mean()),
            "floor_rot_deg": float(np.mean([geodesic_deg(med_rot, r) for r in real_rot])),
            "phase_pct_mean": float(np.mean(phases)),
            "phase_pct_std": float(np.std(phases)),
            "_med_pos": med_pos,
            "_med_rot": med_rot,
        }
    return {"available": True, "n_anchors": mode, "selected": keep, "n_real": len(usable), "anchors": out}


def l2_anchors(ego: Traj, reals: list[Traj], select: list[int] | None = None,
               reference: dict | None = None) -> dict:
    """Absolute pose error at every contact anchor, reported per anchor.

    Object positions are fixed within a layout, so the real TCP pose at each
    gripper toggle is a measurement of that contact pose in base frame - the
    only absolute ground truth available without hand-pose tracking.  Anchors
    are reported individually because not every toggle is task-relevant; use
    ``select`` to name the ones that are.

    ``reference`` is :func:`anchor_reference` output; pass it in to avoid
    recomputing the real-side statistics once per ego episode.
    """
    ref = reference if reference is not None else anchor_reference(reals, select)
    mode = ref.get("n_anchors", 0)
    if not ref.get("available") or len(ego.events) != mode:
        return {
            "available": False,
            "reason": f"event count mismatch (ego {len(ego.events)}, real mode {mode})",
            "n_anchors": mode,
        }

    ego_dirs = ego.event_dirs
    anchors = []
    for k, r in ref["anchors"].items():
        med_pos, med_rot = r["_med_pos"], r["_med_rot"]
        floor_pos_m = r["floor_pos_mm"] / 1000.0
        ego_pos = ego.pos[ego.events[k]]
        err = ego_pos - med_pos
        e_pos = float(np.linalg.norm(err))
        e_rot = geodesic_deg(med_rot, ego.rot[ego.events[k]])
        phase_ego = 100.0 * ego.events[k] / ego.n
        anchors.append(
            {
                "index": k,
                "label": r["label"],
                "gripper_transition": r["gripper_transition"],
                "grasp_ego_before_after": [
                    float(ego.grasp[ego.events[k]]),
                    float(ego.grasp[ego.events[k] + 1]),
                ],
                "direction_agrees_with_ego": int(ego_dirs[k]) == r["dir"],
                "ego_pos_m": ego_pos.tolist(),
                "err_vec_mm": (1000 * err).tolist(),
                "err_pos_mm": 1000 * e_pos,
                "err_rot_deg": e_rot,
                "rho_pos": e_pos / max(floor_pos_m, 1e-9),
                "rho_rot": e_rot / max(r["floor_rot_deg"], 1e-9),
                "phase_ego_pct": phase_ego,
                "phase_z": (phase_ego - r["phase_pct_mean"]) / max(r["phase_pct_std"], 1e-9),
                # kept for the printed table; the authoritative copy lives in
                # L2_anchors.reference, not repeated per episode in the JSON
                "floor_pos_mm": r["floor_pos_mm"],
                "floor_rot_deg": r["floor_rot_deg"],
                "grasp_real_before_after": r["grasp_before_after"],
            }
        )
    return {"available": True, "n_anchors": mode, "selected": ref["selected"], "anchors": anchors}


def _geodesic_median_rotation(rots: R) -> R:
    """The member of the set minimising total geodesic distance to the others."""
    costs = [sum(geodesic_deg(a, b) for b in rots) for a in rots]
    return rots[int(np.argmin(costs))]


def l2_segments(ego: Traj, reals: list[Traj], per_seg: int = 40) -> dict:
    """rho per task phase, using the gripper events as segment boundaries.

    Free-space error and contact-phase error are not equally costly downstream,
    so the aggregate is broken out by segment rather than pooled.
    """
    counts = [len(t.events) for t in reals]
    mode = int(np.bincount(counts).argmax()) if counts else 0
    usable = [t for t in reals if len(t.events) == mode]
    if not usable or len(ego.events) != mode:
        return {"available": False, "reason": f"event count mismatch (ego {len(ego.events)}, real mode {mode})"}

    def seg_idx(traj: Traj) -> list[np.ndarray]:
        ev = traj.events
        bounds = [(0, ev[0])] + [(ev[i], ev[i + 1]) for i in range(len(ev) - 1)] + [(ev[-1], traj.n - 1)]
        return [
            np.clip(np.round(np.linspace(a, b, per_seg)).astype(int), 0, traj.n - 1) for a, b in bounds
        ]

    ego_seg = seg_idx(ego)
    real_seg = [seg_idx(t) for t in usable]
    n_seg = len(ego_seg)

    segs = []
    for s in range(n_seg):
        floor = []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                pi = usable[i].pos[real_seg[i][s]]
                pj = usable[j].pos[real_seg[j][s]]
                floor.append(np.linalg.norm(pi - pj, axis=1).mean())
        err = [
            np.linalg.norm(ego.pos[ego_seg[s]] - usable[i].pos[real_seg[i][s]], axis=1).mean()
            for i in range(len(usable))
        ]
        f = float(np.mean(floor)) if floor else float("nan")
        e = float(np.mean(err))
        segs.append(
            {
                "index": s,
                "label": f"S{s}",
                "floor_mm": 1000 * f,
                "err_mm": 1000 * e,
                "rho": e / max(f, 1e-9),
            }
        )
    return {"available": True, "segments": segs}
