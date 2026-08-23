#!/usr/bin/env python3
"""L3 - set-level metrics: aggregate many ego episodes against the real set.

L1/L2 answer "how far is this one ego trajectory from the real manifold".  L3
answers three questions that only exist once there are several ego episodes:

1. **Is the gap systematic or noisy?**  ``dispersion_ratio`` compares how much
   the ego episodes disagree *with each other* against how much the real ones
   do.  Near 1 means the pipeline is repeatable and the whole gap is a bias that
   some transform could remove; much greater than 1 means the pipeline itself is
   noisy and no fixed correction will help.
2. **Are the two sets distinguishable at all?**  Energy distance with a
   permutation test, plus a classifier two-sample test.  These are the honest
   stopping criteria: when a classifier can no longer tell ego from real, the
   conversion is as good as the reference allows.
3. **How much would one global correction buy?**  Fitted leave-one-out so the
   number is deployable rather than an overfit.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from traj_metrics import N_RESAMPLE, Traj, dtw_dist, dtw_pairs, geodesic_deg, rot_dist


# --------------------------------------------------------------------------
# 1. aggregation of the per-episode layers
# --------------------------------------------------------------------------


def l3_aggregate(per_ego: dict[str, dict], keys: list[str]) -> dict:
    """mean / std / min / max plus the worst episode for each key."""
    out = {}
    for k in keys:
        vals = np.array([v[k] for v in per_ego.values() if k in v and np.isfinite(v[k])])
        if vals.size == 0:
            continue
        names = [n for n, v in per_ego.items() if k in v and np.isfinite(v[k])]
        out[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "worst_episode": names[int(np.argmax(vals))],
            "best_episode": names[int(np.argmin(vals))],
            "n": int(vals.size),
        }
    return out


# --------------------------------------------------------------------------
# 2. dispersion: systematic bias vs pipeline noise
# --------------------------------------------------------------------------


def l3_dispersion(book) -> dict:
    """Within-set spread of ego vs within-set spread of real, per distance field.

    This is the single most decision-relevant L3 number.  A ratio near 1 with a
    large rho says "one transform away"; a large ratio says "fix the variance
    first, a transform cannot help".
    """
    ego = np.where(book.is_ego)[0]
    real = np.where(book.is_real)[0]
    out = {}
    if len(ego) < 2:
        return {"available": False, "reason": f"need >=2 ego episodes, have {len(ego)}"}
    for field in ("abs", "shape", "offset", "rot"):
        M = getattr(book, field)
        ee = M[np.ix_(ego, ego)][np.triu_indices(len(ego), 1)]
        rr = M[np.ix_(real, real)][np.triu_indices(len(real), 1)]
        out[field] = {
            "ego_within_mean": float(ee.mean()),
            "real_within_mean": float(rr.mean()),
            "dispersion_ratio": float(ee.mean() / max(rr.mean(), 1e-12)),
        }
    out["available"] = True
    return out


# --------------------------------------------------------------------------
# 3. distribution-level two-sample tests
# --------------------------------------------------------------------------


def l3_energy_test(book, field: str = "abs", n_perm: int = 2000, seed: int = 0) -> dict:
    """Energy distance between the ego and real sets, with a permutation p-value.

    ``E = 2*mean(D_xy) - mean(D_xx) - mean(D_yy)``, computed straight from the
    pooled distance matrix.  E = 0 iff the two sets are drawn from the same
    distribution, and the permutation test needs no distributional assumption -
    which matters at n ~ 30.
    """
    ego = np.where(book.is_ego)[0]
    real = np.where(book.is_real)[0]
    if len(ego) < 2:
        return {"available": False, "reason": f"need >=2 ego episodes, have {len(ego)}"}
    M = getattr(book, field)

    def energy(a: np.ndarray, b: np.ndarray) -> float:
        xy = M[np.ix_(a, b)].mean()
        xx = M[np.ix_(a, a)][np.triu_indices(len(a), 1)].mean()
        yy = M[np.ix_(b, b)][np.triu_indices(len(b), 1)].mean()
        return float(2 * xy - xx - yy)

    observed = energy(ego, real)
    pool = np.concatenate([ego, real])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for t in range(n_perm):
        perm = rng.permutation(pool)
        null[t] = energy(perm[: len(ego)], perm[len(ego) :])
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "available": True,
        "field": field,
        "energy_distance": observed,
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "n_perm": n_perm,
        "interpretation": "p > 0.05 means ego and real are statistically indistinguishable",
    }


def l3_manifold_overlap(book, field: str = "abs", k: int = 3) -> dict:
    """Precision / coverage against the real manifold.

    ``precision``: fraction of ego episodes that land inside some real episode's
    k-NN ball - "is what we produce plausible?".
    ``coverage``: fraction of real episodes that have an ego episode inside their
    ball - "do we span the variety the robot actually shows?".  A pipeline can
    score high precision and low coverage by collapsing onto one mode.
    """
    ego = np.where(book.is_ego)[0]
    real = np.where(book.is_real)[0]
    if len(real) <= k:
        return {"available": False, "reason": f"need >{k} real episodes"}
    M = getattr(book, field)

    rr = M[np.ix_(real, real)]
    radii = np.sort(rr, axis=1)[:, k]  # index k skips the zero self-distance
    er = M[np.ix_(ego, real)]  # (n_ego, n_real)

    precision = float((er <= radii[None, :]).any(axis=1).mean()) if len(ego) else float("nan")
    coverage = float((er <= radii[None, :]).any(axis=0).mean())
    return {
        "available": True,
        "field": field,
        "k": k,
        "precision": precision,
        "coverage": coverage,
        "real_radius_mean_mm": float(1000 * radii.mean()),
        "nearest_real_mm": (1000 * er.min(axis=1)).tolist(),
    }


def l3_c2st(trajs: list[Traj], n_splits: int = 5, n_perm: int = 200, seed: int = 0,
            include_timing: bool = False) -> dict:
    """Classifier two-sample test on spatial trajectory features.

    AUC -> 0.5 means a classifier cannot separate ego from real, which is the
    end state we want.  Timing features are excluded by default: the human is
    ~2.5x faster than teleop by construction, and letting the classifier use
    duration makes the test trivially saturate on something we do not care about.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"available": False, "reason": "scikit-learn not installed"}

    X = np.array([_traj_features(t, include_timing) for t in trajs])
    y = np.array([1 if t.kind == "ego" else 0 for t in trajs])
    if min(np.bincount(y)) < n_splits:
        return {"available": False, "reason": f"need >={n_splits} of each class"}

    def cv_auc(labels: np.ndarray, rs: int) -> float:
        probs = np.zeros(len(labels))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
        for tr, te in skf.split(X, labels):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=2000).fit(sc.transform(X[tr]), labels[tr])
            probs[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        return float(roc_auc_score(labels, probs))

    observed = cv_auc(y, seed)
    rng = np.random.default_rng(seed)
    null = np.array([cv_auc(rng.permutation(y), seed) for _ in range(n_perm)])
    p = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return {
        "available": True,
        "auc": observed,
        "null_auc_mean": float(null.mean()),
        "p_value": p,
        "n_features": int(X.shape[1]),
        "include_timing": include_timing,
        "interpretation": "auc ~ 0.5 (p > 0.05) means ego and real are indistinguishable",
    }


def _traj_features(t: Traj, include_timing: bool) -> np.ndarray:
    p = t.resampled_pos(N_RESAMPLE)
    rot = t.resampled_rot(N_RESAMPLE)
    mean_rot = rot.mean()
    feats = [
        p.mean(axis=0),                                    # 3  placement
        p.max(axis=0) - p.min(axis=0),                     # 3  extent
        p.std(axis=0),                                     # 3  spread
        [t.path_len_m],                                    # 1
        p[np.linspace(0, N_RESAMPLE - 1, 7).astype(int)].ravel(),  # 21 shape at 7 phases
        mean_rot.as_rotvec(),                              # 3  mean orientation
        [np.mean([geodesic_deg(mean_rot, r) for r in rot])],       # 1 orientation spread
    ]
    if include_timing:
        feats.append([t.duration_s, t.path_len_m / max(t.duration_s, 1e-6)])
    return np.concatenate([np.asarray(f, dtype=float).ravel() for f in feats])


# --------------------------------------------------------------------------
# 4. how much would one global correction buy?
# --------------------------------------------------------------------------


def l3_global_correction(egos: list[Traj], reals: list[Traj], n: int = N_RESAMPLE) -> dict:
    """Fit one translation + one local rotation shared by all ego episodes.

    Reported both self-fitted and leave-one-out.  Only the LOO column is a
    deployable expectation; the self-fitted one is shown next to it so the size
    of the overfit is visible rather than implied.
    """
    if len(egos) < 2:
        return {"available": False, "reason": f"need >=2 ego episodes, have {len(egos)}"}

    real_pos = [t.resampled_pos(n) for t in reals]
    real_rot = [t.resampled_rot(n) for t in reals]
    real_centroid = np.mean([p.mean(axis=0) for p in real_pos], axis=0)

    # per-ego raw ingredients
    per = []
    for e in egos:
        ep = e.resampled_pos(n)
        er = e.resampled_rot(n)
        offsets = []
        for rp, rr in zip(real_pos, real_rot):
            pa, pb = dtw_pairs(ep, rp)
            offsets.append((rr[pb].inv() * er[pa]).as_rotvec())
        per.append(
            {
                "traj": e,
                "pos": ep,
                "rot": er,
                "t": real_centroid - ep.mean(axis=0),
                "R": R.from_rotvec(np.concatenate(offsets)).mean(),
            }
        )

    def score(item, t_fix: np.ndarray, R_fix: R) -> tuple[float, float]:
        # Position uses dtw_dist so the "raw" column reconciles with L1's D_pos;
        # orientation is paired through the same position warping path as L1.
        pos = item["pos"] + t_fix
        rot = item["rot"] * R_fix.inv()
        dp, dr = [], []
        for rp, rr in zip(real_pos, real_rot):
            dp.append(dtw_dist(pos, rp))
            pa, pb = dtw_pairs(pos, rp)
            dr.append(rot_dist(rr[pb], rot[pa]))
        return float(np.mean(dp)), float(np.mean(dr))

    zero_t = np.zeros(3)
    ident = R.identity()
    raw = [score(it, zero_t, ident) for it in per]

    t_all = np.mean([it["t"] for it in per], axis=0)
    R_all = R.from_rotvec([it["R"].as_rotvec() for it in per]).mean()
    selfit = [score(it, t_all, R_all) for it in per]

    loo = []
    for i, it in enumerate(per):
        others = [o for j, o in enumerate(per) if j != i]
        t_o = np.mean([o["t"] for o in others], axis=0)
        R_o = R.from_rotvec([o["R"].as_rotvec() for o in others]).mean()
        loo.append(score(it, t_o, R_o))

    def col(rows, k):
        return float(np.mean([r[k] for r in rows]))

    axis = R_all.as_rotvec()
    ang = float(np.degrees(np.linalg.norm(axis)))
    return {
        "available": True,
        "fitted_translation_mm": (1000 * t_all).tolist(),
        "fitted_rotation_deg": ang,
        "fitted_rotation_axis_local": (axis / max(np.linalg.norm(axis), 1e-12)).tolist(),
        "per_ego_rotation_deg": [float(np.degrees(np.linalg.norm(it["R"].as_rotvec()))) for it in per],
        "rotation_spread_deg": float(
            np.mean([geodesic_deg(R_all, it["R"]) for it in per])
        ),
        "pos_mm": {"raw": 1000 * col(raw, 0), "self_fit": 1000 * col(selfit, 0), "loo": 1000 * col(loo, 0)},
        "rot_deg": {"raw": col(raw, 1), "self_fit": col(selfit, 1), "loo": col(loo, 1)},
        "note": "only the loo column is a deployable expectation",
    }


# --------------------------------------------------------------------------
# 5. per-anchor distributions, ego cloud vs real cloud
# --------------------------------------------------------------------------


def l3_anchor_distributions(egos: list[Traj], reals: list[Traj], select: list[int] | None = None) -> dict:
    """Compare the ego anchor cloud with the real anchor cloud, anchor by anchor.

    Adds what a single-episode anchor error cannot show: whether the ego anchors
    are merely offset (mean shifted, spread matched) or also less repeatable
    (spread inflated).
    """
    counts = [len(t.events) for t in reals]
    mode = int(np.bincount(counts).argmax()) if counts else 0
    usable_r = [t for t in reals if len(t.events) == mode]
    usable_e = [t for t in egos if len(t.events) == mode]
    if not usable_r or not usable_e:
        return {
            "available": False,
            "reason": f"no ego episode has the real mode event count ({mode}); "
            f"ego counts = {sorted({len(t.events) for t in egos})}",
            "n_ego_usable": len(usable_e),
        }

    keep = list(range(mode)) if select is None else [k for k in select if 0 <= k < mode]
    real_dirs = np.array([t.event_dirs for t in usable_r])
    ego_dirs = np.array([t.event_dirs for t in usable_e])
    out = []
    for k in keep:
        rp = np.array([t.pos[t.events[k]] for t in usable_r])
        ep = np.array([t.pos[t.events[k]] for t in usable_e])
        d_real = int(np.sign(real_dirs[:, k].sum()))
        transition = "close" if d_real > 0 else "open"
        n_agree = int((ego_dirs[:, k] == d_real).sum())
        rc, ec = rp.mean(axis=0), ep.mean(axis=0)
        cov = np.cov(rp.T) + np.eye(3) * 1e-9
        delta = ec - rc
        maha = float(np.sqrt(delta @ np.linalg.inv(cov) @ delta))
        r_spread = float(np.linalg.norm(rp - rc, axis=1).mean())
        e_spread = float(np.linalg.norm(ep - ec, axis=1).mean())
        # A spread estimated from one or two samples carries no information;
        # report it as unavailable rather than as a suspiciously precise number.
        enough = len(ep) >= 3
        out.append(
            {
                "index": k,
                "label": f"E{k + 1}_{transition}",
                "gripper_transition": transition,
                "n_ego_dir_agrees": n_agree,
                "n_ego": len(ep),
                "n_real": len(rp),
                "mean_offset_mm": (1000 * delta).tolist(),
                "mean_offset_norm_mm": float(1000 * np.linalg.norm(delta)),
                "mahalanobis": maha,
                "real_spread_mm": 1000 * r_spread,
                "ego_spread_mm": 1000 * e_spread if enough else None,
                "spread_ratio": e_spread / max(r_spread, 1e-12) if enough else None,
            }
        )
    return {"available": True, "n_anchors": mode, "selected": keep, "anchors": out}
