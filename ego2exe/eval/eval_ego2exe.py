#!/usr/bin/env python3
"""Evaluate converted ego EEF trajectories against a set of real-robot episodes.

Everything is reported as ``rho`` - a ratio to the spread the real episodes
already show among themselves.  Absolute millimetres are not comparable across
tasks; ``rho`` is, and ``rho -> 1`` is the point past which further optimisation
is chasing noise the reference set cannot resolve.

Example
-------
    python ego2exe/eval/eval_ego2exe.py \
        --real DATA/20260816_nero_stack_object_horizontal \
        --ego  outputs/new_pipeline/1stack_object_horizontal \
               outputs/new_pipeline/stack_object_*.rgb \
        --anchors 0 2 \
        --out outputs/eval/stack_object_horizontal

Layers
------
L0  gates          reported, never blocking
L1  rho_pos / rho_rot / rho_se3
L2  offset-vs-shape split, per-anchor absolute error, per-segment rho
L3  set-level: dispersion ratio, energy test, manifold overlap, C2ST,
    leave-one-out global correction, per-anchor distributions

Scope of L1 / L2a / L3 (dispersion / energy_test / manifold_overlap /
global_correction / c2st; anchor_distributions excepted): by default, episodes
whose gripper-toggle count doesn't match the real set's mode are excluded (a
stuck grasp signal is a known, recurring pipeline failure and should not
silently pollute pooled statistics) - see --no-grasp-health-filter. Pass
--segment-start/--segment-end to further restrict that scope to the span
between two gripper events, e.g. --segment-start 0 --segment-end 2 for
"1st open -> 1st close -> 2nd open". L0, L2b and L2c always see every loaded
episode regardless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics_l3 import (  # noqa: E402
    l3_aggregate,
    l3_anchor_distributions,
    l3_c2st,
    l3_dispersion,
    l3_energy_test,
    l3_global_correction,
    l3_manifold_overlap,
)
from metrics_layers import (  # noqa: E402
    DistanceBook,
    anchor_reference,
    l0_gates,
    l1_ratios,
    l2_anchors,
    l2_offset_shape,
    l2_segments,
)
from traj_metrics import (  # noqa: E402
    N_RESAMPLE,
    Traj,
    crop_to_segment,
    load_ego_dirs,
    load_real_dir,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# 1: first version carrying scope/run blocks and a version field at all.
REPORT_SCHEMA_VERSION = 1


def as_abs(p: str | Path) -> Path:
    """Absolute as-is; otherwise prefer the shell's cwd, then the repo root.

    DATA/ lives outside the repo in this workspace, so a repo-root-only rule
    would reject the paths people actually type.
    """
    p = Path(p)
    if p.is_absolute():
        return p
    cwd = Path.cwd() / p
    return cwd if cwd.exists() else REPO_ROOT / p


# --------------------------------------------------------------------------
# report formatting
# --------------------------------------------------------------------------

RULE = "=" * 78
THIN = "-" * 78


def _flag(ok) -> str:
    return " ok " if ok is True else "FAIL" if ok is False else " -- "


def print_l0(name: str, g: dict) -> None:
    print(f"  {name}")
    for key, v in g.items():
        if key.startswith("_"):
            continue
        extra = ""
        if key == "grasp_events_match":
            extra = f"ego {v['ego']} vs real mode {v['real_mode']}"
        elif "value" in v:
            unit = "%" if v["value"] <= 1.0 else ""
            extra = f"{100 * v['value']:.1f}{unit or '%'}"
        print(f"    [{_flag(v.get('pass'))}] {key:24s} {extra}")


def print_table(rows: list[list], headers: list[str], aligns: str | None = None) -> None:
    cols = len(headers)
    aligns = aligns or "l" * cols
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in cells)) if cells else len(headers[i]) for i in range(cols)]

    def fmt(vals):
        out = []
        for i, v in enumerate(vals):
            a = aligns[i]
            out.append(v.rjust(widths[i]) if a == "r" else v.center(widths[i]) if a == "c" else v.ljust(widths[i]))
        return "  ".join(out)
    print("  " + fmt(headers))
    print("  " + "  ".join("-" * w for w in widths))
    for r in cells:
        print("  " + fmt(r))


def f(x, nd=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def l0_aggregate(l0: dict) -> dict:
    """mean/min/max of each L0 gate's ``value`` across episodes, plus a fail count.

    The fail count comes from ``pass``, not from thresholding ``value``: a gate's
    "better direction" is not uniform (workspace_entry_rate's value is a pass
    *rate*, higher is better; most others are a violation rate, lower is better),
    so reading pass/fail off value's mean would require remembering which way
    each gate points. The count sidesteps that entirely.
    """
    per_ego = {
        name: {k: v["value"] for k, v in gates.items() if not k.startswith("_") and "value" in v}
        for name, gates in l0.items()
    }
    keys = sorted({k for v in per_ego.values() for k in v})
    agg = l3_aggregate(per_ego, keys)
    for key, stats in agg.items():
        stats["failed"] = sum(1 for gates in l0.values() if gates.get(key, {}).get("pass") is False)
        stats["n_episodes"] = len(l0)
    return agg


def crop_scope(trajs: list[Traj], start: int | None, end: int | None) -> tuple[list[Traj], list[str]]:
    """Crop every trajectory to [start_event, end_event]; return (kept, dropped names)."""
    kept: list[Traj] = []
    dropped: list[str] = []
    for t in trajs:
        c = crop_to_segment(t, start, end)
        (kept if c is not None else dropped).append(c if c is not None else t.name)
    return kept, dropped


def l2_anchor_aggregate(l2a: dict) -> dict:
    """mean/min/max of err_pos_mm / err_rot_deg across episodes, grouped by anchor label."""
    by_label: dict[str, dict[str, dict]] = {}
    for name, v in l2a.items():
        if not v.get("available"):
            continue
        for a in v["anchors"]:
            by_label.setdefault(a["label"], {})[name] = {
                "err_pos_mm": a["err_pos_mm"],
                "err_rot_deg": a["err_rot_deg"],
            }
    return {label: l3_aggregate(per_ego, ["err_pos_mm", "err_rot_deg"]) for label, per_ego in by_label.items()}


# --------------------------------------------------------------------------
# --task / --pipeline / --cohort resolution
# --------------------------------------------------------------------------


def resolve_task_real_dir(data_root: Path, task: str) -> Path:
    """The one *nero* directory under DATA_new/<task>/.

    The task directory is the authority for which real set belongs to the task:
    the jsonl ``metadata.task`` field cannot be used - it reads "stack object"
    for both stack_object_horizontal and stack_object_lean.
    """
    task_dir = data_root / task
    if not task_dir.is_dir():
        avail = ", ".join(sorted(p.name for p in data_root.iterdir() if p.is_dir())) or "(none)"
        raise SystemExit(f"task not found: {task_dir}\navailable tasks: {avail}")
    hits = sorted(p for p in task_dir.iterdir() if p.is_dir() and "nero" in p.name)
    if len(hits) != 1:
        found = ", ".join(p.name for p in hits) or "(none)"
        raise SystemExit(
            f"expected exactly one *nero* directory under {task_dir}, found {len(hits)}: {found}"
        )
    return hits[0]


def resolve_cohort_dirs(pipe_task_dir: Path, cohort: str) -> tuple[list[Path], list[str]]:
    """Ego session dirs for a cohort, plus the cohort names they came from.

    ``all`` means every ``*_ego_*`` directory; otherwise the suffix must match.
    """
    if not pipe_task_dir.is_dir():
        raise SystemExit(
            f"no preprocessed output at {pipe_task_dir}\n"
            f"run: ego2exe/scripts/batch_preprocess_le_ver.sh --task {pipe_task_dir.name} "
            f"--pipeline {pipe_task_dir.parent.name}"
        )
    cohort_dirs = sorted(p for p in pipe_task_dir.iterdir() if p.is_dir() and "_ego_" in p.name)
    if cohort != "all":
        cohort_dirs = [p for p in cohort_dirs if p.name.split("_ego_")[-1] == cohort]
    if not cohort_dirs:
        avail = sorted({p.name.split("_ego_")[-1] for p in pipe_task_dir.iterdir()
                        if p.is_dir() and "_ego_" in p.name})
        raise SystemExit(f"cohort '{cohort}' not found under {pipe_task_dir}\n"
                         f"available cohorts: {', '.join(avail) or '(none)'}")
    sessions = [s for c in cohort_dirs for s in sorted(p for p in c.iterdir() if p.is_dir())]
    return sessions, [c.name.split("_ego_")[-1] for c in cohort_dirs]


def describe_segment(reals: list, start: int | None, end: int | None) -> dict:
    """What a segment index range actually covers, measured on the real set.

    ``events[1]->events[2]`` alone does not say what happened between them, and
    the answer differs per task: on stack_object_horizontal that span is 32%-64%
    of the episode, on stack_object_lean 39%-65%. Comparing the same variant name
    across tasks without this is comparing different fractions of the motion.
    """
    if not reals:
        return {}

    def stat(idx: int | None, default_pct: float) -> tuple[float, str | None]:
        if idx is None:
            return default_pct, None
        pcts, dirs = [], []
        for t in reals:
            if 0 <= idx < len(t.events):
                pcts.append(100.0 * t.events[idx] / t.n)
                dirs.append("close" if t.event_dirs[idx] > 0 else "open")
        if not pcts:
            return float("nan"), None
        return float(np.mean(pcts)), max(set(dirs), key=dirs.count)

    p0, d0 = stat(start, 0.0)
    p1, d1 = stat(end, 100.0)
    arrow = f"{d0 or 'start'} -> {d1 or 'end'}"
    return {
        "start_pct": p0,
        "end_pct": p1,
        "start_transition": d0,
        "end_transition": d1,
        "description": f"{arrow}   {p0:.0f}% -> {p1:.0f}% of the episode "
                       f"(measured on the {len(reals)} real episodes)",
    }


def eval_variant_name(args: argparse.Namespace) -> str:
    """Directory-safe name encoding the eval-time knobs, so runs never collide."""
    parts = []
    if args.segment_start is not None or args.segment_end is not None:
        s = "" if args.segment_start is None else str(args.segment_start)
        e = "" if args.segment_end is None else str(args.segment_end)
        parts.append(f"seg{s}-{e}")
    if args.anchors:
        parts.append("a" + "-".join(str(a) for a in args.anchors))
    if args.no_grasp_health_filter:
        parts.append("nofilter")
    if args.pose_key != "tcp_tip_pose":
        parts.append(args.pose_key)
    if args.real_split_manifest:
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "-", Path(args.real_split_manifest).stem)
        parts.append(f"rs-{stem}-{args.real_split}")
    elif args.max_real:  # 0 means "no cap", matching load_real_dir's convention
        parts.append(f"real{args.max_real}")
    return "_".join(parts) if parts else "full"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def collect_ego_dirs(args: argparse.Namespace) -> list[Path]:
    """Session directories from --ego and/or --ego-list, de-duplicated in order.

    --ego-list is what batch_preprocess_le_ver.sh writes; it also sidesteps the argv
    length limit once a batch runs to dozens of sessions.
    """
    raw: list[str] = list(args.ego or [])
    if args.ego_list:
        listing = as_abs(args.ego_list)
        if not listing.is_file():
            raise SystemExit(f"--ego-list not found: {listing}")
        raw += [
            line.strip()
            for line in listing.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not raw:
        raise SystemExit("provide at least one of --ego or --ego-list")

    seen: set[str] = set()
    out: list[Path] = []
    for p in raw:
        ap = as_abs(p)
        if str(ap) not in seen:
            seen.add(str(ap))
            out.append(ap)
    return out


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, list[Path], Path | None, str]:
    """(real_dir, ego_dirs, out_dir, label) from either --task/--pipeline or the explicit flags."""
    if args.task:
        if not args.pipeline:
            raise SystemExit("--task requires --pipeline")
        data_root = as_abs(args.data_root)
        real_dir = resolve_task_real_dir(data_root, args.task)
        pipe_task_dir = as_abs(args.out_root) / args.pipeline / args.task
        ego_dirs, cohorts = resolve_cohort_dirs(pipe_task_dir, args.cohort)
        variant = eval_variant_name(args)
        out_dir = as_abs(args.out) if args.out else (
            as_abs(args.out_root) / "eval" / f"{args.task}__{args.pipeline}__{args.cohort}__{variant}"
        )
        label = (f"task={args.task}  pipeline={args.pipeline}  "
                 f"cohort={args.cohort}({'+'.join(cohorts)})  variant={variant}")
        return real_dir, ego_dirs, out_dir, label

    if not args.real:
        raise SystemExit("provide --task/--pipeline, or --real with --ego/--ego-list")
    return (as_abs(args.real), collect_ego_dirs(args),
            as_abs(args.out) if args.out else None, "")


def resolve_real_reference(args: argparse.Namespace, real_dir: Path) -> dict:
    """Resolve and fingerprint the exact real episodes used by this run."""
    if args.real_split_manifest:
        manifest_path = as_abs(args.real_split_manifest)
        if not manifest_path.is_file():
            raise SystemExit(f"--real-split-manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if args.task and manifest.get("task") != args.task:
            raise SystemExit(
                f"real split task mismatch: manifest has {manifest.get('task')!r}, "
                f"evaluation requested {args.task!r}"
            )
        names = manifest.get(args.real_split)
        if not isinstance(names, list) or not names:
            raise SystemExit(
                f"split {args.real_split!r} is empty or absent in {manifest_path}"
            )
        if len(names) != len(set(names)):
            raise SystemExit(f"split {args.real_split!r} contains duplicate episode names")
        selected = [str(name) for name in names]
        mode = "manifest"
    else:
        manifest_path = None
        files = sorted(real_dir.glob("*.jsonl"))
        if args.max_real:
            files = files[:args.max_real]
        selected = [path.name for path in files]
        mode = "filename_prefix" if args.max_real else "all"

    fingerprint_payload = json.dumps(
        {
            "real_dir": str(real_dir.resolve()),
            "episode_names": selected,
            "pose_key": args.pose_key,
            "tcp_offset_m": args.tcp_offset_m,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    reference_set_id = hashlib.sha256(fingerprint_payload).hexdigest()[:16]
    return {
        "mode": mode,
        "manifest": str(manifest_path.resolve()) if manifest_path else None,
        "split": args.real_split if manifest_path else None,
        "reference_set_id": reference_set_id,
        "episode_names": selected,
    }


def check_real_leakage(
    ego_dirs: list[Path],
    ego_subdir: str,
    eval_names: list[str],
    *,
    allow: bool,
) -> dict:
    """Refuse evaluation when correction calibration episodes enter its reference set."""
    calibration_names: set[str] = set()
    metadata_files: list[str] = []
    for session_dir in ego_dirs:
        candidates = [session_dir / ego_subdir / "anchor_correction_meta.json"]
        candidates.extend(sorted(session_dir.glob("*/anchor_correction_meta.json")))
        meta_path = next((path for path in candidates if path.is_file()), None)
        if meta_path is None:
            continue
        metadata_files.append(str(meta_path.resolve()))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        calibration_names.update(str(name) for name in meta.get("real_split", {}).get("calibration", []))

    overlap = sorted(calibration_names.intersection(eval_names))
    if overlap and not allow:
        raise SystemExit(
            "REAL-DATA LEAKAGE: the evaluation reference set overlaps the real episodes used "
            f"to calibrate these corrected trajectories ({len(overlap)} file(s)): {overlap}.\n"
            "Evaluate with the split manifest's held-out eval block. Pass --allow-real-leakage "
            "only for an explicitly non-held-out diagnostic."
        )
    return {
        "checked_correction_metadata_files": len(metadata_files),
        "calibration_episode_names": sorted(calibration_names),
        "overlap_episode_names": overlap,
        "override_used": bool(overlap and allow),
    }


def run(args: argparse.Namespace) -> dict:
    real_dir, ego_dirs, out_dir, label = resolve_inputs(args)
    real_reference = resolve_real_reference(args, real_dir)
    leakage = check_real_leakage(
        ego_dirs,
        args.ego_subdir,
        real_reference["episode_names"],
        allow=args.allow_real_leakage,
    )
    real_reference["leakage_check"] = leakage

    print(RULE)
    print("ego2exe trajectory evaluation")
    print(RULE)
    if label:
        print(f"{label}")
    print(f"real : {real_dir}")
    reals = load_real_dir(
        real_dir,
        args.pose_key,
        args.tcp_offset_m,
        max_real=None,
        episode_names=real_reference["episode_names"],
    )
    print(f"       {len(reals)} episodes, "
          f"{np.mean([t.duration_s for t in reals]):.1f} s mean, "
          f"grasp events {sorted({len(t.events) for t in reals})}")
    print(f"       reference {real_reference['reference_set_id']} "
          f"({real_reference['mode']}{':' + real_reference['split'] if real_reference['split'] else ''})")
    if len(reals) < args.min_real:
        print(f"  [warn] only {len(reals)} real episodes; the noise floor is unreliable "
              f"below ~{args.min_real}. Treat every rho below as provisional.")

    egos = load_ego_dirs(ego_dirs, args.ego_subdir, args.ego_hz)
    print(f"ego  : {len(egos)} episodes, {np.mean([t.duration_s for t in egos]):.1f} s mean, "
          f"grasp events {sorted({len(t.events) for t in egos})}")
    if not egos or not reals:
        raise SystemExit("nothing to compare")

    anchors_sel = args.anchors if args.anchors else None

    # ---- L0 -------------------------------------------------------------
    print(f"\n{RULE}\nL0  gates  (reported only - nothing below is skipped)\n{THIN}")
    l0 = {}
    for e in egos:
        l0[e.name] = l0_gates(e, reals, args.bbox_margin)
    if args.verbose:
        for e in egos:
            print_l0(e.name, l0[e.name])
    else:
        # A single FAIL line among 150 was missed in practice; surface only the
        # failures here and keep the full per-episode listing behind --verbose.
        failing = [(n, g) for n, g in l0.items() if g["_n_failed"]]
        if not failing:
            print(f"  all {len(egos)} episode(s) pass every gate")
        else:
            print(f"  {len(failing)}/{len(egos)} episode(s) with at least one failing gate "
                  f"(--verbose for the full listing):")
            for name, g in failing:
                bad = [k for k, v in g.items() if not k.startswith("_") and v.get("pass") is False]
                print(f"    {name}: {', '.join(bad)}")

    # ---- scope for L1 / L2a / L3 -----------------------------------------
    # L0, L2b (anchors) and L2c (segments) always see every loaded episode -
    # anchors/segments are inherently event-indexed and already do their own
    # per-call filtering. L1, L2a and most of L3 instead compare whole-episode
    # *shapes* via one pooled DistanceBook, so they get a shared "scope": the
    # grasp-health filter, then (optionally) a segment crop, both resolved once
    # here and threaded through every book-based metric below.
    real_counts = [len(t.events) for t in reals]
    mode_events = int(np.bincount(real_counts).argmax()) if real_counts else 0

    if args.no_grasp_health_filter:
        reals_scope, egos_scope = reals, egos
    else:
        # Filtered by re-checking the same condition, not "in reals_scope": Traj is a
        # dataclass, so its auto __eq__ would compare the numpy pos/rot/grasp arrays
        # and raise on the ambiguous-truth-value of an array comparison.
        reals_scope = [t for t in reals if len(t.events) == mode_events]
        egos_scope = [t for t in egos if len(t.events) == mode_events]
        dropped_real = [t.name for t in reals if len(t.events) != mode_events]
        dropped_ego = [t.name for t in egos if len(t.events) != mode_events]
        if dropped_real or dropped_ego:
            print(f"\n{RULE}\ngrasp-health filter  "
                  f"(L1 / L2a / L3 only - L0 / L2b / L2c above still see every episode)\n{THIN}")
            print(f"  real-episode mode event count = {mode_events}; episodes with a different "
                  f"count are excluded from shape-based metrics:")
            if dropped_real:
                shown = ", ".join(dropped_real[:5]) + (" ..." if len(dropped_real) > 5 else "")
                print(f"    real: {len(dropped_real)}/{len(reals)} dropped - {shown}")
            if dropped_ego:
                shown = ", ".join(dropped_ego[:5]) + (" ..." if len(dropped_ego) > 5 else "")
                print(f"    ego:  {len(dropped_ego)}/{len(egos)} dropped - {shown}")
            print("    see grasp_events_match in L0 above for which episode and why; pass "
                  "--no-grasp-health-filter to include them anyway")

    segment_info = None
    if args.segment_start is not None or args.segment_end is not None:
        reals_seg, reals_seg_dropped = crop_scope(reals_scope, args.segment_start, args.segment_end)
        egos_seg, egos_seg_dropped = crop_scope(egos_scope, args.segment_start, args.segment_end)
        phase = describe_segment(reals_scope, args.segment_start, args.segment_end)
        segment_info = {
            "start_event": args.segment_start,
            "end_event": args.segment_end,
            "dropped_real": reals_seg_dropped,
            "dropped_ego": egos_seg_dropped,
            **phase,
        }
        print(f"\n{RULE}\nsegment scope: events[{args.segment_start}] -> events[{args.segment_end}]  "
              f"(L1 / L2a / L3 only)\n{THIN}")
        if phase.get("description"):
            print(f"  {phase['description']}")
        print(f"  {len(reals_seg)}/{len(reals_scope)} real, {len(egos_seg)}/{len(egos_scope)} ego "
              f"resolved this span")
        if reals_seg_dropped or egos_seg_dropped:
            print(f"    dropped (event index out of range for that episode): "
                  f"real {reals_seg_dropped}  ego {egos_seg_dropped}")
        reals_scope, egos_scope = reals_seg, egos_seg

    if not reals_scope or not egos_scope:
        raise SystemExit(
            "nothing left in scope after grasp-health/segment filtering - loosen "
            "--segment-start/--segment-end or pass --no-grasp-health-filter"
        )

    # ---- distances ------------------------------------------------------
    print(f"\n{RULE}\ncomputing pooled pairwise distances "
          f"({len(reals_scope)} real, {len(egos_scope)} ego in scope)\n{THIN}")
    # Everything that can change the real-vs-real block goes into the key.
    cache_key = None if args.no_cache else "|".join(str(x) for x in (
        real_dir, real_reference["reference_set_id"], args.pose_key, args.tcp_offset_m,
        args.segment_start, args.segment_end, N_RESAMPLE,
    ))
    book = DistanceBook(reals_scope + egos_scope, cache_key=cache_key)

    # ---- L1 -------------------------------------------------------------
    l1 = l1_ratios(book)
    fl = l1["floors"]
    scope_note = "" if segment_info is None else f"  [segment events[{segment_info['start_event']}]->events[{segment_info['end_event']}]]"
    print(f"\n{RULE}\nL1  headline ratios{scope_note}\n{THIN}")
    print(f"  noise floor (leave-one-out over {len(reals_scope)} real episodes)")
    print(f"    D_pos {1000 * fl['abs']['mean']:6.1f} +/- {1000 * fl['abs']['std']:.1f} mm"
          f"     D_rot {fl['rot']['mean']:5.1f} +/- {fl['rot']['std']:.1f} deg\n")
    if args.verbose:
        rows = [
            [n, f(v["D_pos_mm"], 1), f(v["D_rot_deg"], 1), f(v["rho_pos"]), f(v["rho_rot"]),
             f(v["rho_se3"]), f(v["z_pos"], 1)]
            for n, v in l1["per_ego"].items()
        ]
        print_table(rows, ["episode", "D_pos mm", "D_rot deg", "rho_pos", "rho_rot", "rho_SE3", "z_pos"],
                    "lrrrrrr")
    agg1 = l3_aggregate(l1["per_ego"], ["rho_pos", "rho_rot", "rho_se3", "D_pos_mm", "D_rot_deg"])
    if len(egos_scope) > 1:
        print()
        print_table(
            [[k, f(v["mean"]), f(v["std"]), f(v["min"]), f(v["max"]), v["worst_episode"]]
             for k, v in agg1.items()],
            ["metric", "mean", "std", "min", "max", "worst"], "lrrrrl")

    # ---- L2 -------------------------------------------------------------
    print(f"\n{RULE}\nL2a  offset vs shape\n{THIN}")
    l2os = l2_offset_shape(book)
    fo = l2os["floors_mm"]
    print(f"  floors: abs {fo['abs']:.1f} mm   shape {fo['shape']:.1f} mm   offset {fo['offset']:.1f} mm\n")
    if args.verbose:
        rows = [
            [n, f(v["D_abs_mm"], 1), f(v["D_shape_mm"], 1), f(v["D_offset_mm"], 1),
             f(v["rho_abs"]), f(v["rho_shape"]), f(v["rho_offset"]),
             "[" + ", ".join(f"{c:+.0f}" for c in v["centroid_offset_mm"]) + "]"]
            for n, v in l2os["per_ego"].items()
        ]
        print_table(rows, ["episode", "abs mm", "shape mm", "off mm", "rho_abs", "rho_shape",
                           "rho_offset", "centroid offset mm"], "lrrrrrrl")
    print("\n  read: rho_offset >> rho_shape -> a placement/calibration fix helps;")
    print("        both high              -> shape/perception problem, no single transform helps.")

    print(f"\n{RULE}\nL2b  contact anchors  (absolute ground truth, reported per anchor)\n{THIN}")
    anchor_ref = anchor_reference(reals, anchors_sel)
    l2a = {}
    for e in egos:
        l2a[e.name] = l2_anchors(e, reals, anchors_sel, reference=anchor_ref)
    shown = [n for n, v in l2a.items() if v.get("available")]
    if not shown:
        for n, v in l2a.items():
            print(f"  {n}: unavailable - {v.get('reason')}")
    elif args.verbose:
        for n in shown:
            print(f"  {n}")
            print_table(
                [[a["label"],
                  "%.2f->%.2f" % tuple(a["grasp_real_before_after"]),
                  "yes" if a["direction_agrees_with_ego"] else "NO",
                  f(a["err_pos_mm"], 1), f(a["floor_pos_mm"], 1), f(a["rho_pos"]),
                  f(a["err_rot_deg"], 1), f(a["floor_rot_deg"], 1), f(a["rho_rot"]),
                  "[" + ", ".join(f"{c:+.0f}" for c in a["err_vec_mm"]) + "]",
                  f(a["phase_z"], 1)]
                 for a in l2a[n]["anchors"]],
                ["anchor", "grasp", "ego dir", "err mm", "floor", "rho_p", "err deg", "floor",
                 "rho_r", "err vec mm", "phase z"], "llcrrrrrrlr")
            print("    grasp = real action_grasp before->after the toggle (1 = closed, 0 = open)")
    else:
        print(f"  {len(shown)}/{len(l2a)} episode(s) resolved their anchors; "
              f"per-anchor means are in the SUMMARY below (--verbose for per-episode detail)")
        mismatched = [n for n in shown
                      if any(not a["direction_agrees_with_ego"] for a in l2a[n]["anchors"])]
        if mismatched:
            print(f"  [warn] ego toggles the opposite way at some anchor in {len(mismatched)} "
                  f"episode(s) - those errors are meaningless: {', '.join(mismatched[:4])}")
    if shown:
        missing = [n for n in l2a if n not in shown]
        if missing:
            print(f"  unavailable for {len(missing)} episode(s): {', '.join(missing[:4])}"
                  + (" ..." if len(missing) > 4 else ""))

    print(f"\n{RULE}\nL2c  per-segment rho\n{THIN}")
    l2s = {}
    for e in egos:
        l2s[e.name] = l2_segments(e, reals)
    ok_seg = [n for n, v in l2s.items() if v.get("available")]
    if not ok_seg:
        print(f"  unavailable - {next(iter(l2s.values())).get('reason')}")
    else:
        n_seg = len(l2s[ok_seg[0]]["segments"])
        rows = []
        for s in range(n_seg):
            vals = [l2s[n]["segments"][s] for n in ok_seg]
            rows.append([f"S{s}", f(np.mean([v['floor_mm'] for v in vals]), 1),
                         f(np.mean([v['err_mm'] for v in vals]), 1),
                         f(np.mean([v['rho'] for v in vals])),
                         f(np.std([v['rho'] for v in vals]))])
        print_table(rows, ["segment", "floor mm", "err mm", "rho mean", "rho std"], "lrrrr")

    # ---- L3 -------------------------------------------------------------
    print(f"\n{RULE}\nL3  set-level  ({len(egos_scope)} ego vs {len(reals_scope)} real"
          f"{scope_note})\n{THIN}")
    l3: dict = {}
    l3["dispersion"] = l3_dispersion(book)
    if l3["dispersion"].get("available"):
        print("  dispersion: how much the ego episodes disagree with each other")
        print_table(
            [[k, f(1000 * v["ego_within_mean"] if k != "rot" else v["ego_within_mean"], 1),
              f(1000 * v["real_within_mean"] if k != "rot" else v["real_within_mean"], 1),
              f(v["dispersion_ratio"])]
             for k, v in l3["dispersion"].items() if k != "available"],
            ["field", "ego within", "real within", "ratio"], "lrrr")
        print("    ratio ~ 1 -> repeatable pipeline, the gap is a bias a transform could remove")
        print("    ratio >> 1 -> the pipeline itself is noisy; reduce variance before fitting anything")
    else:
        print(f"  dispersion: unavailable - {l3['dispersion'].get('reason')}")

    l3["energy_test"] = l3_energy_test(book, "abs", args.n_perm)
    if l3["energy_test"].get("available"):
        e = l3["energy_test"]
        print(f"\n  energy distance {e['energy_distance']:.5f} "
              f"(null mean {e['null_mean']:.5f}, p95 {e['null_p95']:.5f})   p = {e['p_value']:.4f}")
        print(f"    {e['interpretation']}")

    l3["manifold_overlap"] = l3_manifold_overlap(book, "abs", args.knn_k)
    if l3["manifold_overlap"].get("available"):
        m = l3["manifold_overlap"]
        near = np.array(m["nearest_real_mm"])
        print(f"\n  manifold overlap (k={m['k']}, real radius {m['real_radius_mean_mm']:.1f} mm)")
        print(f"    precision {m['precision']:.2f}   coverage {m['coverage']:.2f}")
        print(f"    nearest real episode: {near.min():.1f} .. {near.max():.1f} mm "
              f"(mean {near.mean():.1f}) - how far outside the ball, when precision is 0")
        print("    precision = ego lands inside the real manifold; coverage = ego spans its variety")

    l3["c2st"] = ({"available": False, "reason": "skipped via --no-c2st"} if args.no_c2st else
                  l3_c2st(reals_scope + egos_scope, n_perm=args.n_perm_c2st,
                          include_timing=args.c2st_timing))
    if l3["c2st"].get("available"):
        c = l3["c2st"]
        print(f"\n  C2ST AUC {c['auc']:.3f}  (null {c['null_auc_mean']:.3f}, p = {c['p_value']:.4f}, "
              f"{c['n_features']} features)")
        print(f"    {c['interpretation']}")
    else:
        print(f"\n  C2ST unavailable - {l3['c2st'].get('reason')}")

    l3["global_correction"] = l3_global_correction(egos_scope, reals_scope)
    g = l3["global_correction"]
    if g.get("available"):
        print(f"\n  one global correction shared by all {len(egos_scope)} ego episodes")
        print(f"    translation {['%+.1f' % c for c in g['fitted_translation_mm']]} mm     "
              f"rotation {g['fitted_rotation_deg']:.1f} deg "
              f"(per-episode spread {g['rotation_spread_deg']:.1f} deg)")
        print_table(
            [["position mm", f(g["pos_mm"]["raw"], 1), f(g["pos_mm"]["self_fit"], 1), f(g["pos_mm"]["loo"], 1)],
             ["orientation deg", f(g["rot_deg"]["raw"], 1), f(g["rot_deg"]["self_fit"], 1), f(g["rot_deg"]["loo"], 1)]],
            ["quantity", "raw", "self-fit", "leave-one-out"], "lrrr")
        print("    self-fit is an overfit; only the leave-one-out column is deployable")
    else:
        print(f"\n  global correction unavailable - {g.get('reason')}")

    # anchors are absolute per-event poses, not shape comparisons - always use the
    # full episode regardless of grasp-health filter / segment scope above, and
    # they do their own event-count filtering internally.
    l3["anchor_distributions"] = l3_anchor_distributions(egos, reals, anchors_sel)
    ad = l3["anchor_distributions"]
    if ad.get("available"):
        print(f"\n  anchor distributions ({ad['anchors'][0]['n_ego']} ego vs "
              f"{ad['anchors'][0]['n_real']} real)")
        print_table(
            [[a["label"], f"{a['n_ego_dir_agrees']}/{a['n_ego']}", f(a["mean_offset_norm_mm"], 1),
              "[" + ", ".join(f"{c:+.0f}" for c in a["mean_offset_mm"]) + "]",
              f(a["mahalanobis"]), f(a["real_spread_mm"], 1),
              f(a["ego_spread_mm"], 1) if a["ego_spread_mm"] is not None else "n/a",
              f(a["spread_ratio"]) if a["spread_ratio"] is not None else "n/a"]
             for a in ad["anchors"]],
            ["anchor", "ego dir ok", "offset mm", "offset vec mm", "mahal", "real spread",
             "ego spread", "spread ratio"],
            "lcrlrrrr")
        print("    spread_ratio ~ 1 with a large offset -> systematic; >> 1 -> ego anchors are noisy")
    else:
        print(f"\n  anchor distributions unavailable - {ad.get('reason')}")

    # ---- summary --------------------------------------------------------
    print(f"\n{RULE}\nSUMMARY{scope_note}\n{THIN}")
    if segment_info is not None or not args.no_grasp_health_filter:
        print(f"  L1/L2a/L3 scope: {len(egos_scope)}/{len(egos)} ego, "
              f"{len(reals_scope)}/{len(reals)} real  (L0/L2b/L2c below always use all "
              f"{len(egos)}/{len(reals)})")

    n_gate_fail = sum(v["_n_failed"] for v in l0.values())
    l0agg = l0_aggregate(l0)
    print(f"  L0  {n_gate_fail} gate-failure(s) total, summed across all {len(l0agg)} gates x "
          f"{len(egos)} episode(s) - breakdown below")
    print_table(
        [[k, f"{v['failed']}/{v['n_episodes']}", f(v["mean"], 3), f(v["min"], 3), f(v["max"], 3)]
         for k, v in l0agg.items()],
        ["gate", "failed", "value mean", "min", "max"], "lrrrr")

    print(f"\n  L1  noise floor: D_pos {1000 * fl['abs']['mean']:.1f} +/- {1000 * fl['abs']['std']:.1f} mm"
          f"   D_rot {fl['rot']['mean']:.1f} +/- {fl['rot']['std']:.1f} deg   (mean / min / max below)")
    print_table(
        [[k, f(v["mean"]), f(v["min"]), f(v["max"])] for k, v in agg1.items()],
        ["metric", "mean", "min", "max"], "lrrr")
    if segment_info is not None:
        # The floor is recomputed on the cropped span, so a rho change between
        # variants mixes "the error moved" with "the denominator moved".
        print(f"      note: this floor is measured on the segment, not the full episode - "
              f"comparing rho across variants also compares two different denominators. "
              f"Use D_pos_mm / D_rot_deg above for the absolute change.")

    agg2 = l3_aggregate(l2os["per_ego"], ["rho_abs", "rho_shape", "rho_offset"])
    print(f"\n  L2  rho_offset {agg2['rho_offset']['mean']:.2f}   rho_shape {agg2['rho_shape']['mean']:.2f}")
    l2anchor_agg = l2_anchor_aggregate(l2a)
    if l2anchor_agg:
        print("      per-anchor err across episodes (mean / min / max)")
        rows = []
        for label, v in l2anchor_agg.items():
            p, r = v.get("err_pos_mm"), v.get("err_rot_deg")
            rows.append([
                label,
                f(p["mean"], 1) if p else "n/a", f(p["min"], 1) if p else "n/a", f(p["max"], 1) if p else "n/a",
                f(r["mean"], 1) if r else "n/a", f(r["min"], 1) if r else "n/a", f(r["max"], 1) if r else "n/a",
            ])
        print_table(
            rows,
            ["anchor", "err_pos mm mean", "min", "max", "err_rot deg mean", "min", "max"],
            "lrrrrrr")

    if l3["dispersion"].get("available"):
        print(f"\n  L3  dispersion {l3['dispersion']['abs']['dispersion_ratio']:.2f}   "
              f"energy p {l3['energy_test']['p_value']:.3f}   "
              f"C2ST AUC {l3['c2st'].get('auc', float('nan')):.3f}")
    print(RULE)

    report = {
        # Bump when a field changes meaning or disappears. compare_reports.py
        # refuses to mix versions rather than silently comparing different things.
        "schema_version": REPORT_SCHEMA_VERSION,
        "real_dir": str(real_dir),
        "ego_dirs": [str(p) for p in ego_dirs],
        "n_real": len(reals),
        "n_ego": len(egos),
        "real_reference": real_reference,
        "anchors_selected": anchors_sel,
        # What produced this report. Without pose_key/tcp_offset_m recorded here,
        # two reports cannot be told apart when the TCP convention changes - the
        # only clue would be the noise floor shifting, which is far too subtle.
        "run": {
            "task": args.task,
            "pipeline": args.pipeline,
            "cohort": args.cohort if args.task else None,
            "variant": eval_variant_name(args) if args.task else None,
            "pose_key": args.pose_key,
            "tcp_offset_m": args.tcp_offset_m,
            "max_real": args.max_real,
            "real_split_manifest": real_reference["manifest"],
            "real_split": real_reference["split"],
            "reference_set_id": real_reference["reference_set_id"],
            "ego_subdir": args.ego_subdir,
            "ego_hz": args.ego_hz,
        },
        "scope": {
            "note": "L1 / L2a / L3 (except anchor_distributions) were computed on this scope; "
                    "L0 / L2b / L2c always use every loaded episode",
            "grasp_health_filter_applied": not args.no_grasp_health_filter,
            "mode_events": mode_events,
            "n_real_in_scope": len(reals_scope),
            "n_ego_in_scope": len(egos_scope),
            "segment": segment_info,
        },
        "L0": l0,
        "L0_aggregate": l0agg,
        "L1": l1,
        "L1_aggregate": agg1,
        "L2_offset_shape": l2os,
        "L2_offset_shape_aggregate": agg2,
        "L2_anchors": l2a,
        "L2_anchors_reference": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in anchor_ref.get("anchors", {}).items()},
        "L2_anchors_aggregate": l2anchor_agg,
        "L2_segments": l2s,
        "L3": l3,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "eval_report.json"
        with open(path, "w") as fh:
            json.dump(report, fh, indent=2, default=_jsonable)
        print(f"\nWrote {path}")
    return report


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate ego-derived EEF trajectories against real-robot episodes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", default=None,
                   help="task directory name under --data-root, e.g. stack_object_horizontal. "
                        "With --pipeline this derives the real set, the ego sessions and the "
                        "output directory name; --real/--ego are then unnecessary.")
    p.add_argument("--pipeline", default=None,
                   help="pipeline id used by batch_preprocess_le_ver.sh, e.g. h2g_pinch_index")
    p.add_argument("--cohort", default="all",
                   help="operator cohort to evaluate, or 'all' (default) for every cohort of "
                        "the task, e.g. --cohort hyj")
    p.add_argument("--data-root", default=str(REPO_ROOT.parent / "DATA_new"),
                   help="source data root holding <task>/ directories")
    p.add_argument("--out-root", default="outputs",
                   help="output root holding <pipeline>/ and eval/ (default: outputs)")
    p.add_argument("--real", default=None,
                   help="directory of nero_teleop_jsonl episodes; only needed without --task")
    p.add_argument("--ego", nargs="*", default=[], help="ego session directories")
    p.add_argument("--ego-list", default=None,
                   help="file with one ego session directory per line ('#' comments allowed); "
                        "batch_preprocess_le_ver.sh writes this as sessions.txt")
    p.add_argument("--out", default=None,
                   help="directory for eval_report.json; with --task it defaults to "
                        "<out-root>/eval/<task>__<pipeline>__<cohort>__<variant>/")
    p.add_argument("--anchors", type=int, nargs="*", default=None,
                   help="which gripper events to treat as anchors, e.g. --anchors 0 2 "
                        "(default: all). Use this to drop toggles that are not task-relevant.")
    p.add_argument("--segment-start", type=int, default=None,
                   help="restrict L1 / L2a / L3 (shape-based metrics) to the span starting at this "
                        "0-based gripper-event index, e.g. 0 = the 1st toggle (same indexing as "
                        "--anchors). Default: from the episode start. L0 / L2b / L2c are unaffected "
                        "and still see the whole episode.")
    p.add_argument("--segment-end", type=int, default=None,
                   help="restrict L1 / L2a / L3 to the span ending at this 0-based gripper-event "
                        "index, e.g. '--segment-start 0 --segment-end 2' covers 1st open -> "
                        "1st close -> 2nd open without naming the middle event. Default: through "
                        "the episode end.")
    p.add_argument("--no-grasp-health-filter", action="store_true",
                   help="by default, episodes whose grasp-toggle count differs from the real set's "
                        "mode (e.g. a stuck grasp signal) are excluded from L1 / L2a / L3 (shape-based "
                        "metrics), since that pipeline failure mode is a known, recurring bug that "
                        "otherwise pollutes pooled statistics silently - see grasp_events_match in "
                        "L0. Pass this flag to include them anyway.")
    p.add_argument("--ego-subdir", default="robot_eef_scene_camera_axis_corrected")
    p.add_argument("--ego-hz", type=float, default=30.0)
    p.add_argument("--pose-key", default="tcp_tip_pose",
                   choices=["tcp_tip_pose", "tcp_pose", "flange_pose", "fk_pose"],
                   help="tcp_tip_pose (default): recompute the fingertip TCP from flange_pose + "
                        "--tcp-offset-m, matching what the ego side considers 'TCP'. tcp_pose: the "
                        "robot's own reported field, which is the gripper-CENTER point instead - "
                        "~50mm off along the approach axis, kept only for comparison/debugging.")
    p.add_argument("--tcp-offset-m", type=float, default=0.18,
                   help="fingertip offset along the flange's local +X, metres (NERO default 0.18; "
                        "see ego2exe/README.md 'site:tcp'). Only used by --pose-key tcp_tip_pose.")
    p.add_argument("--bbox-margin", type=float, default=0.02, help="workspace bbox margin, metres")
    p.add_argument("--max-real", type=int, default=10,
                   help="use only the first N real episodes, in capture-time order (default: 10). "
                        "A reference set recorded by two operators carries their difference in its "
                        "own dispersion, which deflates every rho measured against it - e.g. "
                        "stack_bowl's floor is 24.6 mm pooled across two operators' full sets but "
                        "20.3-20.5 mm within either one's block. 0 means no cap (all real episodes).")
    p.add_argument(
        "--real-split-manifest", default=None,
        help="JSON split created by correct_eef_with_real_anchors.py. When supplied, load the "
             "exact names in --real-split and ignore --max-real.",
    )
    p.add_argument(
        "--real-split", choices=("calibration", "eval", "reserve"), default="eval",
        help="block selected from --real-split-manifest (default: eval).",
    )
    p.add_argument(
        "--allow-real-leakage", action="store_true",
        help="allow corrected trajectories to be evaluated against calibration episodes. "
             "This is rejected by default because it is not a held-out estimate.",
    )
    p.add_argument("--min-real", type=int, default=10,
                   help="warn below this many real episodes; the noise floor gets unreliable")
    p.add_argument("--knn-k", type=int, default=3, help="k for the manifold-overlap radius")
    p.add_argument("--n-perm", type=int, default=2000, help="permutations for the energy test")
    p.add_argument("--n-perm-c2st", type=int, default=200)
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print the per-episode tables too (L0 gates, L1, L2a, L2b anchors). "
                        "Off by default: at 30 episodes that is ~450 of ~600 lines and buries "
                        "the failures. The full detail is always written to eval_report.json.")
    p.add_argument("--no-c2st", action="store_true",
                   help="skip the classifier two-sample test. It saturates (AUC 1.000) until rho "
                        "is near 2 and costs a few hundred cross-validated fits, so it is pure "
                        "overhead in the meantime.")
    p.add_argument("--no-cache", action="store_true",
                   help="do not reuse the cached real-vs-real distance block")
    p.add_argument("--c2st-timing", action="store_true",
                   help="let C2ST use duration/speed features (off by default: the human is "
                        "systematically ~2.5x faster, which saturates the test trivially)")
    run(p.parse_args())


if __name__ == "__main__":
    main()
