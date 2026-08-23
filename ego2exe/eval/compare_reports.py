#!/usr/bin/env python3
"""Compare several eval_report.json side by side.

Reading a dozen reports one at a time does not scale, and two traps only show up
when they are lined up:

* **Constant columns.** ``rho_pos`` / ``rho_offset`` / ``rho_shape`` are byte-identical
  across hand2gripper variants, because that choice only changes orientation. Those
  columns are marked ``=`` so nobody reads a difference into them.
* **A moving denominator.** The noise floor is recomputed per segment, so comparing
  ``rho`` across ``full`` and ``seg1-2`` compares two different denominators. When the
  floors differ, the absolute ``D_pos_mm`` / ``D_rot_deg`` are shown alongside.

Example
-------
    python ego2exe/eval/compare_reports.py outputs/eval/stack_object_*__all__*
    python ego2exe/eval/compare_reports.py --metric rho_rot,D_rot_deg outputs/eval/*
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPORT_SCHEMA_VERSION = 1

# label -> (json path, digits, lower_is_better, scale)
# The L1 floors are stored in metres; everything else is already in report units.
METRICS: dict[str, tuple[tuple, int, bool, float]] = {
    "n_ego": (("scope", "n_ego_in_scope"), 0, False, 1),
    "floor_mm": (("L1", "floors", "abs", "mean"), 1, False, 1000),
    "floor_deg": (("L1", "floors", "rot", "mean"), 1, False, 1),
    "D_pos_mm": (("L1_aggregate", "D_pos_mm", "mean"), 1, True, 1),
    "D_rot_deg": (("L1_aggregate", "D_rot_deg", "mean"), 1, True, 1),
    "rho_pos": (("L1_aggregate", "rho_pos", "mean"), 2, True, 1),
    "rho_rot": (("L1_aggregate", "rho_rot", "mean"), 2, True, 1),
    "rho_SE3": (("L1_aggregate", "rho_se3", "mean"), 2, True, 1),
    "rho_off": (("L2_offset_shape_aggregate", "rho_offset", "mean"), 2, True, 1),
    "rho_shp": (("L2_offset_shape_aggregate", "rho_shape", "mean"), 2, True, 1),
    "disp": (("L3", "dispersion", "abs", "dispersion_ratio"), 2, True, 1),
    "glob_rot": (("L3", "global_correction", "fitted_rotation_deg"), 1, True, 1),
}
DEFAULT_METRICS = ["n_ego", "floor_mm", "rho_pos", "rho_rot", "rho_SE3", "rho_off", "rho_shp",
                   "disp", "glob_rot"]


def dig(d: dict, path: tuple):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d if isinstance(d, (int, float)) else None


def load(paths: list[Path]) -> tuple[list[dict], list[str]]:
    reports, warnings = [], []
    for p in paths:
        f = p / "eval_report.json" if p.is_dir() else p
        if not f.is_file():
            continue
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            warnings.append(f"{f}: not valid JSON, skipped")
            continue
        v = r.get("schema_version")
        if v != REPORT_SCHEMA_VERSION:
            warnings.append(
                f"{f.parent.name}: schema_version={v} (expected {REPORT_SCHEMA_VERSION}) - skipped. "
                f"Re-run the evaluation to regenerate it."
            )
            continue
        run = r.get("run") or {}
        name = f.parent.name
        r["_key"] = dict(
            name=name,
            task=run.get("task") or "?",
            pipeline=run.get("pipeline") or "?",
            cohort=run.get("cohort") or "?",
            variant=run.get("variant") or "full",
        )
        reports.append(r)
    return reports, warnings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", type=Path,
                    help="eval output directories (or eval_report.json files)")
    # Comma-separated, not nargs="*": a variadic option would greedily swallow the
    # positional paths that follow it.
    ap.add_argument("--metric", default=None,
                    help=f"comma-separated metrics to show (default: {','.join(DEFAULT_METRICS)}). "
                         f"available: {','.join(METRICS)}")
    ap.add_argument("--sort", default="task,variant,pipeline",
                    help="comma-separated key order for rows (default: task,variant,pipeline)")
    ap.add_argument("--show-constant", action="store_true",
                    help="show constant columns in full instead of collapsing them to '='")
    args = ap.parse_args()

    reports, warnings = load(args.paths)
    for w in warnings:
        print(f"[warn] {w}", file=sys.stderr)
    if not reports:
        raise SystemExit("no comparable reports found")

    metrics = ([m.strip() for m in args.metric.split(',') if m.strip()]
               if args.metric else DEFAULT_METRICS)
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise SystemExit(f"unknown metric(s): {', '.join(unknown)}\navailable: {' '.join(METRICS)}")

    keys = [k.strip() for k in args.sort.split(",") if k.strip()]
    reports.sort(key=lambda r: tuple(str(r["_key"].get(k, "")) for k in keys))

    vals = {m: [None if (v := dig(r, METRICS[m][0])) is None else v * METRICS[m][3]
            for r in reports] for m in metrics}

    # A column whose every value is identical carries no information for this
    # comparison - flag it rather than inviting a reading of noise as signal.
    constant = {
        m: len(reports) > 1 and len({None if v is None else round(v, 6) for v in vals[m]}) == 1
        for m in metrics
    }
    best = {}
    for m in metrics:
        nums = [v for v in vals[m] if v is not None]
        if nums and not constant[m] and METRICS[m][2]:
            best[m] = min(nums)

    id_cols = [k for k in ("task", "pipeline", "cohort", "variant")
               if len({r["_key"][k] for r in reports}) > 1 or k in keys]
    shown = [m for m in metrics if args.show_constant or not constant[m]]

    header = [k for k in id_cols] + shown
    rows = []
    for i, r in enumerate(reports):
        row = [str(r["_key"][k]).replace("stack_object_", "").replace("h2g_", "") for k in id_cols]
        for m in shown:
            v, nd = vals[m][i], METRICS[m][1]
            if v is None:
                row.append("n/a")
            else:
                mark = " *" if m in best and abs(v - best[m]) < 1e-9 else ""
                row.append(f"{v:.{nd}f}{mark}")
        rows.append(row)

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    aligns = ["l"] * len(id_cols) + ["r"] * len(shown)

    def line(vals_):
        return "  ".join(v.ljust(widths[i]) if aligns[i] == "l" else v.rjust(widths[i])
                         for i, v in enumerate(vals_))

    print()
    print(line(header))
    print("  ".join("-" * w for w in widths))
    # blank line between groups, grouping on the sort keys minus the finest one
    group_cols = [i for i, k in enumerate(id_cols) if k in keys[:-1]]
    prev = None
    for r in rows:
        cur = tuple(r[i] for i in group_cols) if group_cols else None
        if prev is not None and cur != prev:
            print()
        prev = cur
        print(line(r))
    print()
    if any(m in best for m in shown):
        print("  * = best in column (lower is better)")

    dropped = [m for m in metrics if constant[m] and not args.show_constant]
    if dropped:
        print(f"  = identical across all {len(reports)} reports, hidden: "
              + ", ".join(f"{m}={vals[m][0]:.{METRICS[m][1]}f}" for m in dropped
                          if vals[m][0] is not None))
        print("    (hand2gripper variants change orientation only, so the position columns "
              "are expected to be identical)")

    floors = {round(v, 3) for v in vals.get("floor_mm", []) if v is not None}
    if len(floors) > 1 and "D_pos_mm" not in shown:
        print(f"\n  [warn] the noise floor differs across these rows ({', '.join(f'{x:.1f}' for x in sorted(floors))} mm), "
              f"so rho values have different denominators.\n"
              f"         Add --metric D_pos_mm D_rot_deg ... to compare absolute errors too.")


if __name__ == "__main__":
    main()
