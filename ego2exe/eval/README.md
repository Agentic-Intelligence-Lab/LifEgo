# ego2exe trajectory evaluation

Measures how far converted ego EEF trajectories sit from a set of real-robot
episodes of the same task. For what every reported number means, see
[METRICS.md](METRICS.md).

## Preprocess once, evaluate many times

```bash
# once per (task, pipeline) - WiLoR is cached and shared, so switching pipeline
# only re-runs the cheap export/IK stages
ego2exe/scripts/batch_preprocess_le_ver.sh --task stack_object_horizontal \
    --pipeline h2g_pinch_index --hand2gripper-mode pinch_plane --forward-seed index_tip

# evaluate - pure selection, nothing is recomputed
python ego2exe/eval/eval_ego2exe.py --task stack_object_horizontal --pipeline h2g_pinch_index --cohort all
python ego2exe/eval/eval_ego2exe.py --task stack_object_horizontal --pipeline h2g_pinch_index --cohort hyj
python ego2exe/eval/eval_ego2exe.py --task stack_object_horizontal --pipeline h2g_pinch_index --cohort all \
    --segment-start 0 --segment-end 2
```

Everything is derived from the directory layout - there is no registry file:

```
DATA_new/<task>/<date>_ego_<operator>/     ego cohorts   (the suffix is the cohort name)
DATA_new/<task>/*nero*/                    real set      (exactly one per task)
outputs/_wilor/<session>/                  WiLoR cache, shared across pipelines
outputs/<pipeline>/<task>/<date>_ego_<operator>/<session>/    export + IK
outputs/eval/<task>__<pipeline>__<cohort>__<variant>/         eval_report.json
```

The eval output directory is named from the parameters, so runs never collide and
`ls outputs/eval/` shows the whole matrix:

| flag | contributes |
|---|---|
| (none) | `full` |
| `--segment-start 0 --segment-end 2` | `seg0-2` |
| `--anchors 1 2` | `a1-2` |
| `--no-grasp-health-filter` | `nofilter` |

Four comparisons, each along one axis with the rest fixed:

```
change algorithm  eval/stack_object_horizontal__{h2g_humanego,h2g_pinch_index}__all__full/
change cohort     eval/stack_object_horizontal__h2g_pinch_index__{hyj,xule,ymq,all}__full/
change span       eval/stack_object_horizontal__h2g_pinch_index__all__{full,seg0-2}/
change task       eval/{stack_object_horizontal,stack_object_lean}__h2g_pinch_index__all__full/
```

`--real` / `--ego` / `--ego-list` / `--out` still work for one-off runs outside this
layout; `--task` simply fills them in.

## The whole matrix in one command

```bash
ego2exe/scripts/run_eval_matrix.sh              # 2 tasks x 3 pipelines x full/seg1-2
ego2exe/scripts/run_eval_matrix.sh --cohorts    # + hyj / xule / ymq separately
ego2exe/scripts/run_eval_matrix.sh --dry-run    # print the commands, run nothing
```

Runs every evaluation and prints the comparison tables. Nothing is re-processed —
evaluation is pure selection over what `batch_preprocess_le_ver.sh` already produced, so
the whole matrix takes a few minutes. `--tasks` / `--pipelines` override the lists.

`seg1-2` is the **carry phase** — gripper close to gripper open, i.e. the span
where the object is actually held. It is 32%→64% of the episode on
`stack_object_horizontal` and 39%→65% on `stack_object_lean`; the report prints
the measured range so the same variant name is not mistaken for the same span
across tasks.

## Comparing runs

```bash
python ego2exe/eval/compare_reports.py outputs/eval/stack_object_*__all__*
```

```
task        pipeline       variant   n_ego  floor_mm  rho_rot  rho_SE3  disp  glob_rot
horizontal  finger_center  full         29      18.9    2.79 *   2.96 *  1.62     6.9 *
horizontal  humanego       full         29      18.9    7.54     4.89    1.62    35.8

  * = best in column (lower is better)
  = identical across all reports, hidden: rho_pos=3.18, rho_off=2.87, rho_shp=3.01
    (hand2gripper variants change orientation only, so the position columns are expected
     to be identical)
```

Three things it does that reading the JSONs by hand does not:

- **Collapses constant columns.** `rho_pos` / `rho_offset` / `rho_shape` are
  byte-identical across hand2gripper variants — that choice only changes
  orientation — so they are hidden with their shared value rather than inviting a
  reading of noise as signal. `--show-constant` expands them.
- **Warns when the noise floor moves.** Comparing `rho` across `full` and `seg1-2`
  compares two different denominators, since the floor is recomputed on the cropped
  span. Add `--metric D_pos_mm,D_rot_deg` for the absolute numbers.
- **Refuses to mix schema versions.** Reports carry `schema_version`; older ones are
  skipped with a note to re-run rather than silently compared.

Useful flags: `--metric` picks the columns, comma-separated
(`rho_pos,rho_rot,rho_SE3,rho_off,rho_shp,D_pos_mm,D_rot_deg,floor_mm,floor_deg,disp,glob_rot,n_ego`);
`--sort` reorders rows; `--show-constant` expands the collapsed columns.

## Runtime

About 16 s for 31 real + 29 ego. Two caches make repeated runs cheap:

- The **real-vs-real distance block** (~1/3 of all pairs) is identical on every run
  against the same reference set, so it is cached in `outputs/_cache/real_dtw/`,
  keyed by real dir + pose convention + segment. Episode names are stored and
  re-checked, so adding or removing a real episode invalidates it. `--no-cache` opts out.
- The **per-anchor real-side statistics** are computed once for the whole run instead
  of once per ego episode.

`--no-c2st` skips the classifier two-sample test, which saturates at AUC 1.000 until
`rho` approaches 2 and is pure overhead before then. `--verbose` restores the
per-episode tables (off by default: ~450 of 614 lines at 30 episodes).

> **Never point `--real` at an ego directory.** In the DATA_new layout the ego
> directories carry `.jsonl` files with the identical schema and filenames as the
> real ones - the arm just sat parked, so every pose is frozen. Loading them as a
> reference set would destroy the noise floor silently, so the loader raises a hard
> error when a "reference" set never moves.

Prints a layered report and writes `eval_report.json`.

## Restricting the comparison to a segment

By default L1/L2a/L3 compare the whole episode. To restrict them to a span
between two gripper toggles instead - e.g. "from the 1st open to the 1st close
to the 2nd open" - use `--segment-start`/`--segment-end` with the same 0-based
event indexing as `--anchors`:

```bash
python ego2exe/eval/eval_ego2exe.py --task stack_object_horizontal \
    --pipeline h2g_pinch_index --cohort all --segment-start 0 --segment-end 2
```

Only names the two boundary events - you don't need to name the one in the
middle. Omit `--segment-start`/`--segment-end` to leave that end open (episode
start / episode end). **`--segment-start 0` is the first gripper toggle, not
frame 0** - on this data that toggle sits ~10-17% into the episode, so the
approach phase is excluded; leave the flag off entirely to start from the
beginning. Only L1, L2a, and L3's shape-based metrics (dispersion,
energy test, manifold overlap, global correction, C2ST) are restricted; L0, L2b
(anchors) and L2c (segments) always see the whole episode regardless, since they
already work per-event and per-natural-segment. An episode that doesn't have the
requested event (e.g. a shorter or broken recording) is dropped from the scoped
metrics with a printed reason, not silently misaligned.

The same restriction is applied automatically for a different reason: episodes
whose gripper-toggle count doesn't match the real set's mode (most often a
grasp signal that got stuck, e.g. `action_grasp` never leaving 1.0 for an entire
episode - a recurring failure mode in this pipeline) are excluded from L1/L2a/L3
by default, so one broken episode can't quietly skew the pooled statistics.
Pass `--no-grasp-health-filter` to include them anyway (e.g. to see how much a
known-bad episode actually moves the numbers).

## Preprocessing options

`ego2exe/scripts/batch_preprocess_le_ver.sh --task TASK --pipeline NAME` converts every ego
video of one task. Useful flags:

| flag | effect |
|---|---|
| `--cohort hyj` | only one operator instead of all of them |
| `--hand2gripper-mode` / `--forward-seed` | the pipeline variant being tested |
| `--skip-ik` | stop after the EEF export - the only stage evaluation needs |
| `--replays` | also render replay MP4s (slow, off by default) |
| `--force` | redo export/IK for this pipeline |
| `--force-raw` | also redo the shared WiLoR stage (rare, expensive) |
| `--limit N`, `--dry-run` | try a few first |

Stages are skipped when their outputs exist, so an interrupted run resumes, and a
video that fails is logged to `_logs/<session>.log` while the batch continues;
per-stage status lands in `manifest.tsv`.

`outputs/<pipeline>/pipeline_meta.json` records what the pipeline id means
(hand2gripper mode, seed, axis correction, git commit). Re-running the same
`--pipeline` with a *different* config prints a warning rather than silently
mixing two configs under one name - use a new name instead.

`ego2exe/scripts/preprocess.sh` still handles a single video for quick one-offs.

## The one idea

Every number is a **ratio to a noise floor**:

```
rho = D(ego, real) / D(real, real)
```

Absolute millimetres are not comparable across tasks — the same pipeline scores a
17 mm floor on `stack_object_horizontal` and 57 mm on `stack_bowl`, purely because
the reference sets differ in quality. `rho` is comparable. **`rho -> 1` means the
ego trajectory sits as close to the real set as one real demo sits to the others**,
which is the point past which further optimisation is chasing noise the reference
cannot resolve.

The floor uses a **leave-one-out** convention: the quantity under test is "one ego
vs N real", so the reference must be "one real vs the other N-1", not the all-pairs
mean. Same estimator, comparable spread, and it yields a z-score for free.

## Layers

| Layer | What it answers |
|---|---|
| **L0** | Cheap gates: grasp-event count, workspace entry, table penetration, valid frames, speed feasibility. **Reported, never blocking** — everything below is computed regardless. |
| **L1** | `rho_pos`, `rho_rot`, and their geometric mean `rho_SE3`. Both inputs are dimensionless after normalisation, so the geometric mean is well defined without inventing a mm-per-degree conversion. |
| **L2a** | Splits position error into `rho_offset` (centroid displacement — a placement fix helps) and `rho_shape` (error after removing each centroid — it does not). |
| **L2b** | Absolute pose error at each contact anchor, **all anchors reported by default**. Object positions are fixed within a layout, so the real TCP at each gripper toggle measures that contact pose in base frame — the only absolute ground truth available without hand tracking. Each anchor is labelled with the gripper transition it represents (`open` / `close`), derived from the signal. `--anchors 1 2` narrows to a subset. |
| **L2c** | `rho` per task phase, segmented by the gripper events. Free-space error and contact error are not equally costly downstream. |
| **L3** | Set-level, needs several ego episodes. See below. |

## L3

* **`dispersion_ratio`** — how much the ego episodes disagree *with each other*
  versus how much the real ones do. The most decision-relevant number in the
  report: near 1 with a large `rho` means the pipeline is repeatable and the gap
  is a bias some transform could remove; much greater than 1 means the pipeline
  itself is noisy and no fixed correction will help — reduce variance first.
* **Energy distance + permutation test** — `E = 2·mean(D_xy) − mean(D_xx) −
  mean(D_yy)` straight off the pooled distance matrix. Zero iff the two sets come
  from the same distribution, and the permutation p-value needs no distributional
  assumption, which matters at n ≈ 30.
* **Manifold overlap** — `precision` (do ego episodes land inside the real
  manifold?) and `coverage` (do they span its variety?). A pipeline can score high
  precision and low coverage by collapsing onto one mode. When precision is 0 the
  report also prints the nearest-real distances, so "outside" has a magnitude.
* **C2ST** — classifier two-sample test, AUC → 0.5 is the end state. Timing
  features are excluded by default: the human is ~2.5× faster than teleop by
  construction, and letting the classifier use duration saturates the test on
  something we do not care about. Pass `--c2st-timing` to include them. Needs ≥5
  episodes per class.
* **Global correction** — fits one shared translation + one local rotation across
  all ego episodes, reported **raw / self-fit / leave-one-out**. Only the LOO
  column is a deployable expectation; the self-fit column is printed next to it so
  the size of the overfit is visible rather than implied.
* **Anchor distributions** — per anchor, ego cloud vs real cloud: mean offset,
  Mahalanobis distance under the real covariance, and `spread_ratio`. Distinguishes
  "merely offset" (spread matched) from "also less repeatable" (spread inflated).
  Suppressed below 3 ego episodes, where a spread estimate carries no information.

## Anchor labels

`action_grasp` is **1 = closed, 0 = open** — verified against `gripper_ctrl.value`
(width 0.0016 m at `action_grasp` 1, 0.0990 m at 0). The metadata string
`"normalized_0_closed_1_open_or_mapped"` reads the other way round; the data wins.

Anchor labels are derived from the transition direction of the signal, never from
the event index. On `stack_object_horizontal` the episodes **start closed**, so the
sequence is:

| anchor | phase | transition | meaning |
|---|---|---|---|
| `E1_open` | 16% | open | gripper opens to approach — *not* task-relevant |
| `E2_close` | 32% | close | **the grasp** |
| `E3_open` | 64% | open | **the release** |
| `E4_close` | 81% | close | gripper closes at rest — *not* task-relevant |

So there is one pick-and-place per episode, not two, and the anchors worth
optimising against are `--anchors 1 2`. Assuming index parity (`even = close`)
labels every anchor backwards on this dataset.

The report prints `ego dir` per anchor: whether the ego signal toggles the same
direction as the real one at that anchor. A `NO` there means the two sides are not
describing the same event and the error number is meaningless.

## Gotchas

1. **Pair by layout.** If objects moved between episodes, the floor measures object
   placement rather than operator variation. On `stack_bowl` this inflated the floor
   from ~20 mm to 57 mm and would have loosened every threshold threefold.
2. **The floor needs episodes.** Below ~10 real episodes it is unreliable
   (`stack_bowl` with 4: 20.3° orientation floor; `stack_object` with 31: 5.0°).
   The script warns via `--min-real`.
3. **Do not report DTW and Fréchet as independent evidence** — they are strongly
   correlated (Toohey & Duckham). Only DTW is computed here, deliberately.
4. **Never quote a self-fitted residual as a gain.** Use the LOO column.
5. **`rho = 1` does not mean a policy can learn it.** The final judges are replay
   success rate and policy transfer gain; `rho` is the fast development-loop proxy.

## Baseline: `stack_object_horizontal`, 3 ego vs 31 real

| | value |
|---|---|
| noise floor | `D_pos` 17.0 ± 1.8 mm, `D_rot` 5.0 ± 0.5° |
| L1 | `rho_pos` 3.73, `rho_rot` 7.37, `rho_SE3` 5.23 |
| L2a | `rho_offset` 4.51, `rho_shape` 3.05 |
| L2c | worst segment S0 (approach) `rho` 6.19 |
| L3 | dispersion 2.57, energy p = 0.0005, precision 0.00 / coverage 0.00 |
| L3 global correction | 35.4° rotation → orientation 36.9° raw / 13.9° LOO |

Use these as the regression baseline: any pipeline change should move `rho_SE3`
down from 5.23.
