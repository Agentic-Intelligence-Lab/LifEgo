#!/usr/bin/env bash
# Evaluate the raw pipeline and all real-anchor correction variants on held-out real data.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/run_anchor_correction_eval.sh [options]

Runs the source baseline plus the six default correction variants against the
same held-out `eval` block of each task's real split manifest. No preprocessing
or correction is run here.

Options:
  --task TASK             evaluate one task
  --tasks "a b"           override task list (default: all three tasks)
  --source-pipeline NAME  default: h2g_pinch_index
  --cohort NAME           default: all
  --spaces "a b"          default: ray_depth free_xyz
  --methods "a b"         default: linear min_bending smooth_spline
  --rotations "a b"       default: none; choices: none global_so3
  --include-rot-only       include <source>__rot_global_so3 in the comparison
  --sample-seed N         selects corrected pipeline names (default: 0)
  --split-manifest FILE   only valid with --task; otherwise task-specific defaults
  --data-root DIR         default: ../DATA_new
  --out-root DIR          pipeline/output root (default: outputs)
  --eval-root DIR         report root (default: <out-root>/eval_anchor_correction)
  --full-only             skip the task-critical anchor-to-anchor segment run
  --with-c2st             enable C2ST (disabled by default)
  --dry-run               print commands only
  -h, --help

Environment:
  PY  Python executable (default: python)
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASKS="stack_bowl stack_object_horizontal stack_object_lean"
SINGLE_TASK=""
SOURCE_PIPELINE="h2g_pinch_index"
COHORT="all"
SPACES="ray_depth free_xyz"
METHODS="linear min_bending smooth_spline"
ROTATIONS="none"
SAMPLE_SEED=0
SPLIT_MANIFEST=""
DATA_ROOT="${REPO_ROOT}/../DATA_new"
OUT_ROOT="outputs"
EVAL_ROOT=""
FULL_ONLY=0
C2ST=0
DRY_RUN=0
INCLUDE_ROT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) SINGLE_TASK="$2"; shift 2 ;;
    --tasks) TASKS="$2"; shift 2 ;;
    --source-pipeline) SOURCE_PIPELINE="$2"; shift 2 ;;
    --cohort) COHORT="$2"; shift 2 ;;
    --spaces) SPACES="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --rotations) ROTATIONS="$2"; shift 2 ;;
    --include-rot-only) INCLUDE_ROT_ONLY=1; shift ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --split-manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --eval-root) EVAL_ROOT="$2"; shift 2 ;;
    --full-only) FULL_ONLY=1; shift ;;
    --with-c2st) C2ST=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -n "${SINGLE_TASK}" ]]; then TASKS="${SINGLE_TASK}"; fi
if [[ -n "${SPLIT_MANIFEST}" && $(echo "${TASKS}" | wc -w) -ne 1 ]]; then
  echo "--split-manifest requires exactly one --task" >&2; exit 2
fi
for task in ${TASKS}; do
  case "${task}" in
    stack_bowl|stack_object_horizontal|stack_object_lean) ;;
    *) echo "unsupported task: ${task}" >&2; exit 2 ;;
  esac
done
for rotation in ${ROTATIONS}; do
  [[ "${rotation}" == "none" || "${rotation}" == "global_so3" ]] || {
    echo "unsupported rotation correction: ${rotation}" >&2; exit 2; }
done

PY="${PY:-python}"
abspath() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *) if [[ -e "${PWD}/$1" ]]; then printf '%s/%s' "${PWD}" "$1"
       else printf '%s/%s' "${REPO_ROOT}" "$1"; fi ;;
  esac
}
DATA_ROOT_ABS="$(abspath "${DATA_ROOT}")"
OUT_ROOT_ABS="$(abspath "${OUT_ROOT}")"
[[ -n "${EVAL_ROOT}" ]] || EVAL_ROOT="${OUT_ROOT_ABS}/eval_anchor_correction"
EVAL_ROOT="$(abspath "${EVAL_ROOT}")"
EVAL="${REPO_ROOT}/ego2exe/eval/eval_ego2exe.py"
CMP="${REPO_ROOT}/ego2exe/eval/compare_reports.py"
EXTRA=()
[[ ${C2ST} -eq 0 ]] && EXTRA=(--no-c2st)

PIPELINES=("${SOURCE_PIPELINE}")
[[ ${INCLUDE_ROT_ONLY} -eq 1 ]] && PIPELINES+=("${SOURCE_PIPELINE}__rot_global_so3")
for space in ${SPACES}; do
  if [[ "${space}" == "ray_depth" ]]; then label="ray"; elif [[ "${space}" == "free_xyz" ]]; then label="xyz"; else
    echo "unsupported correction space: ${space}" >&2; exit 2
  fi
  for method in ${METHODS}; do
    [[ "${method}" == "linear" || "${method}" == "min_bending" || "${method}" == "smooth_spline" ]] || {
      echo "unsupported correction method: ${method}" >&2; exit 2; }
    for rotation in ${ROTATIONS}; do
      pipeline="${SOURCE_PIPELINE}__anchor_${label}_${method}_s${SAMPLE_SEED}"
      [[ "${rotation}" == "global_so3" ]] && pipeline="${pipeline}__rot_global_so3"
      PIPELINES+=("${pipeline}")
    done
  done
done

run_eval() {
  local task="$1" pipeline="$2" split="$3" variant="$4" seg_start="$5" seg_end="$6"
  local out_dir="${EVAL_ROOT}/${task}__${pipeline}__${COHORT}__${variant}"
  local cmd=("${PY}" "${EVAL}"
    --task "${task}" --pipeline "${pipeline}" --cohort "${COHORT}"
    --data-root "${DATA_ROOT_ABS}" --out-root "${OUT_ROOT_ABS}" --out "${out_dir}"
    --real-split-manifest "${split}" --real-split eval --max-real 0
    "${EXTRA[@]}")
  if [[ "${task}" == "stack_bowl" ]]; then cmd+=(--anchors 0 1); else cmd+=(--anchors 1 2); fi
  if [[ -n "${seg_start}" ]]; then cmd+=(--segment-start "${seg_start}" --segment-end "${seg_end}"); fi
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '  '; printf '%q ' "${cmd[@]}"; printf '\n'
    return 0
  fi
  mkdir -p "${EVAL_ROOT}"
  local log="${out_dir}.log"
  printf '  %-76s ' "${task}  ${pipeline}  ${variant}"
  if (cd "${REPO_ROOT}" && "${cmd[@]}") >"${log}" 2>&1; then
    echo "ok"
    return 0
  fi
  echo "FAILED (see ${log})"
  return 1
}

echo "=============================================================================="
echo "held-out real-anchor correction evaluation"
echo "=============================================================================="
echo "tasks      : ${TASKS}"
echo "pipelines  : ${#PIPELINES[@]} (baseline + correction variants)"
echo "cohort     : ${COHORT}"
echo "rotations  : ${ROTATIONS}"
echo "rot only   : $([[ ${INCLUDE_ROT_ONLY} -eq 1 ]] && echo yes || echo no)"
echo "reports    : ${EVAL_ROOT}"
echo

N_FAIL=0
for task in ${TASKS}; do
  if [[ -n "${SPLIT_MANIFEST}" ]]; then
    split="$(abspath "${SPLIT_MANIFEST}")"
  else
    split="${OUT_ROOT_ABS}/anchor_calibration_splits/${task}.json"
  fi
  if [[ ${DRY_RUN} -eq 0 && ! -f "${split}" ]]; then
    echo "split manifest not found: ${split}" >&2
    echo "run batch_correct_eef_anchors.sh --task ${task} first" >&2
    exit 1
  fi
  if [[ "${task}" == "stack_bowl" ]]; then seg_start=0; seg_end=1; else seg_start=1; seg_end=2; fi
  for pipeline in "${PIPELINES[@]}"; do
    run_eval "${task}" "${pipeline}" "${split}" "heldout_eval_full" "" "" || N_FAIL=$((N_FAIL + 1))
    if [[ ${FULL_ONLY} -eq 0 ]]; then
      run_eval "${task}" "${pipeline}" "${split}" "heldout_eval_seg${seg_start}-${seg_end}" "${seg_start}" "${seg_end}" || N_FAIL=$((N_FAIL + 1))
    fi
  done
done

[[ ${DRY_RUN} -eq 1 ]] && exit 0
if [[ ${N_FAIL} -gt 0 ]]; then
  echo "${N_FAIL} evaluation run(s) failed; comparison includes successful reports only" >&2
fi

METRICS="floor_mm,D_pos_mm,D_rot_deg,rho_pos,rho_rot,rho_SE3,rho_off,rho_shp,disp,glob_rot"
for task in ${TASKS}; do
  full_reports=()
  seg_reports=()
  for pipeline in "${PIPELINES[@]}"; do
    p="${EVAL_ROOT}/${task}__${pipeline}__${COHORT}__heldout_eval_full"
    [[ -f "${p}/eval_report.json" ]] && full_reports+=("${p}")
    if [[ "${task}" == "stack_bowl" ]]; then seg="0-1"; else seg="1-2"; fi
    p="${EVAL_ROOT}/${task}__${pipeline}__${COHORT}__heldout_eval_seg${seg}"
    [[ -f "${p}/eval_report.json" ]] && seg_reports+=("${p}")
  done
  echo
  echo "=============================================================================="
  echo "${task}: held-out eval, full trajectory"
  echo "=============================================================================="
  if [[ ${#full_reports[@]} -gt 0 ]]; then
    "${PY}" "${CMP}" --metric "${METRICS}" "${full_reports[@]}"
  else
    echo "no successful reports"
  fi
  if [[ ${FULL_ONLY} -eq 0 ]]; then
    echo
    echo "=============================================================================="
    echo "${task}: held-out eval, task-critical anchor segment"
    echo "=============================================================================="
    if [[ ${#seg_reports[@]} -gt 0 ]]; then
      "${PY}" "${CMP}" --metric "${METRICS}" "${seg_reports[@]}"
    else
      echo "no successful reports"
    fi
  fi
done

exit $((N_FAIL > 0 ? 1 : 0))
