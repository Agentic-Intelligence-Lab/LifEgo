#!/usr/bin/env bash
# Apply real-anchor EEF correction to outputs from an existing preprocess pipeline.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/batch_correct_eef_anchors.sh --task TASK [options]

Reads the standard EEF export from an existing pipeline and creates independent
corrected pipelines. It never modifies WiLoR/preprocess outputs.

Default ablation matrix:
  spaces   : ray_depth free_xyz
  methods  : linear min_bending smooth_spline
  rotations: none
  pipelines: <source>__anchor_{ray|xyz}_<method>_s<sample-seed>[__rot_global_so3]

Options:
  --task TASK             required: stack_bowl, stack_object_horizontal, or stack_object_lean
  --source-pipeline NAME  source pipeline (default: h2g_pinch_index)
  --cohort NAME           only one *_ego_NAME cohort (default: all)
  --spaces "a b"          subset of: ray_depth free_xyz
  --methods "a b"         subset of: linear min_bending smooth_spline
  --rotations "a b"       subset of: none global_so3 (default: none)
  --include-rot-only       also create <source>__rot_global_so3 without changing positions
  --rotation-manifest FILE default: <out-root>/rotation_calibration/<task>__<source>.json
  --anchors "K ..."       optional zero-based event indices; otherwise task defaults
  --data-root DIR         default: ../DATA_new
  --out-root DIR          default: outputs
  --split-manifest FILE   default: <out-root>/anchor_calibration_splits/<task>.json
  --target-manifest FILE  default: <out-root>/anchor_target_samples/<task>_s<seed>.json
  --split-seed N          default: 0
  --sample-seed N         default: 0
  --calibration-size N    default: 10
  --eval-size N           default: 10
  --cov-shrinkage X       default: 0.1
  --max-mahalanobis X     default: 0 (untruncated Gaussian)
  --anchor-weight X       smooth_spline only (default: 1)
  --endpoint-weight X     smooth_spline only (default: 1)
  --bend-weight X         smooth_spline only (default: 1000)
  --magnitude-weight X    smooth_spline only (default: 1e-4)
  --limit N               first N sessions per cohort
  --force                 overwrite existing corrected EEF files
  --dry-run               print planned source/output pairs only
  -h, --help

Environment:
  PY  Python executable (default: python)
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK=""
SOURCE_PIPELINE="h2g_pinch_index"
COHORT=""
SPACES="ray_depth free_xyz"
METHODS="linear min_bending smooth_spline"
ROTATIONS="none"
ANCHORS=""
DATA_ROOT="${REPO_ROOT}/../DATA_new"
OUT_ROOT="outputs"
SPLIT_MANIFEST=""
TARGET_MANIFEST=""
ROTATION_MANIFEST=""
SPLIT_SEED=0
SAMPLE_SEED=0
CALIBRATION_SIZE=10
EVAL_SIZE=10
COV_SHRINKAGE=0.1
MAX_MAHALANOBIS=0
ANCHOR_WEIGHT=1
ENDPOINT_WEIGHT=1
BEND_WEIGHT=1000
MAGNITUDE_WEIGHT=1e-4
LIMIT=0
FORCE=0
DRY_RUN=0
INCLUDE_ROT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --source-pipeline) SOURCE_PIPELINE="$2"; shift 2 ;;
    --cohort) COHORT="$2"; shift 2 ;;
    --spaces) SPACES="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --rotations) ROTATIONS="$2"; shift 2 ;;
    --include-rot-only) INCLUDE_ROT_ONLY=1; shift ;;
    --anchors) ANCHORS="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --out-root) OUT_ROOT="$2"; shift 2 ;;
    --split-manifest) SPLIT_MANIFEST="$2"; shift 2 ;;
    --target-manifest) TARGET_MANIFEST="$2"; shift 2 ;;
    --rotation-manifest) ROTATION_MANIFEST="$2"; shift 2 ;;
    --split-seed) SPLIT_SEED="$2"; shift 2 ;;
    --sample-seed) SAMPLE_SEED="$2"; shift 2 ;;
    --calibration-size) CALIBRATION_SIZE="$2"; shift 2 ;;
    --eval-size) EVAL_SIZE="$2"; shift 2 ;;
    --cov-shrinkage) COV_SHRINKAGE="$2"; shift 2 ;;
    --max-mahalanobis) MAX_MAHALANOBIS="$2"; shift 2 ;;
    --anchor-weight) ANCHOR_WEIGHT="$2"; shift 2 ;;
    --endpoint-weight) ENDPOINT_WEIGHT="$2"; shift 2 ;;
    --bend-weight) BEND_WEIGHT="$2"; shift 2 ;;
    --magnitude-weight) MAGNITUDE_WEIGHT="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${TASK}" ]] || { echo "error: --task is required" >&2; exit 2; }
case "${TASK}" in
  stack_bowl|stack_object_horizontal|stack_object_lean) ;;
  *) echo "unsupported task: ${TASK}" >&2; exit 2 ;;
esac
for space in ${SPACES}; do
  [[ "${space}" == "ray_depth" || "${space}" == "free_xyz" ]] || {
    echo "unsupported correction space: ${space}" >&2; exit 2; }
done
for method in ${METHODS}; do
  [[ "${method}" == "linear" || "${method}" == "min_bending" || "${method}" == "smooth_spline" ]] || {
    echo "unsupported correction method: ${method}" >&2; exit 2; }
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
SOURCE_TASK_DIR="${OUT_ROOT_ABS}/${SOURCE_PIPELINE}/${TASK}"
[[ -d "${DATA_ROOT_ABS}/${TASK}" ]] || { echo "task data not found: ${DATA_ROOT_ABS}/${TASK}" >&2; exit 1; }
[[ -d "${SOURCE_TASK_DIR}" ]] || {
  echo "source pipeline output not found: ${SOURCE_TASK_DIR}" >&2
  echo "run batch_preprocess_le_ver.sh first" >&2
  exit 1
}

[[ -n "${SPLIT_MANIFEST}" ]] || SPLIT_MANIFEST="${OUT_ROOT_ABS}/anchor_calibration_splits/${TASK}.json"
[[ -n "${TARGET_MANIFEST}" ]] || TARGET_MANIFEST="${OUT_ROOT_ABS}/anchor_target_samples/${TASK}_s${SAMPLE_SEED}.json"
[[ -n "${ROTATION_MANIFEST}" ]] || ROTATION_MANIFEST="${OUT_ROOT_ABS}/rotation_calibration/${TASK}__${SOURCE_PIPELINE}.json"
SPLIT_MANIFEST="$(abspath "${SPLIT_MANIFEST}")"
TARGET_MANIFEST="$(abspath "${TARGET_MANIFEST}")"
ROTATION_MANIFEST="$(abspath "${ROTATION_MANIFEST}")"

shopt -s nullglob
if [[ -n "${COHORT}" ]]; then
  COHORT_DIRS=("${SOURCE_TASK_DIR}"/*_ego_"${COHORT}")
else
  COHORT_DIRS=("${SOURCE_TASK_DIR}"/*_ego_*)
fi
shopt -u nullglob
[[ ${#COHORT_DIRS[@]} -gt 0 ]] || { echo "no matching source cohorts under ${SOURCE_TASK_DIR}" >&2; exit 1; }

ANCHOR_ARGS=()
if [[ -n "${ANCHORS}" ]]; then
  read -r -a ANCHOR_VALUES <<< "${ANCHORS}"
  ANCHOR_ARGS=(--anchors "${ANCHOR_VALUES[@]}")
fi
OVERWRITE_ARGS=()
[[ ${FORCE} -eq 1 ]] && OVERWRITE_ARGS=(--overwrite)

echo "=============================================================================="
echo "real-anchor EEF correction batch"
echo "=============================================================================="
echo "task             : ${TASK}"
echo "source pipeline  : ${SOURCE_PIPELINE}"
echo "spaces           : ${SPACES}"
echo "methods          : ${METHODS}"
echo "rotations        : ${ROTATIONS}"
echo "rotation only    : $([[ ${INCLUDE_ROT_ONLY} -eq 1 ]] && echo yes || echo no)"
echo "anchors          : ${ANCHORS:-task default}"
echo "real split       : ${SPLIT_MANIFEST}"
echo "frozen targets   : ${TARGET_MANIFEST}"
echo

if [[ " ${ROTATIONS} " == *" global_so3 "* || ${INCLUDE_ROT_ONLY} -eq 1 ]]; then
  shopt -s nullglob
  REAL_DIRS=("${DATA_ROOT_ABS}/${TASK}"/*nero*)
  shopt -u nullglob
  [[ ${#REAL_DIRS[@]} -eq 1 ]] || {
    echo "expected one *nero* real directory, found ${#REAL_DIRS[@]}" >&2; exit 1; }
  fit_cmd=("${PY}" ego2exe/fit_global_rotation_correction.py
    --task "${TASK}"
    --source-task-dir "${SOURCE_TASK_DIR}"
    --real-dir "${REAL_DIRS[0]}"
    --split-manifest "${SPLIT_MANIFEST}"
    --out "${ROTATION_MANIFEST}")
  [[ ${FORCE} -eq 1 ]] && fit_cmd+=(--overwrite)
  if [[ ${DRY_RUN} -eq 1 ]]; then
    printf '  '; printf '%q ' "${fit_cmd[@]}"; printf '\n'
  elif [[ -f "${ROTATION_MANIFEST}" && ${FORCE} -eq 0 ]]; then
    echo "rotation fit     : skip (existing ${ROTATION_MANIFEST})"
  elif ! (cd "${REPO_ROOT}" && "${fit_cmd[@]}"); then
    echo "global rotation calibration failed" >&2; exit 1
  fi
fi

N_OK=0
N_SKIP=0
N_FAIL=0
FAILED=()
if [[ ${INCLUDE_ROT_ONLY} -eq 1 ]]; then
  PIPELINE="${SOURCE_PIPELINE}__rot_global_so3"
  echo "--- ${PIPELINE} (rotation only; source positions preserved) ---"
  for cohort_dir in "${COHORT_DIRS[@]}"; do
    cohort_base="$(basename "${cohort_dir}")"
    shopt -s nullglob
    session_dirs=("${cohort_dir}"/*)
    shopt -u nullglob
    [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt ${#session_dirs[@]} ]] && session_dirs=("${session_dirs[@]:0:${LIMIT}}")
    for source_session in "${session_dirs[@]}"; do
      [[ -d "${source_session}" ]] || continue
      session="$(basename "${source_session}")"
      source_eef="${source_session}/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json"
      out_dir="${OUT_ROOT_ABS}/${PIPELINE}/${TASK}/${cohort_base}/${session}/robot_eef_scene_camera_axis_corrected"
      if [[ ! -f "${source_eef}" ]]; then
        echo "  [missing] ${cohort_base}/${session}: ${source_eef}" >&2
        N_FAIL=$((N_FAIL + 1)); FAILED+=("${PIPELINE}/${cohort_base}/${session} (source missing)")
        continue
      fi
      if [[ ${FORCE} -eq 0 && -f "${out_dir}/robot_eef_trajectory.json" ]]; then
        echo "  [skip] ${cohort_base}/${session}"
        N_SKIP=$((N_SKIP + 1))
        continue
      fi
      cmd=("${PY}" ego2exe/correct_eef_with_real_anchors.py
        --task "${TASK}" --eef "${source_eef}" --out "${out_dir}"
        --data-root "${DATA_ROOT_ABS}"
        --position-correction none
        --rotation-correction global_so3 --rotation-manifest "${ROTATION_MANIFEST}"
        --split-manifest "${SPLIT_MANIFEST}"
        --split-seed "${SPLIT_SEED}" --calibration-size "${CALIBRATION_SIZE}" --eval-size "${EVAL_SIZE}"
        "${OVERWRITE_ARGS[@]}")
      if [[ ${DRY_RUN} -eq 1 ]]; then
        printf '  '; printf '%q ' "${cmd[@]}"; printf '\n'
      elif (cd "${REPO_ROOT}" && "${cmd[@]}"); then
        N_OK=$((N_OK + 1))
      else
        N_FAIL=$((N_FAIL + 1)); FAILED+=("${PIPELINE}/${cohort_base}/${session}")
      fi
    done
  done
fi
for space in ${SPACES}; do
  if [[ "${space}" == "ray_depth" ]]; then space_label="ray"; else space_label="xyz"; fi
  for method in ${METHODS}; do
    for rotation in ${ROTATIONS}; do
      PIPELINE="${SOURCE_PIPELINE}__anchor_${space_label}_${method}_s${SAMPLE_SEED}"
      ROTATION_ARGS=(--rotation-correction none)
      if [[ "${rotation}" == "global_so3" ]]; then
        PIPELINE="${PIPELINE}__rot_global_so3"
        ROTATION_ARGS=(--rotation-correction global_so3 --rotation-manifest "${ROTATION_MANIFEST}")
      fi
      echo "--- ${PIPELINE} ---"
    for cohort_dir in "${COHORT_DIRS[@]}"; do
      cohort_base="$(basename "${cohort_dir}")"
      shopt -s nullglob
      session_dirs=("${cohort_dir}"/*)
      shopt -u nullglob
      [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt ${#session_dirs[@]} ]] && session_dirs=("${session_dirs[@]:0:${LIMIT}}")
      for source_session in "${session_dirs[@]}"; do
        [[ -d "${source_session}" ]] || continue
        session="$(basename "${source_session}")"
        source_eef="${source_session}/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json"
        out_dir="${OUT_ROOT_ABS}/${PIPELINE}/${TASK}/${cohort_base}/${session}/robot_eef_scene_camera_axis_corrected"
        if [[ ! -f "${source_eef}" ]]; then
          echo "  [missing] ${cohort_base}/${session}: ${source_eef}" >&2
          N_FAIL=$((N_FAIL + 1)); FAILED+=("${PIPELINE}/${cohort_base}/${session} (source missing)")
          continue
        fi
        if [[ ${FORCE} -eq 0 && -f "${out_dir}/robot_eef_trajectory.json" ]]; then
          echo "  [skip] ${cohort_base}/${session}"
          N_SKIP=$((N_SKIP + 1))
          continue
        fi
        cmd=("${PY}" ego2exe/correct_eef_with_real_anchors.py
          --task "${TASK}"
          --eef "${source_eef}"
          --out "${out_dir}"
          --data-root "${DATA_ROOT_ABS}"
          --space "${space}"
          --method "${method}"
          "${ROTATION_ARGS[@]}"
          --split-manifest "${SPLIT_MANIFEST}"
          --target-manifest "${TARGET_MANIFEST}"
          --target-key "${cohort_base}/${session}"
          --split-seed "${SPLIT_SEED}"
          --sample-seed "${SAMPLE_SEED}"
          --calibration-size "${CALIBRATION_SIZE}"
          --eval-size "${EVAL_SIZE}"
          --covariance-shrinkage "${COV_SHRINKAGE}"
          --max-mahalanobis "${MAX_MAHALANOBIS}"
          --anchor-weight "${ANCHOR_WEIGHT}"
          --endpoint-weight "${ENDPOINT_WEIGHT}"
          --bend-weight "${BEND_WEIGHT}"
          --magnitude-weight "${MAGNITUDE_WEIGHT}"
          "${ANCHOR_ARGS[@]}" "${OVERWRITE_ARGS[@]}")
        if [[ ${DRY_RUN} -eq 1 ]]; then
          printf '  '; printf '%q ' "${cmd[@]}"; printf '\n'
        elif (cd "${REPO_ROOT}" && "${cmd[@]}"); then
          N_OK=$((N_OK + 1))
        else
          N_FAIL=$((N_FAIL + 1)); FAILED+=("${PIPELINE}/${cohort_base}/${session}")
        fi
      done
    done
    done
  done
done

echo
echo "done: ${N_OK} corrected, ${N_SKIP} skipped, ${N_FAIL} failed"
if [[ ${N_FAIL} -gt 0 ]]; then
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
echo "held-out evaluation: ego2exe/scripts/run_anchor_correction_eval.sh --task ${TASK} --source-pipeline ${SOURCE_PIPELINE}"
