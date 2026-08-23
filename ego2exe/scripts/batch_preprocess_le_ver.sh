#!/usr/bin/env bash
# Convert every ego video of one task through the ego2exe pipeline.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/batch_preprocess_le_ver.sh --task TASK --pipeline NAME [options]

Converts every ego video of one task. Run this ONCE per (task, pipeline); the
evaluation afterwards is pure selection and needs no re-processing.

Layout - outputs mirrors the source tree, with the pipeline name on top:

  DATA_new/<task>/<date>_ego_<operator>/<session>.rgb.mp4        source
  outputs/<pipeline>/<task>/<date>_ego_<operator>/<session>/     this script
  outputs/_wilor/<session>/preprocess/                           WiLoR cache, shared

Everything is derived from the paths - there is no registry to maintain:
  real reference set   DATA_new/<task>/*nero*/     (skipped here, used by eval)
  ego cohorts          DATA_new/<task>/*_ego_*/    (the suffix is the cohort name)

WiLoR lives in outputs/_wilor/ keyed by session only, so switching --pipeline
re-runs just the cheap export/IK stages, never the reconstruction.

Options:
  --task TASK          task directory under --data-root (required)
  --pipeline NAME      pipeline id, e.g. h2g_pinch_index (required)
  --cohort NAME        only this cohort (default: every *_ego_* dir of the task)
  --data-root DIR      source root (default: ../DATA_new relative to the repo)
  --out-root DIR       output root (default: outputs)
  --hand2gripper-mode M  humanego (default) or pinch_plane
  --forward-seed S     index_tip (default) or finger_mcp_centroid; pinch_plane only
  --hand-key K         hand_r (default) or hand_l
  --hands-file NAME    e.g. wilor_hands.json
  --no-axis-correction forwarded to the export stage
  --replays            also render replay MP4s (slow; evaluation does not need them)
  --skip-ik            stop after the EEF export
  --force              redo export/IK for this pipeline even if present
  --force-raw          also redo the shared WiLoR stage (rare, expensive)
  --limit N            only the first N videos per cohort
  --dry-run            list what would run, then exit
  -h, --help

Environment:
  PY                    python executable (default: python)
  GL_BACKEND            MuJoCo render backend for replays (default: egl)
  WILOR_PRETRAINED_DIR  WiLoR checkpoint/cache directory

Example:
  ego2exe/scripts/batch_preprocess_le_ver.sh --task stack_object_horizontal \
      --pipeline h2g_pinch_index --hand2gripper-mode pinch_plane --forward-seed index_tip
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK=""
PIPELINE=""
COHORT=""
DATA_ROOT="${REPO_ROOT}/../DATA_new"
OUT_ROOT="outputs"
H2G_MODE=""
FORWARD_SEED=""
HAND_KEY="hand_r"
HANDS_FILE=""
NO_AXIS_CORRECTION=0
DO_REPLAYS=0
SKIP_IK=0
FORCE=0
FORCE_RAW=0
LIMIT=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)                 TASK="$2"; shift 2 ;;
    --pipeline)             PIPELINE="$2"; shift 2 ;;
    --cohort)               COHORT="$2"; shift 2 ;;
    --data-root)            DATA_ROOT="$2"; shift 2 ;;
    --out-root)             OUT_ROOT="$2"; shift 2 ;;
    --hand2gripper-mode)    H2G_MODE="$2"; shift 2 ;;
    --forward-seed)         FORWARD_SEED="$2"; shift 2 ;;
    --hand-key)             HAND_KEY="$2"; shift 2 ;;
    --hands-file)           HANDS_FILE="$2"; shift 2 ;;
    --no-axis-correction)   NO_AXIS_CORRECTION=1; shift ;;
    --replays)              DO_REPLAYS=1; shift ;;
    --skip-ik)              SKIP_IK=1; shift ;;
    --force)                FORCE=1; shift ;;
    --force-raw)            FORCE_RAW=1; shift ;;
    --limit)                LIMIT="$2"; shift 2 ;;
    --dry-run)              DRY_RUN=1; shift ;;
    -h|--help)              usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "${TASK}" ]]     || { echo "error: --task is required" >&2; exit 2; }
[[ -n "${PIPELINE}" ]] || { echo "error: --pipeline is required" >&2; exit 2; }
if [[ "${PIPELINE}" =~ [^a-zA-Z0-9._-] ]]; then
  echo "error: --pipeline '${PIPELINE}' should only contain [a-zA-Z0-9._-]" >&2; exit 2
fi

PY="${PY:-python}"
GL_BACKEND="${GL_BACKEND:-egl}"

abspath() {
  case "$1" in
    /*) printf '%s' "$1" ;;
    *)  if [[ -e "${PWD}/$1" ]]; then printf '%s/%s' "${PWD}" "$1"
        else printf '%s/%s' "${REPO_ROOT}" "$1"; fi ;;
  esac
}

DATA_ROOT_ABS="$(cd "$(abspath "${DATA_ROOT}")" 2>/dev/null && pwd)" || {
  echo "data root not found: ${DATA_ROOT}" >&2; exit 1; }
OUT_ROOT_ABS="$(abspath "${OUT_ROOT}")"
TASK_DIR="${DATA_ROOT_ABS}/${TASK}"
[[ -d "${TASK_DIR}" ]] || {
  echo "task not found: ${TASK_DIR}" >&2
  echo "available: $(ls "${DATA_ROOT_ABS}" 2>/dev/null | tr '\n' ' ')" >&2; exit 1; }

WILOR_ROOT="${OUT_ROOT_ABS}/_wilor"
PIPE_TASK_DIR="${OUT_ROOT_ABS}/${PIPELINE}/${TASK}"

# --- cohorts: every *_ego_* dir under the task, or just the requested one -----
shopt -s nullglob
if [[ -n "${COHORT}" ]]; then
  COHORT_DIRS=("${TASK_DIR}"/*_ego_"${COHORT}")
else
  COHORT_DIRS=("${TASK_DIR}"/*_ego_*)
fi
shopt -u nullglob
if [[ ${#COHORT_DIRS[@]} -eq 0 ]]; then
  echo "no ego cohort dirs matching '*_ego_${COHORT}*' under ${TASK_DIR}" >&2
  echo "found: $(ls -d "${TASK_DIR}"/*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' ')" >&2
  exit 1
fi

echo "=============================================================================="
echo "ego2exe batch preprocess"
echo "=============================================================================="
echo "task      : ${TASK}   (${TASK_DIR})"
echo "pipeline  : ${PIPELINE}   -> ${PIPE_TASK_DIR}"
echo "wilor     : ${WILOR_ROOT}   (shared across pipelines)"
echo "hand2grip : ${H2G_MODE:-humanego}$([[ "${H2G_MODE}" == pinch_plane ]] && echo " (seed: ${FORWARD_SEED:-index_tip})")"
echo "stages    : wilor, export$([[ ${SKIP_IK} -eq 1 ]] || echo ', ik')$([[ ${DO_REPLAYS} -eq 1 ]] && echo ', replays')"
printf 'cohorts   : '
for c in "${COHORT_DIRS[@]}"; do
  b="$(basename "${c}")"; printf '%s(%d) ' "${b##*_ego_}" "$(ls "${c}"/*.mp4 2>/dev/null | wc -l)"
done
echo; echo

if [[ ${DRY_RUN} -eq 1 ]]; then
  for c in "${COHORT_DIRS[@]}"; do
    cb="$(basename "${c}")"
    shopt -s nullglob; vids=("${c}"/*.mp4); shopt -u nullglob
    for v in "${vids[@]}"; do
      stem="$(basename "${v}")"; stem="${stem%.mp4}"
      echo "  ${PIPE_TASK_DIR}/${cb}/${stem}/  <-  ${cb}/$(basename "${v}")"
    done
  done
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p "${PIPE_TASK_DIR}" "${WILOR_ROOT}"
LOG_DIR="${PIPE_TASK_DIR}/_logs"; mkdir -p "${LOG_DIR}"
MANIFEST_TSV="${PIPE_TASK_DIR}/manifest.tsv"
printf 'cohort\tsession\twilor\texport\tik\tframes\n' > "${MANIFEST_TSV}"

# --- one small meta file per pipeline: what config this pipeline id means -----
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="clean"
git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | grep -q . && GIT_DIRTY="dirty"
AXIS_CORRECTION_FLAG=1
[[ ${NO_AXIS_CORRECTION} -eq 1 ]] && AXIS_CORRECTION_FLAG=0
"${PY}" - "${OUT_ROOT_ABS}/${PIPELINE}/pipeline_meta.json" "${PIPELINE}" "${TASK}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${GIT_COMMIT}" "${GIT_DIRTY}" \
  "${HAND_KEY}" "${HANDS_FILE}" "${AXIS_CORRECTION_FLAG}" "${H2G_MODE}" "${FORWARD_SEED}" <<'PYEOF'
import json, sys, os
(path, pipeline, task, ts, commit, dirty,
 hand_key, hands_file, axis_correction, h2g_mode, forward_seed) = sys.argv[1:12]
h2g_mode = h2g_mode or "humanego"
cfg = {
    "hand2gripper_mode": h2g_mode,
    "forward_seed": (forward_seed or "index_tip") if h2g_mode == "pinch_plane" else None,
    "hand_key": hand_key,
    "hands_file": hands_file or None,
    "axis_correction": axis_correction == "1",
}
try:
    meta = json.load(open(path))
except FileNotFoundError:
    meta = {"pipeline": pipeline, "config": cfg, "tasks": {}}
if meta.get("config") != cfg:
    print(f"  [warn] {os.path.basename(path)} already records a different config for "
          f"pipeline '{pipeline}':\n         stored {meta.get('config')}\n         now    {cfg}\n"
          f"         Use a new --pipeline name, or the results under it will be a mixture.")
meta["config"] = cfg
meta.setdefault("tasks", {})[task] = {"last_run": ts, "git_commit": commit, "git_dirty": dirty == "dirty"}
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(meta, open(path, "w"), indent=2)
PYEOF

WILOR_ARGS=()
[[ -n "${WILOR_PRETRAINED_DIR:-}" ]] && WILOR_ARGS+=(--wilor-pretrained-dir "${WILOR_PRETRAINED_DIR}")
EXPORT_ARGS=(--hand-key "${HAND_KEY}")
[[ -n "${HANDS_FILE}" ]] && EXPORT_ARGS+=(--hands-file "${HANDS_FILE}")
[[ ${NO_AXIS_CORRECTION} -eq 1 ]] && EXPORT_ARGS+=(--no-axis-correction)
[[ -n "${H2G_MODE}" ]] && EXPORT_ARGS+=(--hand2gripper-mode "${H2G_MODE}")
[[ -n "${FORWARD_SEED}" ]] && EXPORT_ARGS+=(--forward-seed "${FORWARD_SEED}")

N_OK=0; N_FAIL=0
FAILED=()

run_stage() {
  local log="$1" label="$2"; shift 2
  if "$@" >>"${log}" 2>&1; then printf '    %-10s ok\n' "${label}"; return 0; fi
  printf '    %-10s FAILED (see %s)\n' "${label}" "${log}"; return 1
}

for COHORT_DIR in "${COHORT_DIRS[@]}"; do
  COHORT_BASE="$(basename "${COHORT_DIR}")"
  COHORT_NAME="${COHORT_BASE##*_ego_}"
  shopt -s nullglob; VIDEOS=("${COHORT_DIR}"/*.mp4); shopt -u nullglob
  [[ "${LIMIT}" -gt 0 && "${LIMIT}" -lt ${#VIDEOS[@]} ]] && VIDEOS=("${VIDEOS[@]:0:${LIMIT}}")
  echo "--- cohort ${COHORT_NAME}: ${#VIDEOS[@]} video(s) ---"

  idx=0
  for VIDEO_ABS in "${VIDEOS[@]}"; do
    idx=$((idx + 1))
    VIDEO_NAME="$(basename "${VIDEO_ABS}")"
    # Must match Path(video).stem in preprocess_wilor_hands.py, which strips only
    # the final suffix - so "foo.rgb.mp4" -> "foo.rgb", keeping the .rgb part.
    SESSION_NAME="${VIDEO_NAME%.mp4}"

    RAW_SESSION_DIR="${WILOR_ROOT}/${SESSION_NAME}"
    ALL_DATA="${RAW_SESSION_DIR}/preprocess/all_data"
    OUT_SESSION_DIR="${PIPE_TASK_DIR}/${COHORT_BASE}/${SESSION_NAME}"
    EEF_DIR="${OUT_SESSION_DIR}/robot_eef_scene_camera_axis_corrected"
    EEF_JSON="${EEF_DIR}/robot_eef_trajectory.json"
    EEF_CSV="${EEF_DIR}/robot_eef_trajectory.csv"
    IK_NPZ="${OUT_SESSION_DIR}/nero_eef_ik/nero_eef_ik.npz"
    LOG="${LOG_DIR}/${SESSION_NAME}.log"; : > "${LOG}"

    echo "[${idx}/${#VIDEOS[@]}] ${SESSION_NAME}"
    st_wilor="-"; st_export="-"; st_ik="-"

    # 1 WiLoR - shared across pipelines, keyed by session only
    if [[ ${FORCE_RAW} -eq 0 && -d "${ALL_DATA}" ]] && compgen -G "${ALL_DATA}/*/wilor_hands.json" >/dev/null; then
      printf '    %-10s skip (cache)\n' "wilor"; st_wilor="skip"
    elif run_stage "${LOG}" "wilor" "${PY}" ego2exe/preprocess_wilor_hands.py \
          --video "${VIDEO_ABS}" --out "${WILOR_ROOT}" "${WILOR_ARGS[@]}"; then
      st_wilor="ok"
    else
      st_wilor="fail"; N_FAIL=$((N_FAIL + 1)); FAILED+=("${SESSION_NAME} (wilor)")
      printf '%s\t%s\t%s\t-\t-\t0\n' "${COHORT_NAME}" "${SESSION_NAME}" "${st_wilor}" >> "${MANIFEST_TSV}"
      continue
    fi

    # 2 EEF export - the only stage the evaluation needs
    if [[ ${FORCE} -eq 0 && -f "${EEF_CSV}" ]]; then
      printf '    %-10s skip (present)\n' "export"; st_export="skip"
    elif run_stage "${LOG}" "export" "${PY}" ego2exe/preprocess_export_eef.py \
          --session "${RAW_SESSION_DIR}" --out "${EEF_DIR}" "${EXPORT_ARGS[@]}"; then
      st_export="ok"
    else
      st_export="fail"; N_FAIL=$((N_FAIL + 1)); FAILED+=("${SESSION_NAME} (export)")
      printf '%s\t%s\t%s\t%s\t-\t0\n' "${COHORT_NAME}" "${SESSION_NAME}" "${st_wilor}" "${st_export}" >> "${MANIFEST_TSV}"
      continue
    fi

    # 3 IK
    if [[ ${SKIP_IK} -eq 1 ]]; then
      st_ik="off"
    elif [[ ${FORCE} -eq 0 && -f "${IK_NPZ}" ]]; then
      printf '    %-10s skip (present)\n' "ik"; st_ik="skip"
    elif run_stage "${LOG}" "ik" "${PY}" ego2exe/retarget_with_mink.py \
          --eef "${EEF_JSON}" --out "${IK_NPZ}"; then
      st_ik="ok"
    else
      st_ik="fail"; FAILED+=("${SESSION_NAME} (ik, non-fatal)")
    fi

    # 4 replays (optional)
    if [[ ${DO_REPLAYS} -eq 1 ]]; then
      REPLAY_DIR="${OUT_SESSION_DIR}/replays"; mkdir -p "${REPLAY_DIR}"
      run_stage "${LOG}" "replay-eef" "${PY}" ego2exe/replay_eef_mujoco.py \
        --eef "${EEF_JSON}" --out "${REPLAY_DIR}/${SESSION_NAME}_eef.mp4" --gl-backend "${GL_BACKEND}" || true
      [[ -f "${IK_NPZ}" ]] && run_stage "${LOG}" "replay-ik" "${PY}" ego2exe/replay_ik_mujoco.py \
        --ik "${IK_NPZ}" --out "${REPLAY_DIR}/${SESSION_NAME}_ik.mp4" --gl-backend "${GL_BACKEND}" || true
    fi

    FRAMES="$(${PY} -c "import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1]))))" "${EEF_CSV}" 2>/dev/null || echo 0)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${COHORT_NAME}" "${SESSION_NAME}" "${st_wilor}" "${st_export}" "${st_ik}" "${FRAMES}" >> "${MANIFEST_TSV}"
    N_OK=$((N_OK + 1))
    printf '    %-10s %s frames\n' "frames" "${FRAMES}"
  done
done

echo
echo "=============================================================================="
echo "done: ${N_OK} session(s) ready, ${N_FAIL} failed   [${PIPELINE} / ${TASK}, commit=${GIT_COMMIT}$([[ ${GIT_DIRTY} == dirty ]] && echo '*')]"
echo "  sessions : ${PIPE_TASK_DIR}/<cohort>/<session>/"
echo "  manifest : ${MANIFEST_TSV}"
echo "  logs     : ${LOG_DIR}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "  failures :"; for f in "${FAILED[@]}"; do echo "      ${f}"; done
fi
echo "=============================================================================="
echo
echo "evaluate (no re-processing needed, any number of times):"
echo "  ${PY} ego2exe/eval/eval_ego2exe.py --task ${TASK} --pipeline ${PIPELINE} --cohort all"
echo "  ${PY} ego2exe/eval/eval_ego2exe.py --task ${TASK} --pipeline ${PIPELINE} --cohort hyj"
echo "  ${PY} ego2exe/eval/eval_ego2exe.py --task ${TASK} --pipeline ${PIPELINE} --cohort all --segment-start 0 --segment-end 2"
