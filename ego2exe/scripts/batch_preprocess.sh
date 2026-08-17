#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/batch_preprocess.sh <video_dir_or_manifest> [out_root]

Batch-runs ego2exe/scripts/preprocess.sh for ego RGB videos. The first argument
can be either:
  - a directory, searched recursively for VIDEO_GLOB
  - a text manifest, one video path per line; empty lines and # comments ignored

Environment overrides:
  PY                    Python executable passed to preprocess.sh
  GL_BACKEND            MuJoCo render backend for MP4 replay (default: egl)
  HAND_KEY              hand_r or hand_l (default: hand_r)
  WILOR_PRETRAINED_DIR  WiLoR checkpoint/cache directory
  VIDEO_GLOB            find pattern for directory mode (default: *.rgb.mp4)
  CONTINUE_ON_ERROR     1 keeps processing after a failed video (default: 1)
  RUN_IK                1 runs mink IK; default is 0 for EEF-only batch preprocessing
  RUN_REPLAY            1 renders replay MP4s; default is 0 for batch safety
  WILOR_COMPACT_OUTPUT  1 omits large WiLoR mesh/pose arrays; default is 1
  BATCH_COOLDOWN_S      sleep seconds after each video (default: 5)
  SKIP_EXISTING         1 skips videos with an existing EEF JSON and IK npz (default: 1)
  LOG_RESOURCE_USAGE    1 prints free/nvidia-smi snapshots per video (default: 1)
  START_INDEX           1-based first video index to process (default: 1)
  MAX_VIDEOS            Max videos to process after START_INDEX; 0 means all (default: 0)
  VIDEO_TIMEOUT_S       Per-video timeout in seconds; 0 disables timeout (default: 0)
  LOG_DIR               Per-video log directory (default: <out_root>/batch_logs)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGO2EXE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EGO2EXE_DIR}/.." && pwd)"

INPUT="$1"
OUT_ROOT="${2:-outputs/new_pipeline}"
VIDEO_GLOB="${VIDEO_GLOB:-*.rgb.mp4}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_IK="${RUN_IK:-0}"
RUN_REPLAY="${RUN_REPLAY:-0}"
WILOR_COMPACT_OUTPUT="${WILOR_COMPACT_OUTPUT:-1}"
BATCH_COOLDOWN_S="${BATCH_COOLDOWN_S:-5}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LOG_RESOURCE_USAGE="${LOG_RESOURCE_USAGE:-1}"
START_INDEX="${START_INDEX:-1}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
VIDEO_TIMEOUT_S="${VIDEO_TIMEOUT_S:-0}"
LOG_DIR="${LOG_DIR:-}"

if [[ -e "${INPUT}" ]]; then
  INPUT_ABS="$(cd "$(dirname "${INPUT}")" && pwd)/$(basename "${INPUT}")"
else
  echo "Input does not exist: ${INPUT}" >&2
  exit 1
fi

declare -a VIDEOS=()
if [[ -d "${INPUT_ABS}" ]]; then
  while IFS= read -r -d '' path; do
    VIDEOS+=("${path}")
  done < <(find "${INPUT_ABS}" -type f -iname "${VIDEO_GLOB}" -print0 | sort -z)
elif [[ -f "${INPUT_ABS}" ]]; then
  MANIFEST_DIR="$(cd "$(dirname "${INPUT_ABS}")" && pwd)"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" ]] && continue
    if [[ "${line}" = /* ]]; then
      VIDEOS+=("${line}")
    else
      VIDEOS+=("${MANIFEST_DIR}/${line}")
    fi
  done < "${INPUT_ABS}"
else
  echo "Input is neither a directory nor a manifest file: ${INPUT_ABS}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

if [[ ${#VIDEOS[@]} -eq 0 ]]; then
  echo "No videos found from input: ${INPUT}" >&2
  exit 1
fi

if [[ "${START_INDEX}" -lt 1 ]]; then
  echo "START_INDEX must be >= 1" >&2
  exit 2
fi
if [[ "${MAX_VIDEOS}" -lt 0 ]]; then
  echo "MAX_VIDEOS must be >= 0" >&2
  exit 2
fi

OUT_ROOT_ABS="${OUT_ROOT}"
if [[ "${OUT_ROOT_ABS}" != /* ]]; then
  OUT_ROOT_ABS="${REPO_ROOT}/${OUT_ROOT_ABS}"
fi
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${OUT_ROOT_ABS}/batch_logs"
elif [[ "${LOG_DIR}" != /* ]]; then
  LOG_DIR="${REPO_ROOT}/${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}"

echo "=== ego2exe batch preprocess ==="
echo "videos:            ${#VIDEOS[@]}"
echo "out_root:          ${OUT_ROOT}"
echo "log_dir:           ${LOG_DIR}"
echo "video_glob:        ${VIDEO_GLOB}"
echo "continue_on_error: ${CONTINUE_ON_ERROR}"
echo "run_ik:           ${RUN_IK}"
echo "run_replay:        ${RUN_REPLAY}"
echo "compact_wilor:     ${WILOR_COMPACT_OUTPUT}"
echo "cooldown_s:        ${BATCH_COOLDOWN_S}"
echo "skip_existing:     ${SKIP_EXISTING}"
echo "start_index:       ${START_INDEX}"
echo "max_videos:        ${MAX_VIDEOS}"
echo "video_timeout_s:   ${VIDEO_TIMEOUT_S}"
echo

log_resources() {
  [[ "${LOG_RESOURCE_USAGE}" == "1" ]] || return 0
  free -h | sed -n '1,2p'
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  fi
}

run_one_video() {
  local video="$1"
  local log_file="$2"
  if [[ "${VIDEO_TIMEOUT_S}" == "0" ]]; then
    RUN_IK="${RUN_IK}" RUN_REPLAY="${RUN_REPLAY}" WILOR_COMPACT_OUTPUT="${WILOR_COMPACT_OUTPUT}" \
      ego2exe/scripts/preprocess.sh "${video}" "${OUT_ROOT}"
  else
    timeout --foreground --kill-after=30s "${VIDEO_TIMEOUT_S}" \
      env RUN_IK="${RUN_IK}" RUN_REPLAY="${RUN_REPLAY}" WILOR_COMPACT_OUTPUT="${WILOR_COMPACT_OUTPUT}" \
      ego2exe/scripts/preprocess.sh "${video}" "${OUT_ROOT}"
  fi 2>&1 | tee "${log_file}"
}

FAILED=0
SKIPPED=0
DONE=0
for i in "${!VIDEOS[@]}"; do
  one_based=$((i + 1))
  if [[ ${one_based} -lt ${START_INDEX} ]]; then
    continue
  fi
  if [[ "${MAX_VIDEOS}" != "0" && ${DONE} -ge ${MAX_VIDEOS} ]]; then
    break
  fi

  video="${VIDEOS[$i]}"
  DONE=$((DONE + 1))
  echo "=== [${one_based}/${#VIDEOS[@]}] ${video} ==="
  session_name="$(basename "${video}")"
  session_name="${session_name%.*}"
  eef_json="${OUT_ROOT_ABS}/${session_name}/robot_eef_scene_camera_axis_corrected/robot_eef_trajectory.json"
  ik_npz="${OUT_ROOT_ABS}/${session_name}/nero_eef_ik/nero_eef_ik.npz"
  if [[ "${SKIP_EXISTING}" == "1" && -f "${eef_json}" && ( "${RUN_IK}" != "1" || -f "${ik_npz}" ) ]]; then
    SKIPPED=$((SKIPPED + 1))
    echo "=== skip existing: ${video} ==="
    echo
    continue
  fi

  echo "--- resources before ---"
  log_resources
  log_file="${LOG_DIR}/$(printf '%04d' "${one_based}")_${session_name}.log"
  echo "--- log: ${log_file} ---"
  if run_one_video "${video}" "${log_file}"; then
    echo "=== done: ${video} ==="
  else
    FAILED=$((FAILED + 1))
    echo "=== failed: ${video} ===" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit 1
    fi
  fi
  echo "--- resources after ---"
  log_resources
  if [[ "${BATCH_COOLDOWN_S}" != "0" ]]; then
    sleep "${BATCH_COOLDOWN_S}"
  fi
  echo
done

echo "=== batch done ==="
echo "selected:  ${DONE}"
echo "total:     ${#VIDEOS[@]}"
echo "skipped:   ${SKIPPED}"
echo "failed:    ${FAILED}"
if [[ ${FAILED} -gt 0 ]]; then
  exit 1
fi
