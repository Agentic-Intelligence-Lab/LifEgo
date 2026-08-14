#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/preprocess.sh <video> [out_root]

Runs the ego2exe preprocessing pipeline:
  1. WiLoR hand reconstruction
  2. EEF export in robot base
  3. mink IK retargeting
  4. headless EEF and IK replay MP4 rendering

Environment overrides:
  PY                    Python executable (default: python)
  GL_BACKEND            MuJoCo render backend for MP4 replay (default: egl)
  WILOR_PRETRAINED_DIR  WiLoR checkpoint/cache directory
  HAND_KEY              hand_r or hand_l (default: hand_r)
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

VIDEO="$1"
OUT_ROOT="${2:-outputs/new_pipeline}"
PY="${PY:-python}"
GL_BACKEND="${GL_BACKEND:-egl}"
HAND_KEY="${HAND_KEY:-hand_r}"

if [[ ! -f "${VIDEO}" ]]; then
  echo "Input video not found: ${VIDEO}" >&2
  exit 1
fi

VIDEO_ABS="$(cd "$(dirname "${VIDEO}")" && pwd)/$(basename "${VIDEO}")"
VIDEO_NAME="$(basename "${VIDEO}")"
SESSION_NAME="${VIDEO_NAME%.*}"

OUT_ROOT_ABS="${OUT_ROOT}"
if [[ "${OUT_ROOT_ABS}" != /* ]]; then
  OUT_ROOT_ABS="${REPO_ROOT}/${OUT_ROOT_ABS}"
fi

SESSION_DIR="${OUT_ROOT_ABS}/${SESSION_NAME}"
EEF_DIR="${SESSION_DIR}/robot_eef_scene_camera_axis_corrected"
EEF_JSON="${EEF_DIR}/robot_eef_trajectory.json"
IK_DIR="${SESSION_DIR}/nero_eef_ik"
IK_NPZ="${IK_DIR}/nero_eef_ik.npz"
REPLAY_DIR="${SESSION_DIR}/replays"
EEF_MP4="${REPLAY_DIR}/${SESSION_NAME}_eef.mp4"
IK_MP4="${REPLAY_DIR}/${SESSION_NAME}_ik.mp4"

cd "${REPO_ROOT}"
mkdir -p "${REPLAY_DIR}"

echo "=== ego2exe preprocess ==="
echo "video:      ${VIDEO_ABS}"
echo "session:    ${SESSION_DIR}"
echo "python:     ${PY}"
echo "gl_backend: ${GL_BACKEND}"
echo

WILOR_ARGS=()
if [[ -n "${WILOR_PRETRAINED_DIR:-}" ]]; then
  WILOR_ARGS+=(--wilor-pretrained-dir "${WILOR_PRETRAINED_DIR}")
fi

echo "=== 1/4 WiLoR hand reconstruction ==="
"${PY}" ego2exe/preprocess_wilor_hands.py \
  --video "${VIDEO_ABS}" \
  --out "${OUT_ROOT_ABS}" \
  "${WILOR_ARGS[@]}"

echo
echo "=== 2/4 Export EEF trajectory ==="
"${PY}" ego2exe/preprocess_export_eef.py \
  --session "${SESSION_DIR}" \
  --out "${EEF_DIR}" \
  --hand-key "${HAND_KEY}"

echo
echo "=== 3/4 Retarget EEF with mink ==="
"${PY}" ego2exe/retarget_with_mink.py \
  --eef "${EEF_JSON}" \
  --out "${IK_NPZ}"

echo
echo "=== 4/4 Render replay MP4s ==="
"${PY}" ego2exe/replay_eef_mujoco.py \
  --eef "${EEF_JSON}" \
  --out "${EEF_MP4}" \
  --gl-backend "${GL_BACKEND}"

"${PY}" ego2exe/replay_ik_mujoco.py \
  --ik "${IK_NPZ}" \
  --out "${IK_MP4}" \
  --gl-backend "${GL_BACKEND}"

echo
echo "=== done ==="
echo "EEF:        ${EEF_JSON}"
echo "IK:         ${IK_NPZ}"
echo "EEF replay: ${EEF_MP4}"
echo "IK replay:  ${IK_MP4}"
