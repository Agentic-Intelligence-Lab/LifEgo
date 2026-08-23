#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/preprocess.sh <video> [out_root] [--exp NAME]
                                [--hand2gripper-mode M] [--forward-seed S]

Runs the ego2exe preprocessing pipeline:
  1. WiLoR hand reconstruction        (raw, shared across --exp)
  2. EEF export in robot base         (experiment-specific if --exp is given)
  3. mink IK retargeting              (experiment-specific if --exp is given)
  4. headless EEF and IK replay MP4 rendering

Without --exp, everything is written under <out_root>/<session>/, matching the
single-run layout this script has always used.

With --exp NAME, WiLoR still writes to <out_root>/<session>/preprocess/ (shared,
expensive, reused across experiments), but the EEF export / IK / replays go to
outputs/experiments/<NAME>/<session>/ instead - so re-running this video after a
hand2gripper or IK change never overwrites a previous result. See
ego2exe/scripts/batch_preprocess_le_ver.sh for the same split applied to a whole folder
of videos, which is the more common way to use --exp.

Options:
  --exp NAME            write export/IK/replays under outputs/experiments/NAME/
  --hand2gripper-mode M humanego (default) or pinch_plane; forwarded to the export stage
  --forward-seed S      index_tip (default) or finger_mcp_centroid; pinch_plane only

Environment overrides:
  PY                    Python executable (default: python)
  GL_BACKEND            MuJoCo render backend for MP4 replay (default: egl)
  WILOR_PRETRAINED_DIR  WiLoR checkpoint/cache directory
  HAND_KEY              hand_r or hand_l (default: hand_r)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -eq 0 ]]; then
  usage
  [[ $# -eq 0 ]] && exit 2 || exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGO2EXE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EGO2EXE_DIR}/.." && pwd)"

VIDEO=""
OUT_ROOT="outputs/new_pipeline"
EXP=""
H2G_MODE=""
FORWARD_SEED=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp) EXP="$2"; shift 2 ;;
    --hand2gripper-mode) H2G_MODE="$2"; shift 2 ;;
    --forward-seed) FORWARD_SEED="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [[ ${#POSITIONAL[@]} -lt 1 || ${#POSITIONAL[@]} -gt 2 ]]; then
  usage >&2
  exit 2
fi
VIDEO="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -eq 2 ]] && OUT_ROOT="${POSITIONAL[1]}"

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

RAW_SESSION_DIR="${OUT_ROOT_ABS}/${SESSION_NAME}"

if [[ -n "${EXP}" ]]; then
  EXP_SESSION_DIR="${REPO_ROOT}/outputs/experiments/${EXP}/${SESSION_NAME}"
else
  EXP_SESSION_DIR="${RAW_SESSION_DIR}"
fi
EEF_DIR="${EXP_SESSION_DIR}/robot_eef_scene_camera_axis_corrected"
EEF_JSON="${EEF_DIR}/robot_eef_trajectory.json"
IK_DIR="${EXP_SESSION_DIR}/nero_eef_ik"
IK_NPZ="${IK_DIR}/nero_eef_ik.npz"
REPLAY_DIR="${EXP_SESSION_DIR}/replays"
EEF_MP4="${REPLAY_DIR}/${SESSION_NAME}_eef.mp4"
IK_MP4="${REPLAY_DIR}/${SESSION_NAME}_ik.mp4"

cd "${REPO_ROOT}"
mkdir -p "${REPLAY_DIR}"

echo "=== ego2exe preprocess ==="
echo "video:      ${VIDEO_ABS}"
echo "raw session:${RAW_SESSION_DIR}"
[[ -n "${EXP}" ]] && echo "experiment: ${EXP}  ->  ${EXP_SESSION_DIR}"
echo "hand2grip:  ${H2G_MODE:-humanego}$([[ "${H2G_MODE}" == pinch_plane ]] && echo " (seed: ${FORWARD_SEED:-index_tip})")"
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
EXPORT_ARGS=(--hand-key "${HAND_KEY}")
[[ -n "${H2G_MODE}" ]] && EXPORT_ARGS+=(--hand2gripper-mode "${H2G_MODE}")
[[ -n "${FORWARD_SEED}" ]] && EXPORT_ARGS+=(--forward-seed "${FORWARD_SEED}")
"${PY}" ego2exe/preprocess_export_eef.py \
  --session "${RAW_SESSION_DIR}" \
  --out "${EEF_DIR}" \
  "${EXPORT_ARGS[@]}"

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
