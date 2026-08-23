#!/usr/bin/env bash
# Run the full evaluation matrix for the stack_object tasks and compare the results.
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ego2exe/scripts/run_eval_matrix.sh [options]

Runs every (task x pipeline x cohort x variant) evaluation, then prints the
comparison tables. Nothing is re-processed - evaluation is pure selection over
what batch_preprocess_le_ver.sh already produced.

  tasks     : stack_object_horizontal, stack_object_lean
  pipelines : h2g_humanego, h2g_pinch_index, h2g_finger_center
  cohorts   : all (+ hyj, xule, ymq with --cohorts)
  variants  : full, seg1-2 (the carry phase: gripper close -> open)

Options:
  --tasks "a b"       override the task list
  --pipelines "a b"   override the pipeline list
  --cohorts           also evaluate each operator separately (4x more runs)
  --with-c2st         run the classifier two-sample test (slow, saturates until rho~2)
  --dry-run           print the commands without running them
  -h, --help

Environment:
  PY   python executable (default: python)
EOF
}

[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { usage; exit 0; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASKS="stack_object_horizontal stack_object_lean"
PIPELINES="h2g_humanego h2g_pinch_index h2g_finger_center"
PER_COHORT=0
C2ST=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)      TASKS="$2"; shift 2 ;;
    --pipelines)  PIPELINES="$2"; shift 2 ;;
    --cohorts)    PER_COHORT=1; shift ;;
    --with-c2st)  C2ST=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${PY:-python}"
cd "${REPO_ROOT}"
EVAL="ego2exe/eval/eval_ego2exe.py"
EXTRA=()
[[ ${C2ST} -eq 0 ]] && EXTRA+=(--no-c2st)

COHORTS="all"
[[ ${PER_COHORT} -eq 1 ]] && COHORTS="all hyj xule ymq"

run() {
  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "  ${PY} ${EVAL} $*"
  else
    printf '  %-58s ' "$(echo "$*" | sed 's/--task //; s/--pipeline //; s/--cohort //; s/--segment-start /seg/; s/--segment-end //')"
    if "${PY}" "${EVAL}" "$@" >/dev/null 2>&1; then echo "ok"; else echo "FAILED"; fi
  fi
}

echo "=============================================================================="
echo "eval matrix: $(echo ${TASKS} | wc -w) task(s) x $(echo ${PIPELINES} | wc -w) pipeline(s) x $(echo ${COHORTS} | wc -w) cohort(s) x 2 variant(s)"
echo "=============================================================================="
for task in ${TASKS}; do
  for pipe in ${PIPELINES}; do
    for coh in ${COHORTS}; do
      # full episode
      run --task "${task}" --pipeline "${pipe}" --cohort "${coh}" "${EXTRA[@]}"
      # carry phase only: gripper close -> open
      run --task "${task}" --pipeline "${pipe}" --cohort "${coh}" \
          --segment-start 1 --segment-end 2 "${EXTRA[@]}"
    done
  done
done

[[ ${DRY_RUN} -eq 1 ]] && exit 0

CMP="ego2exe/eval/compare_reports.py"
for task in ${TASKS}; do
  short="${task#stack_object_}"
  echo
  echo "=============================================================================="
  echo "${task}  -  algorithms, full episode"
  echo "=============================================================================="
  "${PY}" "${CMP}" outputs/eval/"${task}"__*__all__full 2>/dev/null

  echo
  echo "=============================================================================="
  echo "${task}  -  algorithms, carry phase (seg1-2)"
  echo "=============================================================================="
  "${PY}" "${CMP}" outputs/eval/"${task}"__*__all__seg1-2 2>/dev/null

  echo
  echo "=============================================================================="
  echo "${task}  -  full vs carry phase, absolute values alongside rho"
  echo "=============================================================================="
  "${PY}" "${CMP}" --metric floor_mm,D_pos_mm,D_rot_deg,rho_pos,rho_rot,rho_SE3 \
      outputs/eval/"${task}"__*__all__full outputs/eval/"${task}"__*__all__seg1-2 2>/dev/null
done

echo
echo "=============================================================================="
echo "both tasks together"
echo "=============================================================================="
"${PY}" "${CMP}" outputs/eval/stack_object_*__all__full outputs/eval/stack_object_*__all__seg1-2 2>/dev/null

if [[ ${PER_COHORT} -eq 1 ]]; then
  for task in ${TASKS}; do
    echo
    echo "=============================================================================="
    echo "${task}  -  per operator (best pipeline), full episode"
    echo "=============================================================================="
    "${PY}" "${CMP}" --sort task,variant,cohort \
        outputs/eval/"${task}"__h2g_finger_center__*__full 2>/dev/null
  done
fi
