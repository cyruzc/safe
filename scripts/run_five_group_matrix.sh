#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_five_group_matrix.sh <dataset-name> <model-name> [script-options] [-- extra-train-args...]

This script runs the full paper matrix:
  full
  point_centroid
  point_coarse
  safe_centroid
  safe_coarse

Default seeds:
  27 123 456 789 2024

Script options:
  --exp-tag <tag>            Override the train.py experiment tag
  --prior-tag <tag>          Override the SAFE prior tag
  --experiments-root <dir>   Override the experiments root passed to train.py
  --python <bin>             Override the Python executable
  -h, --help                 Show this help message

Examples:
  bash scripts/run_five_group_matrix.sh irstd1k lightweight_unet
  bash scripts/run_five_group_matrix.sh sirst3 lightweight_unet -- --epochs 100 --device cuda:0
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

DATASET_NAME="$1"
MODEL_NAME="$2"
shift 2

PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT_DEFAULT}"
EXP_TAG="${EXP_TAG_DEFAULT}"
PRIOR_TAG="${PRIOR_TAG_DEFAULT}"
SEEDS=("${FIVE_GROUP_SEEDS_DEFAULT[@]}")
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exp-tag)
      [[ $# -ge 2 ]] || die "--exp-tag requires a value."
      EXP_TAG="$2"
      shift 2
      ;;
    --prior-tag)
      [[ $# -ge 2 ]] || die "--prior-tag requires a value."
      PRIOR_TAG="$2"
      shift 2
      ;;
    --experiments-root)
      [[ $# -ge 2 ]] || die "--experiments-root requires a value."
      EXPERIMENTS_ROOT="$2"
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || die "--python requires a value."
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

require_supported_dataset "${DATASET_NAME}"

TOTAL_RUNS=$(( ${#GROUP_ORDER[@]} * ${#SEEDS[@]} ))
printf '[run_five_group_matrix] dataset=%s model=%s exp_tag=%s total_runs=%s\n' \
  "${DATASET_NAME}" "${MODEL_NAME}" "${EXP_TAG}" "${TOTAL_RUNS}"
printf '[run_five_group_matrix] seeds=%s\n' "${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  for group_name in "${GROUP_ORDER[@]}"; do
    printf '[run_five_group_matrix] launching group=%s seed=%s\n' "${group_name}" "${seed}"
    RUN_CMD=(
      "bash" "${SCRIPT_DIR}/run_one_group.sh"
      "${DATASET_NAME}" "${MODEL_NAME}" "${group_name}"
      "--seed" "${seed}"
      "--exp-tag" "${EXP_TAG}"
      "--prior-tag" "${PRIOR_TAG}"
      "--experiments-root" "${EXPERIMENTS_ROOT}"
      "--python" "${PYTHON_BIN}"
    )
    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      RUN_CMD+=(-- "${EXTRA_ARGS[@]}")
    fi
    "${RUN_CMD[@]}"
  done
done
