#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_one_group.sh <dataset-name> <model-name> <group> [script-options] [-- extra-train-args...]

Groups:
  full
  point_centroid
  point_coarse
  safe_centroid
  safe_coarse

Script options:
  --seed <int>               Override the default seed (default: 42)
  --exp-tag <tag>            Override the train.py experiment tag
  --prior-tag <tag>          Override the SAFE prior tag
  --experiments-root <dir>   Override the experiments root passed to train.py
  --python <bin>             Override the Python executable
  -h, --help                 Show this help message

Examples:
  bash scripts/run_one_group.sh irstd1k lightweight_unet full
  bash scripts/run_one_group.sh nuaa_sirst lightweight_unet safe_centroid -- --epochs 100 --device cuda:0
EOF
}

if [[ $# -lt 3 ]]; then
  usage
  exit 1
fi

DATASET_NAME="$1"
MODEL_NAME="$2"
GROUP_NAME="$3"
shift 3

SEED="${ONE_GROUP_SEED_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN_DEFAULT}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT_DEFAULT}"
EXP_TAG="${EXP_TAG_DEFAULT}"
PRIOR_TAG="${PRIOR_TAG_DEFAULT}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      [[ $# -ge 2 ]] || die "--seed requires a value."
      SEED="$2"
      shift 2
      ;;
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
require_supported_group "${GROUP_NAME}"

LABEL_MODE="$(resolve_label_mode "${GROUP_NAME}")"
METHOD="$(resolve_method "${GROUP_NAME}")"
EXPERIMENTS_ROOT_ABS="$(resolve_path_from_repo "${EXPERIMENTS_ROOT}")"

TRAIN_CMD=(
  "${PYTHON_BIN}" "train.py"
  "--dataset-name" "${DATASET_NAME}"
  "--model-name" "${MODEL_NAME}"
  "--label-mode" "${LABEL_MODE}"
  "--method" "${METHOD}"
  "--seed" "${SEED}"
  "--batch-size" "${BATCH_SIZE_DEFAULT}"
  "--lr" "${LR_DEFAULT}"
  "--weight-decay" "${WEIGHT_DECAY_DEFAULT}"
  "--loss-type" "${LOSS_TYPE_DEFAULT}"
  "--scheduler" "${SCHEDULER_DEFAULT}"
  "--exp-tag" "${EXP_TAG}"
  "--experiments-root" "${EXPERIMENTS_ROOT}"
  "--no-amp"
)

if [[ "${METHOD}" == "safe" ]]; then
  PRIOR_ROOT="$(build_prior_root "${EXPERIMENTS_ROOT_ABS}" "${DATASET_NAME}" "${LABEL_MODE}" "${PRIOR_TAG}")"
  INNER_PRIOR_DIR="${PRIOR_ROOT}/priors/${INNER_PRIOR_SUBDIR_DEFAULT}"
  OUTER_PRIOR_DIR="${PRIOR_ROOT}/priors/${OUTER_PRIOR_SUBDIR_DEFAULT}"
  require_file "${PRIOR_ROOT}/manifest.json"
  require_directory "${INNER_PRIOR_DIR}"
  require_directory "${OUTER_PRIOR_DIR}"

  TRAIN_CMD+=(
    "--inner-prior-dir" "${INNER_PRIOR_DIR}"
    "--outer-prior-dir" "${OUTER_PRIOR_DIR}"
    "--inner-loss-weight" "${INNER_LOSS_WEIGHT_DEFAULT}"
    "--outer-loss-weight" "${OUTER_LOSS_WEIGHT_DEFAULT}"
    "--prior-warmup-epochs" "${PRIOR_WARMUP_EPOCHS_DEFAULT}"
  )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

printf '[run_one_group] dataset=%s model=%s group=%s label_mode=%s method=%s seed=%s\n' \
  "${DATASET_NAME}" "${MODEL_NAME}" "${GROUP_NAME}" "${LABEL_MODE}" "${METHOD}" "${SEED}"
if [[ "${METHOD}" == "safe" ]]; then
  printf '[run_one_group] inner_prior=%s\n' "${INNER_PRIOR_DIR}"
  printf '[run_one_group] outer_prior=%s\n' "${OUTER_PRIOR_DIR}"
fi
printf '[run_one_group] command: '
print_command "${TRAIN_CMD[@]}"

cd "${REPO_ROOT}"
"${TRAIN_CMD[@]}"
