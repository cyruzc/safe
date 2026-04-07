#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN_DEFAULT="${PYTHON_BIN:-python}"
EXPERIMENTS_ROOT_DEFAULT="${EXPERIMENTS_ROOT:-experiments}"
EXP_TAG_DEFAULT="${EXP_TAG:-paper_main}"
PRIOR_TAG_DEFAULT="${PRIOR_TAG:-dog_wslcm_p999_p995}"
INNER_PRIOR_SUBDIR_DEFAULT="${INNER_PRIOR_SUBDIR:-inner_dog_p99_9}"
OUTER_PRIOR_SUBDIR_DEFAULT="${OUTER_PRIOR_SUBDIR:-outer_wslcm_p99_5}"

BATCH_SIZE_DEFAULT="${BATCH_SIZE:-16}"
LR_DEFAULT="${LR:-5e-4}"
WEIGHT_DECAY_DEFAULT="${WEIGHT_DECAY:-1e-4}"
LOSS_TYPE_DEFAULT="${LOSS_TYPE:-focal}"
SCHEDULER_DEFAULT="${SCHEDULER:-cosine}"
INNER_LOSS_WEIGHT_DEFAULT="${INNER_LOSS_WEIGHT:-0.02}"
OUTER_LOSS_WEIGHT_DEFAULT="${OUTER_LOSS_WEIGHT:-0.2}"
PRIOR_WARMUP_EPOCHS_DEFAULT="${PRIOR_WARMUP_EPOCHS:-5}"

ONE_GROUP_SEED_DEFAULT="${ONE_GROUP_SEED:-42}"
FIVE_GROUP_SEEDS_DEFAULT=(${FIVE_GROUP_SEEDS:-27 123 456 789 2024})
GROUP_ORDER=(full point_centroid point_coarse safe_centroid safe_coarse)

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

print_command() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

require_supported_dataset() {
  local dataset_name="$1"
  case "${dataset_name}" in
    sirst3|irstd1k|nuaa_sirst|nudt_sirst) ;;
    *)
      die "Unsupported dataset '${dataset_name}'. Expected one of: sirst3, irstd1k, nuaa_sirst, nudt_sirst."
      ;;
  esac
}

require_supported_group() {
  local group_name="$1"
  case "${group_name}" in
    full|point_centroid|point_coarse|safe_centroid|safe_coarse) ;;
    *)
      die "Unsupported group '${group_name}'. Expected one of: full, point_centroid, point_coarse, safe_centroid, safe_coarse."
      ;;
  esac
}

resolve_label_mode() {
  local group_name="$1"
  case "${group_name}" in
    full)
      printf 'full\n'
      ;;
    point_centroid|safe_centroid)
      printf 'centroid\n'
      ;;
    point_coarse|safe_coarse)
      printf 'coarse\n'
      ;;
    *)
      die "Cannot resolve label mode for group '${group_name}'."
      ;;
  esac
}

resolve_method() {
  local group_name="$1"
  case "${group_name}" in
    full|point_centroid|point_coarse)
      printf 'none\n'
      ;;
    safe_centroid|safe_coarse)
      printf 'safe\n'
      ;;
    *)
      die "Cannot resolve method for group '${group_name}'."
      ;;
  esac
}

resolve_path_from_repo() {
  local path_value="$1"
  if [[ "${path_value}" = /* ]]; then
    printf '%s\n' "${path_value}"
  else
    printf '%s/%s\n' "${REPO_ROOT}" "${path_value}"
  fi
}

build_prior_root() {
  local experiments_root_abs="$1"
  local dataset_name="$2"
  local label_mode="$3"
  local prior_tag="$4"
  printf '%s/priors/%s/%s/%s\n' \
    "${experiments_root_abs}" \
    "${dataset_name}" \
    "${label_mode}" \
    "${prior_tag}"
}

require_file() {
  local file_path="$1"
  [[ -f "${file_path}" ]] || die "Required file not found: ${file_path}"
}

require_directory() {
  local dir_path="$1"
  [[ -d "${dir_path}" ]] || die "Required directory not found: ${dir_path}"
}
