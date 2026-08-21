#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"

SHUFFLE_SHARDS="${SHUFFLE_SHARDS:-1}" \
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_dataset_residual_completion}" \
TMP_ROOT="${TMP_ROOT:-$RUN_DIR/.tmp_residual_completion_components}" \
NUM_SAMPLES="${NUM_SAMPLES:-16384}" \
SHARD_SIZE="${SHARD_SIZE:-128}" \
WORKERS="${WORKERS:-16}" \
SEED="${SEED:-20260704}" \
bash "$ROOT/scripts/generate_vehicle_peakset_train_realshape_easy_completion.sh"
