#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_gauss}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_train_realshape_gauss_deadrefine}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EPOCHS="${EPOCHS:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
MAX_QUERIES="${MAX_QUERIES:-32}"
POOLED_TIME="${POOLED_TIME:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-1}"
INERTIA_WEIGHT="${INERTIA_WEIGHT:-0.10}"
DEAD_COMPLETE_TIME_WEIGHT="${DEAD_COMPLETE_TIME_WEIGHT:-6.0}"
DEAD_COMPLETE_VALID_WEIGHT="${DEAD_COMPLETE_VALID_WEIGHT:-2.0}"
RESUME="${RESUME:-$RUN_DIR/checkpoint_best.pt}"

if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  bash scripts/generate_vehicle_peakset_train_realshape_gauss.sh
fi

train_args=(
  --out-dir "$OUT_DIR"
  --dataset-dir "$DATASET_DIR"
  --val-fraction "${VAL_FRACTION:-0.2}"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --shuffle
  --save-every "${SAVE_EVERY:-5}"
  --log-every "${LOG_EVERY:-10}"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --encoder-layers "$ENCODER_LAYERS"
  --decoder-layers "$DECODER_LAYERS"
  --max-queries "$MAX_QUERIES"
  --pooled-time "$POOLED_TIME"
  --inertia-weight "$INERTIA_WEIGHT"
  --dead-complete-time-weight "$DEAD_COMPLETE_TIME_WEIGHT"
  --dead-complete-valid-weight "$DEAD_COMPLETE_VALID_WEIGHT"
)
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
  train_args+=(--resume "$RESUME")
fi
if [[ "$AMP" == "1" ]]; then
  train_args+=(--amp)
fi

PYTHONPATH=src "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
