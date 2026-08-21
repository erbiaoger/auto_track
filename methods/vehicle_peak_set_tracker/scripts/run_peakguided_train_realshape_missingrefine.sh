#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_missing}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_train_realshape_missingrefine}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-24}"
EPOCHS="${EPOCHS:-200}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
MAX_QUERIES="${MAX_QUERIES:-32}"
POOLED_TIME="${POOLED_TIME:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-1}"
INERTIA_WEIGHT="${INERTIA_WEIGHT:-0.80}"
LINEARITY_WEIGHT="${LINEARITY_WEIGHT:-0.40}"
JUMP_WEIGHT="${JUMP_WEIGHT:-0.35}"
SLOPE_VARIATION_WEIGHT="${SLOPE_VARIATION_WEIGHT:-0.50}"
COMPLETE_TIME_WEIGHT="${COMPLETE_TIME_WEIGHT:-8.0}"
COMPLETE_VALID_WEIGHT="${COMPLETE_VALID_WEIGHT:-1.0}"
MISSING_COMPLETE_TIME_WEIGHT="${MISSING_COMPLETE_TIME_WEIGHT:-12.0}"
MISSING_COMPLETE_VALID_WEIGHT="${MISSING_COMPLETE_VALID_WEIGHT:-4.0}"
RESUME="${RESUME:-$OUT_DIR/checkpoint_last.pt}"

if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  bash scripts/generate_vehicle_peakset_train_realshape_missing.sh
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
  --jump-weight "$JUMP_WEIGHT"
  --slope-variation-weight "$SLOPE_VARIATION_WEIGHT"
  --inertia-weight "$INERTIA_WEIGHT"
  --complete-time-weight "$COMPLETE_TIME_WEIGHT"
  --complete-valid-weight "$COMPLETE_VALID_WEIGHT"
  --missing-complete-time-weight "$MISSING_COMPLETE_TIME_WEIGHT"
  --missing-complete-valid-weight "$MISSING_COMPLETE_VALID_WEIGHT"
  --linearity-weight "$LINEARITY_WEIGHT"
)
if [[ -n "$RESUME" && -f "$RESUME" ]]; then
  train_args+=(--resume "$RESUME")
fi
if [[ "$AMP" == "1" ]]; then
  train_args+=(--amp)
fi

PYTHONPATH=src "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
