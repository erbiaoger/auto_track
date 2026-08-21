#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_missing_fixedgauss}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_train_realshape_missing_fixedgauss}"
RESUME="${RESUME:-$RUN_DIR/peakguided_train_realshape_missingrefine/checkpoint_best.pt}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-96}"
LR="${LR:-5e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-10}"
AMP="${AMP:-1}"

if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  echo "missing DATASET_DIR/meta.json: $DATASET_DIR/meta.json" >&2
  exit 1
fi
if [[ ! -f "$RESUME" ]]; then
  echo "missing RESUME checkpoint: $RESUME" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

train_args=(
  --out-dir "$OUT_DIR"
  --dataset-dir "$DATASET_DIR"
  --resume "$RESUME"
  --reset-best-val
  --val-fraction "${VAL_FRACTION:-0.2}"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --num-workers "$NUM_WORKERS"
  --prefetch-factor "$PREFETCH_FACTOR"
  --shuffle
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --hidden-dim "${HIDDEN_DIM:-128}"
  --num-heads "${NUM_HEADS:-4}"
  --encoder-layers "${ENCODER_LAYERS:-2}"
  --decoder-layers "${DECODER_LAYERS:-2}"
  --max-queries "${MAX_QUERIES:-32}"
  --pooled-time "${POOLED_TIME:-64}"
  --no-residual-completion
  --complete-time-weight "${COMPLETE_TIME_WEIGHT:-8.0}"
  --complete-valid-weight "${COMPLETE_VALID_WEIGHT:-1.0}"
  --observed-valid-weight "${OBSERVED_VALID_WEIGHT:-1.0}"
  --missing-complete-time-weight "${MISSING_COMPLETE_TIME_WEIGHT:-12.0}"
  --missing-complete-valid-weight "${MISSING_COMPLETE_VALID_WEIGHT:-4.0}"
  --anchor-weight "${ANCHOR_WEIGHT:-1.0}"
  --anchor-time-weight "${ANCHOR_TIME_WEIGHT:-0.5}"
  --inertia-weight "${INERTIA_WEIGHT:-0.80}"
  --linearity-weight "${LINEARITY_WEIGHT:-0.40}"
  --jump-weight "${JUMP_WEIGHT:-0.35}"
  --slope-variation-weight "${SLOPE_VARIATION_WEIGHT:-0.50}"
  --direction-weight "${DIRECTION_WEIGHT:-0.05}"
  --speed-weight "${SPEED_WEIGHT:-0.05}"
  --count-weight "${COUNT_WEIGHT:-0.01}"
)

if [[ "$AMP" == "1" || "$AMP" == "true" || "$AMP" == "yes" ]]; then
  train_args+=(--amp)
fi

printf 'training_realshape_missing_fixedgauss\n'
printf 'DATASET_DIR=%s\nOUT_DIR=%s\nRESUME=%s\nEPOCHS=%s\nBATCH_SIZE=%s\nLR=%s\n' \
  "$DATASET_DIR" "$OUT_DIR" "$RESUME" "$EPOCHS" "$BATCH_SIZE" "$LR"
printf '%q ' "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
printf '\n'

"$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
