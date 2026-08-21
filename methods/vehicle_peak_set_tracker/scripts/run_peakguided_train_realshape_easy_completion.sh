#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_easy_completion}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_train_realshape_easy_completion}"
BASE_CKPT="${BASE_CKPT:-$RUN_DIR/peakguided_train_realshape_missingrefine/checkpoint_best.pt}"
RESUME="${RESUME:-$BASE_CKPT}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"
DEVICE="${DEVICE:-cuda}"

BATCH_SIZE="${BATCH_SIZE:-24}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
AMP="${AMP:-1}"

HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_HEADS="${NUM_HEADS:-4}"
ENCODER_LAYERS="${ENCODER_LAYERS:-2}"
DECODER_LAYERS="${DECODER_LAYERS:-2}"
MAX_QUERIES="${MAX_QUERIES:-32}"
POOLED_TIME="${POOLED_TIME:-64}"

COMPLETE_TIME_WEIGHT="${COMPLETE_TIME_WEIGHT:-8.0}"
COMPLETE_VALID_WEIGHT="${COMPLETE_VALID_WEIGHT:-1.0}"
MISSING_COMPLETE_TIME_WEIGHT="${MISSING_COMPLETE_TIME_WEIGHT:-16.0}"
MISSING_COMPLETE_VALID_WEIGHT="${MISSING_COMPLETE_VALID_WEIGHT:-4.0}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-1.0}"
ANCHOR_TIME_WEIGHT="${ANCHOR_TIME_WEIGHT:-0.5}"
INERTIA_WEIGHT="${INERTIA_WEIGHT:-0.60}"
LINEARITY_WEIGHT="${LINEARITY_WEIGHT:-0.35}"
JUMP_WEIGHT="${JUMP_WEIGHT:-0.25}"
SLOPE_VARIATION_WEIGHT="${SLOPE_VARIATION_WEIGHT:-0.35}"
NO_OBJECT_WEIGHT="${NO_OBJECT_WEIGHT:-0.02}"

if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  bash scripts/generate_vehicle_peakset_train_realshape_easy_completion.sh
fi
if [[ -n "$RESUME" && ! -f "$RESUME" ]]; then
  echo "resume checkpoint not found: $RESUME" >&2
  exit 2
fi
RESUME_EPOCH=0
if [[ -n "$RESUME" ]]; then
  RESUME_EPOCH="$("$PYTHON_BIN" - "$RESUME" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", 0)))
PY
)"
fi
EPOCHS="${EPOCHS:-$((RESUME_EPOCH + FINETUNE_EPOCHS))}"

train_args=(
  --out-dir "$OUT_DIR"
  --dataset-dir "$DATASET_DIR"
  --val-fraction "${VAL_FRACTION:-0.2}"
  --device "$DEVICE"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --shuffle
  --save-every "$SAVE_EVERY"
  --log-every "$LOG_EVERY"
  --hidden-dim "$HIDDEN_DIM"
  --num-heads "$NUM_HEADS"
  --encoder-layers "$ENCODER_LAYERS"
  --decoder-layers "$DECODER_LAYERS"
  --max-queries "$MAX_QUERIES"
  --pooled-time "$POOLED_TIME"
  --no-object-weight "$NO_OBJECT_WEIGHT"
  --complete-time-weight "$COMPLETE_TIME_WEIGHT"
  --complete-valid-weight "$COMPLETE_VALID_WEIGHT"
  --missing-complete-time-weight "$MISSING_COMPLETE_TIME_WEIGHT"
  --missing-complete-valid-weight "$MISSING_COMPLETE_VALID_WEIGHT"
  --anchor-weight "$ANCHOR_WEIGHT"
  --anchor-time-weight "$ANCHOR_TIME_WEIGHT"
  --jump-weight "$JUMP_WEIGHT"
  --slope-variation-weight "$SLOPE_VARIATION_WEIGHT"
  --inertia-weight "$INERTIA_WEIGHT"
  --linearity-weight "$LINEARITY_WEIGHT"
)
if [[ -n "$RESUME" ]]; then
  train_args+=(--resume "$RESUME")
fi
if [[ "$AMP" == "1" ]]; then
  train_args+=(--amp)
fi

PYTHONPATH=src "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
