#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_crossing_curriculum}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_train_realshape_crossing_curriculum}"
RESUME="${RESUME:-$RUN_DIR/peakguided_train_realshape_missing_fixedgauss/checkpoint_last.pt}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-24}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-50}"
LR="${LR:-3e-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_EVERY="${SAVE_EVERY:-5}"
LOG_EVERY="${LOG_EVERY:-10}"
AMP="${AMP:-1}"
COMPILE="${COMPILE:-1}"
TORCH_THREADS="${TORCH_THREADS:-1}"
if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  bash "$ROOT/scripts/generate_vehicle_peakset_train_crossing_curriculum.sh"
fi
if [[ ! -f "$RESUME" ]]; then
  echo "missing RESUME checkpoint: $RESUME" >&2
  exit 1
fi

RESUME_EPOCH="$("$PYTHON_BIN" - "$RESUME" <<'PY'
import sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(ckpt.get("epoch", 0)))
PY
)"
EPOCHS="${EPOCHS:-$((RESUME_EPOCH + FINETUNE_EPOCHS))}"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

train_args=(
  --out-dir "$OUT_DIR"
  --dataset-dir "$DATASET_DIR"
  --resume "$RESUME"
  --reset-best-val
  --val-fraction "${VAL_FRACTION:-0.2}"
  --device "$DEVICE"
  --torch-threads "$TORCH_THREADS"
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
if [[ "$COMPILE" == "1" || "$COMPILE" == "true" || "$COMPILE" == "yes" ]]; then
  train_args+=(--compile)
fi

printf 'training_realshape_crossing_curriculum\n'
printf 'DATASET_DIR=%s\nOUT_DIR=%s\nRESUME=%s\nEPOCHS=%s\nBATCH_SIZE=%s\nLR=%s\n' \
  "$DATASET_DIR" "$OUT_DIR" "$RESUME" "$EPOCHS" "$BATCH_SIZE" "$LR"
printf '%q ' "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
printf '\n'

PYTHONPATH=src "$PYTHON_BIN" -m autotrack.dl.train_vehicle_peak_set "${train_args[@]}"
