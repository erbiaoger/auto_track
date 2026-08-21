#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/train}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/dataset}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SAVE_EVERY="${SAVE_EVERY:-5}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
LOG_EVERY="${LOG_EVERY:-20}"
AMP="${AMP:-1}"
COMPILE="${COMPILE:-0}"
SHUFFLE="${SHUFFLE:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-0}"
INERTIA_WEIGHT="${INERTIA_WEIGHT:-0.25}"
RESUME="${RESUME:-}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR"

extra_args=()
if [[ "$AMP" == "1" ]]; then
  extra_args+=(--amp)
fi
if [[ "$COMPILE" == "1" ]]; then
  extra_args+=(--compile)
fi
if [[ "$SHUFFLE" == "1" ]]; then
  extra_args+=(--shuffle)
fi
if [[ "$MAX_TRAIN_SAMPLES" != "0" ]]; then
  extra_args+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ "$MAX_VAL_SAMPLES" != "0" ]]; then
  extra_args+=(--max-val-samples "$MAX_VAL_SAMPLES")
fi
resume_args=()
if [[ -n "$RESUME" ]]; then
  resume_args+=(--resume "$RESUME")
fi

uv run python -m autotrack.dl.train_vehicle_peak_set \
  --out-dir "$OUT_DIR" \
  --dataset-dir "$DATASET_DIR" \
  "${resume_args[@]}" \
  --val-fraction "$VAL_FRACTION" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --save-every "$SAVE_EVERY" \
  --num-workers "$NUM_WORKERS" \
  --prefetch-factor "$PREFETCH_FACTOR" \
  --log-every "$LOG_EVERY" \
  --duplicate-weight 0.40 \
  --linearity-weight 1.0 \
  --smoothness-weight 0.5 \
  --inertia-weight "$INERTIA_WEIGHT" \
  --primary-ratio 0.9 \
  --same-direction-ratio 0.9 \
  --crossing-ratio 0.1 \
  --missing-random-ratio-min 0.20 \
  --missing-random-ratio-max 0.45 \
  --missing-segment-count-max 4 \
  --missing-segment-min-len 2 \
  --missing-segment-max-len 8 \
  --per-vehicle-drop-channel-ratio 1.0 \
  --per-vehicle-drop-channel-min 5 \
  --per-vehicle-drop-channel-max 14 \
  --noise-std 0.04 \
  "${extra_args[@]}"

# Parameter notes:
# --run-dir: shared experiment root that holds dataset/train/predict subdirs.
# --out-dir: training outputs, checkpoints, and logs are written here.
# --dataset-dir: exported shard dataset directory to train from.
# --val-fraction: fraction of shard samples reserved for validation.
# --device: training device; use cuda on this machine for the 4090.
# --epochs: number of training epochs for the current curriculum.
# --batch-size: number of synthetic windows per optimization step.
# --save-every: save one intermediate checkpoint every N epochs.
# --resume: checkpoint path to continue from; optimizer state and epoch counter are restored.
# --num-workers: dataloader worker count; default 8 for shard-backed training.
# --prefetch-factor: each worker preloads this many batches.
# --log-every: print one progress line every N batches.
# --amp: enabled by AMP=1, uses CUDA mixed precision for faster training.
# --compile: enabled by COMPILE=1, uses torch.compile; first epoch may be slower.
# --shuffle: enabled by SHUFFLE=1; default 0 because sequential shard reads are much faster and generated samples are already random.
# --max-train-samples / --max-val-samples: set MAX_TRAIN_SAMPLES or MAX_VAL_SAMPLES for quick smoke runs.
# --duplicate-weight: penalty for queries collapsing onto the same vehicle.
# --linearity-weight: penalty that keeps each vehicle trajectory close to a straight line.
# --smoothness-weight: penalty on sharp second-order curvature in peak-time predictions.
# --primary-ratio: probability that the first vehicle direction is the dominant direction.
# --same-direction-ratio: probability that additional vehicles keep the same direction.
# --crossing-ratio: probability that additional vehicles become opposite-direction crossings.
# --missing-random-ratio-min: lower bound for randomly removed observed channels.
# --missing-random-ratio-max: upper bound for randomly removed observed channels.
# --missing-segment-count-max: maximum number of continuous missing segments per vehicle.
# --missing-segment-min-len / --missing-segment-max-len: continuous missing segment length range.
# --per-vehicle-drop-channel-ratio: probability of removing an additional block of channels per vehicle.
# --per-vehicle-drop-channel-min / --per-vehicle-drop-channel-max: extra missing channel block size range.
# --noise-std: additive background noise level in the synthetic generator.
