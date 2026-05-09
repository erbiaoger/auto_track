#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot/train}
OUT_DIR=${OUT_DIR:-models/peak_slot_cpu}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_SAMPLES=${MAX_SAMPLES:-32}
MAX_TRACKS=${MAX_TRACKS:-96}
HIDDEN_DIM=${HIDDEN_DIM:-64}
DECODER_LAYERS=${DECODER_LAYERS:-1}
POOLED_TIME=${POOLED_TIME:-64}
MATCHER=${MATCHER:-hungarian}
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.15}
NONE_WEIGHT=${NONE_WEIGHT:-0.35}
COUNT_LOSS_WEIGHT=${COUNT_LOSS_WEIGHT:-0.05}
MONOTONIC_LOSS_WEIGHT=${MONOTONIC_LOSS_WEIGHT:-0.2}
SMOOTHNESS_LOSS_WEIGHT=${SMOOTHNESS_LOSS_WEIGHT:-0.05}
RESUME=${RESUME:-}
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
SEED=${SEED:-42}

resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
elif [ "$AUTO_RESUME" = "1" ] || [ "$AUTO_RESUME" = "true" ]; then
  resume_args="--auto-resume"
fi

uv run python -m autotrack.dl.train_peak_slot \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device cpu \
  --amp off \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --max-tracks "$MAX_TRACKS" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --pooled-time "$POOLED_TIME" \
  --matcher "$MATCHER" \
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --none-weight "$NONE_WEIGHT" \
  --count-loss-weight "$COUNT_LOSS_WEIGHT" \
  --monotonic-loss-weight "$MONOTONIC_LOSS_WEIGHT" \
  --smoothness-loss-weight "$SMOOTHNESS_LOSS_WEIGHT" \
  --metrics-every 1 \
  --log-every 1 \
  --seed "$SEED" \
  $resume_args
