#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/track_slot/train}
OUT_DIR=${OUT_DIR:-models/track_slot_cuda}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-20}
LR=${LR:-0.0002}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
MAX_TRACKS=${MAX_TRACKS:-96}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DECODER_LAYERS=${DECODER_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-4}
POOLED_CHANNELS=${POOLED_CHANNELS:-8}
POOLED_TIME=${POOLED_TIME:-128}
AMP=${AMP:-on}
AMP_DTYPE=${AMP_DTYPE:-float16}
MATCHER=${MATCHER:-hungarian}
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.05}
VAL_FRACTION=${VAL_FRACTION:-0}
VAL_EVERY=${VAL_EVERY:-5}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
METRICS_EVERY=${METRICS_EVERY:-20}
LOG_EVERY=${LOG_EVERY:-20}
RESUME=${RESUME:-}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
SEED=${SEED:-42}

resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
fi

uv run python -m autotrack.dl.train_track_slot \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-tracks "$MAX_TRACKS" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --pooled-channels "$POOLED_CHANNELS" \
  --pooled-time "$POOLED_TIME" \
  --amp "$AMP" \
  --amp-dtype "$AMP_DTYPE" \
  --channels-last \
  --matcher "$MATCHER" \
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --val-fraction "$VAL_FRACTION" \
  --val-every "$VAL_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --metrics-every "$METRICS_EVERY" \
  --log-every "$LOG_EVERY" \
  --seed "$SEED" \
  $resume_args
