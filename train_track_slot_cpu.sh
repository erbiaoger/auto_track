#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/track_slot/train}
OUT_DIR=${OUT_DIR:-models/track_slot_cpu}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_TRACKS=${MAX_TRACKS:-96}
HIDDEN_DIM=${HIDDEN_DIM:-64}
DECODER_LAYERS=${DECODER_LAYERS:-1}
POOLED_TIME=${POOLED_TIME:-64}
MATCHER=${MATCHER:-hungarian}
SEED=${SEED:-42}

uv run python -m autotrack.dl.train_track_slot \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device cpu \
  --amp off \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-tracks "$MAX_TRACKS" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --pooled-time "$POOLED_TIME" \
  --matcher "$MATCHER" \
  --metrics-every 1 \
  --log-every 1 \
  --seed "$SEED"
