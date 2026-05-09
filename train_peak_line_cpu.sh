#!/usr/bin/env sh
set -eu

DATA_DIR=${DATA_DIR:-datasets/peak_line/train}
OUT_DIR=${OUT_DIR:-models/peak_line_cpu}
EPOCHS=${EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
LR=${LR:-0.0002}
BASE_CHANNELS=${BASE_CHANNELS:-8}
VAL_FRACTION=${VAL_FRACTION:-0.1}
MAX_SAMPLES=${MAX_SAMPLES:-0}
AUTO_RESUME=${AUTO_RESUME:-0}

resume_args=""
if [ "$AUTO_RESUME" = "1" ]; then
  resume_args="$resume_args --auto-resume"
fi
if [ "$MAX_SAMPLES" != "0" ]; then
  resume_args="$resume_args --max-samples $MAX_SAMPLES"
fi

uv run python -m autotrack.dl.train_peak_line \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device cpu \
  --amp off \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --base-channels "$BASE_CHANNELS" \
  --val-fraction "$VAL_FRACTION" \
  $resume_args
