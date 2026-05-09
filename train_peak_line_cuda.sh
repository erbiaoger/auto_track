#!/usr/bin/env sh
set -eu

DATA_DIR=${DATA_DIR:-datasets/peak_line/train}
OUT_DIR=${OUT_DIR:-models/peak_line_cuda}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-0.0002}
BASE_CHANNELS=${BASE_CHANNELS:-8}
VAL_FRACTION=${VAL_FRACTION:-0.1}
POS_WEIGHT=${POS_WEIGHT:-20}
AUTO_RESUME=${AUTO_RESUME:-1}
CHANNELS_LAST=${CHANNELS_LAST:-1}

resume_args=""
if [ "$AUTO_RESUME" = "1" ]; then
  resume_args="$resume_args --auto-resume"
fi
if [ "$CHANNELS_LAST" = "1" ]; then
  resume_args="$resume_args --channels-last"
fi

uv run python -m autotrack.dl.train_peak_line \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --amp on \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --base-channels "$BASE_CHANNELS" \
  --val-fraction "$VAL_FRACTION" \
  --pos-weight "$POS_WEIGHT" \
  $resume_args
