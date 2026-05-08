#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/track_slot/train}
MODEL=${MODEL:-models/track_slot_cuda/checkpoint_best.pt}
OUT_DIR=${OUT_DIR:-models/track_slot_cuda/prediction_check}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SAMPLES=${MAX_SAMPLES:-256}
MAX_CSV_SAMPLES=${MAX_CSV_SAMPLES:-32}
OBJECTNESS_THRESHOLD=${OBJECTNESS_THRESHOLD:-0.3}
VISIBILITY_THRESHOLD=${VISIBILITY_THRESHOLD:-0.3}
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-3}
MAX_PREDICTED_TRACKS=${MAX_PREDICTED_TRACKS:-96}
MATCHER=${MATCHER:-hungarian}

uv run python -m autotrack.dl.predict_track_slot_dataset \
  --data-dir "$DATA_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --max-csv-samples "$MAX_CSV_SAMPLES" \
  --objectness-threshold "$OBJECTNESS_THRESHOLD" \
  --visibility-threshold "$VISIBILITY_THRESHOLD" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --max-predicted-tracks "$MAX_PREDICTED_TRACKS" \
  --matcher "$MATCHER"
