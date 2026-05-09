#!/usr/bin/env sh
set -eu

DATA_DIR=${DATA_DIR:-datasets/peak_line/train}
MODEL=${MODEL:-models/peak_line_cuda/checkpoint_best.pt}
OUT_DIR=${OUT_DIR:-/tmp/peak_line_prediction_check}
DEVICE=${DEVICE:-auto}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SAMPLES=${MAX_SAMPLES:-256}
PLOT_SAMPLES=${PLOT_SAMPLES:-16}
THRESHOLD=${THRESHOLD:-0.5}

uv run python -m autotrack.dl.predict_peak_line_dataset \
  --data-dir "$DATA_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --plot-samples "$PLOT_SAMPLES" \
  --threshold "$THRESHOLD"
