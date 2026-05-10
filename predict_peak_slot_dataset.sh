#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot/test}
MODEL=${MODEL:-models/peak_slot_cuda/checkpoint_best.pt}
OUT_DIR=${OUT_DIR:-models/peak_slot_cuda/prediction_check}
DEVICE=${DEVICE:-cpu}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SAMPLES=${MAX_SAMPLES:-256}
MAX_CSV_SAMPLES=${MAX_CSV_SAMPLES:-32}
PLOT_SAMPLES=${PLOT_SAMPLES:-16}
PLOT_DPI=${PLOT_DPI:-160}
PLOT_STYLE=${PLOT_STYLE:-waveform}
OBJECTNESS_THRESHOLD=${OBJECTNESS_THRESHOLD:-0.5}
PEAK_THRESHOLD=${PEAK_THRESHOLD:-0.4}
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-3}
MAX_PREDICTED_TRACKS=${MAX_PREDICTED_TRACKS:-96}
MATCHER=${MATCHER:-hungarian}

uv run python -m autotrack.dl.predict_peak_slot_dataset \
  --data-dir "$DATA_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --max-csv-samples "$MAX_CSV_SAMPLES" \
  --plot-samples "$PLOT_SAMPLES" \
  --plot-dpi "$PLOT_DPI" \
  --plot-style "$PLOT_STYLE" \
  --objectness-threshold "$OBJECTNESS_THRESHOLD" \
  --peak-threshold "$PEAK_THRESHOLD" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --max-predicted-tracks "$MAX_PREDICTED_TRACKS" \
  --matcher "$MATCHER"
