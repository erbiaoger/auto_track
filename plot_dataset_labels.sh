#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot/train}
OUT_DIR=${OUT_DIR:-models/label_check}
SAMPLE_INDICES=${SAMPLE_INDICES:-}
START_SAMPLE=${START_SAMPLE:-0}
PLOT_SAMPLES=${PLOT_SAMPLES:-16}
PLOT_DPI=${PLOT_DPI:-160}
PLOT_PEAKS=${PLOT_PEAKS:-1}
MAX_GT_TRACKS=${MAX_GT_TRACKS:-0}
POINT_SIZE=${POINT_SIZE:-10}
LINE_WIDTH=${LINE_WIDTH:-1.15}
LINE_ALPHA=${LINE_ALPHA:-0.75}
VMAX_QUANTILE=${VMAX_QUANTILE:-0.995}

sample_args=""
if [ -n "$SAMPLE_INDICES" ]; then
  sample_args="--sample-indices $SAMPLE_INDICES"
fi

peak_args=""
if [ "$PLOT_PEAKS" = "1" ] || [ "$PLOT_PEAKS" = "true" ]; then
  peak_args="--plot-peaks"
fi

uv run python -m autotrack.dl.plot_dataset_labels \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --start-sample "$START_SAMPLE" \
  --plot-samples "$PLOT_SAMPLES" \
  --plot-dpi "$PLOT_DPI" \
  --max-gt-tracks "$MAX_GT_TRACKS" \
  --point-size "$POINT_SIZE" \
  --line-width "$LINE_WIDTH" \
  --line-alpha "$LINE_ALPHA" \
  --vmax-quantile "$VMAX_QUANTILE" \
  $peak_args \
  $sample_args
