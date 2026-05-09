#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

RUN_DIR=${RUN_DIR:-models/peak_slot_cuda}
HISTORY=${HISTORY:-}
OUT_DIR=${OUT_DIR:-}
METRICS=${METRICS:-loss,track_f1,count_mae,time_mae_norm,max_objectness,mean_objectness}
PREFIXES=${PREFIXES:-train,val}
SMOOTH_WINDOW=${SMOOTH_WINDOW:-1}
FORMAT=${FORMAT:-png}
DPI=${DPI:-180}
TITLE=${TITLE:-}
SEPARATE=${SEPARATE:-1}

set -- uv run python -m autotrack.dl.plot_track_slot_history
if [ -n "$HISTORY" ]; then
  set -- "$@" --history "$HISTORY"
else
  set -- "$@" --run-dir "$RUN_DIR"
fi
if [ -n "$OUT_DIR" ]; then
  set -- "$@" --out-dir "$OUT_DIR"
fi
if [ "$SEPARATE" = "1" ] || [ "$SEPARATE" = "true" ]; then
  set -- "$@" --separate
fi

"$@" \
  --metrics "$METRICS" \
  --prefixes "$PREFIXES" \
  --smooth-window "$SMOOTH_WINDOW" \
  --format "$FORMAT" \
  --dpi "$DPI" \
  --title "$TITLE"
