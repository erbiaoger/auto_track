#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

IN_DIR=${IN_DIR:-datasets/track_slot_v2/train}
OUT_DIR=${OUT_DIR:-datasets/peak_slot_v2/train}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-64}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.5}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.02}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}
WORKERS=${WORKERS:-32}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.convert_track_slot_to_peak_slot \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  --workers "$WORKERS" \
  $overwrite_args
