#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

SPLIT=${SPLIT:-train}
IN_DIR=${IN_DIR:-datasets/peak_slot_profile_only_120s_unetprior/$SPLIT}
OUT_DIR=${OUT_DIR:-datasets/peak_slot_v5_120s_lineonly/$SPLIT}
PRIOR_CHANNEL=${PRIOR_CHANNEL:--1}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-96}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.15}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.08}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}
WORKERS=${WORKERS:-8}
MAX_SHARDS=${MAX_SHARDS:-0}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.extract_peak_slot_lineonly_from_prior \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --prior-channel "$PRIOR_CHANNEL" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  --workers "$WORKERS" \
  --max-shards "$MAX_SHARDS" \
  $overwrite_args
