#!/usr/bin/env sh
set -eu

# Segment an unlabeled real DAS .npy array into PeakSlotNet tensor shards.
#
# Default usage:
#   sh segment_real_npy_to_peak_slot.sh
#
# Example with overrides:
#   INPUT=/path/to/gauss_section.npy \
#   OUT_DIR=datasets/peak_slot/my_real_120s_stride60 \
#   WINDOW_SECONDS=120 \
#   STRIDE_SECONDS=60 \
#   sh segment_real_npy_to_peak_slot.sh
#
# Output:
#   The output directory receives meta.json and shard_*.pt files readable by:
#     uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir "$OUT_DIR" ...

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

INPUT=${INPUT:-/Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays/gauss_section.npy}
OUT_DIR=${OUT_DIR:-datasets/peak_slot/xi_gauss_50_120s_stride60}
ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}
FS=${FS:-1000}
DX_M=${DX_M:-100}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
STRIDE_SECONDS=${STRIDE_SECONDS:-60}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
CHANNEL_START=${CHANNEL_START:-0}
CHANNEL_COUNT=${CHANNEL_COUNT:-50}
CLIP_RATIO=${CLIP_RATIO:-1.35}
INPUT_MODE=${INPUT_MODE:-raw}
SPEED_NORM_KMH=${SPEED_NORM_KMH:-150}
X_DTYPE=${X_DTYPE:-float32}
SHARD_SIZE=${SHARD_SIZE:-256}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-64}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.5}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.02}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.segment_real_npy_to_peak_slot \
  --input "$INPUT" \
  --out-dir "$OUT_DIR" \
  --array-layout "$ARRAY_LAYOUT" \
  --fs "$FS" \
  --dx-m "$DX_M" \
  --window-seconds "$WINDOW_SECONDS" \
  --stride-seconds "$STRIDE_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --channel-start "$CHANNEL_START" \
  --channel-count "$CHANNEL_COUNT" \
  --clip-ratio "$CLIP_RATIO" \
  --input-mode "$INPUT_MODE" \
  --speed-norm-kmh "$SPEED_NORM_KMH" \
  --x-dtype "$X_DTYPE" \
  --shard-size "$SHARD_SIZE" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  $overwrite_args
