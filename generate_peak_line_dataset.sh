#!/usr/bin/env sh
set -eu

OUT_DIR=${OUT_DIR:-datasets/peak_line/train}
NUM_SAMPLES=${NUM_SAMPLES:-20000}
SHARD_SIZE=${SHARD_SIZE:-256}
N_CH=${N_CH:-50}
FS=${FS:-1000}
WINDOW_SECONDS=${WINDOW_SECONDS:-240}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-256}
IMAGE_WIDTH=${IMAGE_WIDTH:-1024}
VEHICLES_MIN=${VEHICLES_MIN:-32}
VEHICLES_MAX=${VEHICLES_MAX:-48}
WORKERS=${WORKERS:-8}
SEED=${SEED:-42}
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.84,0.15,0.01}
POINT_DROP_PROB=${POINT_DROP_PROB:-0.10}
NEARBY_DISTRACTOR_PROB=${NEARBY_DISTRACTOR_PROB:-0.20}
NEARBY_DISTRACTOR_COUNT_MAX=${NEARBY_DISTRACTOR_COUNT_MAX:-2}
FALSE_PEAK_PROB_PER_CHANNEL=${FALSE_PEAK_PROB_PER_CHANNEL:-0.08}
INPUT_NOISE_STD=${INPUT_NOISE_STD:-0.0}
LINE_WIDTH=${LINE_WIDTH:-1}
POINT_WIDTH=${POINT_WIDTH:-2}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_peak_line_dataset \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --n-ch "$N_CH" \
  --fs "$FS" \
  --window-seconds "$WINDOW_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --image-height "$IMAGE_HEIGHT" \
  --image-width "$IMAGE_WIDTH" \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --motion-mix "$MOTION_MIX" \
  --motion-weights "$MOTION_WEIGHTS" \
  --point-drop-prob "$POINT_DROP_PROB" \
  --nearby-distractor-prob "$NEARBY_DISTRACTOR_PROB" \
  --nearby-distractor-count-max "$NEARBY_DISTRACTOR_COUNT_MAX" \
  --false-peak-prob-per-channel "$FALSE_PEAK_PROB_PER_CHANNEL" \
  --input-noise-std "$INPUT_NOISE_STD" \
  --line-width "$LINE_WIDTH" \
  --point-width "$POINT_WIDTH" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  $overwrite_args
