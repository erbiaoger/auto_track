#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

OUT_DIR=${OUT_DIR:-datasets/track_slot/train}
NUM_SAMPLES=${NUM_SAMPLES:-20000}
SHARD_SIZE=${SHARD_SIZE:-256}
N_CH=${N_CH:-50}
FS=${FS:-1000}
WINDOW_SECONDS=${WINDOW_SECONDS:-240}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
DX_M=${DX_M:-100}
VEHICLES_MIN=${VEHICLES_MIN:-32}
VEHICLES_MAX=${VEHICLES_MAX:-48}
SPEED_MIN_KMH=${SPEED_MIN_KMH:-70}
SPEED_MAX_KMH=${SPEED_MAX_KMH:-85}
NOISE_STD=${NOISE_STD:-0.0}
AMP_MIN=${AMP_MIN:-6.0}
AMP_MAX=${AMP_MAX:-6.0}
SIGMA_SECONDS=${SIGMA_SECONDS:-0.25}
PRIMARY_RATIO=${PRIMARY_RATIO:-0.8333333333}
INPUT_MODE=${INPUT_MODE:-raw}
X_DTYPE=${X_DTYPE:-float16}
WORKERS=${WORKERS:-8}
SEED=${SEED:-42}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_track_slot_dataset \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --n-ch "$N_CH" \
  --fs "$FS" \
  --window-seconds "$WINDOW_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --dx-m "$DX_M" \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh "$SPEED_MIN_KMH" \
  --speed-max-kmh "$SPEED_MAX_KMH" \
  --noise-std "$NOISE_STD" \
  --amp-min "$AMP_MIN" \
  --amp-max "$AMP_MAX" \
  --sigma-min-s "$SIGMA_SECONDS" \
  --sigma-max-s "$SIGMA_SECONDS" \
  --primary-ratio "$PRIMARY_RATIO" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  $overwrite_args
