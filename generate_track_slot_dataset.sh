#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

OUT_DIR=${OUT_DIR:-datasets/track_slot_v2/train}
NUM_SAMPLES=${NUM_SAMPLES:-40000}
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
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.84,0.15,0.01}
CONSTANT_PERTURB_PROB=${CONSTANT_PERTURB_PROB:-0.05}
CONSTANT_PERTURB_MAX_FRAC=${CONSTANT_PERTURB_MAX_FRAC:-0.01}
CONSTANT_PERTURB_WIDTH_MIN=${CONSTANT_PERTURB_WIDTH_MIN:-1}
CONSTANT_PERTURB_WIDTH_MAX=${CONSTANT_PERTURB_WIDTH_MAX:-2}
SMOOTH_SPEED_MAX_FRAC=${SMOOTH_SPEED_MAX_FRAC:-0.05}
SMOOTH_SPEED_CORR_CHANNELS=${SMOOTH_SPEED_CORR_CHANNELS:-8}
STOP_DURATION_MIN_S=${STOP_DURATION_MIN_S:-1.0}
STOP_DURATION_MAX_S=${STOP_DURATION_MAX_S:-8.0}
STOP_CHANNEL_WIDTH_MIN=${STOP_CHANNEL_WIDTH_MIN:-1}
STOP_CHANNEL_WIDTH_MAX=${STOP_CHANNEL_WIDTH_MAX:-3}
STOP_RESPONSE_SIGMA_SCALE=${STOP_RESPONSE_SIGMA_SCALE:-3.0}
STOP_RESPONSE_AMP_SCALE=${STOP_RESPONSE_AMP_SCALE:-1.2}
RESTART_SPEED_RATIO_MIN=${RESTART_SPEED_RATIO_MIN:-0.95}
RESTART_SPEED_RATIO_MAX=${RESTART_SPEED_RATIO_MAX:-1.05}
NOISE_STD=${NOISE_STD:-0.0}
AMP_MIN=${AMP_MIN:-6.0}
AMP_MAX=${AMP_MAX:-6.0}
SIGMA_SECONDS=${SIGMA_SECONDS:-0.25}
PRIMARY_RATIO=${PRIMARY_RATIO:-0.8333333333}
INTERACTION_RATIO=${INTERACTION_RATIO:-0.3}
INTERACTION_TYPES=${INTERACTION_TYPES:-crossing,overtake,near_parallel}
INTERACTION_TIME_MIN_FRAC=${INTERACTION_TIME_MIN_FRAC:-0.05}
INTERACTION_TIME_MAX_FRAC=${INTERACTION_TIME_MAX_FRAC:-0.95}
ISOLATED_NOISE_RATIO=${ISOLATED_NOISE_RATIO:-0.1}
ISOLATED_NOISE_RATE=${ISOLATED_NOISE_RATE:-6.0}
ISOLATED_NOISE_AMP_MIN=${ISOLATED_NOISE_AMP_MIN:-4.0}
ISOLATED_NOISE_AMP_MAX=${ISOLATED_NOISE_AMP_MAX:-8.0}
ISOLATED_NOISE_SIGMA_MIN=${ISOLATED_NOISE_SIGMA_MIN:-0.08}
ISOLATED_NOISE_SIGMA_MAX=${ISOLATED_NOISE_SIGMA_MAX:-0.35}
INPUT_MODE=${INPUT_MODE:-raw}
X_DTYPE=${X_DTYPE:-float16}
WORKERS=${WORKERS:-32}
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
  --motion-mix "$MOTION_MIX" \
  --motion-weights "$MOTION_WEIGHTS" \
  --constant-perturb-prob "$CONSTANT_PERTURB_PROB" \
  --constant-perturb-max-frac "$CONSTANT_PERTURB_MAX_FRAC" \
  --constant-perturb-width-min "$CONSTANT_PERTURB_WIDTH_MIN" \
  --constant-perturb-width-max "$CONSTANT_PERTURB_WIDTH_MAX" \
  --smooth-speed-max-frac "$SMOOTH_SPEED_MAX_FRAC" \
  --smooth-speed-corr-channels "$SMOOTH_SPEED_CORR_CHANNELS" \
  --stop-duration-min-s "$STOP_DURATION_MIN_S" \
  --stop-duration-max-s "$STOP_DURATION_MAX_S" \
  --stop-channel-width-min "$STOP_CHANNEL_WIDTH_MIN" \
  --stop-channel-width-max "$STOP_CHANNEL_WIDTH_MAX" \
  --stop-response-sigma-scale "$STOP_RESPONSE_SIGMA_SCALE" \
  --stop-response-amp-scale "$STOP_RESPONSE_AMP_SCALE" \
  --restart-speed-ratio-min "$RESTART_SPEED_RATIO_MIN" \
  --restart-speed-ratio-max "$RESTART_SPEED_RATIO_MAX" \
  --noise-std "$NOISE_STD" \
  --amp-min "$AMP_MIN" \
  --amp-max "$AMP_MAX" \
  --sigma-min-s "$SIGMA_SECONDS" \
  --sigma-max-s "$SIGMA_SECONDS" \
  --primary-ratio "$PRIMARY_RATIO" \
  --interaction-ratio "$INTERACTION_RATIO" \
  --interaction-types "$INTERACTION_TYPES" \
  --interaction-time-min-frac "$INTERACTION_TIME_MIN_FRAC" \
  --interaction-time-max-frac "$INTERACTION_TIME_MAX_FRAC" \
  --isolated-noise-ratio "$ISOLATED_NOISE_RATIO" \
  --isolated-noise-rate "$ISOLATED_NOISE_RATE" \
  --isolated-noise-amp-min "$ISOLATED_NOISE_AMP_MIN" \
  --isolated-noise-amp-max "$ISOLATED_NOISE_AMP_MAX" \
  --isolated-noise-sigma-min "$ISOLATED_NOISE_SIGMA_MIN" \
  --isolated-noise-sigma-max "$ISOLATED_NOISE_SIGMA_MAX" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  $overwrite_args
