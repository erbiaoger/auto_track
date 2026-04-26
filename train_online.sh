#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-16}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DECODER_LAYERS=${DECODER_LAYERS:-2}
MAX_QUERIES=${MAX_QUERIES:-128}
POOLED_TIME=${POOLED_TIME:-128}
TRAJECTORY_POINTS=${TRAJECTORY_POINTS:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2000}
VAL_STEPS=${VAL_STEPS:-200}
VAL_EVERY=${VAL_EVERY:-1}
PLOT_EVERY=${PLOT_EVERY:-2}
PLOT_WINDOW_SECONDS=${PLOT_WINDOW_SECONDS:-240}
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.3}
DUPLICATE_LOSS_WEIGHT=${DUPLICATE_LOSS_WEIGHT:-0.2}
DUPLICATE_DISTANCE_TAU=${DUPLICATE_DISTANCE_TAU:-0.04}
DENOISING_LOSS_WEIGHT=${DENOISING_LOSS_WEIGHT:-1.0}
DENOISING_QUERIES=${DENOISING_QUERIES:-32}
DN_POINT_NOISE=${DN_POINT_NOISE:-0.04}
LINE_LOSS_WEIGHT=${LINE_LOSS_WEIGHT:-1.0}
SLOPE_SMOOTH_LOSS_WEIGHT=${SLOPE_SMOOTH_LOSS_WEIGHT:-0.25}
PLOT_OBJECTNESS_THRESHOLD=${PLOT_OBJECTNESS_THRESHOLD:-0.5}
PLOT_VISIBILITY_THRESHOLD=${PLOT_VISIBILITY_THRESHOLD:-0.6}
PLOT_TOP_K=${PLOT_TOP_K:-128}
PLOT_DISPLAY_FLOOR=${PLOT_DISPLAY_FLOOR:-0.08}
LOG_EVERY=${LOG_EVERY:-0}
VEHICLES_MIN=${VEHICLES_MIN:-32}
VEHICLES_MAX=${VEHICLES_MAX:-48}
NOISE_STD=${NOISE_STD:-0.0}
SPEED_MIN_KMH=${SPEED_MIN_KMH:-70}
SPEED_MAX_KMH=${SPEED_MAX_KMH:-85}
SPEED_OUTLIER_RATIO=${SPEED_OUTLIER_RATIO:-0.12}
SLOW_SPEED_MIN_KMH=${SLOW_SPEED_MIN_KMH:-45}
SLOW_SPEED_MAX_KMH=${SLOW_SPEED_MAX_KMH:-60}
FAST_SPEED_MIN_KMH=${FAST_SPEED_MIN_KMH:-95}
FAST_SPEED_MAX_KMH=${FAST_SPEED_MAX_KMH:-120}

# Mirrors the main offline dataset distribution:
# - 240 s windows use about 40 visible vehicles by default.
# - Most vehicles are nearly constant-speed in a narrow typical range; a small
#   fraction are sampled from slow/fast outlier speed ranges.
# - Current online generator is constant-speed only; use offline finetuning for
#   accel/decel/stop-go after this fast pretraining.
uv run python train_trajectory_online.py \
  --out-dir models/trajectory_query_online_v1_cuda \
  --device "$DEVICE" \
  --epochs 400 \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --val-steps "$VAL_STEPS" \
  --val-every "$VAL_EVERY" \
  --plot-every "$PLOT_EVERY" \
  --plot-window-seconds "$PLOT_WINDOW_SECONDS" \
  --plot-objectness-threshold "$PLOT_OBJECTNESS_THRESHOLD" \
  --plot-visibility-threshold "$PLOT_VISIBILITY_THRESHOLD" \
  --plot-top-k "$PLOT_TOP_K" \
  --plot-display-floor "$PLOT_DISPLAY_FLOOR" \
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --duplicate-loss-weight "$DUPLICATE_LOSS_WEIGHT" \
  --duplicate-distance-tau "$DUPLICATE_DISTANCE_TAU" \
  --denoising-loss-weight "$DENOISING_LOSS_WEIGHT" \
  --line-loss-weight "$LINE_LOSS_WEIGHT" \
  --slope-smooth-loss-weight "$SLOPE_SMOOTH_LOSS_WEIGHT" \
  --batch-size "$BATCH_SIZE" \
  --window-seconds 240 \
  --fs 1000 \
  --time-downsample 10 \
  --n-ch 50 \
  --dx-m 100 \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh "$SPEED_MIN_KMH" \
  --speed-max-kmh "$SPEED_MAX_KMH" \
  --speed-outlier-ratio "$SPEED_OUTLIER_RATIO" \
  --slow-speed-min-kmh "$SLOW_SPEED_MIN_KMH" \
  --slow-speed-max-kmh "$SLOW_SPEED_MAX_KMH" \
  --fast-speed-min-kmh "$FAST_SPEED_MIN_KMH" \
  --fast-speed-max-kmh "$FAST_SPEED_MAX_KMH" \
  --noise-std "$NOISE_STD" \
  --amp-min 6.0 \
  --amp-max 6.0 \
  --sigma-min-s 0.06 \
  --sigma-max-s 0.18 \
  --primary-ratio 0.8333333333 \
  --max-queries "$MAX_QUERIES" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --num-heads 4 \
  --pooled-channels 8 \
  --pooled-time "$POOLED_TIME" \
  --trajectory-points "$TRAJECTORY_POINTS" \
  --denoising-queries "$DENOISING_QUERIES" \
  --dn-point-noise "$DN_POINT_NOISE" \
  --num-workers "$NUM_WORKERS" \
  --log-every "$LOG_EVERY"
