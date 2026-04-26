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
PLOT_EVERY=${PLOT_EVERY:-10}
PLOT_WINDOW_SECONDS=${PLOT_WINDOW_SECONDS:-240}
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.05}
PLOT_OBJECTNESS_THRESHOLD=${PLOT_OBJECTNESS_THRESHOLD:-0.35}
PLOT_VISIBILITY_THRESHOLD=${PLOT_VISIBILITY_THRESHOLD:-0.5}
PLOT_TOP_K=${PLOT_TOP_K:-20}
PLOT_DISPLAY_FLOOR=${PLOT_DISPLAY_FLOOR:-0.08}
LOG_EVERY=${LOG_EVERY:-0}
VEHICLES_MIN=${VEHICLES_MIN:-1}
VEHICLES_MAX=${VEHICLES_MAX:-8}
NOISE_STD=${NOISE_STD:-0.0}

# Mirrors the main offline dataset distribution:
# - 360 vehicles/hour = about 6 vehicles/minute on average.
# - Start with a simpler 1-8 vehicle/window curriculum. After the model predicts
#   visible tracks, increase VEHICLES_MAX toward 24 for dense-window finetuning.
# - Current online generator is constant-speed only; use offline finetuning for
#   accel/decel/stop-go after this fast pretraining.
uv run python train_trajectory_online.py \
  --out-dir models/trajectory_query_online_v1_cuda \
  --device "$DEVICE" \
  --epochs 200 \
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
  --batch-size "$BATCH_SIZE" \
  --window-seconds 240 \
  --fs 1000 \
  --time-downsample 10 \
  --n-ch 50 \
  --dx-m 100 \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh 60 \
  --speed-max-kmh 90 \
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
  --num-workers "$NUM_WORKERS" \
  --log-every "$LOG_EVERY"
