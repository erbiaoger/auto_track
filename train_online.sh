#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DECODER_LAYERS=${DECODER_LAYERS:-2}
MAX_QUERIES=${MAX_QUERIES:-128}
POOLED_TIME=${POOLED_TIME:-128}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-2000}
VAL_STEPS=${VAL_STEPS:-200}
VAL_EVERY=${VAL_EVERY:-1}
PLOT_EVERY=${PLOT_EVERY:-10}
PLOT_WINDOW_SECONDS=${PLOT_WINDOW_SECONDS:-240}
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.02}
LOG_EVERY=${LOG_EVERY:-0}

# Mirrors the main offline dataset distribution:
# - 360 vehicles/hour = about 6 vehicles/minute on average.
# - The online generator samples per-window vehicle counts directly, so use a
#   broad 0-24 range to include empty, sparse, dense, and boundary-truncated windows.
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
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --batch-size "$BATCH_SIZE" \
  --window-seconds 60 \
  --fs 1000 \
  --time-downsample 10 \
  --n-ch 50 \
  --dx-m 100 \
  --vehicles-min 0 \
  --vehicles-max 24 \
  --speed-min-kmh 60 \
  --speed-max-kmh 90 \
  --noise-std 0.0 \
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
  --num-workers "$NUM_WORKERS" \
  --log-every "$LOG_EVERY"
