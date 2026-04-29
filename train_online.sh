#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-16}
AMP=${AMP:-auto}
AMP_DTYPE=${AMP_DTYPE:-float16}
EPOCHS=${EPOCHS:-400}
INPUT_MODE=${INPUT_MODE:-auto}
RESUME=${RESUME:-models/trajectory_query_online_v1_cuda/checkpoint_last.pt}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
CACHE_DATASET=${CACHE_DATASET:-1}
CACHE_DTYPE=${CACHE_DTYPE:-float16}
CACHE_BUILD_WORKERS=${CACHE_BUILD_WORKERS:-64}
NUM_WORKERS=${NUM_WORKERS:-0}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DECODER_LAYERS=${DECODER_LAYERS:-2}
MAX_QUERIES=${MAX_QUERIES:-128}
POOLED_TIME=${POOLED_TIME:-128}
TRAJECTORY_POINTS=${TRAJECTORY_POINTS:-32}
STEPS_PER_EPOCH=${STEPS_PER_EPOCH:-10000}
VAL_STEPS=${VAL_STEPS:-100}
VAL_EVERY=${VAL_EVERY:-10}
PLOT_EVERY=${PLOT_EVERY:-2}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-10}
METRICS_EVERY=${METRICS_EVERY:-50}
METRIC_OBJECTNESS_THRESHOLD=${METRIC_OBJECTNESS_THRESHOLD:-0.5}
METRIC_POINT_THRESHOLD=${METRIC_POINT_THRESHOLD:-0.05}
MATCHER=${MATCHER:-greedy}       # hungarian, greedy
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
# - steps-per-epoch is a fixed synthetic window pool; each epoch shuffles and
#   revisits that same pool instead of generating an unbounded random stream.
# - CACHE_DATASET=1 precomputes that fixed pool into RAM. Keep NUM_WORKERS=0 to
#   avoid copying the in-memory cache into multiple DataLoader worker processes.
# - CACHE_BUILD_WORKERS only affects the startup cache-build stage. Training
#   still uses the single in-process RAM cache after generation finishes.
# - AMP=auto enables CUDA mixed precision by default. Batch size stays at 16 by
#   default because the 4090 was already close to full memory.
# - INPUT_MODE=auto uses raw for new models and preserves raw_abs for old
#   two-channel checkpoints. Set INPUT_MODE=raw for a new one-channel model.
# - CHECKPOINT_EVERY=10 and VAL_EVERY=10 reduce per-epoch CPU/disk overhead;
#   PLOT_EVERY=2 keeps frequent visual feedback.
# - METRICS_EVERY=50 avoids synchronizing detailed GPU metrics on every batch.
# - METRIC_* controls vehicle-level precision/recall/F1 reporting only.
# - MATCHER=greedy uses faster approximate GPU-side assignment. Set
#   MATCHER=hungarian for exact CPU Hungarian assignment.
# - RESUME=/path/to/checkpoint_last.pt continues from a saved epoch. EPOCHS is
#   the final target epoch, so epoch 118 with EPOCHS=400 continues at 119.
# - Current online generator is constant-speed only; use offline finetuning for
#   accel/decel/stop-go after this fast pretraining.
cache_args=""
if [ "$CACHE_DATASET" = "1" ] || [ "$CACHE_DATASET" = "true" ]; then
  cache_args="--cache-dataset --cache-dtype $CACHE_DTYPE --cache-build-workers $CACHE_BUILD_WORKERS"
fi
resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
fi

uv run python -m autotrack.dl.train_trajectory_online \
  --out-dir models/trajectory_query_online_v1_cuda \
  --device "$DEVICE" \
  --amp "$AMP" \
  --amp-dtype "$AMP_DTYPE" \
  --input-mode "$INPUT_MODE" \
  --epochs "$EPOCHS" \
  --steps-per-epoch "$STEPS_PER_EPOCH" \
  --val-steps "$VAL_STEPS" \
  --val-every "$VAL_EVERY" \
  --plot-every "$PLOT_EVERY" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --metrics-every "$METRICS_EVERY" \
  --metric-objectness-threshold "$METRIC_OBJECTNESS_THRESHOLD" \
  --metric-point-threshold "$METRIC_POINT_THRESHOLD" \
  --matcher "$MATCHER" \
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
  $cache_args \
  $resume_args \
  --log-every "$LOG_EVERY"
