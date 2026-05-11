#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v3_120s_realistic/train}
OUT_DIR=${OUT_DIR:-models/peak_slot_v3_120s_realistic_cuda}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-1000}
BATCH_SIZE=${BATCH_SIZE:-42}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
LR=${LR:-0.0002}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
MAX_TRACKS=${MAX_TRACKS:-96}
HIDDEN_DIM=${HIDDEN_DIM:-128}
DECODER_LAYERS=${DECODER_LAYERS:-2}
NUM_HEADS=${NUM_HEADS:-4}
POOLED_CHANNELS=${POOLED_CHANNELS:-8}
POOLED_TIME=${POOLED_TIME:-128}
AMP=${AMP:-on}
AMP_DTYPE=${AMP_DTYPE:-float16}
INPUT_TRANSFER_DTYPE=${INPUT_TRANSFER_DTYPE:-auto}
MATCHER=${MATCHER:-auction}         #hungarian
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.15}
NONE_WEIGHT=${NONE_WEIGHT:-0.35}
OBJECT_LOSS_WEIGHT=${OBJECT_LOSS_WEIGHT:-1.25}
COUNT_LOSS_WEIGHT=${COUNT_LOSS_WEIGHT:-0.15}
MONOTONIC_LOSS_WEIGHT=${MONOTONIC_LOSS_WEIGHT:-1.0}
SMOOTHNESS_LOSS_WEIGHT=${SMOOTHNESS_LOSS_WEIGHT:-0.2}
TIME_PRIOR_LOSS_WEIGHT=${TIME_PRIOR_LOSS_WEIGHT:-2.0}
VISIBILITY_PRIOR_LOSS_WEIGHT=${VISIBILITY_PRIOR_LOSS_WEIGHT:-0.75}
SLOT_COMPETITION_LOSS_WEIGHT=${SLOT_COMPETITION_LOSS_WEIGHT:-0.15}
CROSSING_LOSS_WEIGHT=${CROSSING_LOSS_WEIGHT:-0.2}
METRIC_OBJECTNESS_THRESHOLD=${METRIC_OBJECTNESS_THRESHOLD:-0.35}
METRIC_POINT_THRESHOLD=${METRIC_POINT_THRESHOLD:-0.05}
VAL_DATA_DIR=${VAL_DATA_DIR:-}
VAL_FRACTION=${VAL_FRACTION:-0.1}
VAL_EVERY=${VAL_EVERY:-5}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}
METRICS_EVERY=${METRICS_EVERY:-20}
LOG_EVERY=${LOG_EVERY:-20}
TIMING_EVERY=${TIMING_EVERY:-5}
PROFILE_STEPS=${PROFILE_STEPS:-0}
PROFILE_WARMUP=${PROFILE_WARMUP:-2}
RESUME=${RESUME:-}
AUTO_RESUME=${AUTO_RESUME:-1}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}
SEED=${SEED:-42}

resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
elif [ "$AUTO_RESUME" = "1" ] || [ "$AUTO_RESUME" = "true" ]; then
  resume_args="--auto-resume"
fi

val_data_args=""
if [ -n "$VAL_DATA_DIR" ]; then
  val_data_args="--val-data-dir $VAL_DATA_DIR"
fi

loader_args="--num-workers $NUM_WORKERS --prefetch-factor $PREFETCH_FACTOR"
if [ "$PIN_MEMORY" = "1" ] || [ "$PIN_MEMORY" = "true" ]; then
  loader_args="$loader_args --pin-memory"
fi
if [ "$PERSISTENT_WORKERS" = "1" ] || [ "$PERSISTENT_WORKERS" = "true" ]; then
  loader_args="$loader_args --persistent-workers"
fi

uv run python -m autotrack.dl.train_peak_slot \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  $loader_args \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-tracks "$MAX_TRACKS" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --pooled-channels "$POOLED_CHANNELS" \
  --pooled-time "$POOLED_TIME" \
  --amp "$AMP" \
  --amp-dtype "$AMP_DTYPE" \
  --input-transfer-dtype "$INPUT_TRANSFER_DTYPE" \
  --channels-last \
  --matcher "$MATCHER" \
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --none-weight "$NONE_WEIGHT" \
  --object-loss-weight "$OBJECT_LOSS_WEIGHT" \
  --count-loss-weight "$COUNT_LOSS_WEIGHT" \
  --monotonic-loss-weight "$MONOTONIC_LOSS_WEIGHT" \
  --smoothness-loss-weight "$SMOOTHNESS_LOSS_WEIGHT" \
  --time-prior-loss-weight "$TIME_PRIOR_LOSS_WEIGHT" \
  --visibility-prior-loss-weight "$VISIBILITY_PRIOR_LOSS_WEIGHT" \
  --slot-competition-loss-weight "$SLOT_COMPETITION_LOSS_WEIGHT" \
  --crossing-loss-weight "$CROSSING_LOSS_WEIGHT" \
  --metric-objectness-threshold "$METRIC_OBJECTNESS_THRESHOLD" \
  --metric-point-threshold "$METRIC_POINT_THRESHOLD" \
  $val_data_args \
  --val-fraction "$VAL_FRACTION" \
  --val-every "$VAL_EVERY" \
  --val-max-samples "$VAL_MAX_SAMPLES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --metrics-every "$METRICS_EVERY" \
  --log-every "$LOG_EVERY" \
  --timing-every "$TIMING_EVERY" \
  --profile-steps "$PROFILE_STEPS" \
  --profile-warmup "$PROFILE_WARMUP" \
  --seed "$SEED" \
  $resume_args
