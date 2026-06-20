#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v5_120s_lineonly/train}
VAL_DATA_DIR=${VAL_DATA_DIR:-datasets/peak_slot_v5_120s_lineonly/test}
OUT_DIR=${OUT_DIR:-models/peak_slot_v5_120s_lineonly_cuda}
RESUME=${RESUME:-models/peak_slot_profile_only_120s/checkpoint_best.pt}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-80}
BATCH_SIZE=${BATCH_SIZE:-32}
MAX_SAMPLES=${MAX_SAMPLES:-0}
NUM_WORKERS=${NUM_WORKERS:-16}
PIN_MEMORY=${PIN_MEMORY:-1}
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}
WORKER_SHARD_CACHE_SIZE=${WORKER_SHARD_CACHE_SIZE:-2}
LR=${LR:-0.00005}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}
NONE_WEIGHT=${NONE_WEIGHT:-0.25}
GT_COVERAGE_LOSS_WEIGHT=${GT_COVERAGE_LOSS_WEIGHT:-0.8}
VISIBILITY_PRIOR_LOSS_WEIGHT=${VISIBILITY_PRIOR_LOSS_WEIGHT:-1.0}
VAL_EVERY=${VAL_EVERY:-1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1}
LOG_EVERY=${LOG_EVERY:-100}
TIMING_EVERY=${TIMING_EVERY:-1}
AUTO_RESUME=${AUTO_RESUME:-0}
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-1}
AMP=${AMP:-on}
AMP_DTYPE=${AMP_DTYPE:-float16}
INPUT_TRANSFER_DTYPE=${INPUT_TRANSFER_DTYPE:-auto}
CHANNELS_LAST=${CHANNELS_LAST:-1}
SEED=${SEED:-22}

resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
elif [ "$AUTO_RESUME" = "1" ] || [ "$AUTO_RESUME" = "true" ]; then
  resume_args="--auto-resume"
fi

loader_args="--num-workers $NUM_WORKERS --prefetch-factor $PREFETCH_FACTOR --worker-shard-cache-size $WORKER_SHARD_CACHE_SIZE"
if [ "$PIN_MEMORY" = "1" ] || [ "$PIN_MEMORY" = "true" ]; then
  loader_args="$loader_args --pin-memory"
fi
if [ "$PERSISTENT_WORKERS" = "1" ] || [ "$PERSISTENT_WORKERS" = "true" ]; then
  loader_args="$loader_args --persistent-workers"
fi

layout_args=""
if [ "$CHANNELS_LAST" = "1" ] || [ "$CHANNELS_LAST" = "true" ]; then
  layout_args="--channels-last"
fi

uv run python -m autotrack.dl.train_peak_slot \
  --data-dir "$DATA_DIR" \
  --val-data-dir "$VAL_DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  $loader_args \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --none-weight "$NONE_WEIGHT" \
  --gt-coverage-loss-weight "$GT_COVERAGE_LOSS_WEIGHT" \
  --visibility-prior-loss-weight "$VISIBILITY_PRIOR_LOSS_WEIGHT" \
  --amp "$AMP" \
  --amp-dtype "$AMP_DTYPE" \
  --input-transfer-dtype "$INPUT_TRANSFER_DTYPE" \
  --val-every "$VAL_EVERY" \
  --val-max-samples "$VAL_MAX_SAMPLES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --log-every "$LOG_EVERY" \
  --timing-every "$TIMING_EVERY" \
  --seed "$SEED" \
  $layout_args \
  $resume_args
