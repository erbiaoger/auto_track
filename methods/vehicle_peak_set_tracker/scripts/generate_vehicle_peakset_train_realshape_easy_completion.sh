#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_dataset_realshape_easy_completion}"
TMP_ROOT="${TMP_ROOT:-$RUN_DIR/.tmp_easy_completion_components}"
NUM_SAMPLES="${NUM_SAMPLES:-16384}"
SHARD_SIZE="${SHARD_SIZE:-128}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-20260703}"

EASY_NUM_SAMPLES="${EASY_NUM_SAMPLES:-$((NUM_SAMPLES * 50 / 100))}"
COMPLETION_NUM_SAMPLES="${COMPLETION_NUM_SAMPLES:-$((NUM_SAMPLES * 30 / 100))}"
REALSHAPE_NUM_SAMPLES="${REALSHAPE_NUM_SAMPLES:-$((NUM_SAMPLES - EASY_NUM_SAMPLES - COMPLETION_NUM_SAMPLES))}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
rm -rf "$TMP_ROOT"
mkdir -p "$TMP_ROOT"

common_args=(
  --shard-size "$SHARD_SIZE"
  --workers "$WORKERS"
  --sigma-min-s "${SIGMA_MIN_S:-0.42}"
  --sigma-max-s "${SIGMA_MAX_S:-0.60}"
  --dead-channel-indices "${DEAD_CHANNEL_INDICES:-5,6,15,16,22,36,38,45}"
  --intermittent-dead-channel-rates "${INTERMITTENT_DEAD_CHANNEL_RATES:-11:0.69,17:0.37,42:0.75}"
  --clip-ratio "${CLIP_RATIO:-1.35}"
  --overwrite
)

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$TMP_ROOT/easy" \
  --num-samples "$EASY_NUM_SAMPLES" \
  --seed "$SEED" \
  --vehicles-min "${EASY_VEHICLES_MIN:-2}" \
  --vehicles-max "${EASY_VEHICLES_MAX:-6}" \
  --isolated-noise-ratio "${EASY_ISOLATED_NOISE_RATIO:-0.10}" \
  --isolated-noise-rate "${EASY_ISOLATED_NOISE_RATE:-4.0}" \
  --per-vehicle-drop-channel-ratio "${EASY_PER_VEHICLE_DROP_CHANNEL_RATIO:-0.20}" \
  --per-vehicle-drop-channel-min "${EASY_PER_VEHICLE_DROP_CHANNEL_MIN:-1}" \
  --per-vehicle-drop-channel-max "${EASY_PER_VEHICLE_DROP_CHANNEL_MAX:-3}" \
  --missing-random-ratio-min "${EASY_MISSING_RANDOM_RATIO_MIN:-0.00}" \
  --missing-random-ratio-max "${EASY_MISSING_RANDOM_RATIO_MAX:-0.06}" \
  --missing-segment-count-max "${EASY_MISSING_SEGMENT_COUNT_MAX:-1}" \
  --missing-segment-min-len "${EASY_MISSING_SEGMENT_MIN_LEN:-1}" \
  --missing-segment-max-len "${EASY_MISSING_SEGMENT_MAX_LEN:-3}" \
  "${common_args[@]}"

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$TMP_ROOT/completion" \
  --num-samples "$COMPLETION_NUM_SAMPLES" \
  --seed "$((SEED + 100000))" \
  --vehicles-min "${COMPLETION_VEHICLES_MIN:-3}" \
  --vehicles-max "${COMPLETION_VEHICLES_MAX:-8}" \
  --isolated-noise-ratio "${COMPLETION_ISOLATED_NOISE_RATIO:-0.12}" \
  --isolated-noise-rate "${COMPLETION_ISOLATED_NOISE_RATE:-5.0}" \
  --per-vehicle-drop-channel-ratio "${COMPLETION_PER_VEHICLE_DROP_CHANNEL_RATIO:-0.85}" \
  --per-vehicle-drop-channel-min "${COMPLETION_PER_VEHICLE_DROP_CHANNEL_MIN:-2}" \
  --per-vehicle-drop-channel-max "${COMPLETION_PER_VEHICLE_DROP_CHANNEL_MAX:-8}" \
  --missing-random-ratio-min "${COMPLETION_MISSING_RANDOM_RATIO_MIN:-0.05}" \
  --missing-random-ratio-max "${COMPLETION_MISSING_RANDOM_RATIO_MAX:-0.20}" \
  --missing-segment-count-max "${COMPLETION_MISSING_SEGMENT_COUNT_MAX:-2}" \
  --missing-segment-min-len "${COMPLETION_MISSING_SEGMENT_MIN_LEN:-2}" \
  --missing-segment-max-len "${COMPLETION_MISSING_SEGMENT_MAX_LEN:-6}" \
  "${common_args[@]}"

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$TMP_ROOT/realshape" \
  --num-samples "$REALSHAPE_NUM_SAMPLES" \
  --seed "$((SEED + 200000))" \
  --vehicles-min "${REALSHAPE_VEHICLES_MIN:-4}" \
  --vehicles-max "${REALSHAPE_VEHICLES_MAX:-10}" \
  --isolated-noise-ratio "${REALSHAPE_ISOLATED_NOISE_RATIO:-0.30}" \
  --isolated-noise-rate "${REALSHAPE_ISOLATED_NOISE_RATE:-14.0}" \
  --per-vehicle-drop-channel-ratio "${REALSHAPE_PER_VEHICLE_DROP_CHANNEL_RATIO:-0.70}" \
  --per-vehicle-drop-channel-min "${REALSHAPE_PER_VEHICLE_DROP_CHANNEL_MIN:-2}" \
  --per-vehicle-drop-channel-max "${REALSHAPE_PER_VEHICLE_DROP_CHANNEL_MAX:-7}" \
  --missing-random-ratio-min "${REALSHAPE_MISSING_RANDOM_RATIO_MIN:-0.05}" \
  --missing-random-ratio-max "${REALSHAPE_MISSING_RANDOM_RATIO_MAX:-0.18}" \
  --missing-segment-count-max "${REALSHAPE_MISSING_SEGMENT_COUNT_MAX:-2}" \
  --missing-segment-min-len "${REALSHAPE_MISSING_SEGMENT_MIN_LEN:-2}" \
  --missing-segment-max-len "${REALSHAPE_MISSING_SEGMENT_MAX_LEN:-5}" \
  "${common_args[@]}"

uv run python "$ROOT/scripts/merge_vehicle_peakset_shards.py" \
  --out-dir "$OUT_DIR" \
  --component "easy:$TMP_ROOT/easy" \
  --component "completion:$TMP_ROOT/completion" \
  --component "realshape:$TMP_ROOT/realshape" \
  ${SHUFFLE_SHARDS:+--shuffle-shards} \
  --shuffle-seed "${SHUFFLE_SEED:-$SEED}" \
  --overwrite
