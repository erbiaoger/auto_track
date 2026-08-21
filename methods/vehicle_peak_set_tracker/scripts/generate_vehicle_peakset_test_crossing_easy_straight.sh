#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/test_dataset_realshape_easy_straight}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
SHARD_SIZE="${SHARD_SIZE:-8}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-20260705}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --amp-min "${GAUSS_AMP:-0.5}" \
  --amp-max "${GAUSS_AMP:-0.5}" \
  --sigma-min-s "${GAUSS_SIGMA_S:-0.50}" \
  --sigma-max-s "${GAUSS_SIGMA_S:-0.50}" \
  --same-direction-ratio "${SAME_DIRECTION_RATIO:-0.90}" \
  --crossing-ratio "${CROSSING_RATIO:-0.10}" \
  --scene-cluster-ratio "${SCENE_CLUSTER_RATIO:-0.20}" \
  --interaction-ratio "${INTERACTION_RATIO:-0.15}" \
  --interaction-types "${INTERACTION_TYPES:-parallel_crossing,overtake,near_parallel}" \
  --motion-mix "${MOTION_MIX:-constant_sparse,smooth_random}" \
  --motion-weights "${MOTION_WEIGHTS:-0.95,0.05}" \
  --stop-response-amp-scale 1.0 \
  --stop-response-sigma-scale 1.0 \
  --isolated-noise-ratio "${ISOLATED_NOISE_RATIO:-0.10}" \
  --isolated-noise-rate "${ISOLATED_NOISE_RATE:-3}" \
  --isolated-noise-amp-min "${GAUSS_AMP:-0.5}" \
  --isolated-noise-amp-max "${GAUSS_AMP:-0.5}" \
  --isolated-noise-sigma-min-s "${GAUSS_SIGMA_S:-0.50}" \
  --isolated-noise-sigma-max-s "${GAUSS_SIGMA_S:-0.50}" \
  --input-scale "${INPUT_SCALE:-0.50968635}" \
  --vehicles-min "${VEHICLES_MIN:-2}" \
  --vehicles-max "${VEHICLES_MAX:-6}" \
  --per-vehicle-drop-channel-ratio "${PER_VEHICLE_DROP_CHANNEL_RATIO:-0.05}" \
  --per-vehicle-drop-channel-min "${PER_VEHICLE_DROP_CHANNEL_MIN:-0}" \
  --per-vehicle-drop-channel-max "${PER_VEHICLE_DROP_CHANNEL_MAX:-2}" \
  --missing-random-ratio-min "${MISSING_RANDOM_RATIO_MIN:-0.00}" \
  --missing-random-ratio-max "${MISSING_RANDOM_RATIO_MAX:-0.02}" \
  --missing-segment-count-max "${MISSING_SEGMENT_COUNT_MAX:-0}" \
  --missing-segment-min-len "${MISSING_SEGMENT_MIN_LEN:-2}" \
  --missing-segment-max-len "${MISSING_SEGMENT_MAX_LEN:-4}" \
  --overwrite
