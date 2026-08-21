#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_dataset_realshape_crossing_curriculum}"
NUM_SAMPLES="${NUM_SAMPLES:-16384}"
SHARD_SIZE="${SHARD_SIZE:-128}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-20260703}"

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
  --same-direction-ratio "${SAME_DIRECTION_RATIO:-0.45}" \
  --crossing-ratio "${CROSSING_RATIO:-0.55}" \
  --scene-cluster-ratio "${SCENE_CLUSTER_RATIO:-0.45}" \
  --interaction-ratio "${INTERACTION_RATIO:-0.45}" \
  --interaction-types "${INTERACTION_TYPES:-crossing,parallel_crossing,overtake,near_parallel}" \
  --motion-mix "${MOTION_MIX:-constant_sparse,smooth_random}" \
  --motion-weights "${MOTION_WEIGHTS:-0.90,0.10}" \
  --stop-response-amp-scale 1.0 \
  --stop-response-sigma-scale 1.0 \
  --isolated-noise-ratio "${ISOLATED_NOISE_RATIO:-0.20}" \
  --isolated-noise-rate "${ISOLATED_NOISE_RATE:-6}" \
  --isolated-noise-amp-min "${GAUSS_AMP:-0.5}" \
  --isolated-noise-amp-max "${GAUSS_AMP:-0.5}" \
  --isolated-noise-sigma-min-s "${GAUSS_SIGMA_S:-0.50}" \
  --isolated-noise-sigma-max-s "${GAUSS_SIGMA_S:-0.50}" \
  --input-scale "${INPUT_SCALE:-0.50968635}" \
  --vehicles-min "${VEHICLES_MIN:-2}" \
  --vehicles-max "${VEHICLES_MAX:-8}" \
  --per-vehicle-drop-channel-ratio "${PER_VEHICLE_DROP_CHANNEL_RATIO:-0.25}" \
  --per-vehicle-drop-channel-min "${PER_VEHICLE_DROP_CHANNEL_MIN:-1}" \
  --per-vehicle-drop-channel-max "${PER_VEHICLE_DROP_CHANNEL_MAX:-5}" \
  --missing-random-ratio-min "${MISSING_RANDOM_RATIO_MIN:-0.02}" \
  --missing-random-ratio-max "${MISSING_RANDOM_RATIO_MAX:-0.10}" \
  --missing-segment-count-max "${MISSING_SEGMENT_COUNT_MAX:-1}" \
  --missing-segment-min-len "${MISSING_SEGMENT_MIN_LEN:-2}" \
  --missing-segment-max-len "${MISSING_SEGMENT_MAX_LEN:-5}" \
  --overwrite
