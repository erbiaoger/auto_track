#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/test_dataset_realshape_missing_fixedgauss}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"
SHARD_SIZE="${SHARD_SIZE:-8}"
WORKERS="${WORKERS:-32}"
SEED="${SEED:-20260630}"

GAUSS_AMP="${GAUSS_AMP:-0.5}"
GAUSS_SIGMA_S="${GAUSS_SIGMA_S:-0.50}"
INPUT_SCALE="${INPUT_SCALE:-0.50968635}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --include-raw-window \
  --amp-min "$GAUSS_AMP" \
  --amp-max "$GAUSS_AMP" \
  --sigma-min-s "$GAUSS_SIGMA_S" \
  --sigma-max-s "$GAUSS_SIGMA_S" \
  --motion-mix "${MOTION_MIX:-constant_sparse,smooth_random}" \
  --motion-weights "${MOTION_WEIGHTS:-0.85,0.15}" \
  --stop-response-amp-scale 1.0 \
  --stop-response-sigma-scale 1.0 \
  --isolated-noise-amp-min "$GAUSS_AMP" \
  --isolated-noise-amp-max "$GAUSS_AMP" \
  --isolated-noise-sigma-min-s "$GAUSS_SIGMA_S" \
  --isolated-noise-sigma-max-s "$GAUSS_SIGMA_S" \
  --input-scale "$INPUT_SCALE" \
  --per-vehicle-drop-channel-ratio "${PER_VEHICLE_DROP_CHANNEL_RATIO:-0.70}" \
  --per-vehicle-drop-channel-min "${PER_VEHICLE_DROP_CHANNEL_MIN:-2}" \
  --per-vehicle-drop-channel-max "${PER_VEHICLE_DROP_CHANNEL_MAX:-7}" \
  --missing-random-ratio-min "${MISSING_RANDOM_RATIO_MIN:-0.05}" \
  --missing-random-ratio-max "${MISSING_RANDOM_RATIO_MAX:-0.18}" \
  --missing-segment-count-max "${MISSING_SEGMENT_COUNT_MAX:-2}" \
  --missing-segment-min-len "${MISSING_SEGMENT_MIN_LEN:-2}" \
  --missing-segment-max-len "${MISSING_SEGMENT_MAX_LEN:-5}" \
  --overwrite
