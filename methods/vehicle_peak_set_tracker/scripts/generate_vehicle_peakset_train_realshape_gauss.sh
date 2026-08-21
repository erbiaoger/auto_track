#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_dataset_realshape_gauss}"
NUM_SAMPLES="${NUM_SAMPLES:-16384}"
SHARD_SIZE="${SHARD_SIZE:-128}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-20260630}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT_DIR"

uv run python "$ROOT/scripts/export_vehicle_peakset_dataset_realshape.py" \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --overwrite
