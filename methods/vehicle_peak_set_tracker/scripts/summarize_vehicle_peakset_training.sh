#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_DIR/peakguided_train_realshape_missingrefine}"
OUT_JSON="${OUT_JSON:-$TRAIN_DIR/training_summary.json}"
OUT_MD="${OUT_MD:-$TRAIN_DIR/training_summary.md}"
PREDICTION_SUMMARY="${PREDICTION_SUMMARY:-}"

ARGS=(
  --train-dir "$TRAIN_DIR"
  --out-json "$OUT_JSON"
  --out-md "$OUT_MD"
)
if [[ -n "$PREDICTION_SUMMARY" && -f "$PREDICTION_SUMMARY" ]]; then
  ARGS+=(--prediction-summary "$PREDICTION_SUMMARY")
fi

"$PYTHON_BIN" scripts/summarize_vehicle_peakset_training.py "${ARGS[@]}"
