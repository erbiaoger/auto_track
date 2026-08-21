#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_DIR/peakguided_train_realshape_missing_fixedgauss}"
MODEL="${MODEL:-$TRAIN_DIR/checkpoint_last.pt}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/test_dataset_realshape_clean_crossing}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/predict_test_crossing_switch}"

if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  bash "$ROOT/scripts/generate_vehicle_peakset_test_crossing_clean.sh"
fi

MODEL="$MODEL" DATASET_DIR="$DATASET_DIR" OUT_DIR="$OUT_DIR" PREDICT_PROFILE="${PREDICT_PROFILE:-recall}" \
  bash "$ROOT/scripts/predict_vehicle_peakset_test_crossing_cpu.sh"

"$PYTHON_BIN" "$ROOT/scripts/evaluate_vehicle_peakset_predictions.py" \
  --dataset-dir "$DATASET_DIR" \
  --predictions "$OUT_DIR/predictions.jsonl" \
  --out-json "$OUT_DIR/eval_metrics.json" \
  --switch-margin-s "${SWITCH_MARGIN_S:-0.20}" \
  --switch-track-fraction "${SWITCH_TRACK_FRACTION:-0.20}" \
  --switch-track-min-points "${SWITCH_TRACK_MIN_POINTS:-2}"
