#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_DIR/peakguided_train_realshape_missingrefine}"
HISTORY="${HISTORY:-$TRAIN_DIR/train_history.jsonl}"
OUT="${OUT:-$TRAIN_DIR/loss_curve.png}"

"$PYTHON_BIN" scripts/plot_vehicle_peakset_loss.py \
  --history "$HISTORY" \
  --out "$OUT"
