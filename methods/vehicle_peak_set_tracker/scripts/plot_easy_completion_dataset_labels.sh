#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/peakguided_dataset_realshape_easy_completion}"
OUT_DIR="${OUT_DIR:-$RUN_DIR/peakguided_dataset_realshape_easy_completion_plots}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

"$PYTHON_BIN" "$ROOT/scripts/plot_exported_peakset_dataset_labels.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR/easy" \
  --sample-index "${EASY_SAMPLE_INDEX:-0}" \
  --num-samples "${EASY_NUM_PLOTS:-3}"

"$PYTHON_BIN" "$ROOT/scripts/plot_exported_peakset_dataset_labels.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR/completion" \
  --sample-index "${COMPLETION_SAMPLE_INDEX:-8192}" \
  --num-samples "${COMPLETION_NUM_PLOTS:-3}"

"$PYTHON_BIN" "$ROOT/scripts/plot_exported_peakset_dataset_labels.py" \
  --dataset-dir "$DATASET_DIR" \
  --out-dir "$OUT_DIR/realshape" \
  --sample-index "${REALSHAPE_SAMPLE_INDEX:-13107}" \
  --num-samples "${REALSHAPE_NUM_PLOTS:-3}"
