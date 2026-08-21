#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/src"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python3}"

RUN_DIR="${RUN_DIR:-$ROOT/results/vehicle_peakset_run}"
TRAIN_DIR="${TRAIN_DIR:-$RUN_DIR/peakguided_train_realshape_missing_fixedgauss}"
MODEL="${MODEL:-${1:-}}"
if [[ -z "$MODEL" ]]; then
  if [[ -f "$TRAIN_DIR/checkpoint_last.pt" ]]; then
    MODEL="$TRAIN_DIR/checkpoint_last.pt"
  elif [[ -f "$TRAIN_DIR/checkpoint_best.pt" ]]; then
    MODEL="$TRAIN_DIR/checkpoint_best.pt"
  else
    MODEL="$RUN_DIR/checkpoint_best.pt"
  fi
fi
OUT_DIR="${OUT_DIR:-${2:-$RUN_DIR/predict_test_crossing_cpu}}"
DATASET_DIR="${DATASET_DIR:-$RUN_DIR/test_dataset_realshape_clean_crossing}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PLOT_SAMPLES="${PLOT_SAMPLES:-12}"
PLOT_STYLE="${PLOT_STYLE:-waveform}"

export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$MODEL" ]]; then
  echo "checkpoint not found: $MODEL" >&2
  exit 2
fi
if [[ ! -f "$DATASET_DIR/meta.json" ]]; then
  echo "dataset meta not found: $DATASET_DIR/meta.json" >&2
  echo "generate it with: bash $ROOT/scripts/generate_vehicle_peakset_test_crossing_clean.sh" >&2
  exit 3
fi
mkdir -p "$OUT_DIR"

ARGS=(
  --model "$MODEL"
  --dataset-dir "$DATASET_DIR"
  --out-dir "$OUT_DIR"
  --device cpu
  --sample-index "$SAMPLE_INDEX"
  --num-samples "$NUM_SAMPLES"
  --batch-size "$BATCH_SIZE"
  --plot-samples "$PLOT_SAMPLES"
  --plot-style "$PLOT_STYLE"
  --objectness-threshold "${OBJECTNESS_THRESHOLD:-0.25}"
  --complete-valid-threshold "${COMPLETE_VALID_THRESHOLD:-0.35}"
  --observed-threshold "${OBSERVED_THRESHOLD:-0.45}"
  --min-visible-channels "${MIN_VISIBLE_CHANNELS:-5}"
  --min-peak-support-channels "${MIN_PEAK_SUPPORT_CHANNELS:-2}"
  --min-peak-support-ratio "${MIN_PEAK_SUPPORT_RATIO:-0.10}"
  --fused-min-anchor-score "${FUSED_MIN_ANCHOR_SCORE:-0.30}"
  --max-tracks "${MAX_TRACKS:-24}"
  --dedup-line-tolerance-s "${DEDUP_LINE_TOLERANCE_S:-1.0}"
  --dedup-loose-line-tolerance-s "${DEDUP_LOOSE_LINE_TOLERANCE_S:-2.0}"
  --dedup-speed-tolerance-kmh "${DEDUP_SPEED_TOLERANCE_KMH:-12.0}"
  --min-track-observed-ratio "${MIN_TRACK_OBSERVED_RATIO:-0.15}"
  --min-track-total-score "${MIN_TRACK_TOTAL_SCORE:-4.0}"
  --anchor-path-refine
  --anchor-path-top-k "${ANCHOR_PATH_TOP_K:-5}"
  --anchor-path-transition-weight "${ANCHOR_PATH_TRANSITION_WEIGHT:-8.0}"
  --anchor-path-second-diff-weight "${ANCHOR_PATH_SECOND_DIFF_WEIGHT:-2.5}"
  --anchor-path-baseline-weight "${ANCHOR_PATH_BASELINE_WEIGHT:-1.5}"
  --anchor-path-robust-baseline-weight "${ANCHOR_PATH_ROBUST_BASELINE_WEIGHT:-2.5}"
  --anchor-path-robust-baseline-tolerance-s "${ANCHOR_PATH_ROBUST_BASELINE_TOLERANCE_S:-2.0}"
  --anchor-path-robust-baseline-min-support "${ANCHOR_PATH_ROBUST_BASELINE_MIN_SUPPORT:-4}"
  --anchor-path-max-adjacent-jump-s "${ANCHOR_PATH_MAX_ADJACENT_JUMP_S:-8.0}"
  --observed-outlier-repair
  --observed-outlier-residual-s "${OBSERVED_OUTLIER_RESIDUAL_S:-4.0}"
  --observed-outlier-adjacent-s "${OBSERVED_OUTLIER_ADJACENT_S:-6.0}"
  --snap-to-candidates
  --snap-tolerance-s "${SNAP_TOLERANCE_S:-0.25}"
  --postprocess
  --postprocess-refit-missing
  --no-postprocess-refit-observed
  --postprocess-repair-outlier-points
  --postprocess-drop-point-residual-s "${POSTPROCESS_DROP_POINT_RESIDUAL_S:-8.0}"
  --postprocess-repair-point-residual-s "${POSTPROCESS_REPAIR_POINT_RESIDUAL_S:-4.0}"
  --postprocess-drop-point-second-diff-s "${POSTPROCESS_DROP_POINT_SECOND_DIFF_S:-6.0}"
  --postprocess-extend-missing
  --postprocess-extend-max-channels "${POSTPROCESS_EXTEND_MAX_CHANNELS:-8}"
  --postprocess-extend-min-observed "${POSTPROCESS_EXTEND_MIN_OBSERVED:-4}"
  --postprocess-validate-observed-only
  --speed-min-kmh "${SPEED_MIN_KMH:-25.0}"
  --speed-max-kmh "${SPEED_MAX_KMH:-220.0}"
  --segment-speed-min-kmh "${SEGMENT_SPEED_MIN_KMH:-10.0}"
  --segment-speed-max-kmh "${SEGMENT_SPEED_MAX_KMH:-320.0}"
  --max-line-residual-s "${MAX_LINE_RESIDUAL_S:-20.0}"
  --max-adjacent-residual-s "${MAX_ADJACENT_RESIDUAL_S:-15.0}"
)

if [[ "${ANCHOR_PATH_ROBUST_BASELINE:-1}" == "1" ]]; then
  ARGS+=(--anchor-path-robust-baseline)
else
  ARGS+=(--no-anchor-path-robust-baseline)
fi

"$PYTHON_BIN" scripts/predict_vehicle_peak_set_real.py "${ARGS[@]}"
