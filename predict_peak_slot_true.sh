#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v1/xi_gauss_50}
MODEL=${MODEL:-models/peak_slot_v2_120s_cuda/checkpoint_best.pt}
OUT_DIR=${OUT_DIR:-models/peak_slot_v2_120s_cuda/prediction_check}
DEVICE=${DEVICE:-cpu}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SAMPLES=${MAX_SAMPLES:-256}
MAX_CSV_SAMPLES=${MAX_CSV_SAMPLES:-32}
PLOT_SAMPLES=${PLOT_SAMPLES:-16}
PLOT_DPI=${PLOT_DPI:-160}
PLOT_STYLE=${PLOT_STYLE:-waveform}
OBJECTNESS_THRESHOLD=${OBJECTNESS_THRESHOLD:-0.35}
PEAK_THRESHOLD=${PEAK_THRESHOLD:-0.4}
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-2}
MAX_PREDICTED_TRACKS=${MAX_PREDICTED_TRACKS:-96}
MATCHER=${MATCHER:-hungarian}
DECODER_MODE=${DECODER_MODE:-beam_global}
VITERBI_BEAM_SIZE=${VITERBI_BEAM_SIZE:-8}
TIME_PRIOR_WEIGHT=${TIME_PRIOR_WEIGHT:-2.0}
GLOBAL_CONFLICT_PENALTY=${GLOBAL_CONFLICT_PENALTY:-2.0}
VITERBI_TOPK=${VITERBI_TOPK:-16}
VITERBI_CANDIDATE_THRESHOLD=${VITERBI_CANDIDATE_THRESHOLD:-0.01}
VITERBI_SPEED_MIN_KMH=${VITERBI_SPEED_MIN_KMH:-60}
VITERBI_SPEED_MAX_KMH=${VITERBI_SPEED_MAX_KMH:-100}
VITERBI_MAX_SKIP_CHANNELS=${VITERBI_MAX_SKIP_CHANNELS:-4}
VITERBI_INERTIA_PENALTY=${VITERBI_INERTIA_PENALTY:-2.5}
VITERBI_SLOPE_MEMORY=${VITERBI_SLOPE_MEMORY:-0.75}

uv run python -m autotrack.dl.predict_peak_slot_dataset \
  --data-dir "$DATA_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --max-csv-samples "$MAX_CSV_SAMPLES" \
  --plot-samples "$PLOT_SAMPLES" \
  --plot-dpi "$PLOT_DPI" \
  --plot-style "$PLOT_STYLE" \
  --objectness-threshold "$OBJECTNESS_THRESHOLD" \
  --peak-threshold "$PEAK_THRESHOLD" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --max-predicted-tracks "$MAX_PREDICTED_TRACKS" \
  --matcher "$MATCHER" \
  --decoder-mode "$DECODER_MODE" \
  --viterbi-beam-size "$VITERBI_BEAM_SIZE" \
  --time-prior-weight "$TIME_PRIOR_WEIGHT" \
  --global-conflict-penalty "$GLOBAL_CONFLICT_PENALTY" \
  --viterbi-topk "$VITERBI_TOPK" \
  --viterbi-candidate-threshold "$VITERBI_CANDIDATE_THRESHOLD" \
  --viterbi-speed-min-kmh "$VITERBI_SPEED_MIN_KMH" \
  --viterbi-speed-max-kmh "$VITERBI_SPEED_MAX_KMH" \
  --viterbi-max-skip-channels "$VITERBI_MAX_SKIP_CHANNELS" \
  --viterbi-inertia-penalty "$VITERBI_INERTIA_PENALTY" \
  --viterbi-slope-memory "$VITERBI_SLOPE_MEMORY"
