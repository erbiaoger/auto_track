#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

INPUT=${INPUT:-/csim2/zhangzhiyu/MyProjects/auto_track/datasets/03gauss_large.npy}
BASE_NAME=${BASE_NAME:-real_03gauss_large_ponytail_full}
PEAK_DIR=${PEAK_DIR:-datasets/peak_slot/${BASE_NAME}_raw}
DATA_DIR=${DATA_DIR:-datasets/peak_slot_v5_ponytail_real/$BASE_NAME}
OUT_DIR=${OUT_DIR:-predicts/peak_slot_v5_120s_ponytail/$BASE_NAME}
MODEL=${MODEL:-models/peak_slot_v5_120s_ponytail_cuda/checkpoint_best.pt}
UNET_CHECKPOINT=${UNET_CHECKPOINT:-/csim2/zhangzhiyu/MyProjects/waveform_line_task/models/unet_waveform_profile_only_cuda/snapshots/checkpoint_best_epoch10_snapshot.pt}
WAVEFORM_TASK_DIR=${WAVEFORM_TASK_DIR:-/csim2/zhangzhiyu/MyProjects/waveform_line_task}

ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}
FS=${FS:-1000}
DX_M=${DX_M:-100}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
STRIDE_SECONDS=${STRIDE_SECONDS:-60}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
CHANNEL_START=${CHANNEL_START:-0}
CHANNEL_COUNT=${CHANNEL_COUNT:-50}
CLIP_RATIO=${CLIP_RATIO:-1.35}
INPUT_MODE=${INPUT_MODE:-raw}
X_DTYPE=${X_DTYPE:-float32}
SHARD_SIZE=${SHARD_SIZE:-256}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-96}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.15}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.015}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.015}
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}
PRIOR_DEVICE=${PRIOR_DEVICE:-cuda}
PRIOR_BATCH_SIZE=${PRIOR_BATCH_SIZE:-16}
IMAGE_SIZE=${IMAGE_SIZE:-512}
WAVEFORM_LINE_WIDTH=${WAVEFORM_LINE_WIDTH:-1}
WIGGLE_FRACTION=${WIGGLE_FRACTION:-0.28}
ROBUST_PERCENTILE=${ROBUST_PERCENTILE:-99.5}
PRIOR_THRESHOLD=${PRIOR_THRESHOLD:-0.0}
PRIOR_SCALE=${PRIOR_SCALE:-1.0}
PRIOR_PEAK_MIN_HEIGHT=${PRIOR_PEAK_MIN_HEIGHT:-0.08}
PRIOR_PEAK_PROMINENCE=${PRIOR_PEAK_PROMINENCE:-0.02}
CANDIDATE_MERGE_TOLERANCE_S=${CANDIDATE_MERGE_TOLERANCE_S:-0.20}
PRIOR_SCORE_SCALE=${PRIOR_SCORE_SCALE:-0.85}

PRED_DEVICE=${PRED_DEVICE:-cuda}
PRED_BATCH_SIZE=${PRED_BATCH_SIZE:-16}
MAX_SAMPLES=${MAX_SAMPLES:-0}
MAX_CSV_SAMPLES=${MAX_CSV_SAMPLES:-0}
PLOT_SAMPLES=${PLOT_SAMPLES:-64}
PLOT_DPI=${PLOT_DPI:-160}
PLOT_STYLE=${PLOT_STYLE:-waveform}
PLOT_DIRECTION_FILTER=${PLOT_DIRECTION_FILTER:-all}
PREDICTION_DIRECTION_FILTER=${PREDICTION_DIRECTION_FILTER:-all}
OBJECTNESS_THRESHOLD=${OBJECTNESS_THRESHOLD:-0.30}
EXTRA_CANDIDATE_SLOTS=${EXTRA_CANDIDATE_SLOTS:-48}
CANDIDATE_OBJECTNESS_FLOOR=${CANDIDATE_OBJECTNESS_FLOOR:-0.01}
PEAK_THRESHOLD=${PEAK_THRESHOLD:-0.35}
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-2}
MAX_PREDICTED_TRACKS=${MAX_PREDICTED_TRACKS:-96}
MATCHER=${MATCHER:-hungarian}
DECODER_MODE=${DECODER_MODE:-beam_global}
VITERBI_BEAM_SIZE=${VITERBI_BEAM_SIZE:-8}
TIME_PRIOR_WEIGHT=${TIME_PRIOR_WEIGHT:-2.0}
GLOBAL_CONFLICT_PENALTY=${GLOBAL_CONFLICT_PENALTY:-2.0}
CONFLICT_MODE=${CONFLICT_MODE:-soft}
DUPLICATE_OVERLAP_RATIO=${DUPLICATE_OVERLAP_RATIO:-0.75}
MIN_UNIQUE_SUPPORT_CHANNELS=${MIN_UNIQUE_SUPPORT_CHANNELS:-3}
VITERBI_TOPK=${VITERBI_TOPK:-24}
VITERBI_CANDIDATE_THRESHOLD=${VITERBI_CANDIDATE_THRESHOLD:-0.01}
VITERBI_SPEED_MIN_KMH=${VITERBI_SPEED_MIN_KMH:-60}
VITERBI_SPEED_MAX_KMH=${VITERBI_SPEED_MAX_KMH:-100}
VITERBI_MAX_SKIP_CHANNELS=${VITERBI_MAX_SKIP_CHANNELS:-4}
VITERBI_INERTIA_PENALTY=${VITERBI_INERTIA_PENALTY:-2.5}
VITERBI_SLOPE_MEMORY=${VITERBI_SLOPE_MEMORY:-0.75}

SEG_OVERWRITE=${SEG_OVERWRITE:-1}
PRIOR_OVERWRITE=${PRIOR_OVERWRITE:-1}

seg_overwrite_args=""
if [ "$SEG_OVERWRITE" = "1" ] || [ "$SEG_OVERWRITE" = "true" ]; then
  seg_overwrite_args="--overwrite"
fi
prior_overwrite_args=""
if [ "$PRIOR_OVERWRITE" = "1" ] || [ "$PRIOR_OVERWRITE" = "true" ]; then
  prior_overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.segment_real_npy_to_peak_slot \
  --input "$INPUT" \
  --out-dir "$PEAK_DIR" \
  --array-layout "$ARRAY_LAYOUT" \
  --fs "$FS" \
  --dx-m "$DX_M" \
  --window-seconds "$WINDOW_SECONDS" \
  --stride-seconds "$STRIDE_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --channel-start "$CHANNEL_START" \
  --channel-count "$CHANNEL_COUNT" \
  --clip-ratio "$CLIP_RATIO" \
  --input-mode "$INPUT_MODE" \
  --speed-norm-kmh 150 \
  --x-dtype "$X_DTYPE" \
  --shard-size "$SHARD_SIZE" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  $seg_overwrite_args

uv run python -m autotrack.dl.add_unet_prior_to_peak_slot \
  --in-dir "$PEAK_DIR" \
  --out-dir "$DATA_DIR" \
  --unet-checkpoint "$UNET_CHECKPOINT" \
  --waveform-task-dir "$WAVEFORM_TASK_DIR" \
  --device "$PRIOR_DEVICE" \
  --batch-size "$PRIOR_BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --waveform-line-width "$WAVEFORM_LINE_WIDTH" \
  --wiggle-fraction "$WIGGLE_FRACTION" \
  --robust-percentile "$ROBUST_PERCENTILE" \
  --prior-threshold "$PRIOR_THRESHOLD" \
  --prior-scale "$PRIOR_SCALE" \
  --x-dtype preserve \
  --candidate-source raw_prior_union \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  --prior-peak-min-height "$PRIOR_PEAK_MIN_HEIGHT" \
  --prior-peak-prominence "$PRIOR_PEAK_PROMINENCE" \
  --candidate-merge-tolerance-s "$CANDIDATE_MERGE_TOLERANCE_S" \
  --prior-score-scale "$PRIOR_SCORE_SCALE" \
  $prior_overwrite_args

uv run python -m autotrack.dl.predict_peak_slot_dataset \
  --data-dir "$DATA_DIR" \
  --model "$MODEL" \
  --out-dir "$OUT_DIR" \
  --device "$PRED_DEVICE" \
  --batch-size "$PRED_BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --max-csv-samples "$MAX_CSV_SAMPLES" \
  --plot-samples "$PLOT_SAMPLES" \
  --plot-dpi "$PLOT_DPI" \
  --plot-style "$PLOT_STYLE" \
  --plot-direction-filter "$PLOT_DIRECTION_FILTER" \
  --prediction-direction-filter "$PREDICTION_DIRECTION_FILTER" \
  --objectness-threshold "$OBJECTNESS_THRESHOLD" \
  --extra-candidate-slots "$EXTRA_CANDIDATE_SLOTS" \
  --candidate-objectness-floor "$CANDIDATE_OBJECTNESS_FLOOR" \
  --peak-threshold "$PEAK_THRESHOLD" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --max-predicted-tracks "$MAX_PREDICTED_TRACKS" \
  --matcher "$MATCHER" \
  --decoder-mode "$DECODER_MODE" \
  --viterbi-beam-size "$VITERBI_BEAM_SIZE" \
  --time-prior-weight "$TIME_PRIOR_WEIGHT" \
  --global-conflict-penalty "$GLOBAL_CONFLICT_PENALTY" \
  --conflict-mode "$CONFLICT_MODE" \
  --duplicate-overlap-ratio "$DUPLICATE_OVERLAP_RATIO" \
  --min-unique-support-channels "$MIN_UNIQUE_SUPPORT_CHANNELS" \
  --viterbi-topk "$VITERBI_TOPK" \
  --viterbi-candidate-threshold "$VITERBI_CANDIDATE_THRESHOLD" \
  --viterbi-speed-min-kmh "$VITERBI_SPEED_MIN_KMH" \
  --viterbi-speed-max-kmh "$VITERBI_SPEED_MAX_KMH" \
  --viterbi-max-skip-channels "$VITERBI_MAX_SKIP_CHANNELS" \
  --viterbi-inertia-penalty "$VITERBI_INERTIA_PENALTY" \
  --viterbi-slope-memory "$VITERBI_SLOPE_MEMORY" \
  --no-ground-truth-csv
