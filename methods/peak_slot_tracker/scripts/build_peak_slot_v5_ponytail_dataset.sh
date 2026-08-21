#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

SPLIT=${SPLIT:-train}
PROFILE_TRACK_DIR=${PROFILE_TRACK_DIR:-datasets/track_slot_profile_only_120s/$SPLIT}
HARD_TRACK_DIR=${HARD_TRACK_DIR:-datasets/track_slot_v4_120s_noisy_badch/$SPLIT}
PROFILE_OUT_DIR=${PROFILE_OUT_DIR:-datasets/peak_slot_v5_120s_ponytail_profile/$SPLIT}
HARD_OUT_DIR=${HARD_OUT_DIR:-datasets/peak_slot_v5_120s_ponytail_hard/$SPLIT}
OUT_DIR=${OUT_DIR:-datasets/peak_slot_v5_120s_ponytail/$SPLIT}
UNET_CHECKPOINT=${UNET_CHECKPOINT:-/csim2/zhangzhiyu/MyProjects/waveform_line_task/models/unet_waveform_profile_only_cuda/snapshots/checkpoint_best_epoch10_snapshot.pt}
WAVEFORM_TASK_DIR=${WAVEFORM_TASK_DIR:-/csim2/zhangzhiyu/MyProjects/waveform_line_task}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-16}
IMAGE_SIZE=${IMAGE_SIZE:-512}
WAVEFORM_LINE_WIDTH=${WAVEFORM_LINE_WIDTH:-1}
WIGGLE_FRACTION=${WIGGLE_FRACTION:-0.28}
ROBUST_PERCENTILE=${ROBUST_PERCENTILE:-99.5}
PRIOR_THRESHOLD=${PRIOR_THRESHOLD:-0.0}
PRIOR_SCALE=${PRIOR_SCALE:-1.0}
X_DTYPE=${X_DTYPE:-preserve}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-96}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.15}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.015}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.015}
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}
PRIOR_PEAK_MIN_HEIGHT=${PRIOR_PEAK_MIN_HEIGHT:-0.08}
PRIOR_PEAK_PROMINENCE=${PRIOR_PEAK_PROMINENCE:-0.02}
CANDIDATE_MERGE_TOLERANCE_S=${CANDIDATE_MERGE_TOLERANCE_S:-0.20}
PRIOR_SCORE_SCALE=${PRIOR_SCORE_SCALE:-0.85}
MAX_SHARDS=${MAX_SHARDS:-0}
OVERWRITE=${OVERWRITE:-1}
RESUME=${RESUME:-0}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

resume_args=""
if [ "$RESUME" = "1" ] || [ "$RESUME" = "true" ]; then
  resume_args="--resume"
fi

build_one() {
  track_dir="$1"
  out_dir="$2"
  uv run python -m autotrack.dl.build_peak_slot_unetprior_from_track \
    --track-dir "$track_dir" \
    --out-dir "$out_dir" \
    --unet-checkpoint "$UNET_CHECKPOINT" \
    --waveform-task-dir "$WAVEFORM_TASK_DIR" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --image-size "$IMAGE_SIZE" \
    --waveform-line-width "$WAVEFORM_LINE_WIDTH" \
    --wiggle-fraction "$WIGGLE_FRACTION" \
    --robust-percentile "$ROBUST_PERCENTILE" \
    --prior-threshold "$PRIOR_THRESHOLD" \
    --prior-scale "$PRIOR_SCALE" \
    --x-dtype "$X_DTYPE" \
    --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
    --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
    --peak-min-height "$PEAK_MIN_HEIGHT" \
    --peak-prominence "$PEAK_PROMINENCE" \
    --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
    --candidate-source raw_prior_union \
    --prior-peak-min-height "$PRIOR_PEAK_MIN_HEIGHT" \
    --prior-peak-prominence "$PRIOR_PEAK_PROMINENCE" \
    --candidate-merge-tolerance-s "$CANDIDATE_MERGE_TOLERANCE_S" \
    --prior-score-scale "$PRIOR_SCORE_SCALE" \
    --workers 1 \
    --max-shards "$MAX_SHARDS" \
    $resume_args \
    $overwrite_args
}

build_one "$PROFILE_TRACK_DIR" "$PROFILE_OUT_DIR"
build_one "$HARD_TRACK_DIR" "$HARD_OUT_DIR"

uv run python -m autotrack.dl.merge_peak_slot_datasets \
  --in-dir "$PROFILE_OUT_DIR" \
  --in-dir "$HARD_OUT_DIR" \
  --out-dir "$OUT_DIR" \
  $overwrite_args
