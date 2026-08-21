#!/usr/bin/env sh
set -eu

# 用途：
#   使用 `realism_profile.json` 中提取出来的固定坏道、高概率坏道和真实峰形参数，
#   生成一套完全脱离原始真实背景的纯模拟 TrackSlotNet 数据。
#   这条流程不会再混入真实 `.npy` 窗口，因此所有车辆都来自生成器本身，
#   理论上不会再出现“真实窗里还有车但没有标签”的情况。
#
# 用法：
#   sh generate_track_slot_dataset_profile_only.sh
#
#   PROFILE=datasets/profiles/xi_gauss_50_realbg/realism_profile.json \
#   OUT_DIR=datasets/track_slot_profile_only_120s/train \
#   NUM_SAMPLES=20000 \
#   VEHICLES_MIN=6 \
#   VEHICLES_MAX=18 \
#   ISOLATED_NOISE_RATE=28 \
#   SMOOTH_SPEED_MAX_FRAC=0.16 \
#   TRACK_TIME_JITTER_MAX_S=0.7 \
#   sh generate_track_slot_dataset_profile_only.sh
#
# 输出：
#   - <OUT_DIR>/meta.json
#   - <OUT_DIR>/shard_*.pt

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

PROFILE=${PROFILE:-datasets/profiles/xi_gauss_50_realbg/realism_profile.json}
OUT_DIR=${OUT_DIR:-datasets/track_slot_profile_only_120s/train}
NUM_SAMPLES=${NUM_SAMPLES:-20000}
SHARD_SIZE=${SHARD_SIZE:-128}
SEED=${SEED:-52}
N_CH=${N_CH:-50}
FS=${FS:-1000}
DX_M=${DX_M:-100}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
VEHICLES_MIN=${VEHICLES_MIN:-6}
VEHICLES_MAX=${VEHICLES_MAX:-18}
SPEED_MIN_KMH=${SPEED_MIN_KMH:-60}
SPEED_MAX_KMH=${SPEED_MAX_KMH:-100}
PRIMARY_RATIO=${PRIMARY_RATIO:-0.83}
INTERACTION_RATIO=${INTERACTION_RATIO:-0.25}
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.24,0.66,0.10}
CONSTANT_PERTURB_PROB=${CONSTANT_PERTURB_PROB:-0.18}
CONSTANT_PERTURB_MAX_FRAC=${CONSTANT_PERTURB_MAX_FRAC:-0.05}
CONSTANT_PERTURB_WIDTH_MIN=${CONSTANT_PERTURB_WIDTH_MIN:-2}
CONSTANT_PERTURB_WIDTH_MAX=${CONSTANT_PERTURB_WIDTH_MAX:-4}
SMOOTH_SPEED_MAX_FRAC=${SMOOTH_SPEED_MAX_FRAC:-0.14}
SMOOTH_SPEED_CORR_CHANNELS=${SMOOTH_SPEED_CORR_CHANNELS:-4}
TRACK_TIME_JITTER_MAX_S=${TRACK_TIME_JITTER_MAX_S:-0.7}
TRACK_TIME_JITTER_CORR_CHANNELS=${TRACK_TIME_JITTER_CORR_CHANNELS:-3}
TRACK_TIME_JITTER_MIN_GAP_RATIO=${TRACK_TIME_JITTER_MIN_GAP_RATIO:-0.30}
ISOLATED_NOISE_RATIO=${ISOLATED_NOISE_RATIO:-1.0}
ISOLATED_NOISE_RATE=${ISOLATED_NOISE_RATE:-28.0}
CLIP_RATIO=${CLIP_RATIO:-1.35}
INPUT_MODE=${INPUT_MODE:-raw}
X_DTYPE=${X_DTYPE:-float16}
WORKERS=${WORKERS:-0}
PROFILE_STRENGTH=${PROFILE_STRENGTH:-1.0}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_track_slot_dataset \
  --out-dir "$OUT_DIR" \
  --profile "$PROFILE" \
  --profile-strength "$PROFILE_STRENGTH" \
  --realism-preset "profile_only_bad_channel_layout" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --n-ch "$N_CH" \
  --fs "$FS" \
  --dx-m "$DX_M" \
  --window-seconds "$WINDOW_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh "$SPEED_MIN_KMH" \
  --speed-max-kmh "$SPEED_MAX_KMH" \
  --primary-ratio "$PRIMARY_RATIO" \
  --interaction-ratio "$INTERACTION_RATIO" \
  --motion-mix "$MOTION_MIX" \
  --motion-weights "$MOTION_WEIGHTS" \
  --constant-perturb-prob "$CONSTANT_PERTURB_PROB" \
  --constant-perturb-max-frac "$CONSTANT_PERTURB_MAX_FRAC" \
  --constant-perturb-width-min "$CONSTANT_PERTURB_WIDTH_MIN" \
  --constant-perturb-width-max "$CONSTANT_PERTURB_WIDTH_MAX" \
  --smooth-speed-max-frac "$SMOOTH_SPEED_MAX_FRAC" \
  --smooth-speed-corr-channels "$SMOOTH_SPEED_CORR_CHANNELS" \
  --track-time-jitter-max-s "$TRACK_TIME_JITTER_MAX_S" \
  --track-time-jitter-corr-channels "$TRACK_TIME_JITTER_CORR_CHANNELS" \
  --track-time-jitter-min-gap-ratio "$TRACK_TIME_JITTER_MIN_GAP_RATIO" \
  --isolated-noise-ratio "$ISOLATED_NOISE_RATIO" \
  --isolated-noise-rate "$ISOLATED_NOISE_RATE" \
  --clip-ratio "$CLIP_RATIO" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  --workers "$WORKERS" \
  $overwrite_args
