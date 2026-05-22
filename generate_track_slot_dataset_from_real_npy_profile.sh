#!/usr/bin/env sh
set -eu

# 用途：
#   使用 `realism_profile.json` 驱动真实背景 TrackSlotNet 数据生成。
#   这是 profile 驱动模拟流程的第 2 步。
#
# 用法：
#   sh generate_track_slot_dataset_from_real_npy_profile.sh
#   PROFILE=/tmp/real_profile/realism_profile.json \
#   OUT_DIR=datasets/track_slot_realbg_profile/train \
#   NUM_SAMPLES=512 \
#   sh generate_track_slot_dataset_from_real_npy_profile.sh
#
# 输出：
#   - <OUT_DIR>/meta.json
#   - <OUT_DIR>/shard_*.pt

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

REAL_BG_INPUT="datasets/xi/large/03gauss_large.npy"
REAL_BG_WINDOW_STRIDE_SECONDS="60"

PROFILE=${PROFILE:-datasets/profiles/xi_gauss_50_realbg/realism_profile.json}
OUT_DIR=${OUT_DIR:-datasets/track_slot_realbg_120s_profile/train}
ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}
NUM_SAMPLES=${NUM_SAMPLES:-40000}
SHARD_SIZE=${SHARD_SIZE:-128}
SEED=${SEED:-52}
FS=${FS:-1000}
DX_M=${DX_M:-100}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}
CHANNEL_START=${CHANNEL_START:-0}
CHANNEL_COUNT=${CHANNEL_COUNT:-50}
PROFILE_STRENGTH=${PROFILE_STRENGTH:-1.0}
WINDOW_SAMPLER=${WINDOW_SAMPLER:-profile_weighted}
ARTIFACT_POLICY=${ARTIFACT_POLICY:-hybrid}
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.84,0.15,0.01}
CLIP_RATIO=${CLIP_RATIO:-1.35}
INPUT_MODE=${INPUT_MODE:-raw}
X_DTYPE=${X_DTYPE:-float16}
OVERWRITE=${OVERWRITE:-1}

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_track_slot_dataset_from_real_npy \
  --input "$REAL_BG_INPUT" \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --array-layout "$ARRAY_LAYOUT" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --seed "$SEED" \
  --fs "$FS" \
  --dx-m "$DX_M" \
  --window-seconds "$WINDOW_SECONDS" \
  --window-stride-seconds "$REAL_BG_WINDOW_STRIDE_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --channel-start "$CHANNEL_START" \
  --channel-count "$CHANNEL_COUNT" \
  --profile-strength "$PROFILE_STRENGTH" \
  --window-sampler "$WINDOW_SAMPLER" \
  --artifact-policy "$ARTIFACT_POLICY" \
  --motion-mix "$MOTION_MIX" \
  --motion-weights "$MOTION_WEIGHTS" \
  --clip-ratio "$CLIP_RATIO" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  $overwrite_args
