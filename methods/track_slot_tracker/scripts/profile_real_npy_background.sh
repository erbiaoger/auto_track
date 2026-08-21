#!/usr/bin/env sh
set -eu

# 用途：
#   对固定真实 `.npy` 背景做统计画像，输出 `profile.json`、`realism_profile.json`
#   和 `report.md`。这是 profile 驱动模拟流程的第 1 步。
#
# 用法：
#   sh profile_real_npy_background.sh
#   OUT_DIR=/tmp/real_profile WINDOW_STRIDE_SECONDS=300 sh profile_real_npy_background.sh
#
# 输出：
#   - <OUT_DIR>/profile.json
#   - <OUT_DIR>/realism_profile.json
#   - <OUT_DIR>/report.md

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

REAL_BG_INPUT="datasets/xi/large/03gauss_large.npy"

OUT_DIR=${OUT_DIR:-datasets/profiles/xi_gauss_50_realbg}
ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}
FS=${FS:-1000}
DX_M=${DX_M:-100}
CHANNEL_START=${CHANNEL_START:-0}
CHANNEL_COUNT=${CHANNEL_COUNT:-50}
WINDOW_SECONDS=${WINDOW_SECONDS:-120}
WINDOW_STRIDE_SECONDS=${WINDOW_STRIDE_SECONDS:-600}
PEAK_HEIGHT=${PEAK_HEIGHT:-0.02}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}
PEAK_DISTANCE_SECONDS=${PEAK_DISTANCE_SECONDS:-0.15}
ZERO_THRESHOLD=${ZERO_THRESHOLD:-1e-8}
COMPONENT_TIME_DOWNSAMPLE=${COMPONENT_TIME_DOWNSAMPLE:-100}
CLIP_RATIO=${CLIP_RATIO:-1.35}
PROXY_SPEED_MIN_KMH=${PROXY_SPEED_MIN_KMH:-55}
PROXY_SPEED_MAX_KMH=${PROXY_SPEED_MAX_KMH:-110}
PROXY_MATCH_SLACK_S=${PROXY_MATCH_SLACK_S:-0.08}
WINDOW_CATALOG_LIMIT=${WINDOW_CATALOG_LIMIT:-4096}

uv run python -m autotrack.simulation.profile_real_npy_background \
  --input "$REAL_BG_INPUT" \
  --out-dir "$OUT_DIR" \
  --array-layout "$ARRAY_LAYOUT" \
  --fs "$FS" \
  --dx-m "$DX_M" \
  --channel-start "$CHANNEL_START" \
  --channel-count "$CHANNEL_COUNT" \
  --window-seconds "$WINDOW_SECONDS" \
  --window-stride-seconds "$WINDOW_STRIDE_SECONDS" \
  --peak-height "$PEAK_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-distance-seconds "$PEAK_DISTANCE_SECONDS" \
  --zero-threshold "$ZERO_THRESHOLD" \
  --component-time-downsample "$COMPONENT_TIME_DOWNSAMPLE" \
  --clip-ratio "$CLIP_RATIO" \
  --proxy-speed-min-kmh "$PROXY_SPEED_MIN_KMH" \
  --proxy-speed-max-kmh "$PROXY_SPEED_MAX_KMH" \
  --proxy-match-slack-s "$PROXY_MATCH_SLACK_S" \
  --window-catalog-limit "$WINDOW_CATALOG_LIMIT"
