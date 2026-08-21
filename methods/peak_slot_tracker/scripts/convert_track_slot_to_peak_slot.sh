#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

IN_DIR=${IN_DIR:-datasets/track_slot_v4_120s_noisy_badch/train}      # 输入 track_slot 数据目录
OUT_DIR=${OUT_DIR:-datasets/peak_slot_v4_120s_noisy_badch/train}     # 输出 peak_slot 数据目录
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-64}       # 每道最多保留的峰候选数
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.15}                     # 同一道峰候选最小间隔 s
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.02}                             # 峰候选最小高度阈值
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}                             # 峰候选最小显著性阈值
PEAK_MATCH_TOLERANCE_S=${PEAK_MATCH_TOLERANCE_S:-0.25}               # GT 与候选峰匹配容差 s
WORKERS=${WORKERS:-100}                                               # 转换并行 worker 数
OVERWRITE=${OVERWRITE:-1}                                            # 是否覆盖已有输出目录

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.convert_track_slot_to_peak_slot \
  --in-dir "$IN_DIR" \
  --out-dir "$OUT_DIR" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE" \
  --peak-match-tolerance-s "$PEAK_MATCH_TOLERANCE_S" \
  --workers "$WORKERS" \
  $overwrite_args
