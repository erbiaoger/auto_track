#!/usr/bin/env sh
set -eu

# 用途：
#   对比一批生成数据目录与 `realism_profile.json` 的分布差异，输出校准报告。
#   这是 profile 驱动模拟流程的第 4 步，适合在训练前先做分布验收。
#
# 用法：
#   sh calibrate_realbg_generator.sh
#   PROFILE=/tmp/real_profile/realism_profile.json \
#   DATA_DIRS="datasets/track_slot_realbg_120s_heavy/train,datasets/track_slot_realbg_120s_profile/train" \
#   OUT_DIR=/tmp/realbg_calibration \
#   sh calibrate_realbg_generator.sh
#
# 输出：
#   - <OUT_DIR>/calibration_summary.json
#   - <OUT_DIR>/calibration_report.md

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

PROFILE=${PROFILE:-datasets/profiles/xi_gauss_50_realbg/realism_profile.json}
OUT_DIR=${OUT_DIR:-predicts/realbg_calibration}
DATA_DIRS=${DATA_DIRS:-datasets/track_slot_realbg_120s_heavy/train,datasets/track_slot_realbg_120s_profile/train}
MAX_SAMPLES=${MAX_SAMPLES:-64}
PEAK_CANDIDATES_PER_CHANNEL=${PEAK_CANDIDATES_PER_CHANNEL:-64}
PEAK_MIN_DISTANCE_S=${PEAK_MIN_DISTANCE_S:-0.15}
PEAK_MIN_HEIGHT=${PEAK_MIN_HEIGHT:-0.02}
PEAK_PROMINENCE=${PEAK_PROMINENCE:-0.02}

set -- \
  --profile "$PROFILE" \
  --out-dir "$OUT_DIR" \
  --max-samples "$MAX_SAMPLES" \
  --peak-candidates-per-channel "$PEAK_CANDIDATES_PER_CHANNEL" \
  --peak-min-distance-s "$PEAK_MIN_DISTANCE_S" \
  --peak-min-height "$PEAK_MIN_HEIGHT" \
  --peak-prominence "$PEAK_PROMINENCE"

IFS=','
for item in $DATA_DIRS; do
  set -- "$@" --data-dir "$item"
done
unset IFS

uv run python -m autotrack.dl.calibrate_realbg_generator "$@"
