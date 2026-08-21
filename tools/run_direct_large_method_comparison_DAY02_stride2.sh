#!/usr/bin/env bash
set -euo pipefail

# One-click reproduction of the large-vehicle comparison figure with dense
# sliding windows.  The defaults are intentionally kept here so the run is
# reproducible without remembering a long command line.
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
DEVICE="${DEVICE:-cuda}"
START_S="${START_S:-0}"
DURATION_S="${DURATION_S:-600}"
WINDOW_S="${WINDOW_S:-120}"
STRIDE_S="${STRIDE_S:-2}"
OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/runs/direct_large_method_comparison_DAY02_0_600s_stride2_stable.png}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_PATH%.png}.json}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python 环境不存在或不可执行：$PYTHON_BIN" >&2
  echo "可通过 PYTHON_BIN=/path/to/python 重新指定。" >&2
  exit 1
fi

export PYTHONPATH="$ROOT_DIR/common/src:$ROOT_DIR/vehicle_replay_web/backend:$ROOT_DIR/methods/hybrid_vehicle_tracker/src:$ROOT_DIR/methods/kalman_seed_tracker/src:$ROOT_DIR/methods/hungarian_assignment_tracker/src:$ROOT_DIR/methods/graph_search_tracker/src:$ROOT_DIR/methods/peak_slot_tracker/src:$ROOT_DIR/methods/vehicle_peak_set_tracker/src:$ROOT_DIR/compatibility/autotrack_legacy${PYTHONPATH:+:$PYTHONPATH}"

echo "运行 DAY02 大车方法对比："
echo "  时间范围: ${START_S}–$((START_S + DURATION_S)) s"
echo "  识别窗口: ${WINDOW_S} s"
echo "  滑动步长: ${STRIDE_S} s"
echo "  计算设备: ${DEVICE}"
echo "  输出文件: ${OUTPUT_PATH}"
echo "提示：2 秒步长会显著增加运行时间，完整五方法对比可能需要较长时间。"

exec "$PYTHON_BIN" "$ROOT_DIR/tools/compare_peak_only_methods_DAY02.py" \
  --device "$DEVICE" \
  --start-s "$START_S" \
  --duration-s "$DURATION_S" \
  --window-s "$WINDOW_S" \
  --stride-s "$STRIDE_S" \
  --output "$OUTPUT_PATH" \
  --summary "$SUMMARY_PATH" \
  "$@"
