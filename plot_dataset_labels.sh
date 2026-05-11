#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v2_120s/train}               # 待检查的数据目录
OUT_DIR=${OUT_DIR:-models/peak_slot_v2_120s_cuda/label_check}        # 图片和 CSV 输出目录
SAMPLE_INDICES=${SAMPLE_INDICES:-}                                    # 指定样本编号，逗号分隔
START_SAMPLE=${START_SAMPLE:-0}                                       # 未指定编号时的起始样本号
PLOT_SAMPLES=${PLOT_SAMPLES:-16}                                      # 连续绘制的样本数量
PLOT_DPI=${PLOT_DPI:-160}                                             # 输出图片 DPI
PLOT_PEAKS=${PLOT_PEAKS:-1}                                           # 是否额外画 peak candidates
PLOT_STYLE=${PLOT_STYLE:-heatmap}                                     # 背景绘图风格：heatmap 或 waveform
MAX_GT_TRACKS=${MAX_GT_TRACKS:-0}                                     # 每个样本最多绘制多少条 GT，0 表示全部
POINT_SIZE=${POINT_SIZE:-10}                                          # GT 点大小
LINE_WIDTH=${LINE_WIDTH:-1.15}                                        # GT 线宽
LINE_ALPHA=${LINE_ALPHA:-0.75}                                        # GT 线透明度
VMAX_QUANTILE=${VMAX_QUANTILE:-0.995}                                 # 热图灰度截断分位数

sample_args=""
if [ -n "$SAMPLE_INDICES" ]; then
  sample_args="--sample-indices $SAMPLE_INDICES"
fi

peak_args=""
if [ "$PLOT_PEAKS" = "1" ] || [ "$PLOT_PEAKS" = "true" ]; then
  peak_args="--plot-peaks"
fi

uv run python -m autotrack.dl.plot_dataset_labels \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --start-sample "$START_SAMPLE" \
  --plot-samples "$PLOT_SAMPLES" \
  --plot-dpi "$PLOT_DPI" \
  --plot-style "$PLOT_STYLE" \
  --max-gt-tracks "$MAX_GT_TRACKS" \
  --point-size "$POINT_SIZE" \
  --line-width "$LINE_WIDTH" \
  --line-alpha "$LINE_ALPHA" \
  --vmax-quantile "$VMAX_QUANTILE" \
  $peak_args \
  $sample_args
