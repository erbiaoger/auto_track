#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v2_120s/xi_gauss_50_120s_stride60}              # 真实/无标签数据目录
MODEL=${MODEL:-models/peak_slot_v4_120s_noisy_badch_cuda/checkpoint_best.pt}     # checkpoint 路径
OUT_DIR=${OUT_DIR:-models/peak_slot_v4_120s_noisy_badch_cuda/prediction_check}   # 预测结果输出目录
DEVICE=${DEVICE:-cpu}                                                 # 推理设备
BATCH_SIZE=${BATCH_SIZE:-16}                                          # 推理 batch 大小
MAX_SAMPLES=${MAX_SAMPLES:-256}                                       # 最多预测多少个样本
MAX_CSV_SAMPLES=${MAX_CSV_SAMPLES:-32}                                # 最多导出多少个样本的 CSV
PLOT_SAMPLES=${PLOT_SAMPLES:-16}                                      # 最多绘制多少张图
PLOT_DPI=${PLOT_DPI:-160}                                             # 绘图 DPI
PLOT_STYLE=${PLOT_STYLE:-waveform}                                    # 绘图风格：waveform 或 heatmap
PLOT_DIRECTION_FILTER=${PLOT_DIRECTION_FILTER:-forward}               # 绘图方向：all, forward, reverse
PREDICTION_DIRECTION_FILTER=${PREDICTION_DIRECTION_FILTER:-all}       # 导出/统计方向：all, forward, reverse
OBJECTNESS_THRESHOLD=${OBJECTNESS_THRESHOLD:-0.35}                    # objectness 阈值
EXTRA_CANDIDATE_SLOTS=${EXTRA_CANDIDATE_SLOTS:-32}                   # 额外解码的低 objectness slot 数
CANDIDATE_OBJECTNESS_FLOOR=${CANDIDATE_OBJECTNESS_FLOOR:-0.02}       # 额外候选 slot 的最低 objectness
PEAK_THRESHOLD=${PEAK_THRESHOLD:-0.4}                                 # 点级峰选择阈值
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-2}                       # 一条轨迹至少保留多少可见道
MAX_PREDICTED_TRACKS=${MAX_PREDICTED_TRACKS:-96}                      # 每个样本最多输出多少条轨迹
MATCHER=${MATCHER:-hungarian}                                         # 评估时使用的匹配器
DECODER_MODE=${DECODER_MODE:-beam_global}                             # 解码模式
VITERBI_BEAM_SIZE=${VITERBI_BEAM_SIZE:-8}                             # beam size
TIME_PRIOR_WEIGHT=${TIME_PRIOR_WEIGHT:-2.0}                           # 时间先验权重
GLOBAL_CONFLICT_PENALTY=${GLOBAL_CONFLICT_PENALTY:-2.0}               # 跨 slot 冲突惩罚
CONFLICT_MODE=${CONFLICT_MODE:-soft}                                  # 冲突处理：soft, hard, off
DUPLICATE_OVERLAP_RATIO=${DUPLICATE_OVERLAP_RATIO:-0.75}              # soft 模式下判定重复的最小重合比例
MIN_UNIQUE_SUPPORT_CHANNELS=${MIN_UNIQUE_SUPPORT_CHANNELS:-3}         # 近车轨迹至少有多少独立峰支撑时保留
VITERBI_TOPK=${VITERBI_TOPK:-16}                                      # 每道保留多少候选峰
VITERBI_CANDIDATE_THRESHOLD=${VITERBI_CANDIDATE_THRESHOLD:-0.01}      # 候选峰最低概率阈值
VITERBI_SPEED_MIN_KMH=${VITERBI_SPEED_MIN_KMH:-60}                    # 解码允许最小速度
VITERBI_SPEED_MAX_KMH=${VITERBI_SPEED_MAX_KMH:-100}                   # 解码允许最大速度
VITERBI_MAX_SKIP_CHANNELS=${VITERBI_MAX_SKIP_CHANNELS:-4}             # 最多允许跳过多少道
VITERBI_INERTIA_PENALTY=${VITERBI_INERTIA_PENALTY:-2.5}               # 速度惯性惩罚
VITERBI_SLOPE_MEMORY=${VITERBI_SLOPE_MEMORY:-0.75}                    # 斜率记忆系数

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
  --viterbi-slope-memory "$VITERBI_SLOPE_MEMORY"
