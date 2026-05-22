#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

# 固定真实背景源和背景窗采样步长，不作为常用外部参数暴露。
REAL_BG_INPUT="/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy"
REAL_BG_WINDOW_STRIDE_SECONDS="60"

OUT_DIR=${OUT_DIR:-datasets/track_slot_realbg_120s_heavy/train}              # 输出 track_slot shard 目录
ARRAY_LAYOUT=${ARRAY_LAYOUT:-time_channel}                                    # 输入数组布局
NUM_SAMPLES=${NUM_SAMPLES:-200}                                              # 生成样本总数
SHARD_SIZE=${SHARD_SIZE:-128}                                                 # 每个 shard 的样本数
SEED=${SEED:-52}                                                              # 随机种子
FS=${FS:-1000}                                                                # 采样率 Hz
DX_M=${DX_M:-100}                                                             # 通道间距 m
WINDOW_SECONDS=${WINDOW_SECONDS:-120}                                         # 每窗时长 s
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}                                        # 时间降采样倍率
CHANNEL_START=${CHANNEL_START:-0}                                             # 起始通道
CHANNEL_COUNT=${CHANNEL_COUNT:-50}                                            # 通道数
BACKGROUND_SCALE_MIN=${BACKGROUND_SCALE_MIN:-0.95}                            # 背景缩放最小值
BACKGROUND_SCALE_MAX=${BACKGROUND_SCALE_MAX:-1.05}                            # 背景缩放最大值
BACKGROUND_OFFSET_STD=${BACKGROUND_OFFSET_STD:-0.0}                           # 背景整体偏置扰动
VEHICLES_MIN=${VEHICLES_MIN:-1}                                               # 每窗最少车辆数
VEHICLES_MAX=${VEHICLES_MAX:-12}                                              # 每窗最多车辆数
SPEED_MIN_KMH=${SPEED_MIN_KMH:-70}                                            # 最低车速 km/h
SPEED_MAX_KMH=${SPEED_MAX_KMH:-86}                                            # 最高车速 km/h
SPEED_NORM_KMH=${SPEED_NORM_KMH:-150}                                         # 标签速度归一化常数
FIXED_AMP=${FIXED_AMP:-6.0}                                                   # 车辆高斯窗幅值
SIGMA_SECONDS=${SIGMA_SECONDS:-0.25}                                          # 车辆高斯窗 sigma s
PRIMARY_RATIO=${PRIMARY_RATIO:-0.83}                                          # 正向车辆比例
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-4}                               # 轨迹至少可见通道数
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}               # 运动模型组合
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.84,0.15,0.01}                              # 运动模型权重
CONSTANT_PERTURB_PROB=${CONSTANT_PERTURB_PROB:-0.05}                          # 稀疏速度扰动概率
CONSTANT_PERTURB_MAX_FRAC=${CONSTANT_PERTURB_MAX_FRAC:-0.01}                  # 稀疏速度扰动幅度比例
CONSTANT_PERTURB_WIDTH_MIN=${CONSTANT_PERTURB_WIDTH_MIN:-1}                   # 稀疏扰动最小宽度
CONSTANT_PERTURB_WIDTH_MAX=${CONSTANT_PERTURB_WIDTH_MAX:-2}                   # 稀疏扰动最大宽度
SMOOTH_SPEED_MAX_FRAC=${SMOOTH_SPEED_MAX_FRAC:-0.05}                          # 平滑速度起伏最大比例
SMOOTH_SPEED_CORR_CHANNELS=${SMOOTH_SPEED_CORR_CHANNELS:-8}                   # 平滑速度起伏相关宽度
STOP_DURATION_MIN_S=${STOP_DURATION_MIN_S:-1.0}                               # stop-go 最短停顿
STOP_DURATION_MAX_S=${STOP_DURATION_MAX_S:-8.0}                               # stop-go 最长停顿
STOP_CHANNEL_WIDTH_MIN=${STOP_CHANNEL_WIDTH_MIN:-1}                           # stop-go 最小通道宽度
STOP_CHANNEL_WIDTH_MAX=${STOP_CHANNEL_WIDTH_MAX:-3}                           # stop-go 最大通道宽度
STOP_RESPONSE_SIGMA_SCALE=${STOP_RESPONSE_SIGMA_SCALE:-3.0}                   # stop-go 局部 sigma 放大倍数
STOP_RESPONSE_AMP_SCALE=${STOP_RESPONSE_AMP_SCALE:-1.2}                       # stop-go 局部幅值放大倍数
RESTART_SPEED_RATIO_MIN=${RESTART_SPEED_RATIO_MIN:-0.95}                      # 停车后恢复速度最小比例
RESTART_SPEED_RATIO_MAX=${RESTART_SPEED_RATIO_MAX:-1.05}                      # 停车后恢复速度最大比例
ISOLATED_NOISE_RATIO=${ISOLATED_NOISE_RATIO:-0.85}                            # 启用孤立峰的样本比例
ISOLATED_NOISE_RATE=${ISOLATED_NOISE_RATE:-220.0}                             # 每窗期望孤立峰数量
ISOLATED_NOISE_AMP_MIN=${ISOLATED_NOISE_AMP_MIN:-0.6}                         # 孤立峰最小幅值
ISOLATED_NOISE_AMP_MAX=${ISOLATED_NOISE_AMP_MAX:-5.5}                         # 孤立峰最大幅值
ISOLATED_NOISE_SIGMA_MIN_S=${ISOLATED_NOISE_SIGMA_MIN_S:-0.04}                # 孤立峰最小 sigma s
ISOLATED_NOISE_SIGMA_MAX_S=${ISOLATED_NOISE_SIGMA_MAX_S:-0.22}                # 孤立峰最大 sigma s
TRACK_DROP_CHANNEL_RATIO=${TRACK_DROP_CHANNEL_RATIO:-1.0}                     # 每辆车独立缺道的启用比例
TRACK_DROP_CHANNEL_MIN=${TRACK_DROP_CHANNEL_MIN:-6}                           # 每辆车最少缺失多少个通道
TRACK_DROP_CHANNEL_MAX=${TRACK_DROP_CHANNEL_MAX:-12}                          # 每辆车最多缺失多少个通道
RANDOM_DEAD_CHANNEL_RATIO=${RANDOM_DEAD_CHANNEL_RATIO:-0.85}                  # 启用整窗额外坏道的样本比例
RANDOM_DEAD_CHANNEL_MIN=${RANDOM_DEAD_CHANNEL_MIN:-6}                         # 整窗额外坏道最少条数
RANDOM_DEAD_CHANNEL_MAX=${RANDOM_DEAD_CHANNEL_MAX:-8}                        # 整窗额外坏道最多条数
DEAD_CHANNEL_INDICES=${DEAD_CHANNEL_INDICES:-}                                # 固定整窗额外坏道列表
ZERO_BACKGROUND_RATIO=${ZERO_BACKGROUND_RATIO:-0.9}                           # 启用额外缺块的样本比例
ZERO_BACKGROUND_RATE=${ZERO_BACKGROUND_RATE:-32.0}                            # 每窗期望额外缺块数
ZERO_BACKGROUND_CHANNEL_MIN=${ZERO_BACKGROUND_CHANNEL_MIN:-1}                 # 缺块最小通道宽度
ZERO_BACKGROUND_CHANNEL_MAX=${ZERO_BACKGROUND_CHANNEL_MAX:-3}                 # 缺块最大通道宽度
ZERO_BACKGROUND_DURATION_MIN_S=${ZERO_BACKGROUND_DURATION_MIN_S:-0.6}         # 缺块最短持续时间
ZERO_BACKGROUND_DURATION_MAX_S=${ZERO_BACKGROUND_DURATION_MAX_S:-4.0}         # 缺块最长持续时间
CLIP_RATIO=${CLIP_RATIO:-1.35}                                                # 输入裁剪比例
INPUT_MODE=${INPUT_MODE:-raw}                                                 # 输入特征模式
X_DTYPE=${X_DTYPE:-float16}                                                   # 保存 x 的数据类型
OVERWRITE=${OVERWRITE:-1}                                                     # 是否覆盖已有输出

overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_track_slot_dataset_from_real_npy \
  --input "$REAL_BG_INPUT" \
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
  --background-scale-min "$BACKGROUND_SCALE_MIN" \
  --background-scale-max "$BACKGROUND_SCALE_MAX" \
  --background-offset-std "$BACKGROUND_OFFSET_STD" \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh "$SPEED_MIN_KMH" \
  --speed-max-kmh "$SPEED_MAX_KMH" \
  --speed-norm-kmh "$SPEED_NORM_KMH" \
  --fixed-amp "$FIXED_AMP" \
  --sigma-seconds "$SIGMA_SECONDS" \
  --primary-ratio "$PRIMARY_RATIO" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --motion-mix "$MOTION_MIX" \
  --motion-weights "$MOTION_WEIGHTS" \
  --constant-perturb-prob "$CONSTANT_PERTURB_PROB" \
  --constant-perturb-max-frac "$CONSTANT_PERTURB_MAX_FRAC" \
  --constant-perturb-width-min "$CONSTANT_PERTURB_WIDTH_MIN" \
  --constant-perturb-width-max "$CONSTANT_PERTURB_WIDTH_MAX" \
  --smooth-speed-max-frac "$SMOOTH_SPEED_MAX_FRAC" \
  --smooth-speed-corr-channels "$SMOOTH_SPEED_CORR_CHANNELS" \
  --stop-duration-min-s "$STOP_DURATION_MIN_S" \
  --stop-duration-max-s "$STOP_DURATION_MAX_S" \
  --stop-channel-width-min "$STOP_CHANNEL_WIDTH_MIN" \
  --stop-channel-width-max "$STOP_CHANNEL_WIDTH_MAX" \
  --stop-response-sigma-scale "$STOP_RESPONSE_SIGMA_SCALE" \
  --stop-response-amp-scale "$STOP_RESPONSE_AMP_SCALE" \
  --restart-speed-ratio-min "$RESTART_SPEED_RATIO_MIN" \
  --restart-speed-ratio-max "$RESTART_SPEED_RATIO_MAX" \
  --isolated-noise-ratio "$ISOLATED_NOISE_RATIO" \
  --isolated-noise-rate "$ISOLATED_NOISE_RATE" \
  --isolated-noise-amp-min "$ISOLATED_NOISE_AMP_MIN" \
  --isolated-noise-amp-max "$ISOLATED_NOISE_AMP_MAX" \
  --isolated-noise-sigma-min-s "$ISOLATED_NOISE_SIGMA_MIN_S" \
  --isolated-noise-sigma-max-s "$ISOLATED_NOISE_SIGMA_MAX_S" \
  --per-vehicle-drop-channel-ratio "$TRACK_DROP_CHANNEL_RATIO" \
  --per-vehicle-drop-channel-min "$TRACK_DROP_CHANNEL_MIN" \
  --per-vehicle-drop-channel-max "$TRACK_DROP_CHANNEL_MAX" \
  --random-dead-channel-ratio "$RANDOM_DEAD_CHANNEL_RATIO" \
  --random-dead-channel-min "$RANDOM_DEAD_CHANNEL_MIN" \
  --random-dead-channel-max "$RANDOM_DEAD_CHANNEL_MAX" \
  --dead-channel-indices "$DEAD_CHANNEL_INDICES" \
  --zero-background-ratio "$ZERO_BACKGROUND_RATIO" \
  --zero-background-rate "$ZERO_BACKGROUND_RATE" \
  --zero-background-channel-min "$ZERO_BACKGROUND_CHANNEL_MIN" \
  --zero-background-channel-max "$ZERO_BACKGROUND_CHANNEL_MAX" \
  --zero-background-duration-min-s "$ZERO_BACKGROUND_DURATION_MIN_S" \
  --zero-background-duration-max-s "$ZERO_BACKGROUND_DURATION_MAX_S" \
  --clip-ratio "$CLIP_RATIO" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  $overwrite_args
