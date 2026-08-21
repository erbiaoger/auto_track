#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

REALISM_PRESET=${REALISM_PRESET:-xi_gauss_50_noisy_badch}             # 真实化预设名称，仅写入元数据便于追踪
OUT_DIR=${OUT_DIR:-datasets/track_slot_v4_120s_noisy_badch/train}    # track_slot 数据集输出目录
NUM_SAMPLES=${NUM_SAMPLES:-40000}                                    # 总样本数
SHARD_SIZE=${SHARD_SIZE:-256}                                        # 每个 shard 的样本数
N_CH=${N_CH:-50}                                                     # 通道数
FS=${FS:-1000}                                                       # 采样率 Hz
WINDOW_SECONDS=${WINDOW_SECONDS:-120}                                # 窗口长度 s
TIME_DOWNSAMPLE=${TIME_DOWNSAMPLE:-10}                               # 时间降采样步长
DX_M=${DX_M:-100}                                                    # 通道间距 m
VEHICLES_MIN=${VEHICLES_MIN:-24}                                     # 每窗最少车辆数
VEHICLES_MAX=${VEHICLES_MAX:-36}                                     # 每窗最多车辆数
SPEED_MIN_KMH=${SPEED_MIN_KMH:-70}                                   # 最低车速 km/h
SPEED_MAX_KMH=${SPEED_MAX_KMH:-85}                                   # 最高车速 km/h
MOTION_MIX=${MOTION_MIX:-constant_sparse,smooth_random,stop_go}      # 运动模型混合
MOTION_WEIGHTS=${MOTION_WEIGHTS:-0.84,0.15,0.01}                     # 运动模型权重
CONSTANT_PERTURB_PROB=${CONSTANT_PERTURB_PROB:-0.05}                 # 稀疏局部速度扰动概率
CONSTANT_PERTURB_MAX_FRAC=${CONSTANT_PERTURB_MAX_FRAC:-0.01}         # 稀疏速度扰动最大比例
CONSTANT_PERTURB_WIDTH_MIN=${CONSTANT_PERTURB_WIDTH_MIN:-1}          # 稀疏扰动最小通道宽度
CONSTANT_PERTURB_WIDTH_MAX=${CONSTANT_PERTURB_WIDTH_MAX:-2}          # 稀疏扰动最大通道宽度
SMOOTH_SPEED_MAX_FRAC=${SMOOTH_SPEED_MAX_FRAC:-0.05}                 # 平滑速度起伏最大比例
SMOOTH_SPEED_CORR_CHANNELS=${SMOOTH_SPEED_CORR_CHANNELS:-8}          # 平滑速度起伏相关宽度
STOP_DURATION_MIN_S=${STOP_DURATION_MIN_S:-1.0}                      # stop-go 最短停顿时间 s
STOP_DURATION_MAX_S=${STOP_DURATION_MAX_S:-8.0}                      # stop-go 最长停顿时间 s
STOP_CHANNEL_WIDTH_MIN=${STOP_CHANNEL_WIDTH_MIN:-1}                  # stop-go 最小影响宽度
STOP_CHANNEL_WIDTH_MAX=${STOP_CHANNEL_WIDTH_MAX:-3}                  # stop-go 最大影响宽度
STOP_RESPONSE_SIGMA_SCALE=${STOP_RESPONSE_SIGMA_SCALE:-3.0}          # stop-go 附近高斯窗宽度放大倍数
STOP_RESPONSE_AMP_SCALE=${STOP_RESPONSE_AMP_SCALE:-1.2}              # stop-go 附近高斯窗幅值放大倍数
RESTART_SPEED_RATIO_MIN=${RESTART_SPEED_RATIO_MIN:-0.95}             # stop 后恢复速度最小比例
RESTART_SPEED_RATIO_MAX=${RESTART_SPEED_RATIO_MAX:-1.05}             # stop 后恢复速度最大比例
NOISE_STD=${NOISE_STD:-0.04}                                         # 连续白噪声强度
COLORED_NOISE_STD=${COLORED_NOISE_STD:-0.08}                         # 连续相关噪声强度
COLORED_NOISE_CORR_S=${COLORED_NOISE_CORR_S:-0.8}                    # 相关噪声时间相关长度 s
CHANNEL_BIAS_STD=${CHANNEL_BIAS_STD:-0.03}                           # 通道常值偏置强度
CHANNEL_GAIN_STD=${CHANNEL_GAIN_STD:-0.08}                           # 通道增益扰动强度
BASELINE_DRIFT_STD=${BASELINE_DRIFT_STD:-0.04}                       # 慢变基线漂移强度
BASELINE_DRIFT_CORR_S=${BASELINE_DRIFT_CORR_S:-6.0}                  # 基线漂移相关长度 s
DEAD_CHANNEL_INDICES=${DEAD_CHANNEL_INDICES:-}                       # 固定坏道列表，逗号分隔；当前默认不启用
RANDOM_DEAD_CHANNEL_RATIO=${RANDOM_DEAD_CHANNEL_RATIO:-0.30}         # 启用随机坏道的样本比例
RANDOM_DEAD_CHANNEL_MIN=${RANDOM_DEAD_CHANNEL_MIN:-2}                # 随机坏道最少条数
RANDOM_DEAD_CHANNEL_MAX=${RANDOM_DEAD_CHANNEL_MAX:-10}               # 随机坏道最多条数
ZERO_BACKGROUND_RATIO=${ZERO_BACKGROUND_RATIO:-1.0}                  # 启用局部缺失块的样本比例
ZERO_BACKGROUND_RATE=${ZERO_BACKGROUND_RATE:-45.0}                   # 每个样本期望缺失块数量
ZERO_BACKGROUND_CHANNEL_MIN=${ZERO_BACKGROUND_CHANNEL_MIN:-1}        # 缺失块最小通道宽度
ZERO_BACKGROUND_CHANNEL_MAX=${ZERO_BACKGROUND_CHANNEL_MAX:-4}        # 缺失块最大通道宽度
ZERO_BACKGROUND_DURATION_MIN_S=${ZERO_BACKGROUND_DURATION_MIN_S:-0.8} # 缺失块最短持续时间 s
ZERO_BACKGROUND_DURATION_MAX_S=${ZERO_BACKGROUND_DURATION_MAX_S:-4.0} # 缺失块最长持续时间 s
AMP_MIN=${AMP_MIN:-6.0}                                              # 车辆高斯窗最小幅值
AMP_MAX=${AMP_MAX:-6.0}                                              # 车辆高斯窗最大幅值
SIGMA_SECONDS=${SIGMA_SECONDS:-0.25}                                 # 车辆高斯窗标准差 s
PRIMARY_RATIO=${PRIMARY_RATIO:-0.8333333333}                         # 正向车辆比例
INTERACTION_RATIO=${INTERACTION_RATIO:-0.3}                          # 含交汇/超车/近距并行样本比例
INTERACTION_TYPES=${INTERACTION_TYPES:-crossing,overtake,near_parallel} # 困难交互类型
INTERACTION_TIME_MIN_FRAC=${INTERACTION_TIME_MIN_FRAC:-0.05}         # 交互最早发生时间占比
INTERACTION_TIME_MAX_FRAC=${INTERACTION_TIME_MAX_FRAC:-0.95}         # 交互最晚发生时间占比
ISOLATED_NOISE_RATIO=${ISOLATED_NOISE_RATIO:-1.0}                    # 启用孤立高斯干扰的样本比例
ISOLATED_NOISE_RATE=${ISOLATED_NOISE_RATE:-280.0}                    # 每个样本期望孤立高斯干扰数量
ISOLATED_NOISE_AMP_MIN=${ISOLATED_NOISE_AMP_MIN:-1.0}                # 孤立高斯干扰最小幅值
ISOLATED_NOISE_AMP_MAX=${ISOLATED_NOISE_AMP_MAX:-6.0}                # 孤立高斯干扰最大幅值
ISOLATED_NOISE_SIGMA_MIN=${ISOLATED_NOISE_SIGMA_MIN:-0.08}           # 孤立高斯干扰最小 sigma s
ISOLATED_NOISE_SIGMA_MAX=${ISOLATED_NOISE_SIGMA_MAX:-0.35}           # 孤立高斯干扰最大 sigma s
INPUT_MODE=${INPUT_MODE:-raw}                                        # 输入特征模式
X_DTYPE=${X_DTYPE:-float16}                                          # 保存到磁盘的数据类型
MIN_VISIBLE_CHANNELS=${MIN_VISIBLE_CHANNELS:-4}                      # 车辆至少可见通道数；避免专门生成短可见轨迹
WORKERS=${WORKERS:-100}                                               # 并行生成 worker 数
SEED=${SEED:-42}                                                     # 随机种子
OVERWRITE=${OVERWRITE:-1}                                            # 是否覆盖已有输出目录


overwrite_args=""
if [ "$OVERWRITE" = "1" ] || [ "$OVERWRITE" = "true" ]; then
  overwrite_args="--overwrite"
fi

uv run python -m autotrack.dl.generate_track_slot_dataset \
  --out-dir "$OUT_DIR" \
  --realism-preset "$REALISM_PRESET" \
  --num-samples "$NUM_SAMPLES" \
  --shard-size "$SHARD_SIZE" \
  --n-ch "$N_CH" \
  --fs "$FS" \
  --window-seconds "$WINDOW_SECONDS" \
  --time-downsample "$TIME_DOWNSAMPLE" \
  --dx-m "$DX_M" \
  --vehicles-min "$VEHICLES_MIN" \
  --vehicles-max "$VEHICLES_MAX" \
  --speed-min-kmh "$SPEED_MIN_KMH" \
  --speed-max-kmh "$SPEED_MAX_KMH" \
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
  --noise-std "$NOISE_STD" \
  --colored-noise-std "$COLORED_NOISE_STD" \
  --colored-noise-corr-s "$COLORED_NOISE_CORR_S" \
  --channel-bias-std "$CHANNEL_BIAS_STD" \
  --channel-gain-std "$CHANNEL_GAIN_STD" \
  --baseline-drift-std "$BASELINE_DRIFT_STD" \
  --baseline-drift-corr-s "$BASELINE_DRIFT_CORR_S" \
  --dead-channel-indices "$DEAD_CHANNEL_INDICES" \
  --random-dead-channel-ratio "$RANDOM_DEAD_CHANNEL_RATIO" \
  --random-dead-channel-min "$RANDOM_DEAD_CHANNEL_MIN" \
  --random-dead-channel-max "$RANDOM_DEAD_CHANNEL_MAX" \
  --zero-background-ratio "$ZERO_BACKGROUND_RATIO" \
  --zero-background-rate "$ZERO_BACKGROUND_RATE" \
  --zero-background-channel-min "$ZERO_BACKGROUND_CHANNEL_MIN" \
  --zero-background-channel-max "$ZERO_BACKGROUND_CHANNEL_MAX" \
  --zero-background-duration-min-s "$ZERO_BACKGROUND_DURATION_MIN_S" \
  --zero-background-duration-max-s "$ZERO_BACKGROUND_DURATION_MAX_S" \
  --amp-min "$AMP_MIN" \
  --amp-max "$AMP_MAX" \
  --sigma-min-s "$SIGMA_SECONDS" \
  --sigma-max-s "$SIGMA_SECONDS" \
  --primary-ratio "$PRIMARY_RATIO" \
  --interaction-ratio "$INTERACTION_RATIO" \
  --interaction-types "$INTERACTION_TYPES" \
  --interaction-time-min-frac "$INTERACTION_TIME_MIN_FRAC" \
  --interaction-time-max-frac "$INTERACTION_TIME_MAX_FRAC" \
  --isolated-noise-ratio "$ISOLATED_NOISE_RATIO" \
  --isolated-noise-rate "$ISOLATED_NOISE_RATE" \
  --isolated-noise-amp-min "$ISOLATED_NOISE_AMP_MIN" \
  --isolated-noise-amp-max "$ISOLATED_NOISE_AMP_MAX" \
  --isolated-noise-sigma-min "$ISOLATED_NOISE_SIGMA_MIN" \
  --isolated-noise-sigma-max "$ISOLATED_NOISE_SIGMA_MAX" \
  --min-visible-channels "$MIN_VISIBLE_CHANNELS" \
  --input-mode "$INPUT_MODE" \
  --x-dtype "$X_DTYPE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  $overwrite_args
