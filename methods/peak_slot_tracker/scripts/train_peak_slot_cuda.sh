#!/usr/bin/env sh
set -eu

# 用法示例：
# 1) 正常训练当前 v4 noisy/bad-channel 数据：
#    DEVICE=cuda sh train_peak_slot_cuda.sh
#
# 2) 用旧的 v3 checkpoint 作为初始化，再训练当前 v4 noisy/bad-channel 数据：
#    DATA_DIR=datasets/peak_slot_v4_120s_noisy_badch/train \
#    OUT_DIR=models/peak_slot_v4_120s_noisy_badch_cuda \
#    RESUME=models/peak_slot_v3_120s_realistic_cuda/checkpoint_best.pt \
#    RESUME_MODEL_ONLY=1 \
#    AUTO_RESUME=0 \
#    DEVICE=cuda \
#    EPOCHS=200 \
#    sh train_peak_slot_cuda.sh
#
# 说明：
# `RESUME_MODEL_ONLY=1` 只导入模型权重，不恢复 optimizer 和历史训练状态。
# 当训练数据分布已经切到 v4 noisy/bad-channel 时，这种做法比完整 resume 更稳。

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$(cd "$SCRIPT_DIR/../.." && pwd)"

DATA_DIR=${DATA_DIR:-datasets/peak_slot_v4_120s_noisy_badch/train}   # 训练数据目录
OUT_DIR=${OUT_DIR:-models/peak_slot_v4_120s_noisy_badch_cuda}        # 模型输出目录
DEVICE=${DEVICE:-cuda}                                                # 训练设备
EPOCHS=${EPOCHS:-1000}                                                # 目标总 epoch 数
BATCH_SIZE=${BATCH_SIZE:-32}                                          # batch 大小
MAX_SAMPLES=${MAX_SAMPLES:-0}                                         # 限制训练样本数，0 表示全部
NUM_WORKERS=${NUM_WORKERS:-16}                                        # DataLoader worker 数
PIN_MEMORY=${PIN_MEMORY:-1}                                           # 是否开启 pin memory
PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-1}                           # 是否复用 DataLoader worker
PREFETCH_FACTOR=${PREFETCH_FACTOR:-4}                                 # DataLoader 预取倍数
WORKER_SHARD_CACHE_SIZE=${WORKER_SHARD_CACHE_SIZE:-2}                 # 每个 DataLoader worker 最多缓存多少个 shard，避免内存随 epoch 增长
MULTIPROCESSING_CONTEXT=${MULTIPROCESSING_CONTEXT:-auto}              # DataLoader worker 启动方式；Unix 下 auto 默认 fork
DATALOADER_TIMEOUT=${DATALOADER_TIMEOUT:-0}                           # DataLoader 卡住多久后报错，0 表示不超时
LR=${LR:-0.0002}                                                      # 学习率
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001}                                  # 权重衰减
MAX_TRACKS=${MAX_TRACKS:-96}                                          # 模型最大 slot 数
HIDDEN_DIM=${HIDDEN_DIM:-128}                                         # Transformer 隐层维度
DECODER_LAYERS=${DECODER_LAYERS:-2}                                   # Decoder 层数
NUM_HEADS=${NUM_HEADS:-4}                                             # 多头注意力头数
POOLED_CHANNELS=${POOLED_CHANNELS:-8}                                 # 通道池化尺寸
POOLED_TIME=${POOLED_TIME:-128}                                       # 时间池化尺寸
AMP=${AMP:-on}                                                        # 是否开启混合精度
AMP_DTYPE=${AMP_DTYPE:-float16}                                       # 混合精度类型
INPUT_TRANSFER_DTYPE=${INPUT_TRANSFER_DTYPE:-auto}                    # 输入传输到设备时的数据类型
MATCHER=${MATCHER:-independent}                                      # 匹配器类型；independent 为高吞吐近似 GPU 匹配
NO_OBJECT_WEIGHT=${NO_OBJECT_WEIGHT:-0.15}                            # 无目标类别权重
NONE_WEIGHT=${NONE_WEIGHT:-0.35}                                      # none 类别权重
OBJECT_LOSS_WEIGHT=${OBJECT_LOSS_WEIGHT:-1.25}                        # objectness loss 权重
COUNT_LOSS_WEIGHT=${COUNT_LOSS_WEIGHT:-0.15}                          # 计数 loss 权重
MONOTONIC_LOSS_WEIGHT=${MONOTONIC_LOSS_WEIGHT:-1.0}                   # 单调性 loss 权重
SMOOTHNESS_LOSS_WEIGHT=${SMOOTHNESS_LOSS_WEIGHT:-0.2}                 # 平滑性 loss 权重
TIME_PRIOR_LOSS_WEIGHT=${TIME_PRIOR_LOSS_WEIGHT:-2.0}                 # 时间先验 loss 权重
VISIBILITY_PRIOR_LOSS_WEIGHT=${VISIBILITY_PRIOR_LOSS_WEIGHT:-0.75}    # 可见性先验 loss 权重
SLOT_COMPETITION_LOSS_WEIGHT=${SLOT_COMPETITION_LOSS_WEIGHT:-0.15}    # slot 竞争 loss 权重
CROSSING_LOSS_WEIGHT=${CROSSING_LOSS_WEIGHT:-0.2}                     # 交叉场景 loss 权重
GT_COVERAGE_LOSS_WEIGHT=${GT_COVERAGE_LOSS_WEIGHT:-0.5}               # 每个 GT 至少被一个 slot 解释的覆盖 loss 权重
GT_COVERAGE_TEMPERATURE=${GT_COVERAGE_TEMPERATURE:-0.2}               # GT coverage softmin 温度
CLOSE_PAIR_SEPARATION_LOSS_WEIGHT=${CLOSE_PAIR_SEPARATION_LOSS_WEIGHT:-0.3} # 近车 pair 分离 loss 权重
CLOSE_PAIR_MARGIN=${CLOSE_PAIR_MARGIN:-0.5}                           # 近车 pair 分离 margin
CLOSE_PAIR_MIN_COMMON_CHANNELS=${CLOSE_PAIR_MIN_COMMON_CHANNELS:-8}   # 判定近车 pair 的最少共同可见通道数
CLOSE_PAIR_MIN_GAP_S=${CLOSE_PAIR_MIN_GAP_S:-0.15}                    # 判定近车 pair 的最小平均时间差
CLOSE_PAIR_MAX_GAP_S=${CLOSE_PAIR_MAX_GAP_S:-1.5}                     # 判定近车 pair 的最大平均时间差
METRIC_OBJECTNESS_THRESHOLD=${METRIC_OBJECTNESS_THRESHOLD:-0.35}      # 评估 objectness 阈值
METRIC_POINT_THRESHOLD=${METRIC_POINT_THRESHOLD:-0.05}                # 评估点阈值
VAL_DATA_DIR=${VAL_DATA_DIR:-}                                        # 独立验证集目录，留空则使用切分
VAL_FRACTION=${VAL_FRACTION:-0.1}                                     # 训练集尾部分出的验证比例
VAL_EVERY=${VAL_EVERY:-5}                                             # 每多少 epoch 验证一次
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:-0}                                 # 验证样本上限，0 表示全部
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-5}                               # 每多少 epoch 保存一次 checkpoint
METRICS_EVERY=${METRICS_EVERY:-0}                                     # 每多少 step 汇总一次重指标；0 表示训练热路径关闭
LOG_EVERY=${LOG_EVERY:-200}                                           # 每多少 step 打印一次日志
TIMING_EVERY=${TIMING_EVERY:-1}                                       # 每多少 epoch 打印一次耗时统计
PROFILE_STEPS=${PROFILE_STEPS:-8}                                     # profile 的 step 数
PROFILE_WARMUP=${PROFILE_WARMUP:-2}                                   # profile 预热 step 数
RESUME=${RESUME:-}                                                    # 手动指定 resume checkpoint；可指向旧 v3 模型
AUTO_RESUME=${AUTO_RESUME:-1}                                         # 是否自动从当前 OUT_DIR 的 last checkpoint 续训
RESUME_MODEL_ONLY=${RESUME_MODEL_ONLY:-0}                             # 是否只恢复模型权重；迁移到新数据分布时建议设为 1
SEED=${SEED:-22}                                                      # 随机种子

resume_args=""
if [ -n "$RESUME" ]; then
  resume_args="--resume $RESUME"
  if [ "$RESUME_MODEL_ONLY" = "1" ] || [ "$RESUME_MODEL_ONLY" = "true" ]; then
    resume_args="$resume_args --resume-model-only"
  fi
elif [ "$AUTO_RESUME" = "1" ] || [ "$AUTO_RESUME" = "true" ]; then
  resume_args="--auto-resume"
fi

val_data_args=""
if [ -n "$VAL_DATA_DIR" ]; then
  val_data_args="--val-data-dir $VAL_DATA_DIR"
fi

loader_args="--num-workers $NUM_WORKERS --prefetch-factor $PREFETCH_FACTOR"
loader_args="$loader_args --worker-shard-cache-size $WORKER_SHARD_CACHE_SIZE --multiprocessing-context $MULTIPROCESSING_CONTEXT --dataloader-timeout $DATALOADER_TIMEOUT"
if [ "$PIN_MEMORY" = "1" ] || [ "$PIN_MEMORY" = "true" ]; then
  loader_args="$loader_args --pin-memory"
fi
if [ "$PERSISTENT_WORKERS" = "1" ] || [ "$PERSISTENT_WORKERS" = "true" ]; then
  loader_args="$loader_args --persistent-workers"
fi

uv run python -m autotrack.dl.train_peak_slot \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  $loader_args \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --max-tracks "$MAX_TRACKS" \
  --hidden-dim "$HIDDEN_DIM" \
  --decoder-layers "$DECODER_LAYERS" \
  --num-heads "$NUM_HEADS" \
  --pooled-channels "$POOLED_CHANNELS" \
  --pooled-time "$POOLED_TIME" \
  --amp "$AMP" \
  --amp-dtype "$AMP_DTYPE" \
  --input-transfer-dtype "$INPUT_TRANSFER_DTYPE" \
  --channels-last \
  --matcher "$MATCHER" \
  --no-object-weight "$NO_OBJECT_WEIGHT" \
  --none-weight "$NONE_WEIGHT" \
  --object-loss-weight "$OBJECT_LOSS_WEIGHT" \
  --count-loss-weight "$COUNT_LOSS_WEIGHT" \
  --monotonic-loss-weight "$MONOTONIC_LOSS_WEIGHT" \
  --smoothness-loss-weight "$SMOOTHNESS_LOSS_WEIGHT" \
  --time-prior-loss-weight "$TIME_PRIOR_LOSS_WEIGHT" \
  --visibility-prior-loss-weight "$VISIBILITY_PRIOR_LOSS_WEIGHT" \
  --slot-competition-loss-weight "$SLOT_COMPETITION_LOSS_WEIGHT" \
  --crossing-loss-weight "$CROSSING_LOSS_WEIGHT" \
  --gt-coverage-loss-weight "$GT_COVERAGE_LOSS_WEIGHT" \
  --gt-coverage-temperature "$GT_COVERAGE_TEMPERATURE" \
  --close-pair-separation-loss-weight "$CLOSE_PAIR_SEPARATION_LOSS_WEIGHT" \
  --close-pair-margin "$CLOSE_PAIR_MARGIN" \
  --close-pair-min-common-channels "$CLOSE_PAIR_MIN_COMMON_CHANNELS" \
  --close-pair-min-gap-s "$CLOSE_PAIR_MIN_GAP_S" \
  --close-pair-max-gap-s "$CLOSE_PAIR_MAX_GAP_S" \
  --metric-objectness-threshold "$METRIC_OBJECTNESS_THRESHOLD" \
  --metric-point-threshold "$METRIC_POINT_THRESHOLD" \
  $val_data_args \
  --val-fraction "$VAL_FRACTION" \
  --val-every "$VAL_EVERY" \
  --val-max-samples "$VAL_MAX_SAMPLES" \
  --checkpoint-every "$CHECKPOINT_EVERY" \
  --metrics-every "$METRICS_EVERY" \
  --log-every "$LOG_EVERY" \
  --timing-every "$TIMING_EVERY" \
  --profile-steps "$PROFILE_STEPS" \
  --profile-warmup "$PROFILE_WARMUP" \
  --seed "$SEED" \
  $resume_args
