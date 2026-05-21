# 当前主线模型、Loss 与训练方法说明

本文档以当前仓库代码实现为准，说明 `auto_track` 里深度学习主线分支当前实际使用的模型、loss、训练方法，以及其背后的物理与数学原理。

结论先行：

- 当前代码默认主线模型是 `PeakSlotNet`，核心实现位于 `autotrack/dl/peak_slot_model.py`。
- 当前默认训练入口是 `train_peak_slot_cuda.sh` -> `autotrack/dl/train_peak_slot.py`。
- 当前默认训练脚本已经切到 `v4 noisy/bad-channel` 数据分布。
- 但当前仓库里已经存在的历史模型目录主要是 `models/peak_slot_v2_120s_cuda` 和 `models/peak_slot_v3_120s_realistic_cuda`，尚未看到已产出的 `v4` 模型目录。

因此，若说“当前代码默认用什么”，答案是：

- `PeakSlotNet`
- 训练目标面向 `peak_slot_v4_120s_noisy_badch`
- 训练脚本默认输出目录为 `models/peak_slot_v4_120s_noisy_badch_cuda`

若说“仓库里当前已经训练出来、能直接看到的模型版本”，答案是：

- `v2`
- `v3 realistic`

## 1. 当前主线为什么不是旧的 TrackSlotNet

仓库中同时保留了两条深度学习路线：

- `TrackSlotNet`：直接预测每条车轨在每个通道上的连续时间位置。
- `PeakSlotNet`：先在每个通道上检测峰候选，再让网络从候选里选出属于同一辆车的峰。

当前主线已经切到 `PeakSlotNet`，原因很明确：

1. 它不再强行回归“任意连续时间”，而是只从真实峰候选里选点。
2. 对交汇、近距离并行、噪声强峰、缺道、缺块等情况更稳。
3. 输出轨迹天然落在真实峰上，物理可解释性更强。

从表示上看，`TrackSlotNet` 学的是“连续曲线回归”，而 `PeakSlotNet` 学的是“候选峰上的集合分配问题”。

## 2. 数据表示与问题定义

### 2.1 输入

模型输入 `x` 是一个 DAS 通道-时间热图张量：

- 形状大致为 `[B, in_channels, C, T_down]`
- `C` 是通道数，当前默认 `50`
- `T_down` 是时间降采样后的长度

默认训练数据先由 `generate_track_slot_dataset.py` 生成，再由 `convert_track_slot_to_peak_slot.py` 转成 PeakSlotNet 使用的数据格式。

### 2.2 PeakSlotNet 的标签

对每个样本、每个通道，先找出最多 `K=64` 个峰候选：

```text
peak_time[c, k]
peak_amp[c, k]
peak_valid[c, k]
peak_index[c, k]
```

其中：

- `peak_time`：候选峰的归一化时间位置，范围约为 `[0, 1]`
- `peak_amp`：峰值幅值
- `peak_valid`：该候选是否存在
- `peak_index`：在降采样时间轴上的离散索引

GT 不再直接给“该车在该通道的连续时间”，而是给：

```text
gt_peak_index[g, c]
```

表示第 `g` 条 GT 轨迹在通道 `c` 上对应哪个峰候选。

若该车在该通道不可见，则标签为 `K`，即最后一个 `none` 类。

### 2.3 任务本质

因此，这个任务可以表述为：

给定一个通道-时间热图，以及每个通道的一组候选峰，预测固定数量的 `Q` 个车辆 slot，每个 slot 在每个通道上选择：

- 某个真实峰候选，或
- `none`

并额外预测：

- 该 slot 是否真的对应一辆车
- 行驶方向
- 速度
- 每个通道上的时间先验
- 每个通道上的可见性先验

## 3. 当前模型结构

当前模型类是 `PeakSlotPredictor`。

默认配置来自 `ModelConfig`：

```text
n_channels = 50
max_tracks = 96
peak_candidates = 64
hidden_dim = 128
num_heads = 4
decoder_layers = 2
pooled_channels = 8
pooled_time = 128
dropout = 0.1
```

### 3.1 整体结构概览

可以把模型理解成 4 个部分：

1. 2D 卷积 backbone：从 DAS 热图中提局部时空特征
2. Transformer decoder：用固定 `Q=96` 个 query/slot 去查询全局特征
3. 候选峰编码器：把每个候选峰编码成向量
4. 多头输出：分别输出 objectness、方向、速度、峰选择 logits、时间先验、可见性先验

### 3.2 Backbone

backbone 由 4 个卷积块组成，每个卷积块内部都是：

```text
Conv2d -> GroupNorm -> GELU -> Conv2d -> GroupNorm -> GELU
```

作用：

- 在二维 `(channel, time)` 平面中提取局部结构
- 识别车辆轨迹在热图上的斜线、局部峰簇、交汇结构

卷积后特征会被双线性插值到固定大小：

```text
(pooled_channels, pooled_time) = (8, 128)
```

这样后续 Transformer 看到的 token 数固定，训练更稳定。

### 3.3 Transformer Decoder 与固定 slot

模型使用固定数量的 slot embedding：

```text
slot_embed: Embedding(max_tracks=96, hidden_dim)
```

它们像 DETR 里的 object queries，一开始并不对应具体哪辆车，而是在训练中通过匹配逐渐学会“每个 query 去解释一条 GT 轨迹”。

Decoder 输出 `hs[b, q, h]` 后，每个 slot 都会生成：

- `objectness_logits[b, q]`
- `direction_logits[b, q, 2]`
- `speed[b, q]`
- `time_prior[b, q, c]`
- `visibility_prior_logits[b, q, c]`

### 3.4 候选峰编码与峰选择

每个候选峰被编码为 4 维输入：

```text
[peak_time, peak_amp, valid_flag, channel_position]
```

然后经 MLP 映射到 `hidden_dim`：

```text
Linear(4 -> hidden) -> GELU -> Linear(hidden -> hidden) -> LayerNorm
```

每个通道还有独立的 `channel_embed`。

最后用 slot 向量与候选峰向量做点积，得到：

```text
peak_scores[b, q, c, k]
```

再拼接一个 `none` 分支，得到：

```text
peak_logits[b, q, c, k_or_none]
```

这一步的本质是一个条件分类问题：

- 条件：当前 slot 的全局语义
- 分类对象：该通道上的各个候选峰 + none

## 4. 输出量的物理意义

### 4.1 objectness

`objectness_logits[q]` 表示这个 slot 是否真的对应一辆车。

因为 `Q=96` 通常大于窗口中的真实车辆数，所以大量 slot 应该被判为“空”。

### 4.2 direction

`direction_logits[q]` 预测轨迹方向，当前是二分类：

- `0 -> forward`
- `1 -> reverse`

方向直接决定单调性约束的符号，也影响匹配代价与后处理解码。

### 4.3 speed

`speed[q]` 是归一化速度标量。实际速度通过：

```text
v_kmh = speed[q] * speed_norm_kmh
```

默认 `speed_norm_kmh = 150.0`。

### 4.4 time_prior

`time_prior[q, c]` 是该 slot 在每个通道上的时间先验，不直接替代峰选择，而是告诉解码器：

- 这辆车大概应当出现在这个时间附近

交汇时，两个候选峰都可能“局部看起来像对的”，但时间先验能帮助保持车辆身份不切换。

### 4.5 visibility_prior

`visibility_prior_logits[q, c]` 预测该车在该通道上是否可见。

这有助于：

- 学会边界通道缺失
- 学会死道/丢块造成的局部不可见
- 让 time_prior 的监督只施加在可见通道上

## 5. 数学问题的本质

PeakSlotNet 其实把问题从“连续轨迹回归”改写成了：

```text
多实例集合预测 + 每通道离散候选分类 + 一对一匹配
```

设：

- `q = 1...Q` 是预测 slot
- `g = 1...G` 是 GT 轨迹
- `c = 1...C` 是通道
- `k = 1...K` 是候选峰

模型要学习：

1. 哪些 slot 是有效车辆
2. 每个有效 slot 对应哪条 GT
3. 对应后，在每个通道选哪一个峰

这本质上是一个带结构约束的离散组合学习问题。

## 6. 匹配方法与其数学意义

### 6.1 为什么要匹配

因为网络输出的是固定 `Q=96` 个 slot，但 GT 中车辆数 `G` 是可变的。

所以训练时必须先做“预测 slot 与 GT 轨迹”的分配，再计算监督损失。

### 6.2 当前支持的匹配器

代码支持：

- `hungarian`
- `greedy`
- `auction`
- `independent`

当前训练脚本默认：

```text
MATCHER=independent
```

这是一个高吞吐近似 GPU 匹配器，用于提升训练速度。

验证和更精确评估时，通常更推荐 `hungarian` 或 `auction`。

### 6.3 匹配代价

当前实现中的匹配代价为：

```text
cost(q, g) =
  3.0 * peak_cost(q, g)
  - 1.0 * objectness_prob(q)
  - 0.2 * direction_prob(q, dir_g)
  + 0.1 * |speed(q) - speed_g|
```

其中：

- `peak_cost(q, g)`：该 slot 在 GT 可见通道上选中 GT 峰的平均负对数似然
- `objectness_prob(q)`：slot 是真实车辆的概率
- `direction_prob(q, dir_g)`：方向预测对 GT 方向的概率
- `|speed(q) - speed_g|`：预测速度与 GT 速度的 L1 差

这表示匹配时优先考虑：

1. 峰选择是否对
2. slot 是否像一条真车轨迹
3. 方向和速度是否接近

## 7. 当前总 Loss 组成

当前 `peak_slot_set_loss()` 的总损失为：

```text
L =
  w_obj * L_obj
+ w_peak * L_peak
+ w_count * L_count
+ w_dir * L_dir
+ w_speed * L_speed
+ w_mono * L_mono
+ w_smooth * L_smooth
+ w_tprior * L_time_prior
+ w_vprior * L_visibility_prior
+ w_comp * L_slot_competition
+ w_cross * L_crossing
+ w_cov * L_gt_coverage
+ w_close * L_close_pair_separation
```

当前训练脚本默认权重为：

```text
object_loss_weight = 1.25
peak_loss_weight = 1.0
count_loss_weight = 0.15
direction_loss_weight = 0.5
speed_loss_weight = 0.25
monotonic_loss_weight = 1.0
smoothness_loss_weight = 0.2
time_prior_loss_weight = 2.0
visibility_prior_loss_weight = 0.75
slot_competition_loss_weight = 0.15
crossing_loss_weight = 0.2
gt_coverage_loss_weight = 0.5
close_pair_separation_loss_weight = 0.3
```

下面逐项解释。

### 7.1 Objectness Loss

```text
L_obj = BCEWithLogits(objectness_logits, obj_target, weight=obj_weight)
```

其中未匹配 slot 的权重较小：

```text
no_object_weight = 0.15
```

物理意义：

- 只少量惩罚空 slot，避免因为负样本太多而压制正样本学习

数学意义：

- 这是一个类别极不平衡的二分类校准项

### 7.2 峰分类 Loss

```text
L_peak = CrossEntropy(peak_logits, gt_peak_index)
```

但若 GT 在某通道不可见，对 `none` 类的权重降低为：

```text
none_weight = 0.35
```

这是整个系统的核心监督。

物理意义：

- 监督网络在每个通道上选中“真正属于该车”的那一个峰

数学意义：

- 这是一个条件离散分类问题，不再回归连续时间

### 7.3 Count Loss

```text
L_count = SmoothL1(sum(sigmoid(objectness)) , GT_count)
```

物理意义：

- 约束模型输出的有效车辆数接近真实车辆数

数学意义：

- 用 soft count 替代硬阈值计数，可微分

### 7.4 Direction Loss

```text
L_dir = CrossEntropy(direction_logits, gt_direction)
```

方向决定轨迹在通道轴上的时间单调性符号。

### 7.5 Speed Loss

```text
L_speed = SmoothL1(pred_speed, gt_speed)
```

速度预测本身不是最终轨迹点，但它参与：

- slot 与 GT 的匹配
- 后处理 Viterbi 解码时的速度先验

### 7.6 Monotonic Loss

设预测的期望时间为：

```text
E[t_{q,c}] = sum_k p_{q,c,k} * peak_time_{c,k}
```

若方向为 `forward`，则沿通道方向时间应单调增加；
若为 `reverse`，则应单调减少。

代码等价形式：

```text
L_mono = mean( relu( -(t_{c+1} - t_c) * sign(direction) ) )
```

物理意义：

- 车辆沿光纤方向传播时，轨迹在通道-时间图上应有一致方向

数学意义：

- 这是对一阶差分符号的约束

### 7.7 Smoothness Loss

对二阶差分加 SmoothL1 惩罚：

```text
d2_c = t_{c+1} - 2 t_c + t_{c-1}
L_smooth = SmoothL1(d2_c, 0)
```

物理意义：

- 车辆速度不会在相邻道之间剧烈振荡

数学意义：

- 二阶差分接近 0 表示局部近似线性，等价于斜率平滑

### 7.8 Time Prior Loss

```text
L_time_prior = SmoothL1(pred_time_prior, gt_time)
```

但只在 GT 可见通道上计算。

物理意义：

- 让 slot 在每个通道上学会“这辆车大概出现在哪”
- 在交汇、噪声峰竞争时提供全局身份保持线索

### 7.9 Visibility Prior Loss

```text
L_visibility_prior = BCEWithLogits(pred_visibility_prior, gt_visibility)
```

物理意义：

- 学会车辆在哪些通道上应该可见、哪些通道应该缺失

### 7.10 Slot Competition Loss

代码思路是：

- 先取每个 slot 对每个峰的概率
- 再乘上该 slot 的 objectness
- 然后惩罚不同 slot 在同一峰上的重叠占用

可理解为：

```text
L_comp ~ sum_{c,k} sum_{q1 != q2} p(q1,c,k) p(q2,c,k)
```

物理意义：

- 两辆车不应长期抢同一个峰

数学意义：

- 抑制 slot 间的高重叠软分配

### 7.11 Crossing Margin Loss

该项专门针对交汇和近交汇误切换。

对已匹配 slot，要求 GT 峰的 logit 至少比其他竞争峰高一个 margin：

```text
L_cross = mean( relu(margin + bad_logit - good_logit) )
```

当前内部 margin 固定为：

```text
margin = 0.75
```

物理意义：

- 在交汇区域，不仅要“选对”，还要“比错峰明显更自信”

### 7.12 GT Coverage Loss

该项要求每条 GT 至少被某个 slot 解释得足够好。

实现方式是对所有 slot 的 GT 峰 NLL 做 softmin 聚合：

```text
weights = softmax(-nll / T)
L_cov = sum_q weights_q * nll_q
```

默认温度：

```text
T = 0.2
```

物理意义：

- 防止某些 GT 轨迹被整体遗漏

数学意义：

- softmin 近似 `min_q nll(q,g)`，保持可导

### 7.13 Close-Pair Separation Loss

这是当前实现里非常关键的一项，专门处理“近车并行”。

先找出满足以下条件的 GT 轨迹对：

- 共同可见通道数足够多
- 平均时间间隔在 `[0.15s, 1.5s]`

然后要求：

- 负责 GT1 的 slot 对 GT1 的解释优于 GT2
- 负责 GT2 的 slot 对 GT2 的解释优于 GT1

形式上近似为：

```text
L_close =
  0.5 * mean(
    relu(m + nll(q1,g1) - nll(q1,g2)) +
    relu(m + nll(q2,g2) - nll(q2,g1))
  )
```

默认：

```text
close_pair_margin = 0.5
close_pair_min_common_channels = 8
close_pair_min_gap_s = 0.15
close_pair_max_gap_s = 1.5
```

物理意义：

- 两辆相距很近的车应被两个不同 slot 稳定区分，而不是并成一条

## 8. 训练方法

## 8.1 数据流程

当前主线训练流程是：

```text
generate_track_slot_dataset.py
  -> 生成 track_slot tensor shards

convert_track_slot_to_peak_slot.py
  -> 生成 peak_slot tensor shards

train_peak_slot.py
  -> 训练 PeakSlotNet
```

默认 shell 入口：

```sh
DEVICE=cuda EPOCHS=1000 BATCH_SIZE=32 sh train_peak_slot_cuda.sh
```

## 8.2 当前默认训练配置

`train_peak_slot_cuda.sh` 当前默认值：

```text
DATA_DIR = datasets/peak_slot_v4_120s_noisy_badch/train
OUT_DIR = models/peak_slot_v4_120s_noisy_badch_cuda
DEVICE = cuda
EPOCHS = 1000
BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 1e-4
MAX_TRACKS = 96
HIDDEN_DIM = 128
DECODER_LAYERS = 2
NUM_HEADS = 4
POOLED_CHANNELS = 8
POOLED_TIME = 128
AMP = on
AMP_DTYPE = float16
MATCHER = independent
VAL_FRACTION = 0.1
VAL_EVERY = 5
CHECKPOINT_EVERY = 5
SEED = 22
```

## 8.3 优化器与数值训练策略

当前训练器使用：

```text
optimizer = AdamW(lr=2e-4, weight_decay=1e-4)
```

并配合：

- CUDA AMP 混合精度
- `GradScaler`
- 梯度裁剪 `grad_clip = 1.0`
- `channels_last`
- 多 worker DataLoader
- shard 缓存

这说明当前训练方法不是单纯“把模型跑起来”，而是明显针对吞吐做过工程优化。

## 8.4 验证策略

默认验证方式有两种：

1. 从训练 shard 的尾部切出 `10%` 做验证
2. 指定独立 `VAL_DATA_DIR`

默认：

```text
VAL_FRACTION = 0.1
VAL_EVERY = 5
```

最佳模型以验证 loss 为主保存为：

- `checkpoint_last.pt`
- `checkpoint_best.pt`

## 8.5 自动续训与迁移训练

脚本支持：

- `AUTO_RESUME=1`：从 `checkpoint_last.pt` 自动续训
- `RESUME_MODEL_ONLY=1`：只加载模型参数，不恢复优化器状态

后者适用于：

- 数据分布已切换，例如从 `v3 realistic` 迁移到 `v4 noisy/bad-channel`

这是合理的，因为当数据分布变化较大时，直接恢复旧优化器动量未必稳定。

## 9. 推理与训练的差别

训练时，峰选择主要由神经网络 loss 决定。

推理时，默认还会再套一层物理约束解码，核心是 `beam_global` 模式的 beam-Viterbi 路径搜索。

默认推理配置包括：

```text
objectness_threshold = 0.35
peak_threshold = 0.4
viterbi_speed_min_kmh = 60
viterbi_speed_max_kmh = 100
viterbi_max_skip_channels = 4
time_prior_weight = 2.0
global_conflict_penalty = 2.0
```

它不是训练 loss 的一部分，但它把物理知识进一步用于最终解码：

- 方向一致
- 速度在合理范围内
- 允许少量跳道
- 惩罚轨迹抖动
- 惩罚多个 slot 抢同一峰

## 10. 物理原理

## 10.1 车辆轨迹在 DAS 图上的几何意义

设：

- 通道坐标为 `x`
- 时间坐标为 `t`
- 车辆速度为 `v`

若车辆匀速运动，则有：

```text
x = x0 + v (t - t0)
```

等价改写：

```text
t = t0 + (x - x0) / v
```

因此在 `channel-time` 图上，车辆轨迹近似是一条斜线。

若用离散通道表示，且通道间距为 `dx_m`，相邻可见点时间差应满足：

```text
dt ≈ dx / v
```

这正是：

- 单调性约束
- 速度窗约束
- 平滑性约束

这些 loss 与后处理规则的物理基础。

## 10.2 为什么交汇会困难

两辆车交汇时，在某些通道上会出现：

- 时间位置非常接近的两个峰
- 局部看都“像是对的”

此时如果只看局部分类，很容易发生 identity switch。

所以系统必须额外引入：

- 全局 slot 表示
- 方向预测
- 速度预测
- time prior
- crossing margin
- slot competition
- global conflict penalty

这些机制本质上都在做同一件事：

局部峰看起来相似时，用全局轨迹一致性来决定归属。

## 10.3 为什么不能只做连续时间回归

连续回归的核心问题是：

- 模型可能回归到“并不存在峰”的位置
- 真实观测是离散可见峰，不是光滑理想曲线

PeakSlotNet 改成候选峰选择后，等于把输出空间约束到“真实可观测事件集合”上：

```text
预测轨迹点 ∈ 检测到的峰候选集合
```

这使得：

- 输出更物理
- 交汇时更稳
- 对缺失通道更自然

## 11. 数学原理总结

从数学角度，当前主线方法可以概括为：

1. 用卷积网络提取时空局部特征
2. 用 Transformer query 表示可变数量车辆实例
3. 用一对一匹配把固定 slot 与可变 GT 集合对齐
4. 用离散峰分类替代连续时间回归
5. 用一阶/二阶差分损失注入运动学先验
6. 用 coverage / competition / crossing / close-pair loss 解决多车竞争与交汇问题
7. 用 Viterbi 风格物理解码进一步将局部概率转成全局最优轨迹

所以它不是单一网络，而是一套：

```text
候选峰检测
+ 实例集合预测
+ 结构化匹配
+ 物理先验正则
+ 约束解码
```

的组合系统。

## 12. 当前主线的最准确描述

如果要用一句尽量准确的话概括当前系统，可以写成：

> 当前 `auto_track` 深度学习主线采用 `PeakSlotNet`。它先在每个 DAS 通道上提取峰候选，再用固定数量的 Transformer slot 对车辆实例做集合预测，并通过峰分类、一对一匹配、计数约束、方向速度监督、单调性/平滑性正则、交汇与近车分离损失来学习轨迹归属；推理阶段再结合速度窗、方向一致性、time prior 和全局冲突惩罚做 beam-Viterbi 物理解码。

## 13. 相关代码位置

- 模型定义：`autotrack/dl/peak_slot_model.py`
- 训练入口：`autotrack/dl/train_peak_slot.py`
- 默认训练脚本：`train_peak_slot_cuda.sh`
- 数据生成：`autotrack/dl/generate_track_slot_dataset.py`
- 数据转换：`autotrack/dl/convert_track_slot_to_peak_slot.py`
- 旧版设计说明：`docs/peak_slot_network.md`

## 14. 建议

如果你后面要继续做实验，我建议以后统一把版本描述拆成三层：

1. `代码主线版本`
2. `默认训练配置版本`
3. `当前最佳已训练模型版本`

因为现在仓库里这三者已经不完全相同：

- 代码主线：PeakSlotNet v4 思路
- 默认训练配置：`peak_slot_v4_120s_noisy_badch`
- 已存在训练产物：主要还是 `v2/v3`

这样写文档时最不容易混淆。
