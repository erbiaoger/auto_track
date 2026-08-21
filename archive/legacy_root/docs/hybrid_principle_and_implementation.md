# Hybrid 车辆轨迹识别：原理与实现

本文总结当前项目中 Hybrid vehicle tracker 的设计思想、推理链路、训练方式和工程接口。
文档以当前 active preset `hybrid_v9_morphology_16384` 和
`hybrid_vehicle_tracker_20260724` 中的实现为准；早期 v3/v6/v7 结果只作为历史版本，
不与当前实现混用。

## 1. 一句话概括

Hybrid 是一个“学习感知 + 物理约束 + 图优化”的 DAS 车辆跟踪器：

```text
Raw / Pre / Gauss
        │
        ▼
20 Hz 五平面特征 + 真实站点坐标
        │
        ▼
多模态 ResUNet：检测中心线、慢度、交叉概率、轨迹 embedding
        │
        ├──────────────► learned Hough
        │
        └──────────────► 模态证据 Hough（physics-only）
                                  │
                                  ▼
                    强/弱峰观测 + Hough seeds
                                  │
                                  ▼
                    物理速度门控候选图
                                  │
                                  ▼
                         两跳 Edge-GNN
                                  │
                                  ▼
                       Beam search 生成候选轨迹
                                  │
                                  ▼
                 MILP 全局选择 + bounded-slowness 拟合
                                  │
                                  ▼
               observed 点 / expected 补点 / VehicleTrack
```

核心目标不是让神经网络直接回归一条轨迹，而是让神经网络提供多模态证据，随后由速度、
方向、站点间距和轨迹连续性约束负责筛掉不符合物理规律的峰关联。

## 2. 为什么要采用 Hybrid

DAS 数据中车辆信号通常表现为站点-时间平面上的斜线，但实际输入还会包含：

- 噪声、孤立峰和局部成组干扰峰；
- 固定坏道、临时 outage 和连续缺道；
- 同向近似平行车辆和交叉车辆；
- 车辆在窗口边界进入或离开，导致轨迹只显示一部分；
- 站点实际间距不均匀，且数组顺序可能与车辆行驶方向相反。

单纯峰值检测容易把干扰峰连成线；单纯深度模型又可能受输入形态变化影响，或者产生不满足
速度约束的预测。因此 Hybrid 将两类信息分工：

| 信息 | 负责的部分 |
| --- | --- |
| ResUNet | 从 Raw/Pre/Gauss 的联合形态中提取事件、慢度、交叉和身份特征 |
| Hough | 把站点-时间上的局部证据汇聚成全局车辆直线假设 |
| 物理门控 | 只允许方向正确、速度位于 60–90 km/h 的观测连边 |
| Edge-GNN | 根据节点证据、边速度、embedding 和交叉状态重新评估边可信度 |
| Beam search | 在每个 Hough seed 下保留有限个高分路径 |
| MILP | 在所有候选路径之间做全局去冲突选择 |
| 有界拟合 | 在速度范围内修正慢度，补出缺道站点并计算诊断量 |

## 3. 输入与物理坐标

### 3.1 三个对齐模态

`HybridVehicleTracker.predict()` 接受形状为 `[time, station]` 的三个数组：

- `raw`：原始或预处理后的波形能量；
- `pre`：Pre 模态，车辆事件中心通常表现为负谷；
- `gauss`：高斯峰/峰窗模态，作为显式的峰证据。

三个数组必须形状一致，默认采样率为 `1000 Hz`，默认识别窗口为 `120 s`。

### 3.2 mapping 是物理约束的基础

mapping JSON 提供每个通道的 `station_id`、`channel_index` 和物理位置。实现会把位置解析成米，
并要求通道连续、物理位置严格递增。算法使用真实位置差 `dx`，不使用“通道号 × 100 m”代替，
因此可以正确处理非均匀站距。

方向由 `motion_direction` 显式表示：

- `+1`：时间随物理位置增大而增加；
- `-1`：时间增加时物理位置减小。

当前 DAY11 preset 使用 `-1`，即高位置端向低位置端行驶。模型配置和关联配置的方向必须一致。

## 4. 特征构造

实现位置：[`data/features.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/data/features.py)。

原始 1000 Hz 数据按 `1000 / 20 = 50` 个采样点聚合为 20 Hz 的特征图，得到五个平面：

| 平面 | 构造方式 | 作用 |
| --- | --- | --- |
| Raw plane | Raw 的 RMS，经 `log1p` 和 robust z-score 后截断到 `[-1, 1]` | 保留波形能量和局部形态 |
| Pre score | Pre 的时间 bin 最大值，再做分位数归一化 | 提供 Pre 模态事件强度 |
| Gauss score | Gauss 的时间 bin 最大值并截断到 `[0, 1]` | 提供峰形态和峰位置证据 |
| Quality plane | 有限值比例 × 通道波动性，并按全局参考归一化 | 标记坏道或低质量站点 |
| Coordinate plane | 站点相对物理位置归一化 | 让网络看到真实空间位置 |

网络输入张量形状为 `[batch, 5, station, time_bin]`，当前 120 s 窗口对应约 `50 × 2400` 的
站点-时间网格。

由于 Pre 的事件是负谷，推理时还构造 `pre_event_score = 1 - pre_score`，用于与 Gauss、Raw
组成独立的物理证据平面。

## 5. 感知网络：MultiModalResUNet

实现位置：[`models/resunet.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/resunet.py)。

网络是一个轻量二维 ResUNet：

1. 残差 stem；
2. 两级下采样 encoder；
3. 四个时间维膨胀率为 `2/4/8/16` 的 context residual block；
4. 两级带 skip connection 的 decoder；
5. 从 decoder feature map 输出多任务 head。

当前 v9 参数为 `base_channels=12`、`embedding_dim=32`，输出包括：

- `centerline_logits`：车辆中心线/事件概率；
- `slowness`：局部慢度预测；
- `crossing_logits`：交叉或关联歧义概率；
- `embedding`：L2 归一化的 32 维局部轨迹身份特征；
- `feature_map`：供 Physical Hough 使用的中间特征。

推理时，learned network probability 不是单独使用，而是按当前实现融合为：

```text
network_probability
  = 0.55 × learned_probability
  + 0.20 × pre_event_score
  + 0.15 × gauss_score
  + 0.10 × raw_score
```

同时保留一张 physics-only 模态证据图：

```text
modal_probability
  = 0.45 × pre_event_score
  + 0.40 × gauss_score
  + 0.15 × raw_score
```

这样做的意义是：当合成训练得到的网络对真实峰形态不够鲁棒时，模态证据仍可以支持物理 Hough
和后续关联，而不是完全依赖 learned probability。

## 6. Physical Hough：把局部证据变成全局轨迹假设

实现位置：[`models/physical_hough.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/physical_hough.py)。

车辆轨迹在物理位置 `x` 和时间 `t` 上近似为：

```text
t(x) = b + s × (x - x0)
```

其中 `s = dt/dx` 是慢度，速度关系为：

```text
v(km/h) = 3.6 / |s|
```

当前速度范围 `60–90 km/h` 对应慢度范围：

```text
3.6 / 90 <= |s| <= 3.6 / 60  (s/m)
```

实现不在所有任意直线上搜索，而是：

- 使用 17 个 slope bins；
- 使用 `0.5 s` 的 intercept 网格；
- slope 符号由 `motion_direction` 决定；
- 对每条候选直线通过双线性采样汇聚站点证据；
- 计算 mean、max、std、active fraction、top-quarter mean 和站点覆盖率；
- 用小型 MLP 输出 learned Hough score，同时保留 raw physical score。

Hough 搜索是可微的，训练时可以用 GT 车辆的 slope/intercept 对网格邻域进行监督；推理时从
learned Hough 和 physics-only Hough 各自选取 top-k seed，再合并去重。当前 `hough_top_k=64`，
要求至少有 `min_observations=5` 个有效站点支撑。

## 7. 观测点提取

实现位置：[`data/peaks.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/data/peaks.py)。

每个通道上同时提取三类候选：

1. Gauss 强峰：`gauss >= strong_gauss_threshold`，当前阈值为 `0.50`；
2. Pre 弱峰：对 `1 - calibrated_pre` 做高分位数峰检测，当前使用 `0.997` 分位数且不低于 `0.65`；
3. 网络峰：在 20 Hz network probability 上找局部峰，再映射回原始采样时间。

候选峰的最小时间间隔当前为 `1.25 s`。来自不同来源且相差不超过 `0.20 s` 的峰会合并，
并为每个观测保存：

- 通道、站点 ID、物理位置和精确时间；
- Gauss、Pre、Raw、网络和 crossing 分数；
- 是否为强观测；
- ResUNet embedding；
- `evidence_score`。

`evidence_score` 是四种证据的互补融合：

```text
evidence = 1 - (1-gauss) × (1-pre) × (1-raw) × (1-network)
```

最后按证据分数截断到 `max_candidates=1200`，避免后续图搜索规模失控。

## 8. 物理门控候选图与 Edge-GNN

实现位置：[`association/graph.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/graph.py)、
[`models/edge_gnn.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/edge_gnn.py)。

### 8.1 候选边

两个观测只有在物理上可能属于同一辆车时才连边。对源点和目标点计算：

```text
dx = motion_direction × (x_target - x_source)
dt = t_target - t_source
v  = 3.6 × dx / dt
```

边必须满足：

- `dx > 0`，即沿行驶方向前进；
- `dt > 0`；
- 速度在 `60–90 km/h` 范围内，并允许 `±0.3 s` 的边时间容差；
- 物理间隔不超过 `600 m`。

每条边的 9 维特征包括归一化距离、时间、速度位置、速度偏差、两端证据、embedding 相似度、
跳过的通道数和速度得分。初始 heuristic score 综合了：

```text
0.40 × 两端证据
0.30 × 速度带内得分
0.15 × 距离 gap 得分
0.15 × embedding 相似度
```

两端均为强观测时再增加 bonus。

### 8.2 两跳 Edge-GNN

Edge-GNN 先编码节点和边，然后进行两轮 message passing：

1. 用源节点、目标节点和当前边状态更新边；
2. 将边消息聚合到节点并更新节点；
3. 重新分类边，得到 learned edge probability；
4. 输出节点 merge probability，帮助标记交叉区域。

最终边分数为：

```text
edge_score = 0.65 × learned_edge_score + 0.35 × heuristic_edge_score
```

当 merge 分数较高且 crossing 分数达到阈值时，观测允许被两个候选轨迹共享，以处理交叉车辆，
而普通观测容量仍为 1。

## 9. Seed 引导的 Beam Search

实现位置：[`association/beam.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/beam.py)。

对每个 Hough seed，先保留满足 seed 直线时间残差 `<= 0.4 s` 的观测，然后按行驶方向排序，
在候选图上做动态规划。状态包含：

- 当前观测 ID 序列；
- 累计路径得分；
- 上一条边的速度；
- 强观测数量。

状态转移得分主要由边分数、目标观测证据、seed 残差和速度平滑惩罚组成。每个节点最多保留
`beam_width=24` 个状态，每个 seed 最多返回 `paths_per_seed=4` 条路径。候选路径还需满足：

- 至少 `5` 个观测；
- 至少 `2` 个强观测；
- 物理跨度至少 `500 m`。

高度重叠的路径（重叠比例达到 `80%`）会先去重。

除深度/物理 Hough seed 外，关联阶段还会从强观测对生成 observation-pair seeds，以补充网络
Hough 漏检或 seed 不稳定的情况。所有 seed 会在 dense evidence 上重新打分，要求至少有足够的
20 Hz 证据支持。

## 10. MILP 全局选轨

实现位置：[`association/milp.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/milp.py)。

Beam search 得到的是多个局部候选，可能共享观测点。Hybrid 将每条候选路径视为一个 0/1 变量，
最大化所选路径总分：

```text
max Σ path_score[p] × z[p]
```

普通观测的容量约束为：

```text
Σ paths_using_observation[i] z[p] <= 1
```

交叉区域中被标记为 ambiguous 的观测容量可放宽到 2。求解器使用
`scipy.optimize.milp`，时间限制为 30 秒；如果 MILP 失败，则回退到按分数排序的 greedy 选择。

## 11. 有界慢度拟合与输出点

实现位置：[`association/refine.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/refine.py)。

选中的路径会在完整 mapping 站点集合上做分段慢度拟合：

- 每个站点区间有一个慢度变量；
- 每个慢度变量都限制在 `60–90 km/h` 对应的慢度范围；
- 对相邻区间慢度加入平滑约束；
- 通过迭代加权的 `lsq_linear` 降低异常点影响；
- 使用 robust linear diagnostic 计算中位时间残差和整体速度。

最终每个站点可能得到两种点：

- `observed=true`：来自真实检测到的峰观测；
- `observed=false`：由拟合轨迹在缺道/漏检站点上补出的 expected point。

补点不会伪装成真实观测。输出还包含 `median_speed_kmh`、`local_speeds_kmh`、`confidence`、
物理跨度、最大 gap、是否进入/离开窗口和是否发生交叉歧义。

## 12. 端到端推理流程

入口是 [`tracker.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/tracker.py) 中的
`HybridVehicleTracker.predict()`：

1. 读取 mapping，并检查模型与关联器方向一致；
2. 按 `start_s` 和 `duration_s` 从三个全速率数组切出窗口；
3. 构造五平面 20 Hz 特征；
4. 运行 ResUNet（如果加载 checkpoint）；
5. 组合 learned/modal evidence；
6. 分别运行 learned Hough 和 physics-only Hough；
7. 提取强弱观测点；
8. 构造物理速度门控图，并用 Edge-GNN 重打分；
9. 合并 Hough seed 与 observation-pair seed；
10. Beam search 生成候选路径并去重；
11. MILP 全局选择；
12. 有界慢度拟合、补 expected 点并生成 `TrackBatch`；
13. 将中间结果保存在 `last_artifacts`，便于绘图和诊断。

最小 Python 调用示例：

```python
from hybrid_vehicle_tracker import HybridVehicleTracker, load_tracker_config

tracker = HybridVehicleTracker(
    load_tracker_config("configs/day11_120s_v9.yaml")
)
batch = tracker.predict(
    raw, pre, gauss,
    "datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json",
    start_s=0,
    duration_s=120,
)
```

CLI 预测入口：

```bash
cd hybrid_vehicle_tracker_20260724
.venv/bin/hvt-predict --config configs/day11_120s_v9.yaml
```

## 13. 训练实现

训练入口是 [`cli/train.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/cli/train.py)。
当前训练使用参数化合成场景，不读取真实 DAY11 样本做梯度训练。合成器会注入车辆、孤立峰、
坏道、连续缺口、临时 outage、边界车辆、交叉和近似平行车辆，并保存车辆级 GT。

当前 v9 训练配置的主要设置：

- 10–12 辆车/场景，速度 `60–90 km/h`；
- 至少 8 个可见站点；
- 70–110 个孤立/干扰事件；
- 8%–20% 随机缺失比例；
- 80% 场景包含连续 gap；
- 60% 场景包含 edge dropout；
- 35% 场景包含近似平行轨迹；
- 0.8–6.0 s 的多站 outage；
- 训练设备固定为 CUDA，不会静默回退 CPU。

### 13.1 分阶段训练

`--stage all` 按以下顺序训练：

1. `perception`：训练 ResUNet 感知头；
2. `hough`：训练感知和 Physical Hough；
3. `gnn`：训练 Edge-GNN；
4. `joint`：联合训练全部模块，并加入 EMA teacher consistency。

感知损失由以下部分组成：

```text
focal + dice(centerline)
+ 0.50 × smooth-L1(slowness)
+ 0.35 × focal(crossing)
+ 0.20 × track embedding loss
```

Hough 使用稀疏 focal loss，并额外对正例和 hard negative 做 ranking loss；训练权重为总损失的
`0.5`。GNN 使用带正例权重的 edge BCE，并以 `0.20` 权重加入 merge BCE。联合阶段使用噪声扰动
输入和 EMA teacher 输出的一致性损失，权重为 `0.10`。

常用命令：

```bash
cd hybrid_vehicle_tracker_20260724
.venv/bin/hvt-train --config configs/synthetic_training_v9.yaml --stage all
```

checkpoint 包含模型 state dict、模型配置、训练阶段、模拟器版本和“未使用真实样本做梯度训练”
等元数据。当前网页 preset 指向：

```text
checkpoints/v9_synthetic_morphology_16384/hybrid_final.pt
```

## 14. 网页实时回放

网页相关实现位于 `web/` 和 `src/hybrid_vehicle_tracker/web/`。运行方式是：

1. 全日 Raw/Pre/Gauss 数组以 mmap 方式读取；
2. 页面按秒接收用于显示的波形帧；
3. 缓冲区达到 `window_s` 后，将窗口放入独立预测 worker；
4. 默认 `window_s=120 s`、`stride_s=60 s`；
5. 相邻窗口的轨迹通过 Hungarian assignment 进行匹配；
6. 匹配成功则复用稳定的全局 ID，例如 `V0001`；
7. 导出 `tracks.jsonl`、`track_points.csv` 和 session manifest。

跨窗口匹配主要使用：

- 重叠站点上的时间残差；
- 没有共同站点时的轨迹拟合残差；
- 速度差；
- 共同观测数量和轨迹跨度。

启动示例：

```bash
cd hybrid_vehicle_tracker_20260724
cd web/frontend && npm install && npm run build && cd ../..
.venv/bin/hvt-web --device cuda --host 127.0.0.1 --port 8001
```

## 15. 当前 v9 关键配置

配置文件：[`configs/day11_120s_v9.yaml`](../hybrid_vehicle_tracker_20260724/configs/day11_120s_v9.yaml)。

| 类别 | 参数 | 当前值 | 含义 |
| --- | --- | ---: | --- |
| 数据 | `sample_rate_hz` | 1000 | 输入采样率 |
| 数据 | `feature_rate_hz` | 20 | 网络特征率 |
| 模型 | `base_channels` | 12 | ResUNet 基础通道数 |
| 模型 | `embedding_dim` | 32 | 局部轨迹 embedding 维度 |
| Hough | `hough_slopes` | 17 | 慢度候选数量 |
| Hough | `hough_intercept_step_s` | 0.5 | 截距网格步长 |
| 关联 | `speed_min/max_kmh` | 60 / 90 | 速度物理门控范围 |
| 关联 | `edge_time_tolerance_s` | 0.3 | 候选边时间容差 |
| 关联 | `seed_time_tolerance_s` | 0.4 | seed 到观测的时间容差 |
| 关联 | `max_gap_m` | 600 | 最大物理跨站 gap |
| 关联 | `min_observations` | 5 | 最少观测点数 |
| 关联 | `min_span_m` | 500 | 最小物理跨度 |
| 关联 | `beam_width` | 24 | 每个节点保留的 Beam 状态数 |
| 关联 | `paths_per_seed` | 4 | 每个 seed 返回的路径数 |
| 关联 | `max_candidates` | 1200 | 观测候选上限 |
| 方向 | `motion_direction` | -1 | DAY11 高位置到低位置 |

## 16. 输出与置信度语义

推理结果会输出轨迹、观测点、拟合点以及诊断文件。诊断中会记录设备、checkpoint、站点数、
观测数、边数、Hough seed 数、候选路径数和最终轨迹数。

需要特别注意：当前 DAY11 没有人工真值，因此输出中的：

```text
track_status = physics_candidate_unvalidated
```

`confidence` 表示内部多模态证据、物理残差和观测数量的一致性分数，不是经过标定的 precision，
也不能直接解释为真实车辆概率。只有合成数据或后续人工标注数据，才可以计算 precision/recall、
F1 和速度 MAE。

另外，`observed=false` 的点是拟合得到的 expected 点，不代表原始数据中实际检测到了峰。

## 17. 代码入口索引

| 功能 | 文件 |
| --- | --- |
| 总推理 API | [`tracker.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/tracker.py) |
| 配置定义 | [`config.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/config.py) |
| 五平面特征 | [`data/features.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/data/features.py) |
| 峰观测提取 | [`data/peaks.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/data/peaks.py) |
| 站点 mapping | [`data/mapping.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/data/mapping.py) |
| ResUNet | [`models/resunet.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/resunet.py) |
| Physical Hough | [`models/physical_hough.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/physical_hough.py) |
| Edge-GNN | [`models/edge_gnn.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/models/edge_gnn.py) |
| 候选图与物理门控 | [`association/graph.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/graph.py) |
| Beam search | [`association/beam.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/beam.py) |
| MILP 选轨 | [`association/milp.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/milp.py) |
| 有界拟合与补点 | [`association/refine.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/association/refine.py) |
| 分阶段训练 | [`cli/train.py`](../hybrid_vehicle_tracker_20260724/src/hybrid_vehicle_tracker/cli/train.py) |
| 当前 v9 配置 | [`day11_120s_v9.yaml`](../hybrid_vehicle_tracker_20260724/configs/day11_120s_v9.yaml) |

## 18. 设计取舍总结

Hybrid 的关键取舍可以概括为：

1. **网络负责“看见”和“区分”**：从多模态数据中提供事件、交叉和身份特征；
2. **物理模型负责“能不能这样运动”**：使用真实位置、方向和速度范围限制候选空间；
3. **图模型负责“两个点是否属于同一辆车”**：结合局部证据和上下文重新评估边；
4. **全局优化负责“多条车轨如何互不冲突”**：通过 MILP 处理候选轨迹之间的竞争；
5. **拟合负责“把离散检测变成可解释轨迹”**：补齐缺道点，同时显式区分 observed 与 expected。

因此，Hybrid 不是简单地把多个模型串联，而是将深度模型的表征能力与 DAS 车辆运动的可解释
约束放进同一条推理链中。
