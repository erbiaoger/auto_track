# 原始能量斜线一致性 Peak 筛选

## 1. 目的

DAS 台站的置信度分布并不完全一致。对所有台站使用同一个阈值时，容易出现两种问题：

- 阈值太低：弱噪声和孤立 peak 很多，车辆轨迹被淹没；
- 阈值太高：强台站还能保留车辆，弱台站却断道，斜向轨迹不连续。

本方法不直接把“置信度最高”当成车辆，而是加入原始波形的空间连续性约束：

1. 从每个台站的原始波形计算短时 RMS 能量；
2. 根据台站间距和车辆速度范围，扫描物理上可能的斜线；
3. 保留原始能量沿斜线连续增强的候选；
4. 只保留距离这些斜线较近的 probability peak；
5. 可选地根据每个台站在线上候选的置信度，自动生成台站独立阈值。

红色参考线是原始数据能量扫描得到的候选轨迹，不是神经网络预测结果。最终保留的彩色 peak 必须同时满足“置信度候选”和“原始能量斜线一致”。

## 2. 物理关系

令台站编号为 `i`，事件到达时间为 `t_i`，使用直线模型：

```text
t_i = t0 + slope * i
```

相邻台站间距为 `d`，车辆速度为 `v` 时：

```text
abs(slope) = d / v
```

例如台站间距 100 m、车辆速度 75 km/h：

```text
0.1 km / 75 km/h * 3600 s/h ≈ 4.8 s/台站
```

因此默认速度范围 60–100 km/h 对应约 3.6–6.0 秒/台站，并同时扫描正、负两个方向。

## 3. 代码位置

核心模块：

[raw_energy_line_filter.py](/csim2/zhangzhiyu/MyProjects/auto_track/autotrack/dl/raw_energy_line_filter.py)

主要函数：

- `compute_energy_envelope`：计算单台站短时 RMS 能量；
- `compute_energy_matrix`：得到 `[时间, 台站]` 能量矩阵；
- `scan_raw_energy_lines`：扫描并返回原始能量斜线；
- `pick_probability_peaks`：从单道置信度曲线提取候选 peak；
- `match_peaks_to_lines`：将 peak 与斜线匹配；
- `derive_station_thresholds`：根据各台站在线候选的置信度生成独立阈值；
- `run_line_consistent_filter`：对已经提取好的 peak 运行斜线筛选；
- `run_from_probability_arrays`：从原始波形和 probability 数组直接完成候选提取与斜线筛选；
- `gaussian_curve_from_picks`：将最终 peak 转成高斯曲线。

## 4. 命令行用法

输入目录中的 raw 和 probability 文件名需要一一对应，并且文件名中包含类似 `_1907001D_EHZ_` 的台站编号。

推荐使用 Excel 排列映射 JSON，模块会读取其中的 `workbook_station_rows` 作为台站顺序：

```sh
uv run python -m autotrack.dl.raw_energy_line_filter \
  --raw-dir datasets/peaks_20260723/threshold_test/day11_raw/11 \
  --probability-dir datasets/peaks_20260723/threshold_test/day11_pre/new11 \
  --order-json datasets/peaks_20260723/converted_50ch/arrays/gauss_DAY11.mapping.json \
  --out-dir datasets/peaks_20260723/threshold_test/line_filter_cli_DAY11 \
  --fs-hz 1000 \
  --station-spacing-m 100 \
  --speed-min-kmh 60 \
  --speed-max-kmh 100 \
  --candidate-threshold 0.3 \
  --min-gap-s 2.0 \
  --line-count 70 \
  --line-match-tolerance-s 0.8
```

输出文件：

- `lines.json`：原始能量斜线的得分、斜率、截距；
- `selected_peaks.csv`：筛选后 peak 的台站、时间、置信度和所属斜线；
- `summary.json`：台站数、候选数量、筛选后数量和参数。

这个命令只做 peak 筛选，不运行 PeakSlotNet 模型。

## 5. Python 调用示例

```python
from autotrack.dl.raw_energy_line_filter import run_from_probability_arrays

result = run_from_probability_arrays(
    raw_signals=raw_signals,              # list[np.ndarray], 每道原始波形
    probability_signals=probability_signals,
    fs_hz=1000.0,
    candidate_threshold=0.3,
    min_gap_s=2.0,
    flip_probability=True,
    station_spacing_m=100.0,
    speed_min_kmh=60.0,
    speed_max_kmh=100.0,
    line_count=70,
    line_match_tolerance_s=0.8,
)

for station_index, times in enumerate(result.selected_times_s):
    print(station_index, times)
```

若已经有 `peak_times_s` 和 `peak_scores`，不需要重新处理 probability：

```python
from autotrack.dl.raw_energy_line_filter import run_line_consistent_filter

result = run_line_consistent_filter(
    raw_signals,
    candidate_times_s=peak_times_s,
    candidate_scores=peak_scores,
    fs_hz=1000.0,
    station_spacing_m=100.0,
    speed_min_kmh=60.0,
    speed_max_kmh=100.0,
    line_count=70,
    line_match_tolerance_s=0.8,
)
```

## 6. 台站独立阈值

如果希望先根据原始能量斜线得到每道阈值：

```python
from autotrack.dl.raw_energy_line_filter import derive_station_thresholds

station_thresholds = derive_station_thresholds(
    result.candidate_times_s,
    result.candidate_scores,
    result.lines,
    quantile=0.5,
    lower=0.3,
    upper=0.9,
    tolerance_s=0.8,
)
```

`quantile=0.5` 表示取每道斜线候选置信度的中位数。弱台站会得到相对较低的阈值，强台站或没有可靠斜线候选的台站会得到相对较高的阈值。

需要注意：台站独立阈值适合生成较完整的候选集；若目标是尽量少的孤立 peak，仍建议在阈值后再运行 `match_peaks_to_lines`。

## 7. 参数建议

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `candidate_threshold` | 0.3 | 第一阶段候选阈值；越低召回越高，但孤立点越多 |
| `min_gap_s` | 2.0 s | 同一道相邻 peak 的最小间隔 |
| `station_spacing_m` | 100 m | 相邻台站空间间距 |
| `speed_min_kmh` | 60 | 最低扫描速度 |
| `speed_max_kmh` | 100 | 最高扫描速度 |
| `line_count` | 70 | 保留的原始能量斜线数量 |
| `line_match_tolerance_s` | 0.8 s | peak 到斜线的最大时间距离 |
| `energy_window_s` | 1.0 s | 原始 RMS 能量窗口 |

经验上：

- 车辆较少、希望多找候选：`candidate_threshold=0.3`，`line_count=70–100`；
- 噪声较多、希望少误检：提高 `candidate_threshold` 或降低 `line_count`；
- 斜线断点较多：可以把 `line_match_tolerance_s` 从 0.8 调到 1.0，但不要盲目增大；
- 速度范围必须符合实际车辆速度，否则会扫描出不合理的斜线。

## 8. 结果解释与限制

这不是车辆真值标注，也不是最终轨迹关联器。原始数据中如果存在长距离同步噪声、机械振动或固定干扰，也可能形成高分斜线。因此建议：

1. 先查看 `lines.json` 和 `selected_peaks.csv` 的斜率、覆盖台站数；
2. 再查看原始波形与筛选 peak 的叠加图；
3. 最后才把筛选结果送入 PeakSlotNet 或经典轨迹关联器。

斜线筛选的作用是提供高质量候选和先验约束，不应替代最终模型或人工检查。

## 9. 台站级 fallback 阈值

当严格阈值已经压住噪声、但某些台站在明显车辆斜线上仍然没有 peak 时，不能直接把所有台站的阈值一起降低。推荐使用两级候选：

1. 先用严格的台站阈值保留基础候选；
2. 再用较低的台站 fallback 阈值生成备用候选；
3. 只有备用候选同时靠近“原始能量高分斜线”，并且该斜线至少得到若干台站支持时，才把它加入最终候选。

本仓库提供了可复现脚本：

```sh
PYTHONPATH=. uv run python scripts/build_station_fallback_dataset.py \
  --raw-dir datasets/peaks_20260723/threshold_test/day11_raw/11 \
  --probability-dir datasets/peaks_20260723/threshold_test/day11_pre/new11 \
  --mapping datasets/peaks_20260723/converted_50ch/arrays/gauss_DAY11.mapping.json \
  --base-thresholds datasets/peaks_20260723/threshold_test/adaptive_station_q05_line_filter_DAY11/station_thresholds_q05.json \
  --fallback-thresholds datasets/peaks_20260723/threshold_test/adaptive_station_q02_gauss_DAY11/adaptive_station_thresholds.csv \
  --lines datasets/peaks_20260723/threshold_test/adaptive_station_q05_line_filter_DAY11/lines.json \
  --output-dir datasets/peaks_20260723/converted_50ch_station_fallback_q75_s8 \
  --line-score-quantile 0.75 \
  --min-fallback-support 8
```

`station_fallback_thresholds.csv` 会记录每个台站的基础阈值、fallback 阈值和实际新增数量；`selected_lines.json` 会记录每条斜线的原始能量得分及基础/fallback 台站支持数。脚本同时输出正常时间轴和全局时间反转后的高斯数组，后者用于当前 repro 流程。

本次 Day11 使用 q05 作为基础、q02 作为 fallback，原始能量斜线分数取前 25%，至少 8 个台站支持。最终选中 13 条斜线，仅新增 76 个候选，避免把 q02 的全部低阈值噪声引入模型。
