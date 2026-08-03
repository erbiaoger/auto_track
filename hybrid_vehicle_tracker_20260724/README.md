# Hybrid Vehicle Tracker 20260724

独立的 DAS 连续车辆轨迹识别工程。系统融合多模态 ResUNet、物理坐标 Deep Hough、
两跳 Edge-GNN 与全局 MILP，按 mapping 配置的运动方向识别 60–90 km/h 车辆（DAY11 为
高位置→低位置、时间递增），并
显式处理噪声、缺道、粘连峰和交叉车辆。

本工程不导入或修改同级 `repro_vehicle_pipeline_20260629`。数据集作为只读外部输入，
所有 checkpoint、预测和报告只写入当前工程。

训练数据全部由本工程的参数化模拟器从零生成：不会读取 DAY10/12/13/14 的真实背景，
也不会把 DAY11 事件当作车辆标签。DAY11 的 Raw/Pre/Gauss 数组只在预测和无标签形态
审计时读取；训练仅使用从 Gauss 审计提取的公开统计常量，不使用真实样本。逐车辆“模拟 GT vs 识别结果”由
`scripts/compare_synthetic_scene.py` 写入 `reports/synthetic_scene_compare_peakset_v6_long_exact/`；此前的
聚合 benchmark 不能替代这个逐车对照。

## 方法

```text
raw / pre / gauss
       │
       ▼
20 Hz 五平面特征（含站点质量与真实坐标）
       │
       ▼
多模态 ResUNet ──► 中心线 / 慢度 / 交叉概率 / embedding
       │
       ▼
可微物理 Hough（只搜索 60–90 km/h、时间递增方向）
       │
       ▼
强弱两级候选峰 ──► 真实距离门控图 ──► 两跳 Edge-GNN
       │
       ▼
二阶 beam search ──► scipy.optimize.milp 全局选轨
       │
       ▼
有界慢度拟合与 observed / expected 点分离
```

速度门控使用 mapping 中的物理位置，不使用通道号乘 100 m。DAY11 映射中
87.5 km 到 87.7 km 的 200 m 间隔会被正确保留。DAY11 车迹在原始站点顺序中是
“高位置→低位置”，因此配置显式使用 `motion_direction: -1`；这表示时间沿行驶方向
递增，不能把斜率简单取绝对值。
默认边与全局线时间容差分别收紧为 0.3 s 和 0.4 s（均严于 0.8 s 上限），用于抑制
高密度峰集合中偶然满足速度带的随机连线。

## CUDA 环境

训练被明确限制为 GPU；代码不会静默回退 CPU。项目锁定官方 CUDA 12.6 构建：

```bash
uv sync --extra dev
.venv/bin/python scripts/check_cuda.py
```

当前锁文件使用 Python 3.10 和 `torch==2.13.0+cu126`，适配本机 NVIDIA 560.35.05
驱动及 RTX 4090。

## 训练

正式训练是**全参数化合成训练**，不会读取 DAY10、DAY12、DAY13、DAY14 或其他日期
的真实背景。模拟器从零生成有色 Raw 波形、Pre 负谷和宽 Gauss 包，并同步注入带精确
真值的车辆。训练入口会拒绝包含 `background_days` 或 `background_root` 的配置。

正式训练使用 `vehicle_peakset_complex_v2` 的 long-exact 版本。它在本工程内重新实现旧工程
`test_dataset_realshape_clean_crossing` 的车辆包形态（旧工程只读参考，没有被导入），
并用 DAY11 Gauss 前 120 秒的统计校准事件高度、约 1.18 s FWHM、每站事件占用和约
33.6% 的孤立峰比例。真实数组只用于一次统计审计，**不复制任何真实时间、波形或样本
进入训练**。每个 120 s 场景包含 10–12 辆 60–90 km/h 车辆，其中至少 4 辆锚定在
时间/空间边界（入窗/出窗角）；同时生成 50–82 个有形态的孤立/局部成组干扰峰、固定
坏道、2–6 站缺口、随机 0.8–5.5 s 多站 outage 和受控交叉。干扰峰与车辆峰使用完全
相同的 Gauss 高度/宽度和 Raw/Pre 窗口，必须由跨站连续性和物理速度门控拒绝，不是靠
人工降低峰高。完整域差异审计见
`reports/domain_gap_peakset_v6_long_exact/`。

```bash
.venv/bin/hvt-train --config configs/synthetic_training.yaml --stage all

# 逐车复核：保存模拟 Raw/Pre/Gauss、GT 车辆清单、每种方法的预测和叠加图
PYTHONPATH=src .venv/bin/python scripts/compare_synthetic_scene.py \
  --config configs/day11_120s.yaml --output reports/synthetic_scene_compare_peakset_v6_long_exact \
  --checkpoint checkpoints/v6_peakset_long_exact/hybrid_final.pt \
  --scene-index 0 --seed 21260707 --device cuda
```

该命令另外输出 `vehicle_peakset_overlay.png`：横轴是真实物理 offset（DAY11 方向时
按行驶方向翻转显示），灰色虚线/圆点是注入 GT，彩色线和方块/空心圆是完整关联器的
观测点/预测缺道点，样式与旧工程给出的 overlay 一致。

阶段顺序固定为：

1. ResUNet 感知；
2. ResUNet + 可微 Hough；
3. Edge-GNN；
4. 联合训练与 EMA 教师一致性。

旧 v2 的密集随机背景仍保留为回归对照，但不再用于正式训练。正式 peak-set 训练集
包含 10–12 辆车、实际非均匀站距、固定坏道、临时 outage、连续缺口和高密度受控交叉；
车辆峰和干扰峰共享同一固定 Gauss 高度/宽度及 Raw/Pre 窗口，区分只依赖跨站连续性和物理速度；
45–59、91–110 km/h 及反方向轨迹只作为低幅硬负样本。

如需复核“参考了什么”而不是复制数据，可执行：

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_gauss_reference.py \
  --output reports/day11_gauss_reference_stats_v1.json
```

该审计会得到 363 个强峰、122 个孤立峰（0.3361），并明确标记
`training_samples_exported=false`。

## DAY11 前 120 秒预测

配置已经固定三个输入、真实 mapping、`start_s=0` 和 `duration_s=120`：

```bash
.venv/bin/hvt-predict --config configs/day11_120s.yaml
```

预测目录包含：

- `tracks.jsonl`：逐车辆物理一致候选（不是人工真值意义上的确认车辆）；
- `observations.csv`：所有真实候选及多模态证据；
- `track_points.csv`：观测点和缺道预测点，后者始终为 `observed=false`；
- `tracks_overlay.png`、`network_probability.png`、`physical_hough.png`、
  `association_graph.png`；
- `diagnostics.json`、解析后的配置和 Markdown 报告。

当前 v6 long-exact checkpoint 在 DAY11 前 120 s 产生 10 条物理候选；
200 组站点循环位移校准后的平均误轨为 0.01（未校准平均 0.05）。这只是无标签条件下
的候选筛选/误关联控制，不是 precision/recall；旧 v3 严格审计结果仍保留在
`runs/day11_0_120_v3_direction/`。
推理同时保留 learned Hough 和 Gauss+Pre modal Hough；当合成域网络概率压低真实峰时，
物理 Hough 仍可用真实峰的跨站速度连续性恢复候选。

### 用阈值脚本从原始波形生成高斯窗再识别

如果输入来自 `threshold_test/day11_raw/11` 和 `threshold_test/day11_pre/new11`，不要直接
把概率数组当 Gauss 输入。下面的独立包装脚本只读调用数据目录中的
`0002pre2guass.py`：使用其 `FLIP_PROB=True`、全局阈值、2 s 最小峰间隔和 σ=0.5 s
高斯叠加，然后按 `raw_DAY11.mapping.json` 的 50 个选中站点组装 `[120000, 50]` 数组。
脚本强制使用 CUDA，不会静默回退 CPU：

```bash
cd /csim2/zhangzhiyu/MyProjects/auto_track/hybrid_vehicle_tracker_20260724
uv run python scripts/convert_threshold_and_predict.py \
  --threshold 0.9 --duration-s 120 \
  --gauss-dir /csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/run_threshold_090/gauss \
  --cache-dir runs/day11_threshold_090_real/modal \
  --output-dir runs/day11_threshold_090_real/tracks
```

本次实际运行匹配 55 对 Raw/预测文件，按真实 mapping 保留 50 道，生成的中间数组、转换
清单和追踪结果在 `runs/day11_threshold_090_real/`；结果中的 18 条是物理约束候选，
不是有人工真值确认的车辆。`diagnostics.json` 会记录 `device=cuda`、阈值、站点顺序、
速度带和 `track_status=physics_candidate_unvalidated`。阈值改为 0.7 时只需改
`--threshold 0.7` 和输出目录，避免覆盖 0.9 结果。

`tracks_overlay.png` 是阈值拾峰生成的 Gauss 窗叠加图：灰色曲线对应各物理站点的 Gauss
峰窗，彩色轨迹上的实测点为实心方块，缺道或轨迹拟合补出的点为空心圆；底图不使用 Raw
波形。

如需查看多个局部案例，可执行 `uv run python scripts/render_overlay_cases.py`，结果写入
`runs/day11_threshold_090_real/tracks/overlay_cases/`，包含完整窗口和四个 30 秒子窗口。

固定阈值 0.9、滑动 120 秒窗口可执行：

```bash
uv run python scripts/render_sliding_window_cases.py \
  --threshold 0.9 --stride-s 60 --count 5
```

默认生成 `0–120、60–180、120–240、180–300、240–360 s` 五张完整叠加图。

### DAY11 全日一次转换

如果要处理完整 DAY11，先把原始 Raw/Pre 按阈值一次性转换成模型输入缓存，后续滑窗只切片
缓存，不再重复读取 55 个站点或重复运行阈值脚本：

```bash
uv run python scripts/precompute_full_day_threshold.py \
  --threshold 0.9 \
  --output-dir runs/day11_threshold_090_full_day_cache
```

输出为 0–3600 s、50 道的三个数组：`raw_threshold_full_DAY11.npy`、
`pre_probability_full_DAY11.npy` 和 `gauss_threshold_full_DAY11.npy`，形状均为
`(3600000, 50)`；`full_day_manifest.json` 记录站点顺序和每道拾峰数。模型预测时直接向
`HybridVehicleTracker.predict(..., start_s=..., duration_s=120)` 传入这三个缓存数组即可。

完整缓存上的 GPU 滑窗预测：

```bash
uv run python scripts/predict_all_from_full_cache.py \
  --cache-dir runs/day11_threshold_090_full_day_cache \
  --output-root runs/day11_threshold_090_all_v9_gpu \
  --device cuda
```

该命令只转换一次缓存，然后预测全部 59 个窗口；每窗保存 `tracks.jsonl`、
`observations.jsonl`、`track_points.csv`、叠加图和 `summary.json`，总耗时写入
`prediction_timing.json`。

## 实时网页回放

网页服务只读取 `runs/day11_threshold_090_full_day_cache/` 的三个 mmap 数组；不会在默认回放
过程中再生成窗口级模态文件。启动前构建一次 React/Vite 页面：

```bash
cd /csim2/zhangzhiyu/MyProjects/auto_track/hybrid_vehicle_tracker_20260724
cd web/frontend && npm install && npm run build && cd ../..
.venv/bin/hvt-web --device cuda --host 127.0.0.1 --port 8001
```

默认参数是轨迹识别窗口 120 秒、窗口移动/识别间隔 60 秒。注意：如果只把“窗口长度”改为
20 秒，识别间隔仍然是 60 秒；想要每 20 秒识别一次，必须把 `stride_s`（页面中的“每隔多久识别”）
改为 20。窗口为 120 秒、间隔为 20 秒时，首次识别仍要等到 120 秒缓冲完成，之后在 140、160、180 秒…
继续识别。也可以在启动时修改：

```bash
.venv/bin/hvt-web --device cuda --window-s 180 --stride-s 30
```

页面工具栏中的“识别窗口”和“移动/识别间隔”输入框可以运行中修改，点击“应用窗口参数”
会从 0 秒新建回放会话并按新参数重新累计车辆。也可以通过 API 传参：
`POST /api/replay/start` 的 JSON 支持 `start_s`、`window_s`、`stride_s`、`speed`、
`waveform_downsample`；WebSocket 的 `configure` 控制消息支持同名字段。页面也可直接修改
“波形降采样 ×”：填 `10` 表示 10 倍降采样（100 Hz），填 `20` 表示 20 倍降采样（50 Hz）。

浏览器打开 `http://127.0.0.1:8001/`。服务启动时一次性 mmap 全日缓存、加载真实 mapping 和
v9 CUDA checkpoint；页面每秒接收 20 Hz 的 Gauss/Raw 显示帧，同时附带可配置降采样的 Raw 波形，
默认 20 倍（原始 1000 Hz → 50 Hz）。选择 Raw 或 Gauss+Raw 时，波形会按物理站点位置展开绘制。缓冲达到
配置的窗口长度后按配置的间隔把窗口交给单独 GPU worker。相邻窗口通过 Hungarian 门控拼接为稳定的
`V0001` 形式全局 ID。

常用接口：`GET /api/health`、`GET /api/replay/state`、`POST /api/replay/start`、
`PATCH /api/replay/control`、`POST /api/replay/export`，以及 `WS /ws/replay`。默认只绑定
localhost；局域网使用 `--host 0.0.0.0 --access-token <token>`，WebSocket 以 `?token=` 传递令牌。
导出才会写入 `runs/web_sessions/<session_id>/`，导出结果包含稳定 ID 的 JSONL 和逐点 CSV。

v7 增量训练 checkpoint 为 `checkpoints/v7_synthetic_longer_exact/hybrid_final.pt`，对应
配置 `configs/day11_120s_v7.yaml`。v7 仍只使用本工程合成数据训练；DAY11 仅用于测试。
24 个合成测试窗口的完整方案 F1=0.854、速度 MAE=0.169 km/h（v6 对应 F1=0.810、
MAE=0.245 km/h）。

Python API：

```python
from hybrid_vehicle_tracker import HybridVehicleTracker, load_tracker_config

tracker = HybridVehicleTracker(load_tracker_config("configs/day11_120s.yaml"))
batch = tracker.predict(raw, pre, gauss, mapping, start_s=0, duration_s=120)
```

## 逐车对照、负对照与测试

站点独立循环位移负对照保留每站峰值统计、破坏跨站车辆连续性，用于估计误关联阈值：

```bash
.venv/bin/hvt-evaluate --config configs/day11_120s.yaml --device cuda --null-samples 200 \
  --output reports/day11_null_control_peakset_v6_modal_hough_200.json
.venv/bin/hvt-evaluate --config configs/day11_120s.yaml --device cuda --synthetic-samples 24 \
  --output reports/synthetic_benchmark_peakset_v6_long_exact_24.json
.venv/bin/pytest -q
```

第一版曾输出的 11 条直线已被域差异审计判定为不可信随机峰关联，并保留在 legacy 目录
供复盘。DAY11 没有人工真值，不能把候选数当作召回率；预测报告会明确记录运动方向、
物理间距和“候选未验证”语义。合成场景的 F1/MAE 只在保存的 GT 上计算，不能外推为
DAY11 的真实 precision/recall。

DAY11 尚无人工真值，所以真实报告只陈述物理一致性、证据支撑和负对照，不把候选数
当作 precision/recall。只有合成真值和后续人工标注数据用于 F1 与速度 MAE。

固定 seed=21260707 的逐车复核结果在
`reports/synthetic_scene_compare_peakset_v6_long_exact/comparison.json`：模拟器注入 11 辆车，
传统 Hough 找到 8/11（F1=0.842），ResUNet+Hough 找到 7/11（F1=0.778），完整
GNN+MILP 找到 9/11（F1=0.900），速度 MAE=0.020 km/h。聚合 24 场景中完整方案
F1=0.810、速度 MAE=0.245 km/h；复杂交叉/缺道仍是有意保留的难例，不能把单场景结果
夸大为 DAY11 的真实 precision/recall。`vehicle_peakset_overlay.png` 按旧工程的
waveform overlay 样式画出逐点 GT 和预测；`comparison.png` 同时画出 GT（蓝虚线）和预测
（红线）；`simulation_manifest.json` 列出每辆车的速度、斜率、观测/缺道站点和精确
时间；`scene_arrays.npz` 保存生成的 Raw/Pre/Gauss 与网络概率。

## 参考

- [Semantic Perception–Topological Reasoning for DAS](https://doi.org/10.1016/j.eswa.2026.131886)
- [Intelligent Traffic Monitoring with DAS](https://arxiv.org/abs/2403.02791)及其[公开代码](https://github.com/TTMuTian/itm)
- [Deep Hough Transform](https://arxiv.org/abs/2003.04676)
- [MPNTrack](https://github.com/dvl-tum/mot_neural_solver)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)

本工程未复制 ITM 或 Deep Hough 源码；可微物理 Hough、网络和关联器均在本工程中独立
实现，避免引入无明确许可证或仅限非商业用途的代码。
