# Auto Track

车辆轨迹识别 workspace。算法工程统一放在 [`methods/`](methods/) 下；公共类型、mapping 和协议在 [`common/`](common/)，真实共享数据在 [`shared_data/`](shared_data/)，网页在 [`vehicle_replay_web/`](vehicle_replay_web/)，迁移工具在 [`tools/`](tools/)。

## 算法工程

```text
methods/
├── hybrid_vehicle_tracker/
├── peak_slot_tracker/
├── vehicle_peak_set_tracker/
├── graph_search_tracker/
├── track_slot_tracker/
├── compact_slot_tracker/
├── trajectory_query_tracker/
├── query_mask_tracker/
├── trajectory_energy_tracker/
├── single_vehicle_trace_tracker/
├── single_vehicle_focus_tracker/
├── vehicle_proposal_trace_pipeline/
├── vehicle_set_tracker/
├── kalman_seed_tracker/
└── hungarian_assignment_tracker/
```

每个工程都包含 `src/`、`configs/`、`scripts/`、`tests/`、`data/`、`checkpoints/`、`results/` 和 `archive/`。

## 启动网页

```bash
./vehicle_replay_web/scripts/run_web.sh
```

规范入口是 `vehicle_replay_web/scripts/run_web.sh`。网页保留现有深度学习方法，并提供三种传统物理方法：图搜索（CuPy GPU）、匈牙利逐站匹配（SciPy CPU）和卡尔曼滤波（NumPy/SciPy CPU）。

## 兼容路径

兼容旧 import、旧命令和日期工程的软链接或 wrapper 统一放在
[`compatibility/`](compatibility/)；根目录不再放这些别名。历史备份、日志和旧工程保存在
[`archive/`](archive/) 中；没有删除训练数据、checkpoint 或结果。
