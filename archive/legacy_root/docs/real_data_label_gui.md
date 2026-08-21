# Real Data Label GUI

本文档说明真实数据自动标注 + 手动校准 GUI 的工作流、标签格式和当前交互约束。

## 1. 目标

该 GUI 的目标不是替代现有主提取 GUI，而是在真实 DAS 数据上形成一个更稳定的标注闭环：

1. 复用现有图搜索轨迹提取方法，对当前窗口自动生成候选轨迹。
2. 允许人工在同一张 `channel-time` 图上对候选轨迹做点级校准。
3. 将结果保存为结构化 JSON 和平铺 CSV，便于后续训练集生成或脚本后处理。

## 2. 数据流

### 2.1 输入

- SAC 文件夹，包含 `*.sac`
- 单个真实 DAS `.npy`

内部统一转为：

- 数据阵列：`[channel, time]`
- 时间采样率：`fs_hz`
- 通道间距：`dx_m`

### 2.2 自动标注

当前窗口点击 `Auto Label Current Window` 后，GUI 调用已有后端：

- `AutoTrackBackend.run_auto_extract(...)`
- 提取引擎仍是当前项目里的图搜索方法
- 范围限定为当前窗口 `current_window_only=True`

自动结果会直接替换当前窗口内已有标签点，避免同一窗口内自动结果和旧人工结果混杂。

### 2.3 人工校准

当前版本支持三种模式：

- `Select track`
  用于点击现有轨迹点并选中轨迹。
- `Add / move point`
  在最近通道上新增一个点；若该轨迹在该通道已有点，则直接改写。
- `Delete point`
  删除当前选中轨迹里离点击位置最近的点。

这里采用“每条轨迹每个通道最多一个点”的简化编辑约束，原因是：

- 当前图搜索轨迹本身就是按通道组织的单值路径
- 这样可避免点级交互把轨迹编辑成自交或一通道多解
- 速度估计和可视化会更稳定

## 3. 标签工程格式

GUI 保存的 `manual_labels.json` 结构包含：

- `source_path`
- `source_kind`
- `fs_hz`
- `dx_m`
- `created_utc`
- `updated_utc`
- `tracks`

每条轨迹包含：

- `track_id`
- `direction`
- `total_score`
- `mean_speed_kmh`
- `source`
- `note`
- `points`

每个点包含：

- `ch_idx`
- `t_idx`
- `time_s`
- `offset_m`
- `amp`
- `score`
- `source`

其中：

- 自动标注点默认 `source="auto"`
- 人工改动后的点默认 `source="manual"`

## 4. CSV 导出

`manual_labels.csv` 按点展开，一行对应一个轨迹点。适合：

- 快速筛查轨迹点是否落在合理时间范围
- 给后续数据生成脚本做中间输入
- 在 Pandas / MATLAB 中直接分析

## 5. 当前限制

当前版本没有实现以下高级编辑操作：

- 轨迹拆分
- 轨迹合并
- 框选批量删除
- 跨窗口自动传播修正

如果后续真实数据标注量继续增大，优先建议补：

1. 轨迹拆分/合并
2. 撤销/重做
3. 批量窗口自动标注
4. 从 `manual_labels.json` 直接生成训练 shard 的转换脚本
