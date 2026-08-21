# autotrack.labeling

真实数据标注工程的数据层。

- `track_label_project.py`：维护可编辑轨迹、窗口替换、点级增删改、JSON/CSV 保存导出。
- `__init__.py`：导出 GUI 会直接使用的标签工程类。

典型用途：

1. 从图搜索自动提取结果生成当前窗口候选标签。
2. 在 GUI 里对标签做手动校准。
3. 将结果保存成 `manual_labels.json` 和 `manual_labels.csv`。
