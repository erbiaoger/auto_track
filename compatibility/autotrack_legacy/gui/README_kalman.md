# Kalman GUI

`kalman_track_gui.py` 是从原始 `KF03.py` 迁移出来的完整卡尔曼交互 GUI，
专门用于真实 `.npy` DAS 数据。

启动：

```bash
uvr -m autotrack.gui.kalman_track_gui \
  --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
  --out-dir predicts/kalman_gui_windows
```

核心操作：

- 左键：从最近峰值点出发跑一条卡尔曼轨迹
- 中键：当前临时轨迹在该通道上吸附到最近峰值
- 右键：补全或裁剪轨迹
- `S`：保存当前临时轨迹到当前窗口轨迹列表
- `F`：保存当前窗口 bundle，并前进到下一个窗口
