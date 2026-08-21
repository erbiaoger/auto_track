# autotrack.kalman

基于卡尔曼滤波的轨迹跟踪代码。

- `kf03_real_npy.py`：从原始 `KF03.py` 整理出来的真实 `.npy` 版本。支持直接读取
  `/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy` 这类真实 DAS 数组，
  并从单个种子点出发输出一条卡尔曼跟踪轨迹。
- `__init__.py`：导出加载器、默认参数和跟踪类。

推荐运行方式：

```bash
uvr -m autotrack.kalman.kf03_real_npy \
  --seed-channel-idx 25 \
  --seed-time-index 180000 \
  --out-dir /tmp/kf03_real_npy_demo
```
