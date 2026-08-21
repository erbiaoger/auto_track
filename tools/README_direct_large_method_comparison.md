# DAY02 大车方法对比一键运行

运行 120 秒识别窗口、每 2 秒滑动一次的完整四方法对比图（含跨窗去重与平滑）：

```bash
./tools/run_direct_large_method_comparison_DAY02_stride2.sh
```

默认输出：

```text
runs/direct_large_method_comparison_DAY02_0_600s_stride2_stable.png
runs/direct_large_method_comparison_DAY02_0_600s_stride2_stable.json
```

默认参数：

- 设备：`cuda`
- 时间范围：`0–600 s`
- 窗口长度：`120 s`
- 滑动步长：`2 s`

可临时覆盖参数，例如只先跑 120 秒：

```bash
DURATION_S=120 ./tools/run_direct_large_method_comparison_DAY02_stride2.sh
```

也可以直接传入脚本参数覆盖命令行选项：

```bash
./tools/run_direct_large_method_comparison_DAY02_stride2.sh --stride-s 5
```
