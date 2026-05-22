# 基于真实背景 Profile 的工作流

这份文档是当前基于真实背景 profile 的 PeakSlotNet 数据生产说明。后续如果要复现我现在这条流程，直接看这一份就够了。

## 目标

把“纯手工调参数”改成一条可重复的流程：

1. 对真实 DAS 背景做统计画像。
2. 从 profile 中提取生成器默认参数和背景窗采样权重。
3. 生成带真实背景的 `track_slot` 数据。
4. 转成 `peak_slot` 数据。
5. 在训练前先做一次分布校准和验收。

下面所有命令都默认在项目根目录执行。

## 完整流程

如果你要跑当前推荐的 profile 驱动流程，直接按这个顺序执行：

```sh
sh profile_real_npy_background.sh
sh generate_track_slot_dataset_from_real_npy_profile.sh
IN_DIR=datasets/track_slot_realbg_120s_profile/train OUT_DIR=datasets/peak_slot_realbg_120s_profile/train sh convert_track_slot_to_peak_slot.sh


sh calibrate_realbg_generator.sh
```

这 4 步会产出：

- 更新后的真实背景 profile
- `track_slot_realbg_120s_profile` 数据集
- 对应的 `peak_slot` 数据集
- `predicts/realbg_calibration/` 下的校准报告

## 分步说明

### 1. 生成真实数据 profile

```sh
sh profile_real_npy_background.sh
```

默认输出目录：

```text
datasets/profiles/xi_gauss_50_realbg
```

关键文件：

- `profile.json`
- `realism_profile.json`
- `report.md`

其中 `realism_profile.json` 是后续生成器真正使用的核心文件，里面包含：

- 背景窗口采样权重
- 稳定坏道索引
- 生成器默认参数
- 无标签车辆代理统计

### 2. 生成新的 `track_slot` 数据

```sh
sh generate_track_slot_dataset_from_real_npy_profile.sh
```

默认输出目录：

```text
datasets/track_slot_realbg_120s_profile/train
```

这一步会读取：

- 固定真实背景 `.npy`
- `datasets/profiles/xi_gauss_50_realbg/realism_profile.json`

并生成新的 `track_slot` 张量分片。当前这版生成逻辑已经包含：

- 固定坏道优先复现
- 峰密度下调
- 结构化、非均匀的伪影放置

### 3. 转成 `peak_slot`

```sh
IN_DIR=datasets/track_slot_realbg_120s_profile/train \
OUT_DIR=datasets/peak_slot_realbg_120s_profile/train \
sh convert_track_slot_to_peak_slot.sh
```

默认输出目录：

```text
datasets/peak_slot_realbg_120s_profile/train
```

### 4. 做校准对比

```sh
sh calibrate_realbg_generator.sh
```

默认输出文件：

```text
predicts/realbg_calibration/calibration_summary.json
predicts/realbg_calibration/calibration_report.md
```

这一步会把生成数据和真实 profile 做对比，并指出当前最主要的剩余失配项。

## 训练

如果前面 4 步都完成了，接下来训练 PeakSlotNet：

```sh
DATA_DIR=datasets/peak_slot_realbg_120s_profile/train \
OUT_DIR=models/peak_slot_realbg_120s_profile_cuda \
DEVICE=cuda \
sh train_peak_slot_cuda.sh
```

## 在真实数据上做预测

如果你想用训练好的模型直接看真实数据效果：

```sh
MODEL=models/peak_slot_realbg_120s_profile_cuda/checkpoint_best.pt \
DATA_DIR=datasets/peak_slot/xi_gauss_50_120s_stride60_saved_arrays04 \
OUT_DIR=predicts/peak_slot_realbg_120s_profile_cuda \
sh predict_peak_slot_dataset.sh
```

## 小样本 Smoke Test

如果你只想先验证整条链路能不能跑通，用下面这套最小命令：

```sh
OUT_DIR=/tmp/real_profile_smoke sh profile_real_npy_background.sh
PROFILE=/tmp/real_profile_smoke/realism_profile.json OUT_DIR=/tmp/track_slot_profile_smoke NUM_SAMPLES=32 SHARD_SIZE=16 sh generate_track_slot_dataset_from_real_npy_profile.sh
IN_DIR=/tmp/track_slot_profile_smoke OUT_DIR=/tmp/peak_slot_profile_smoke sh convert_track_slot_to_peak_slot.sh
PROFILE=/tmp/real_profile_smoke/realism_profile.json DATA_DIRS=/tmp/track_slot_profile_smoke OUT_DIR=/tmp/realbg_calibration_smoke sh calibrate_realbg_generator.sh
```

这套 smoke test 主要验证四件事：

- profile 生成正常
- profile 驱动数据生成正常
- `track_slot -> peak_slot` 转换正常
- 校准报告生成正常

## 说明

- 显式传入的 CLI 参数仍然优先于 profile 默认值。
- `window-sampler=profile_weighted` 表示使用 `realism_profile.json` 中记录的加权窗口采样。
- `artifact-policy=manual` 表示保留旧的手工参数行为。
- `artifact-policy=hybrid` 或 `profile_matched` 表示让 profile 注入车辆和伪影的默认参数。
- 当前实现仍然是 `profile + real background mix`，不是 `profile-only`。
- 也就是说，生成数据时仍然需要真实背景 `.npy` 文件。

## 关键代码文件

- `autotrack/simulation/profile_real_npy_background.py`
  输出 `profile.json`、`realism_profile.json` 和 `report.md`。
- `autotrack/dl/generate_track_slot_dataset_from_real_npy.py`
  支持 `--profile`、`--profile-strength`、`--window-sampler`、`--artifact-policy`。
- `autotrack/dl/calibrate_realbg_generator.py`
  用 `realism_profile.json` 对已有生成数据做分布评分。
