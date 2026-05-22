# BaFang Auto Track

This project extracts vehicle trajectories from DAS channel-time data. It keeps
classic peak/graph extraction paths and deep-learning paths side by side.

## Folders

- `autotrack/`: importable Python package for extraction, training, inference,
  GUI, CLI, and simulation code.
- `autotrack/dl/`: PyTorch models, tensor dataset generation, training,
  inference, evaluation, and checkpoint helpers.
- `autotrack/core/`: shared trajectory types, classic extraction engines, and
  the deep-learning adapter used by the backend.
- `autotrack/gui/`: PyQt applications for interactive tracking and label review.
- `autotrack/cli/`: command-line classic extraction variants.
- `autotrack/simulation/`: accelerated synthetic SAC/label generation helpers.
- `docs/`: algorithm and network design notes.
- `datasets/`: generated or checked-in data used for training and tests.
- `models/`: trained checkpoints and training outputs.
- `notebooks/`: exploratory notebooks.


## Launch GUI interface

```sh
uvr -m autotrack.gui.auto_track_gui
```

Real-data auto-label + manual calibration GUI:

```sh
uv run python -m autotrack.gui.real_data_label_gui \
  --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
  --fs 1000 \
  --dx-m 100
```

This GUI reuses the current classic graph-search extractor to auto-label only
the current window, then lets you manually select a track, add/move a point,
delete a point, create a track, delete a track, and save the result as
`manual_labels.json` plus `manual_labels.csv`.

The GUI import box accepts either:

- a SAC folder containing `*.sac` files, or
- one real DAS `.npy` array file such as `/Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy`.
- one tensor dataset shard `.pt/.pth` file such as `datasets/track_slot_realbg_120s/train/shard_000000.pt`.

For `.npy` input, the GUI assumes the project-default real-data layout
`[time, channel]` and converts it internally to `[channel, time]`.
If the file does not use the project-default `1000 Hz / 100 m`, fill the GUI
fields `NPY fs (Hz)` and `NPY dx (m)` before import.

For tensor shard input, the GUI initially loads the first sample stored in the
shard and uses the shard `meta.json` to recover `fs`, `dx_m`, and window
duration. After import, place the mouse over the plot and use the scroll wheel
to switch between samples inside the same shard.


## TrackSlotNet Workflow

Generate tensor shards without SAC I/O:

```sh
WORKERS=8 sh generate_track_slot_dataset.sh
```

Generate TrackSlotNet shards from real `.npy` background windows with a
heavier preset that adds more isolated Gaussian peaks, more extra bad channels,
and more missing blocks:

```sh
sh generate_track_slot_dataset_from_real_npy.sh
```

Common overrides:

```sh
OUT_DIR=datasets/track_slot_realbg_120s_heavy/train \
NUM_SAMPLES=4000 \
ISOLATED_NOISE_RATE=320 \
TRACK_DROP_CHANNEL_MIN=6 \
TRACK_DROP_CHANNEL_MAX=12 \
RANDOM_DEAD_CHANNEL_MIN=8 \
RANDOM_DEAD_CHANNEL_MAX=16 \
sh generate_track_slot_dataset_from_real_npy.sh
```

The default generator now targets the v4 noisy/bad-channel setting for
Gaussian-window DAS outputs. Motion defaults are
`constant_sparse,smooth_random,stop_go` with weights `0.84,0.15,0.01`; signal
defaults add continuous white/colored noise, channel bias/gain variation,
baseline drift, denser isolated Gaussian windows, random dead channels, and
random missing channel-time blocks. The default sample count is
`NUM_SAMPLES=40000`.
Direction sampling keeps the expected traffic prior (`PRIMARY_RATIO=0.8333333333`).

Profile-driven real-background workflow:

```sh
sh profile_real_npy_background.sh
sh generate_track_slot_dataset_from_real_npy_profile.sh
IN_DIR=datasets/track_slot_realbg_120s_profile/train OUT_DIR=datasets/peak_slot_realbg_120s_profile/train sh convert_track_slot_to_peak_slot.sh
sh calibrate_realbg_generator.sh
```

`profile_real_npy_background.sh` writes `realism_profile.json`, which stores
window-level sampling weights, sparse-artifact statistics, and unlabeled
vehicle proxy statistics. `generate_track_slot_dataset_from_real_npy.py` now
accepts `--profile`, `--profile-strength`, `--window-sampler`, and
`--artifact-policy`, so profile defaults can drive generation without removing
explicit CLI control.

Train on CUDA:

```sh
DEVICE=cuda EPOCHS=2 BATCH_SIZE=64 sh train_track_slot_cuda.sh
```

`train_track_slot_cuda.sh` defaults to `AUTO_RESUME=1`, so rerunning it will
continue from `models/track_slot_cuda/checkpoint_last.pt` if that file exists.
`EPOCHS` is the target total epoch count; to continue after epoch 180, set
`EPOCHS` to a value larger than 180.

Plot training history:

```sh
RUN_DIR=models/track_slot_cuda sh plot_track_slot_history.sh
```

Plot data and GT labels for manual checking:

```sh
DATA_DIR=datasets/track_slot/train SAMPLE_INDICES=6 OUT_DIR=/tmp/track_slot_label_check sh plot_dataset_labels.sh
DATA_DIR=datasets/peak_slot/train SAMPLE_INDICES=6 OUT_DIR=/tmp/peak_slot_label_check PLOT_PEAKS=1 sh plot_dataset_labels.sh
```

Run a CPU smoke test:

```sh
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir /tmp/track_slot_data --num-samples 32 --shard-size 16 --window-seconds 10 --time-downsample 20 --workers 2 --overwrite
uv run python -m autotrack.dl.train_track_slot --data-dir /tmp/track_slot_data --out-dir /tmp/track_slot_smoke --device cpu --epochs 1 --batch-size 2
```

Predict directly on generated tensor shards:

```sh
MODEL=/tmp/track_slot_smoke/checkpoint_best.pt DATA_DIR=/tmp/track_slot_data OUT_DIR=/tmp/track_slot_predict sh predict_track_slot_dataset.sh
```

This writes `summary.json`, prediction CSV files, and heatmap overlay figures
under `<OUT_DIR>/plots/`. Shard prediction applies per-slot monotonic trimming
and trajectory NMS by default before writing CSV and overlay figures.

Full method documentation:

```text
docs/track_slot_method_complete.md
```

Infer on SAC data with a trained checkpoint:

```sh
uv run python -m autotrack.dl.infer_trajectory_model --model-family track_slot --model models/track_slot_cuda/checkpoint_best.pt --data-folder datasets/test/sim_1001
```

## PeakSlotNet Workflow

PeakSlotNet first detects peak candidates on each DAS channel, then predicts
which peaks belong to the same vehicle slot. Predicted tracks therefore pass
through detected peaks rather than arbitrary regressed times.

Convert existing TrackSlotNet shards:

```sh
sh convert_track_slot_to_peak_slot.sh
```

Train on CUDA:

```sh
DEVICE=cuda EPOCHS=20 BATCH_SIZE=32 sh train_peak_slot_cuda.sh
```

`train_peak_slot_cuda.sh` reserves a fixed tail split by default
(`VAL_FRACTION=0.1`) and evaluates it every 5 epochs. Set `VAL_DATA_DIR` to use
a separately generated peak-slot validation directory instead. Count/objectness
training is weighted more strongly by default through `OBJECT_LOSS_WEIGHT=1.25`
and `COUNT_LOSS_WEIGHT=0.15`. Step timing is summarized every 5 epochs by
default with `TIMING_EVERY=5`.

Predict directly on converted shards:

```sh
MODEL=models/peak_slot_cuda/checkpoint_best.pt DATA_DIR=datasets/peak_slot/train sh predict_peak_slot_dataset.sh
```

`predict_peak_slot_dataset.sh` defaults to the tuned balanced inference
configuration (`OBJECTNESS_THRESHOLD=0.45`, `MIN_VISIBLE_CHANNELS=4`,
`EXTRA_CANDIDATE_SLOTS=8`) while keeping cross-slot conflict suppression enabled
with `GLOBAL_CONFLICT_PENALTY=2.0`.

When synthetic validation is strong but real-data prediction is poor, compare
the two domains explicitly:

```sh
uv run python -m autotrack.dl.analyze_peak_slot_domain_gap \
  --reference-dir datasets/peak_slot_v3_120s_realistic/test \
  --target-dir datasets/peak_slot/xi_gauss_50_120s_stride60_saved_arrays04 \
  --model models/peak_slot_v4_120s_noisy_badch_cuda/checkpoint_best.pt \
  --out-dir /tmp/peak_slot_domain_gap \
  --max-samples 64 \
  --device cpu
```

This writes `summary.json` and `report.md` so you can see whether the failure is
coming from input normalization drift, peak-candidate clutter, or model
objectness/count bias.

Infer on SAC data:

```sh
uv run python -m autotrack.dl.infer_trajectory_model --model-family peak_slot --model models/peak_slot_cuda/checkpoint_best.pt --data-folder datasets/test/sim_1001
```

## Python Environment

Use the project environment through `uv run`. Do not invoke a different Python
environment for training or dataset generation.
