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

## TrackSlotNet Workflow

Generate tensor shards without SAC I/O:

```sh
WORKERS=8 sh generate_track_slot_dataset.sh
```

The default generator uses realistic motion augmentation and noisy DAS
backgrounds. Motion defaults are `constant_sparse,smooth_random,stop_go` with
weights `0.84,0.15,0.01`; signal defaults also include white noise, correlated
low-frequency noise, channel bias/gain variation, slow baseline drift, and
isolated pulse interference. The default `REALISM_PRESET=xi_gauss_50` also
uses 32-48 vehicles per 120 s window, fixed dead channels
`5,6,15,16,22,36,38,42,45,48`, random dead channels, and random zero background
blocks to better match the checked real `gauss_section.npy` data. Override
`NOISE_STD=0` and the other noise/dead-channel environment variables only for
clean ablation runs. The default sample count is `NUM_SAMPLES=80000`.
Direction sampling keeps the expected traffic prior (`PRIMARY_RATIO=0.8333333333`).

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

`predict_peak_slot_dataset.sh` defaults to recall-oriented inference
(`OBJECTNESS_THRESHOLD=0.35`, `MIN_VISIBLE_CHANNELS=2`,
`VITERBI_BEAM_SIZE=8`) while keeping cross-slot conflict suppression enabled
with `GLOBAL_CONFLICT_PENALTY=2.0`. Raise the threshold if false positives
become too high.

Infer on SAC data:

```sh
uv run python -m autotrack.dl.infer_trajectory_model --model-family peak_slot --model models/peak_slot_cuda/checkpoint_best.pt --data-folder datasets/test/sim_1001
```

## Python Environment

Use the project environment through `uv run`. Do not invoke a different Python
environment for training or dataset generation.
