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

The default generator uses realistic motion augmentation:
`constant_sparse,smooth_random,stop_go` with weights `0.84,0.15,0.01`.

Train on CUDA:

```sh
DEVICE=cuda EPOCHS=2 BATCH_SIZE=64 sh train_track_slot_cuda.sh
```

Plot training history:

```sh
RUN_DIR=models/track_slot_cuda sh plot_track_slot_history.sh
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
under `<OUT_DIR>/plots/`.

Full method documentation:

```text
docs/track_slot_method_complete.md
```

Infer on SAC data with a trained checkpoint:

```sh
uv run python -m autotrack.dl.infer_trajectory_model --model-family track_slot --model models/track_slot_cuda/checkpoint_best.pt --data-folder datasets/test/sim_1001
```

## Python Environment

Use the project environment through `uv run`. Do not invoke a different Python
environment for training or dataset generation.
