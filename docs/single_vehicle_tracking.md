# Single Vehicle Tracking

This is the new tracking direction for the one-vehicle-at-a-time task.

## Goal

Recover one physically consistent vehicle trajectory per window with:

- no duplicate pickup
- no missing interior channels when the evidence is present
- stable behavior across crossings and short dropouts
- a clear separation between candidate scoring, path decoding, and smoothing

The previous PeakSlotNet path remains preserved under `bak/20260627_peakslot_snapshot/`.

## Current Pipeline

The current mainline is now split into two layers:

1. A dense multi-vehicle proposal net marks likely vehicle energy over the whole window.
2. Candidate peaks are extracted from that proposal-enhanced map.
3. A single best path is decoded with a physically constrained dynamic program.
4. Sparse gaps are bridged with local Hungarian assignment against real peak candidates.
5. A constant-velocity Kalman smoother refines the final channel-time path.

The implementation lives in:

- [`autotrack/core/single_vehicle_tracker.py`](/csim2/zhangzhiyu/MyProjects/auto_track/autotrack/core/single_vehicle_tracker.py)
- [`autotrack/dl/single_vehicle_net.py`](/csim2/zhangzhiyu/MyProjects/auto_track/autotrack/dl/single_vehicle_net.py)
- [`autotrack/dl/multi_vehicle_pipeline.py`](/csim2/zhangzhiyu/MyProjects/auto_track/autotrack/dl/multi_vehicle_pipeline.py)

Training and evaluation entry points:

```sh
uv run python -m autotrack.dl.train_single_vehicle --out-dir models/single_vehicle_cuda --device cuda --epochs 20
uv run python -m autotrack.dl.evaluate_single_vehicle --model models/single_vehicle_cuda/checkpoint_best.pt --out-dir /tmp/single_vehicle_eval --device cuda
uv run python -m autotrack.dl.run_single_vehicle_exact --out-dir /tmp/sv_exact_run --samples 64 --epochs 8 --batch-size 1 --hidden-dim 32 --device cpu
uv run python -m autotrack.dl.validate_single_vehicle_realbg_exact --model models/single_vehicle_cuda/checkpoint_best.pt --out-dir /tmp/sv_realbg_exact --realbg-npy datasets/03gauss_large.npy --samples 32 --device cpu
uv run python -m autotrack.dl.run_single_vehicle_realbg_exact --out-dir /tmp/sv_realbg_exact_run --realbg-npy datasets/03gauss_large.npy --samples 64 --epochs 5 --batch-size 1 --hidden-dim 32 --device cpu
```

Real-background training is also supported with a `.npy` window source:

```sh
uv run python -m autotrack.dl.train_single_vehicle --out-dir models/single_vehicle_realbg_cuda --device cuda --background-npy datasets/03gauss_large.npy --background-layout time_channel --background-channel-start 0
```

If you already have `manual_labels.json`, you can finetune in one command:

```sh
uv run python -m autotrack.dl.train_single_vehicle_from_labels --out-dir models/single_vehicle_labels_cuda --source-npy /path/to/real.npy --labels-json /path/to/manual_labels.json --device cpu --epochs 10
```

The current real-background smoke result on `datasets/03gauss_large.npy` uses
64 train windows, 16 validation windows, 5 epochs, and reaches:

- `track_found_rate = 1`
- `direction_acc = 1`
- `mean_time_abs_error_s = 0.0106`
- `mean_speed_abs_error_kmh = 0.70`

For repeatable comparisons, build a fixed benchmark file first:

```sh
uv run python -m autotrack.dl.build_single_vehicle_benchmark --out-file /tmp/single_vehicle_bench.pt --samples 64 --background-npy datasets/03gauss_large.npy --background-layout time_channel --background-channel-start 0
uv run python -m autotrack.dl.evaluate_single_vehicle --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --benchmark-file /tmp/single_vehicle_bench.pt --out-dir /tmp/single_vehicle_bench_eval
uv run python -m autotrack.dl.build_single_vehicle_benchmark_from_labels --out-file /tmp/single_vehicle_labels_bench.pt --source-npy /path/to/real.npy --labels-json /path/to/manual_labels.json
uv run python -m autotrack.dl.train_single_vehicle --out-dir models/single_vehicle_labels_cuda --benchmark-file /tmp/single_vehicle_labels_bench.pt --device cpu --epochs 10
uv run python -m autotrack.dl.train_single_vehicle_from_labels --out-dir models/single_vehicle_labels_cuda --source-npy /path/to/real.npy --labels-json /path/to/manual_labels.json --device cpu --epochs 10
uv run python -m autotrack.dl.predict_single_vehicle_real_npy --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --input /tmp/realbg_vehicle.npy --out-dir /tmp/realbg_vehicle_out --array-layout time_channel --channel-start 0 --channel-count 50 --window-seconds 10 --window-stride-seconds 10 --max-windows 16
uv run python -m autotrack.dl.plot_single_vehicle_benchmark_compare --benchmark-file /tmp/sv_same_dir_hard4.pt --model /tmp/sv_competing_mixed_model/checkpoint_best.pt --sample-index 6 --out-file /tmp/sv_same_dir_hard4_compare.png
uv run python -m autotrack.dl.validate_single_vehicle_realbg_exact --model models/single_vehicle_cuda/checkpoint_best.pt --out-dir /tmp/sv_realbg_exact --realbg-npy datasets/03gauss_large.npy --samples 32 --device cpu
```

The single-vehicle network is not a "find all cars at once" model. It is a
candidate refiner:

- input: one cropped candidate vehicle window, with optional priors
- output: one physically consistent trajectory for that window
- role: sharpen the chosen path, repair short gaps, and suppress local confusion
- the decoder keeps a small set of path hypotheses and ranks them with the
  network prior before final Kalman/Hungarian repair

The multi-vehicle entry point is a separate top-down pipeline:

1. Run the dense proposal model on the full segment, or on sliding windows.
2. Extract coarse candidate tracks from the proposal-enhanced map.
3. Crop around each candidate and run the single-vehicle decoder / network.
4. Merge, deduplicate, and resolve conflicts at the trajectory level.

The recommended real-data path is the windowed hybrid mode with the proposal
network enabled. That keeps the heavy decoding local, but still lets the full
segment contribute graph candidates and trajectory-level de-duplication.
The current calibrated simulation defaults are `window_seconds=10`,
`window_stride_seconds=5`, `dedup_min_overlap_channels=2`, and
`dedup_min_overlap_ratio=0.45`.

Run it with:

```sh
uv run python -m autotrack.dl.predict_multi_vehicle_real_npy --input /path/to/real.npy --out-dir /tmp/multi_vehicle_out --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --device cuda --plot
uv run python -m autotrack.dl.train_vehicle_proposal --out-dir models/vehicle_proposal_cuda --device cuda --epochs 20
uv run python -m autotrack.dl.predict_multi_vehicle_real_npy --input /path/to/real.npy --out-dir /tmp/multi_vehicle_out --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --proposal-model models/vehicle_proposal_cuda/checkpoint_best.pt --candidate-mode windowed --window-seconds 60 --window-stride-seconds 30 --device cuda --plot
```

`build_single_vehicle_benchmark_from_labels.py` defaults to multiple windows per
track, but boundary checks can reduce the final count when a candidate crop would
leave the labeled vehicle too close to the window edge.

The real-file inference path writes:

- `predicted_track_points.csv` for per-window fragments
- `merged_track_points.csv` for the stitched global trajectory

It also skips obvious blank windows with a cheap robust activity gate before
calling the heavier decoder. The real-file CLI now sorts candidate windows by
activity by default, so the most likely vehicle regions are decoded first.

## Why This Direction

PeakSlotNet is designed for multi-instance slot assignment. That is useful when the task is to keep many vehicles separated, but it is the wrong bias for the current target:

- the task is one vehicle at a time
- the final path must be continuous
- missing channels should be repaired by motion continuity, not by slot competition

The new path turns the problem into a single best trajectory decode, which is
the right place for Kalman smoothing and local graph matching. For multi-car
segments, the decoder is applied per candidate, not globally across all cars.

## Verification

The current synthetic smoke tests are:

```sh
uv run pytest -q tests/test_single_vehicle_tracker.py tests/test_single_vehicle_net.py
```

The current test suite also keeps the older graph/boundary helpers green:

```sh
uv run pytest -q tests/test_single_vehicle_tracker.py tests/test_track_fusion.py tests/test_boundary_completion.py
```

## Next Step

The next implementation step is a dedicated single-vehicle neural scorer trained on one-vehicle synthetic windows, with the decoder above kept as the post-processing layer.
