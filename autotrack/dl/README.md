# autotrack.dl

Deep-learning code for DAS vehicle trajectory recognition.

## TrackSlotNet Files

- `generate_track_slot_dataset.py`: creates tensor shards for TrackSlotNet
  training. It now also accepts `--profile`, so a `realism_profile.json` can
  inject fixed/probabilistic bad-channel structure plus real peak-shape
  defaults into a fully synthetic labeled dataset.
- `generate_track_slot_dataset_from_real_npy.py`: creates TrackSlotNet shards by sampling real `.npy` background windows and overlaying synthetic vehicles. Use this when pure synthetic backgrounds are too far from the real sparse DAS distribution.
  It now supports `--profile`, `--profile-strength`, `--window-sampler`, and
  `--artifact-policy`, so the generator can default to a
  `realism_profile.json`-driven mode while still letting explicit CLI
  arguments override any profile-derived defaults.
  training. It writes `meta.json` and `shard_*.pt`; it does not write or read
  SAC files. The default motion mix is `constant_sparse,smooth_random,stop_go`
  with rare stop-go events. The default shell preset targets v4 noisy/bad-channel
  Gaussian-window data: continuous white/colored noise, channel bias/gain
  variation, baseline drift, dense isolated Gaussian windows, random dead
  channels, and missing channel-time blocks are enabled. Direction sampling keeps
  the expected traffic prior.
- `predict_track_slot_dataset.py`: runs a trained TrackSlotNet checkpoint
  directly on generated tensor shards and writes prediction CSV, metrics, and
  heatmap overlay figures. It applies monotonic trimming and trajectory NMS by
  default before writing predictions.
- `plot_track_slot_history.py`: reads `train_history.jsonl` and plots epoch
  curves for loss, F1, count error, objectness, and other metrics.
- `plot_dataset_labels.py`: plots tensor-shard heatmaps with GT labels. It can
  inspect both raw TrackSlotNet `time` labels and converted PeakSlotNet
  `gt_peak_index` labels.
- `track_slot_model.py`: model, Hungarian/greedy set loss, metrics, inference,
  NMS, and checkpoint helpers for `model_family=track_slot`.
- `train_track_slot.py`: trains TrackSlotNet from generated shards, including
  objectness count calibration plus monotonic and smoothness trajectory losses.

## Compact Slot Files

- `compact_slot_model.py`: lightweight channel-first slot model for the
  50-channel vehicle task. It keeps Hungarian matching and slot outputs, but
  replaces the heavier 2D CNN/decoder stack with a small temporal encoder plus
  channel GRU and slot attention block.
- `train_compact_slot.py`: training entrypoint for the compact slot model. It
  reuses the shard loading and logging harness, but swaps in the compact model.
- `predict_compact_slot_dataset.py`: compact-model shard evaluation with loss
  and slot metrics. It now uses a soft-count based slot selection step so the
  prediction count can be calibrated against the synthetic benchmark.
- `predict_compact_slot_real_npy.py`: sliding-window real `.npy` inference for
  the compact slot model, including per-window plotting and merged track output.
  Its defaults now match the calibrated synthetic settings
  (`objectness_threshold=0.15`, `visibility_threshold=0.35`,
  `objectness_count_scale=1.05`). The evaluation path now also uses a
  normalized point threshold of `0.12`, which matches the current synthetic
  smoke scale better than the older `0.05` default.
- `calibrate_compact_slot_inference.py`: threshold / soft-count sweep for the
  compact slot decoder on a shard dataset.

## PeakSlotNet Files

- `convert_track_slot_to_peak_slot.py`: converts existing TrackSlotNet shards
  into peak-candidate shards with `peak_time`, `peak_amp`, `peak_valid`, and
  `gt_peak_index`.
- `build_reference_style_peakslot_dataset.py`: one-command orchestration for
  the reference-style TrackSlot -> PeakSlot + waveform-line prior workflow.
  It generates the intermediate track dataset, converts it to peak-slot, adds
  the U-Net prior channel, and can render the same inspection plots as the
  sample directory you pointed to.
- `analyze_peak_slot_domain_gap.py`: compares a reference peak-slot dataset and
  a target peak-slot dataset, and can optionally run a PeakSlotNet checkpoint on
  both to quantify objectness/count drift. It writes `summary.json` and
  `report.md`, which is useful when synthetic validation looks good but real
  data over-predicts.
- `calibrate_realbg_generator.py`: compares one or more generated `track_slot`
  or `peak_slot` datasets against `realism_profile.json`, scores their domain
  mismatch, and writes `calibration_summary.json` plus
  `calibration_report.md`.
- `segment_real_npy_to_peak_slot.py`: cuts an unlabeled real DAS `.npy` array
  into overlapping PeakSlotNet shards for direct prediction/inspection.
- `raw_energy_line_filter.py`: filters probability peaks with raw-waveform
  energy diagonals constrained by station spacing and vehicle speed. It can
  scan both directions, derive station-specific thresholds, and export the
  line-consistent peak table before any model prediction. See
  `docs/raw_energy_line_filter.md` for the full Chinese explanation and
  command examples.
- `peak_slot_model.py`: PeakSlotNet model, Hungarian/greedy set loss, metrics,
  checkpoint helpers, and SAC-window inference for `model_family=peak_slot`.
- `train_peak_slot.py`: trains PeakSlotNet from converted peak-candidate shards.
  It supports either a fixed tail-shard validation split with `--val-fraction`
  or a separate validation dataset through `--val-data-dir`; validation can be
  run every N epochs with `--val-every`.
- `predict_peak_slot_dataset.py`: predicts selected peak candidates, writes
  CSV files, and draws overlay figures where predictions lie on detected peaks.
  Its default thresholds favor recall, but cross-slot conflict suppression
  remains enabled to reduce duplicate tracks. It also supports
  `--fusion-mode graph_extend` to extend PeakSlotNet fragments with the classic
  graph search before writing metrics and plots.

## Single-Vehicle Files

- `single_vehicle_net.py`: dense one-vehicle heatmap scorer plus global motion
  heads. It is the new neural front-end for the one-vehicle task.
- `single_vehicle_tracker.py`: single-path decoder with graph continuity,
  Hungarian gap bridging, and Kalman smoothing.
- `train_single_vehicle.py`: synthetic one-vehicle trainer built on
  `online_synth_dataset.py`. It can also sample windows from a real background
  `.npy` file and overlay one synthetic vehicle on top.
- `evaluate_single_vehicle.py`: end-to-end synthetic benchmark for a trained
  single-vehicle checkpoint. It also supports the real-background `.npy`
  smoke path.
- `build_single_vehicle_benchmark.py`: writes a fixed benchmark `.pt` file so
  single-vehicle models can be compared on the exact same windows.
- `build_single_vehicle_benchmark_realistic.py`: realistic single-vehicle
  benchmark generator tuned for 50 channels, 100 m spacing, and 70-90 km/h
  motion priors.
- `build_single_vehicle_benchmark_from_labels.py`: converts
  `manual_labels.json` plus a real `.npy` source into a fixed benchmark `.pt`
  file.
- `train_single_vehicle_from_labels.py`: one-command wrapper that builds the
  label benchmark and finetunes the single-vehicle model in the same run.
- `train_single_vehicle.py`: accepts `--benchmark-file` for direct finetuning on
  a fixed benchmark built from labels.
- `predict_single_vehicle_real_npy.py`: slides the single-vehicle model over a
  real `.npy` file and exports window-level fragments plus a stitched global
  trajectory. It also includes a cheap activity gate to skip obvious blank
  windows before running the heavier decoder. The current default
  `window_seconds=60` is calibrated for one-vehicle trajectories that span a
  longer time slice in the real 50-channel windows, and the default ranking
  now uses the model score instead of pure activity so the scan is more
  target-aware. The `--preset real_vehicle` mode locks in the validated
  long-window inference settings without needing to remember the individual
  flags.
- `predict_single_vehicle_real_vehicle.py`: convenience wrapper that always
  runs the validated `real_vehicle` preset.
- `train_single_vehicle_real_vehicle.py`: one-command wrapper that rebuilds
  the mixed clean / real-background / TrackSlot-derived benchmark and
  finetunes the single-vehicle model on it.
- `run_single_vehicle_real_vehicle.py`: end-to-end wrapper that trains the
  mixed model and immediately runs a raw real-data smoke check. Use
  `--quick` for a smaller smoke-friendly run.
- `validate_single_vehicle_real_vehicle.py`: rebuilds the validation mix,
  evaluates a checkpoint on it, and runs a raw real-data smoke check.
- `build_single_vehicle_benchmark_exact.py`: generates a near-perfect
  single-vehicle benchmark with no competing tracks, no dropouts, and no
  noise. Use this to measure the upper bound of the redesigned pipeline.
- `train_single_vehicle_exact.py`: trains the redesigned one-vehicle model on
  the exact benchmark. It can either build the benchmark internally or reuse
  a prebuilt benchmark file.
- `validate_single_vehicle_exact.py`: builds the exact benchmark, trains the
  model, evaluates it, and writes an overlay plot for the first sample.
- `run_single_vehicle_exact.py`: one-command exact-mode wrapper that forwards
  to the exact validation flow.
- `validate_single_vehicle_realbg_exact.py`: evaluates a checkpoint on a
  clean single-vehicle benchmark sampled from the real background array. Use
  this to measure how the redesigned pipeline behaves on real noise without
  competing vehicles.
- `train_single_vehicle_realbg_exact.py`: finetunes the single-vehicle model
  on clean real-background windows only.
- `run_single_vehicle_realbg_exact.py`: one-command clean real-background
  workflow that finetunes, evaluates, and plots the first sample.
- `validate_single_vehicle_real_vehicle.py`: rebuilds the mixed validation set,
  evaluates a checkpoint on it, and runs a raw real-data smoke check.
- `predict_multi_vehicle_real_npy.py`: multi-vehicle `.npy` inference entry
  point. It now defaults to a hybrid candidate mode that scans overlapping
  blocks with the single-vehicle network, also keeps global graph candidates,
  and then stitches / de-duplicates the resulting track fragments at the
  trajectory level. The current calibrated defaults use `window_seconds=10`,
  `window_stride_seconds=5`, `dedup_min_overlap_channels=2`, and
  `dedup_min_overlap_ratio=0.45`.
- `multi_vehicle_pipeline.py`: shared multi-instance candidate extraction and
  refinement helpers used by the multi-vehicle CLI.
- `evaluate_multi_vehicle_benchmark.py`: synthetic benchmark evaluator for the
  multi-vehicle hybrid pipeline.

## Legacy / Compatible Files

- `trajectory_set_model.py`: older query polyline model and shared utilities.
- `query_mask_instance_model.py`: query mask instance model.
- `train_trajectory_online.py`: online synthetic trainer for legacy query
  models.
- `train_trajectory_model.py`: SAC/tracks.json training path.
- `infer_trajectory_model.py`: CLI inference wrapper.
- `evaluate_trajectory_model.py`: CLI evaluation against simulated `tracks.json`.

## Common Commands

```sh
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir datasets/track_slot/train --num-samples 1024 --shard-size 128 --workers 8 --overwrite
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir datasets/track_slot_profile_only/train --profile datasets/profiles/xi_gauss_50_realbg/realism_profile.json --num-samples 1024 --shard-size 128 --vehicles-min 6 --vehicles-max 18 --overwrite
uv run python -m autotrack.dl.generate_track_slot_dataset_from_real_npy --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy --out-dir datasets/track_slot_realbg/train --num-samples 1024 --shard-size 128 --window-seconds 120 --window-stride-seconds 60 --channel-count 50 --vehicles-min 6 --vehicles-max 24 --overwrite
uv run python -m autotrack.dl.train_track_slot --data-dir datasets/track_slot/train --out-dir models/track_slot_cuda --device cuda --amp on
uv run python -m autotrack.dl.train_track_slot --data-dir datasets/track_slot/train --out-dir models/track_slot_cuda --device cuda --amp on --epochs 200 --auto-resume
uv run python -m autotrack.dl.plot_track_slot_history --run-dir models/track_slot_cuda --separate
uv run python -m autotrack.dl.plot_dataset_labels --data-dir datasets/track_slot/train --out-dir /tmp/track_slot_label_check --sample-indices 6
uv run python -m autotrack.dl.predict_track_slot_dataset --data-dir datasets/track_slot/train --model models/track_slot_cuda/checkpoint_best.pt --out-dir /tmp/track_slot_prediction_check --device cuda --max-samples 128
uv run python -m autotrack.dl.convert_track_slot_to_peak_slot --in-dir datasets/track_slot/train --out-dir datasets/peak_slot/train --overwrite
uv run python -m autotrack.dl.build_reference_style_peakslot_dataset --out-dir /tmp/peak_slot_unetprior_ref --unet-checkpoint models/peak_slot_profile_only_120s_unetprior_cuda/checkpoint_best.pt --num-samples 4 --shard-size 4 --plot-peaks --overwrite
uv run python -m autotrack.dl.segment_real_npy_to_peak_slot --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays/gauss_section.npy --out-dir datasets/peak_slot/xi_gauss_50_120s_stride60 --window-seconds 120 --stride-seconds 60 --fs 1000 --channel-start 0 --channel-count 50 --overwrite
uv run python -m autotrack.dl.plot_dataset_labels --data-dir datasets/peak_slot/train --out-dir /tmp/peak_slot_label_check --sample-indices 6 --plot-peaks
uv run python -m autotrack.dl.train_peak_slot --data-dir datasets/peak_slot/train --out-dir models/peak_slot_cuda --device cuda --amp on
uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir datasets/peak_slot/train --model models/peak_slot_cuda/checkpoint_best.pt --out-dir /tmp/peak_slot_prediction_check --device cuda --max-samples 128
uv run python -m autotrack.dl.train_compact_slot --data-dir datasets/track_slot/train --out-dir models/compact_slot_cuda --device cuda --epochs 20
uv run python -m autotrack.dl.calibrate_compact_slot_inference --data-dir datasets/track_slot/train --model models/compact_slot_cuda/checkpoint_best.pt --out-json /tmp/compact_slot_calibration/report.json --device cuda --max-samples 64
uv run python -m autotrack.dl.predict_compact_slot_real_npy --model models/compact_slot_cuda/checkpoint_best.pt --input /path/to/real.npy --out-dir /tmp/compact_slot_real_npy --device cuda --array-layout time_channel --channel-count 50 --window-seconds 10 --window-stride-seconds 10
uv run python -m autotrack.dl.train_single_vehicle --out-dir models/single_vehicle_cuda --device cuda --epochs 20 --train-samples 4096 --val-samples 512
uv run python -m autotrack.dl.evaluate_single_vehicle --model models/single_vehicle_cuda/checkpoint_best.pt --out-dir /tmp/single_vehicle_eval --device cuda --samples 256
uv run python -m autotrack.dl.build_single_vehicle_benchmark --out-file /tmp/single_vehicle_bench.pt --samples 64 --background-npy datasets/03gauss_large.npy --background-layout time_channel --background-channel-start 0
uv run python -m autotrack.dl.evaluate_single_vehicle --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --benchmark-file /tmp/single_vehicle_bench.pt --out-dir /tmp/single_vehicle_bench_eval
uv run python -m autotrack.dl.build_single_vehicle_benchmark_from_labels --out-file /tmp/single_vehicle_labels_bench.pt --source-npy /path/to/real.npy --labels-json /path/to/manual_labels.json
uv run python -m autotrack.dl.train_single_vehicle --out-dir models/single_vehicle_labels_cuda --benchmark-file /tmp/single_vehicle_labels_bench.pt --device cpu --epochs 10
uv run python -m autotrack.dl.train_single_vehicle_from_labels --out-dir models/single_vehicle_labels_cuda --source-npy /path/to/real.npy --labels-json /path/to/manual_labels.json --device cpu --epochs 10
uv run python -m autotrack.dl.predict_single_vehicle_real_npy --model models/single_vehicle_realbg_cuda/checkpoint_best.pt --input datasets/03gauss_large.npy --out-dir /tmp/single_vehicle_realbg_infer --array-layout time_channel --channel-start 0 --channel-count 50
uv run python -m autotrack.dl.analyze_peak_slot_domain_gap --reference-dir datasets/peak_slot_v3_120s_realistic/test --target-dir datasets/peak_slot/xi_gauss_50_120s_stride60_saved_arrays04 --model models/peak_slot_v4_120s_noisy_badch_cuda/checkpoint_best.pt --out-dir /tmp/peak_slot_domain_gap --max-samples 64 --device cpu
```

The top-level helper shell below wraps the real-background generator with the
current heavier sparse-artifact preset:

```sh
sh generate_track_slot_dataset_from_real_npy.sh
```

The profile-driven workflow adds three more top-level shells:

```sh
sh profile_real_npy_background.sh
sh generate_track_slot_dataset_from_real_npy_profile.sh
sh generate_track_slot_dataset_profile_only.sh
sh calibrate_realbg_generator.sh
```

`--auto-resume` reads `<out-dir>/checkpoint_last.pt` when present. `--epochs`
means the final total epoch count, not the number of additional epochs.
