# autotrack.dl

Deep-learning code for DAS vehicle trajectory recognition.

## TrackSlotNet Files

- `generate_track_slot_dataset.py`: creates tensor shards for TrackSlotNet
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

## PeakSlotNet Files

- `convert_track_slot_to_peak_slot.py`: converts existing TrackSlotNet shards
  into peak-candidate shards with `peak_time`, `peak_amp`, `peak_valid`, and
  `gt_peak_index`.
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
- `peak_slot_model.py`: PeakSlotNet model, Hungarian/greedy set loss, metrics,
  checkpoint helpers, and SAC-window inference for `model_family=peak_slot`.
- `train_peak_slot.py`: trains PeakSlotNet from converted peak-candidate shards.
  It supports either a fixed tail-shard validation split with `--val-fraction`
  or a separate validation dataset through `--val-data-dir`; validation can be
  run every N epochs with `--val-every`.
- `predict_peak_slot_dataset.py`: predicts selected peak candidates, writes
  CSV files, and draws overlay figures where predictions lie on detected peaks.
  Its default thresholds favor recall, but cross-slot conflict suppression
  remains enabled to reduce duplicate tracks.

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
uv run python -m autotrack.dl.generate_track_slot_dataset_from_real_npy --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy --out-dir datasets/track_slot_realbg/train --num-samples 1024 --shard-size 128 --window-seconds 120 --window-stride-seconds 60 --channel-count 50 --vehicles-min 6 --vehicles-max 24 --overwrite
uv run python -m autotrack.dl.train_track_slot --data-dir datasets/track_slot/train --out-dir models/track_slot_cuda --device cuda --amp on
uv run python -m autotrack.dl.train_track_slot --data-dir datasets/track_slot/train --out-dir models/track_slot_cuda --device cuda --amp on --epochs 200 --auto-resume
uv run python -m autotrack.dl.plot_track_slot_history --run-dir models/track_slot_cuda --separate
uv run python -m autotrack.dl.plot_dataset_labels --data-dir datasets/track_slot/train --out-dir /tmp/track_slot_label_check --sample-indices 6
uv run python -m autotrack.dl.predict_track_slot_dataset --data-dir datasets/track_slot/train --model models/track_slot_cuda/checkpoint_best.pt --out-dir /tmp/track_slot_prediction_check --device cuda --max-samples 128
uv run python -m autotrack.dl.convert_track_slot_to_peak_slot --in-dir datasets/track_slot/train --out-dir datasets/peak_slot/train --overwrite
uv run python -m autotrack.dl.segment_real_npy_to_peak_slot --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays/gauss_section.npy --out-dir datasets/peak_slot/xi_gauss_50_120s_stride60 --window-seconds 120 --stride-seconds 60 --fs 1000 --channel-start 0 --channel-count 50 --overwrite
uv run python -m autotrack.dl.plot_dataset_labels --data-dir datasets/peak_slot/train --out-dir /tmp/peak_slot_label_check --sample-indices 6 --plot-peaks
uv run python -m autotrack.dl.train_peak_slot --data-dir datasets/peak_slot/train --out-dir models/peak_slot_cuda --device cuda --amp on
uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir datasets/peak_slot/train --model models/peak_slot_cuda/checkpoint_best.pt --out-dir /tmp/peak_slot_prediction_check --device cuda --max-samples 128
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
sh calibrate_realbg_generator.sh
```

`--auto-resume` reads `<out-dir>/checkpoint_last.pt` when present. `--epochs`
means the final total epoch count, not the number of additional epochs.
