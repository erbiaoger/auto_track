# autotrack.dl

Deep-learning code for DAS vehicle trajectory recognition.

## TrackSlotNet Files

- `generate_track_slot_dataset.py`: creates tensor shards for TrackSlotNet
  training. It writes `meta.json` and `shard_*.pt`; it does not write or read
  SAC files. The default motion mix is `constant_sparse,smooth_random,stop_go`
  with rare stop-go events. Generated inputs include white noise, correlated
  noise, channel bias/gain variation, slow baseline drift, and isolated pulse
  interference by default. The default shell preset also adds fixed real-data
  dead channels, random dead channels, and random zero background blocks.
  Direction sampling keeps the expected traffic prior.
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
```

`--auto-resume` reads `<out-dir>/checkpoint_last.pt` when present. `--epochs`
means the final total epoch count, not the number of additional epochs.
