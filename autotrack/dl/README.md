# autotrack.dl

Deep-learning code for DAS vehicle trajectory recognition.

## TrackSlotNet Files

- `generate_track_slot_dataset.py`: creates tensor shards for TrackSlotNet
  training. It writes `meta.json` and `shard_*.pt`; it does not write or read
  SAC files. The default motion mix is `constant_sparse,smooth_random,stop_go`
  with rare stop-go events.
- `predict_track_slot_dataset.py`: runs a trained TrackSlotNet checkpoint
  directly on generated tensor shards and writes prediction CSV, metrics, and
  heatmap overlay figures. It applies monotonic trimming and trajectory NMS by
  default before writing predictions.
- `plot_track_slot_history.py`: reads `train_history.jsonl` and plots epoch
  curves for loss, F1, count error, objectness, and other metrics.
- `track_slot_model.py`: model, Hungarian/greedy set loss, metrics, inference,
  NMS, and checkpoint helpers for `model_family=track_slot`.
- `train_track_slot.py`: trains TrackSlotNet from generated shards, including
  objectness count calibration plus monotonic and smoothness trajectory losses.

## PeakSlotNet Files

- `convert_track_slot_to_peak_slot.py`: converts existing TrackSlotNet shards
  into peak-candidate shards with `peak_time`, `peak_amp`, `peak_valid`, and
  `gt_peak_index`.
- `peak_slot_model.py`: PeakSlotNet model, Hungarian/greedy set loss, metrics,
  checkpoint helpers, and SAC-window inference for `model_family=peak_slot`.
- `train_peak_slot.py`: trains PeakSlotNet from converted peak-candidate shards.
- `predict_peak_slot_dataset.py`: predicts selected peak candidates, writes
  CSV files, and draws overlay figures where predictions lie on detected peaks.

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
uv run python -m autotrack.dl.predict_track_slot_dataset --data-dir datasets/track_slot/train --model models/track_slot_cuda/checkpoint_best.pt --out-dir /tmp/track_slot_prediction_check --device cuda --max-samples 128
uv run python -m autotrack.dl.convert_track_slot_to_peak_slot --in-dir datasets/track_slot/train --out-dir datasets/peak_slot/train --overwrite
uv run python -m autotrack.dl.train_peak_slot --data-dir datasets/peak_slot/train --out-dir models/peak_slot_cuda --device cuda --amp on
uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir datasets/peak_slot/train --model models/peak_slot_cuda/checkpoint_best.pt --out-dir /tmp/peak_slot_prediction_check --device cuda --max-samples 128
```

`--auto-resume` reads `<out-dir>/checkpoint_last.pt` when present. `--epochs`
means the final total epoch count, not the number of additional epochs.
