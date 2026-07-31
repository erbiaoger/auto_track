# Compact Slot Model

This is the new lightweight mainline for the 50-channel vehicle task.
It replaces the heavier 2D CNN + deep decoder stack with a channel-first
encoder and a small slot-attention head.

## Why this design

The target data has three strong priors:

- 50 fixed channels with 100 m spacing
- typical vehicle speeds around 70-90 km/h
- 2-minute windows containing about 10-20 vehicles

That means a full-resolution, heavy decoder is unnecessary. Most vehicles are
close to linear in channel-time space, and the main problem is slot separation
under crossing and partial gaps. The compact model keeps the set-prediction
formulation, but reduces the backbone to:

1. Temporal CNN per channel
2. Bidirectional GRU across channels
3. Learned slot queries with attention

The output interface stays compatible with the shard datasets:

- `objectness_logits`
- `direction_logits`
- `speed`
- `time`
- `visibility_logits`

## Inference behavior

Prediction uses soft slot counting, slot ranking, local peak refinement,
Kalman smoothing, and trajectory de-duplication.
The real-data CLI uses the same defaults as the calibrated synthetic smoke:
`objectness_threshold=0.15`, `visibility_threshold=0.35`, and
`objectness_count_scale=1.05`.

## Entry points

```sh
uv run python -m autotrack.dl.train_compact_slot --data-dir datasets/track_slot_profile_only_120s/train --out-dir /tmp/compact_slot --device cuda --amp on
uv run python -m autotrack.dl.predict_compact_slot_dataset --data-dir datasets/track_slot_profile_only_120s/train --model /tmp/compact_slot/checkpoint_best.pt --out-dir /tmp/compact_slot_eval --device cuda
uv run python -m autotrack.dl.predict_compact_slot_real_npy --model /tmp/compact_slot/checkpoint_best.pt --input datasets/03gauss_large.npy --out-dir /tmp/compact_slot_real --device cuda --channel-count 50
uv run python -m autotrack.dl.calibrate_compact_slot_inference --data-dir datasets/track_slot_profile_only_120s/train --model /tmp/compact_slot/checkpoint_best.pt --out-json /tmp/compact_slot_calibration/report.json --device cuda
```

## Status

This branch is new. It has smoke-tested training, shard evaluation, and real
`npy` inference paths, but it still needs longer training on the profile-driven
50-channel datasets before it can be judged on final accuracy.
