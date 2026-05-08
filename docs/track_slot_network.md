# TrackSlotNet

TrackSlotNet is an instance-level trajectory recognition model for clean DAS
Gaussian heatmap data. It is designed for the case where each vehicle already
appears as a smooth ridge in the channel-time plane, and the goal is to make the
network identify which ridge belongs to which vehicle.

## Task

For one input window:

```text
x: [1, C, T_down]
```

the model predicts a fixed number of slots:

```text
objectness_logits: [Q]
time:              [Q, C]
visibility_logits: [Q, C]
direction_logits:  [Q, 2]
speed:             [Q]
```

Each slot is one candidate vehicle. The number of vehicles is variable because
inference keeps only slots whose objectness passes a threshold, then removes
duplicate slots by trajectory NMS.

## Physical Representation

For a constant-speed vehicle moving over evenly spaced DAS channels, the center
time at channel `c` is approximately:

```text
t(c) = t0 + direction_sign * c * dx / v
```

The observed training input is a Gaussian response around this center time:

```text
x(c, t) += A * exp(-0.5 * ((t - t(c)) / sigma)^2)
```

The model does not need to segment pixels. It learns the instance-level mapping
from a set of Gaussian ridges to a set of vehicle slots. For every slot, `time`
stores the normalized center time at each channel and `visibility` marks whether
the vehicle is visible in the current window at that channel.

## Matching and Loss

Training uses Hungarian matching only to assign predicted slots to GT vehicles.
The cost matrix is small:

```text
[Q, GT]
```

The matching cost compares only compact trajectory attributes:

```text
time error on visible GT channels
visibility error
direction probability
speed error
objectness
```

No `[Q, C, T]` mask is generated. After matching, the loss is computed on GPU:

```text
loss =
  loss_objectness
+ 8.0 * loss_time
+ 1.0 * loss_visibility
+ 0.5 * loss_direction
+ 0.5 * loss_speed
```

`loss_time` is weighted by GT visibility, so invisible channel entries do not
contribute to the time regression.

## Inference

Inference does not use graph search or clustering.

1. Run the model on the downsampled input window.
2. Keep slots with `sigmoid(objectness) >= threshold`.
3. For each kept slot, keep channel points with
   `sigmoid(visibility) >= threshold`.
4. Drop slots with too few visible channels.
5. Apply trajectory NMS: if two slots overlap on enough channels and their
   median time difference is below the tolerance, keep the higher-score slot.

The remaining slots are converted directly into `Track` and `TrackPoint`
objects for the existing GUI and CSV exporter.

## Data Flow

```text
generate_track_slot_dataset.py
    -> meta.json + shard_*.pt
train_track_slot.py
    -> checkpoint_last.pt / checkpoint_best.pt
predict_track_slot_dataset.py
    -> summary.json / predicted_tracks.csv / sample_summary.csv / plots/*.png
infer_trajectory_model.py --model-family track_slot
    -> auto_tracks_deep.csv
evaluate_trajectory_model.py --model-family track_slot
    -> precision / recall / F1 / time error
```

## CPU and GPU Runs

CPU smoke test:

```sh
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir /tmp/track_slot_data --num-samples 32 --shard-size 16 --window-seconds 10 --time-downsample 20 --workers 2 --overwrite
uv run python -m autotrack.dl.train_track_slot --data-dir /tmp/track_slot_data --out-dir /tmp/track_slot_smoke --device cpu --epochs 1 --batch-size 2
uv run python -m autotrack.dl.predict_track_slot_dataset --data-dir /tmp/track_slot_data --model /tmp/track_slot_smoke/checkpoint_best.pt --out-dir /tmp/track_slot_predict --device cpu --max-samples 16
```

CUDA training:

```sh
WORKERS=8 sh generate_track_slot_dataset.sh
DEVICE=cuda EPOCHS=2 BATCH_SIZE=64 sh train_track_slot_cuda.sh
DEVICE=cuda sh predict_track_slot_dataset.sh
```
