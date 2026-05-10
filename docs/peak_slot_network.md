# PeakSlotNet

PeakSlotNet is an instance-level vehicle trajectory network for clean DAS
heatmaps. It changes the representation from continuous time regression to
peak-candidate assignment.

## Core Idea

For each channel `c`, first detect up to `K` peak candidates:

```text
peak_time[c, k], peak_amp[c, k], peak_valid[c, k]
```

The network outputs `Q` vehicle slots. For each slot and channel it chooses one
candidate peak or the final `none` class:

```text
peak_logits[q, c, 0:K+1]
```

The class `K` means no vehicle point on that channel. This means exported
trajectory points can only be detected peaks; empty boundary channels are not
forced to receive a regressed time.

## Inference Decoding

Prediction now defaults to a Viterbi-style physical decoder instead of
independent per-channel argmax. For each active slot, the decoder searches a
peak path using:

```text
emission = log_softmax(peak_logits[q, ch, k])
hard speed window = 60-100 km/h
max channel skip = 4
soft penalties = speed mismatch + skipped channels + slope changes
```

This is a post-processing step, not part of the neural network and not
backpropagated during training. It keeps selected points on detected peak
candidates while reducing local reversals, cross-car jumps, and zigzag paths.
Use `--no-viterbi-decoder` in `predict_peak_slot_dataset.py` to compare with
the legacy argmax decoder.

## Matching and Loss

Training still uses Hungarian matching on the small `[Q, GT]` matrix. The
matching cost is:

```text
3.0 * visible_peak_nll
-1.0 * objectness
+0.2 * direction_cost
+0.1 * speed_cost
```

After matching, the default loss is:

```text
loss =
  loss_objectness
+ loss_peak_ce
+ 0.05 * loss_count
+ 0.5 * loss_direction
+ 0.25 * loss_speed
+ 1.0 * loss_monotonic
+ 0.2 * loss_smooth
```

`loss_peak_ce` is the main term. Visible GT channel points are supervised to
their candidate index. Invisible channel points are supervised to `none` with
default `none_weight=0.35`.

`loss_count` calibrates the number of active slots:

```text
SmoothL1(sum(sigmoid(objectness)), GT_count)
```

Monotonic and smoothness losses are weak regularizers computed from the
expected peak time distribution. They reduce reversals and jitter, but the hard
position constraint is the peak candidate selection itself.

## Data Flow

```text
generate_track_slot_dataset.py
    -> track_slot shards
convert_track_slot_to_peak_slot.py
    -> peak_slot shards with peak candidates and gt_peak_index
train_peak_slot.py
    -> checkpoint_last.pt / checkpoint_best.pt
predict_peak_slot_dataset.py
    -> summary.json / CSV / overlay plots
infer_trajectory_model.py --model-family peak_slot
    -> auto_tracks_deep.csv
```

## Commands

```sh
sh convert_track_slot_to_peak_slot.sh
DEVICE=cuda EPOCHS=20 BATCH_SIZE=32 sh train_peak_slot_cuda.sh
MODEL=models/peak_slot_cuda/checkpoint_best.pt DATA_DIR=datasets/peak_slot/train sh predict_peak_slot_dataset.sh
```

CPU smoke test:

```sh
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir /tmp/track_slot_data --num-samples 32 --shard-size 16 --window-seconds 10 --time-downsample 20 --vehicles-min 2 --vehicles-max 4 --workers 2 --overwrite
uv run python -m autotrack.dl.convert_track_slot_to_peak_slot --in-dir /tmp/track_slot_data --out-dir /tmp/peak_slot_data --peak-candidates-per-channel 16 --overwrite
uv run python -m autotrack.dl.train_peak_slot --data-dir /tmp/peak_slot_data --out-dir /tmp/peak_slot_smoke --device cpu --epochs 1 --batch-size 2 --max-samples 32
uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir /tmp/peak_slot_data --model /tmp/peak_slot_smoke/checkpoint_best.pt --out-dir /tmp/peak_slot_pred --device cpu --plot-samples 8
```
