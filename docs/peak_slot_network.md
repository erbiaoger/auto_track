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
time_prior[q, c]
visibility_prior_logits[q, c]
```

The class `K` means no vehicle point on that channel. This means exported
trajectory points can only be detected peaks; empty boundary channels are not
forced to receive a regressed time.

`time_prior` is a continuous per-slot guide for where the vehicle should be on
each channel. It does not replace peak selection. During decoding it adds a
soft penalty when a candidate peak is far from the prior, so crossings are less
likely to switch to the other vehicle after the intersection. Old checkpoints
that do not contain the prior heads still load; in that case the time-prior
penalty is disabled.

## Inference Decoding

Prediction now defaults to `beam_global`, a beam Viterbi physical decoder plus
global conflict-aware selection, instead of independent per-channel argmax. For
each active slot, the decoder searches peak paths using:

```text
emission = log_softmax(peak_logits[q, ch, k])
prior penalty = abs(peak_time[ch, k] - time_prior[q, ch])
candidate threshold = 0.01
beam size = 4
hard speed window = 60-100 km/h
max channel skip = 4
soft penalties = speed mismatch + skipped channels (strong) + slope changes + speed inertia
```

This is a post-processing step, not part of the neural network and not
backpropagated during training. It keeps selected points on detected peak
candidates while reducing local reversals, cross-car jumps, and zigzag paths.
Use `--no-viterbi-decoder` in `predict_peak_slot_dataset.py` to compare with
the legacy argmax decoder. The Viterbi candidate threshold is intentionally
lower than the exported point threshold so weak but physically consistent
middle-channel peaks can bridge otherwise broken trajectories.
The inertia term keeps a smoothed running slope for each partial path and
penalizes candidates that do not continue from the previously implied speed,
which reduces identity switches after crossing points.
`beam_global` keeps several alternatives for each slot, then greedily selects
one path per slot while penalizing repeated use of the same peak candidates.
This makes crossing cases less brittle because the decoder can reject a locally
high-scoring path that steals peaks from another trajectory.

## Matching and Loss

Training uses one-to-one matching on the small `[Q, GT]` matrix. The default is
exact CPU/SciPy Hungarian matching; `--matcher auction` enables a torch auction
matcher that stays on the tensor device and is useful when CPU matching stalls
GPU training. The matching cost is:

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
+ 2.0 * loss_time_prior
+ 0.5 * loss_visibility_prior
+ 0.1 * loss_slot_competition
+ 0.2 * loss_crossing_margin
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

The prior losses supervise matched slots to output the GT candidate time and
visibility per channel. The slot competition loss discourages different slots
from assigning high probability to the same peak candidates. The crossing
margin term raises the margin between the matched GT candidate and other
candidate peaks on the same channel, which is targeted at crossing and
near-crossing switch errors. It reuses the main matching result, so enabling it
does not run a second Hungarian/auction assignment.

## Synthetic Interaction Data

The 120 s branch defaults generate shorter windows in separate directories:

```text
window_seconds = 120
vehicles_min/max = 32/48
default generated samples = 80000
primary_ratio = 0.8333333333
realism_preset = xi_gauss_50
track-slot data = datasets/track_slot_v3_120s_realistic/train
peak-slot data = datasets/peak_slot_v3_120s_realistic/train
model output = models/peak_slot_v3_120s_realistic_cuda
```

`generate_track_slot_dataset.py` now supports interaction-heavy training
samples:

```text
--interaction-ratio 0.3
--interaction-types crossing,overtake,near_parallel
--interaction-time-min-frac 0.05
--interaction-time-max-frac 0.95
--noise-std 0.35
--colored-noise-std 0.18
--channel-bias-std 0.08
--channel-gain-std 0.12
--baseline-drift-std 0.10
--dead-channel-indices 5,6,15,16,22,36,38,42,45,48
--random-dead-channel-ratio 0.35
--random-dead-channel-min 1 --random-dead-channel-max 5
--zero-background-ratio 1.0
--zero-background-rate 160
--zero-background-channel-min 1 --zero-background-channel-max 8
--zero-background-duration-min-s 2.0 --zero-background-duration-max-s 10.0
--primary-ratio 0.8333333333
--isolated-noise-ratio 1.0
--isolated-noise-rate 18
--isolated-noise-amp-min 1 --isolated-noise-amp-max 6
--isolated-noise-sigma-min 0.08 --isolated-noise-sigma-max 0.35
```

The interaction time is sampled across the full window, not only the middle.
The difficult cases include opposite-direction crossings, same-direction
overtakes, near-parallel close tracks, and isolated Gaussian peaks that may be
stronger than nearby vehicle peaks. Background noise is added before robust
input normalization so the saved training tensors are no longer ideal clean
Gaussian traces.

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

For inspection plots, the default prediction command is recall-oriented:
`OBJECTNESS_THRESHOLD=0.35`, `MIN_VISIBLE_CHANNELS=2`, and
`VITERBI_BEAM_SIZE=8`, while cross-slot conflict suppression stays enabled with
`GLOBAL_CONFLICT_PENALTY=2.0`. If overlays show too many false positives, first
raise `OBJECTNESS_THRESHOLD`.

## Commands

```sh
sh generate_track_slot_dataset.sh
sh convert_track_slot_to_peak_slot.sh
DEVICE=cuda EPOCHS=20 BATCH_SIZE=32 sh train_peak_slot_cuda.sh
MODEL=models/peak_slot_v2_120s_cuda/checkpoint_best.pt DATA_DIR=datasets/peak_slot_v2_120s/test sh predict_peak_slot_dataset.sh
```

Training defaults reserve a fixed 10% tail-shard validation split and evaluate
it every 5 epochs:

```text
VAL_FRACTION=0.1
VAL_EVERY=5
OBJECT_LOSS_WEIGHT=1.25
COUNT_LOSS_WEIGHT=0.15
METRIC_OBJECTNESS_THRESHOLD=0.35
```

For a fully independent validation set, generate another peak-slot directory
with a different `SEED` and pass it as `VAL_DATA_DIR=/path/to/peak_slot_val`.

CPU smoke test:

```sh
uv run python -m autotrack.dl.generate_track_slot_dataset --out-dir /tmp/track_slot_data --num-samples 32 --shard-size 16 --window-seconds 10 --time-downsample 20 --vehicles-min 2 --vehicles-max 4 --workers 2 --overwrite
uv run python -m autotrack.dl.convert_track_slot_to_peak_slot --in-dir /tmp/track_slot_data --out-dir /tmp/peak_slot_data --peak-candidates-per-channel 16 --overwrite
uv run python -m autotrack.dl.train_peak_slot --data-dir /tmp/peak_slot_data --out-dir /tmp/peak_slot_smoke --device cpu --epochs 1 --batch-size 2 --max-samples 32
uv run python -m autotrack.dl.predict_peak_slot_dataset --data-dir /tmp/peak_slot_data --model /tmp/peak_slot_smoke/checkpoint_best.pt --out-dir /tmp/peak_slot_pred --device cpu --plot-samples 8
```
