# PeakLineNet: Sparse Peaks to Continuous Trajectory Lines

PeakLineNet solves a different problem from TrackSlotNet and PeakSlotNet. It
does not identify individual vehicles. It learns a semantic channel-time line
mask:

```text
sparse peak point image -> probability that any vehicle trajectory passes here
```

The goal is to filter noisy peak candidates and reconnect missing local peak
observations before instance-level assignment.

## Physical View

The DAS peak image is a sampled channel-time observation. A vehicle moving along
the channel axis creates one response peak per channel when the vehicle reaches
that channel. If channel spacing is `dx` and local velocity is `v_j`, adjacent
arrival times satisfy

```math
\tau(c+1)=\tau(c)+\frac{dx}{v_j}.
```

For a constant-speed vehicle this is nearly a straight line in the
channel-time plane. Real trajectories can bend slightly because of sparse local
speed perturbations, smooth speed variation, or rare stop-go events. The label
therefore uses the sampled physical trajectory times and renders a continuous
soft line instead of forcing a strict straight-line model.

## Data Representation

Each generated shard contains:

```text
x            [N, 1, H, W] sparse peak-point image
line_mask    [N, 1, H, W] per-vehicle trajectory polylines
point_target [N, 1, H, W] clean true peak points
```

`line_mask` is generated first from the true trajectory. For each vehicle, the
visible trajectory points are mapped into rendered image coordinates and plotted
as a polyline. The label contains only these trajectory lines.

The input `x` is generated afterwards as peak points only. Realistic
imperfections are added as extra or missing points:

- true peak dropout,
- nearby shifted Gaussian distractors around some true vehicle points,
- isolated false peaks on random channels,
- optional weak background noise, disabled by default.

These corruptions are applied only to `x`, not to `line_mask`.

## Network

The model is a lightweight 2D U-Net:

1. encoder convolution blocks extract local point and line evidence,
2. stride operations reduce time/channel resolution for larger context,
3. decoder blocks upsample and fuse skip features,
4. a `1x1` head outputs `line_logits [B, 1, H, W]`.

The output probability is

```math
P(c,t)=\sigma(z(c,t)),
```

where `z` is the predicted line logit.

## Loss

Positive pixels are sparse, so plain BCE would be dominated by background. The
training objective is:

```text
loss = BCEWithLogits(pos_weight=20)
     + DiceLoss
     + 0.25 * FocalLoss(gamma=2)
```

The weighted BCE prevents all-background collapse. Dice rewards overlap of the
thin line region. Focal loss focuses the gradient on harder pixels such as
broken line segments and distractor peaks.

## Role in the Pipeline

PeakLineNet is a semantic pre-filter:

```text
raw/peak points -> PeakLineNet line probability -> cleaner peak candidates -> PeakSlotNet instance slots
```

It is not a replacement for PeakSlotNet. It can provide a trajectory-likelihood
map that suppresses isolated false peaks and highlights missing or weak parts of
a likely vehicle line.

## Commands

Generate a small dataset:

```sh
uv run python -m autotrack.dl.generate_peak_line_dataset \
  --out-dir /tmp/peak_line_data \
  --num-samples 32 \
  --shard-size 16 \
  --window-seconds 10 \
  --time-downsample 20 \
  --vehicles-min 2 \
  --vehicles-max 4 \
  --workers 2 \
  --overwrite
```

Train a CPU smoke test:

```sh
uv run python -m autotrack.dl.train_peak_line \
  --data-dir /tmp/peak_line_data \
  --out-dir /tmp/peak_line_smoke \
  --device cpu \
  --epochs 1 \
  --batch-size 2
```

Predict and plot:

```sh
uv run python -m autotrack.dl.predict_peak_line_dataset \
  --data-dir /tmp/peak_line_data \
  --model /tmp/peak_line_smoke/checkpoint_best.pt \
  --out-dir /tmp/peak_line_pred \
  --device cpu \
  --plot-samples 8
```
