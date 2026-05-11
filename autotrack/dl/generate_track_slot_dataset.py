"""Generate tensor shards for TrackSlotNet training.

Purpose:
    This script creates a disk dataset tailored for the track-slot model. It
    directly writes PyTorch `.pt` shards and does not create SAC files. Each
    sample is a noisy synthetic DAS heatmap window plus instance trajectory
    labels, so training can read tensors instead of parsing waveform files.

Example:
    uv run python -m autotrack.dl.generate_track_slot_dataset \
        --out-dir datasets/track_slot/train \
        --num-samples 20000 \
        --shard-size 256 \
        --window-seconds 240 \
        --time-downsample 10 \
        --vehicles-min 32 \
        --vehicles-max 48 \
        --motion-mix constant_sparse,smooth_random,stop_go \
        --motion-weights 0.84,0.15,0.01 \
        --workers 8

Outputs:
    <out-dir>/meta.json
        Dataset configuration and shard inventory.
    <out-dir>/shard_000000.pt, ...
        Each shard is a dictionary containing:
        - x:          [N, in_channels, C, T_down]
        - time:       [N, G, C]
        - visibility: [N, G, C]
        - direction:  [N, G]
        - speed:      [N, G]
        - gt_valid:   [N, G]

Notes:
    Use this generator before `train_track_slot.py`. For CPU-only experiments,
    keep `--num-samples` small. For CUDA training, generate a larger dataset
    once and reuse the shards across runs. Motion models are integrated as
    channel-segment speed profiles, so labels keep the same compact
    `[vehicle, channel]` time representation while becoming more realistic.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TrackSlotNet tensor training shards.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for meta.json and .pt shards.")
    parser.add_argument("--realism-preset", default="custom", help="Human-readable preset name recorded in meta.json.")
    parser.add_argument("--num-samples", type=int, default=20000, help="Total generated windows.")
    parser.add_argument("--shard-size", type=int, default=256, help="Samples per .pt shard.")
    parser.add_argument("--n-ch", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Original sampling rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=240.0, help="Window length in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time-axis stride stored in x.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--vehicles-min", type=int, default=32, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=48, help="Maximum vehicles per window and label slots per sample.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=85.0, help="Maximum vehicle speed.")
    parser.add_argument(
        "--motion-mix",
        default="constant_sparse,smooth_random,stop_go",
        help="Comma-separated motion models: constant_sparse, smooth_random, stop_go.",
    )
    parser.add_argument("--motion-weights", default="0.84,0.15,0.01", help="Comma-separated motion model weights.")
    parser.add_argument("--constant-perturb-prob", type=float, default=0.05, help="Sparse local speed perturb event probability per segment.")
    parser.add_argument("--constant-perturb-max-frac", type=float, default=0.01, help="Maximum absolute sparse perturbation fraction.")
    parser.add_argument("--constant-perturb-width-min", type=int, default=1, help="Minimum sparse perturbation width in channel segments.")
    parser.add_argument("--constant-perturb-width-max", type=int, default=2, help="Maximum sparse perturbation width in channel segments.")
    parser.add_argument("--smooth-speed-max-frac", type=float, default=0.05, help="Maximum absolute smooth speed variation fraction.")
    parser.add_argument("--smooth-speed-corr-channels", type=int, default=8, help="Smoothing width for random speed variation in channel segments.")
    parser.add_argument("--stop-duration-min-s", type=float, default=1.0, help="Minimum stop-go delay in seconds.")
    parser.add_argument("--stop-duration-max-s", type=float, default=8.0, help="Maximum stop-go delay in seconds.")
    parser.add_argument("--stop-channel-width-min", type=int, default=1, help="Minimum number of path positions with widened stop response.")
    parser.add_argument("--stop-channel-width-max", type=int, default=3, help="Maximum number of path positions with widened stop response.")
    parser.add_argument("--stop-response-sigma-scale", type=float, default=3.0, help="Gaussian sigma multiplier near a stop event.")
    parser.add_argument("--stop-response-amp-scale", type=float, default=1.2, help="Gaussian amplitude multiplier near a stop event.")
    parser.add_argument("--restart-speed-ratio-min", type=float, default=0.95, help="Minimum restart speed ratio after a stop.")
    parser.add_argument("--restart-speed-ratio-max", type=float, default=1.05, help="Maximum restart speed ratio after a stop.")
    parser.add_argument("--noise-std", type=float, default=0.35, help="White Gaussian noise std added to the downsampled heatmap before normalization.")
    parser.add_argument("--colored-noise-std", type=float, default=0.18, help="Low-frequency correlated noise std added to each channel.")
    parser.add_argument("--colored-noise-corr-s", type=float, default=0.8, help="Approximate time correlation length in seconds for colored noise.")
    parser.add_argument("--channel-bias-std", type=float, default=0.08, help="Per-channel constant baseline offset std.")
    parser.add_argument("--channel-gain-std", type=float, default=0.12, help="Per-channel multiplicative gain variation std applied to the final heatmap.")
    parser.add_argument("--baseline-drift-std", type=float, default=0.10, help="Slow baseline drift std added per channel.")
    parser.add_argument("--baseline-drift-corr-s", type=float, default=6.0, help="Approximate time correlation length in seconds for baseline drift.")
    parser.add_argument("--dead-channel-indices", default="", help="Comma-separated channel indices that are always completely zeroed.")
    parser.add_argument("--random-dead-channel-ratio", type=float, default=0.0, help="Fraction of samples with extra random completely zeroed channels.")
    parser.add_argument("--random-dead-channel-min", type=int, default=1, help="Minimum extra random dead channels when enabled.")
    parser.add_argument("--random-dead-channel-max", type=int, default=5, help="Maximum extra random dead channels when enabled.")
    parser.add_argument("--zero-background-ratio", type=float, default=0.0, help="Fraction of samples with random channel-time zero background blocks before vehicle pulses.")
    parser.add_argument("--zero-background-rate", type=float, default=12.0, help="Expected random zero background blocks when enabled.")
    parser.add_argument("--zero-background-channel-min", type=int, default=1, help="Minimum channels per zero background block.")
    parser.add_argument("--zero-background-channel-max", type=int, default=4, help="Maximum channels per zero background block.")
    parser.add_argument("--zero-background-duration-min-s", type=float, default=0.5, help="Minimum zero background block duration.")
    parser.add_argument("--zero-background-duration-max-s", type=float, default=4.0, help="Maximum zero background block duration.")
    parser.add_argument("--amp-min", type=float, default=6.0, help="Minimum Gaussian pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=6.0, help="Maximum Gaussian pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.25, help="Minimum Gaussian sigma in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.25, help="Maximum Gaussian sigma in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=0.8333333333, help="Fraction of forward-direction vehicles.")
    parser.add_argument("--interaction-ratio", type=float, default=0.3, help="Fraction of samples seeded with crossing/overtake/near-parallel pairs.")
    parser.add_argument("--interaction-types", default="crossing,overtake,near_parallel", help="Comma-separated interaction types.")
    parser.add_argument("--interaction-time-min-frac", type=float, default=0.05, help="Earliest interaction time as window fraction.")
    parser.add_argument("--interaction-time-max-frac", type=float, default=0.95, help="Latest interaction time as window fraction.")
    parser.add_argument("--isolated-noise-ratio", type=float, default=1.0, help="Fraction of samples with isolated Gaussian noise peaks.")
    parser.add_argument("--isolated-noise-rate", type=float, default=18.0, help="Expected isolated Gaussian noise peaks when enabled.")
    parser.add_argument("--isolated-noise-amp-min", type=float, default=1.0, help="Minimum isolated Gaussian noise amplitude.")
    parser.add_argument("--isolated-noise-amp-max", type=float, default=6.0, help="Maximum isolated Gaussian noise amplitude.")
    parser.add_argument(
        "--isolated-noise-sigma-min-s",
        "--isolated-noise-sigma-min",
        dest="isolated_noise_sigma_min_s",
        type=float,
        default=0.08,
        help="Minimum isolated Gaussian noise sigma in seconds.",
    )
    parser.add_argument(
        "--isolated-noise-sigma-max-s",
        "--isolated-noise-sigma-max",
        dest="isolated_noise_sigma_max_s",
        type=float,
        default=0.35,
        help="Maximum isolated Gaussian noise sigma in seconds.",
    )
    parser.add_argument("--min-visible-channels", type=int, default=2, help="Reject sampled vehicles with fewer visible channels.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization denominator.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Robust clipping ratio for input normalization.")
    parser.add_argument("--input-mode", default="raw", choices=["raw", "raw_abs"], help="Input channels stored in x.")
    parser.add_argument("--x-dtype", default="float16", choices=["float16", "float32"], help="Stored dtype for x tensor.")
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel shard workers. 0 or 1 runs sequentially; each worker writes independent shard files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing non-empty output directory.")
    return parser.parse_args()


def _rand_uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    return float(lo + torch.rand((), generator=gen).item() * (hi - lo))


def _prepare_input(data_ds: torch.Tensor, *, clip_ratio: float, input_mode: str) -> torch.Tensor:
    abs_vals = torch.abs(data_ds)
    q995 = torch.quantile(abs_vals.flatten(), 0.995)
    rms = torch.sqrt(torch.mean(abs_vals * abs_vals))
    scale = torch.clamp(torch.maximum(q995, 3.0 * rms), min=1e-6)
    clip = float(max(1e-6, clip_ratio))
    raw = torch.clamp(data_ds / scale, -clip, clip) / clip
    if str(input_mode) == "raw":
        return raw.unsqueeze(0).to(torch.float32)
    abs_feat = torch.clamp(abs_vals / scale, 0.0, clip) / clip
    return torch.stack([raw, abs_feat], dim=0).to(torch.float32)


def _split_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def _parse_float_csv(text: str) -> list[float]:
    return [float(item) for item in _split_csv(text)]


def _parse_int_csv(text: str) -> list[int]:
    return [int(item) for item in _split_csv(text)]


def _motion_models_and_weights(args: argparse.Namespace) -> tuple[list[str], list[float]]:
    allowed = {"constant_sparse", "smooth_random", "stop_go"}
    models = _split_csv(str(args.motion_mix))
    weights = _parse_float_csv(str(args.motion_weights))
    if not models:
        raise ValueError("--motion-mix must contain at least one model")
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"Unknown motion model(s): {', '.join(unknown)}")
    if len(weights) != len(models):
        raise ValueError("--motion-weights must have the same item count as --motion-mix")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("--motion-weights must be non-negative")
    total = float(sum(weights))
    if total <= 0.0:
        raise ValueError("--motion-weights must sum to a positive value")
    return models, [float(weight / total) for weight in weights]


def _choose_motion_model(args: argparse.Namespace, gen: torch.Generator) -> str:
    models, weights = _motion_models_and_weights(args)
    idx = int(torch.multinomial(torch.tensor(weights, dtype=torch.float32), 1, generator=gen).item())
    return models[idx]


def _apply_constant_sparse_perturbations(
    speed_kmh: torch.Tensor,
    args: argparse.Namespace,
    gen: torch.Generator,
) -> torch.Tensor:
    n_seg = int(speed_kmh.numel())
    if n_seg <= 0:
        return speed_kmh
    prob = float(max(0.0, args.constant_perturb_prob))
    max_frac = float(max(0.0, args.constant_perturb_max_frac))
    width_min = int(max(1, args.constant_perturb_width_min))
    width_max = int(max(width_min, args.constant_perturb_width_max))
    out = speed_kmh.clone()
    if prob <= 0.0 or max_frac <= 0.0:
        return out
    for start in range(n_seg):
        if float(torch.rand((), generator=gen).item()) >= prob:
            continue
        width = int(torch.randint(width_min, width_max + 1, (1,), generator=gen).item())
        end = min(n_seg, start + width)
        delta = _rand_uniform(gen, -max_frac, max_frac)
        out[start:end] *= float(1.0 + delta)
    return out


def _apply_smooth_random_speed(
    speed_kmh: torch.Tensor,
    args: argparse.Namespace,
    gen: torch.Generator,
) -> torch.Tensor:
    n_seg = int(speed_kmh.numel())
    if n_seg <= 0:
        return speed_kmh
    max_frac = float(max(0.0, args.smooth_speed_max_frac))
    if max_frac <= 0.0:
        return speed_kmh
    corr = int(max(1, args.smooth_speed_corr_channels))
    noise = torch.randn((n_seg,), generator=gen, dtype=torch.float32)
    if corr > 1 and n_seg > 1:
        kernel = torch.ones((1, 1, min(corr, n_seg)), dtype=torch.float32) / float(min(corr, n_seg))
        pad_left = int(kernel.shape[-1] // 2)
        pad_right = int(kernel.shape[-1] - 1 - pad_left)
        padded = torch.nn.functional.pad(noise.view(1, 1, -1), (pad_left, pad_right), mode="replicate")
        noise = torch.nn.functional.conv1d(padded, kernel).view(-1)
    max_abs = float(torch.max(torch.abs(noise)).item())
    if max_abs <= 1e-9:
        return speed_kmh
    amplitude = _rand_uniform(gen, 0.0, max_frac)
    factor = 1.0 + noise / max_abs * float(amplitude)
    return speed_kmh * factor


def _integrate_segment_times(
    segment_speed_kmh: torch.Tensor,
    *,
    n_ch: int,
    dx_m: float,
    anchor_pos: int,
    anchor_time: float,
) -> torch.Tensor:
    t_path = torch.empty((n_ch,), dtype=torch.float32)
    if n_ch <= 1:
        t_path.fill_(float(anchor_time))
        return t_path
    speed_mps = torch.clamp(segment_speed_kmh.to(torch.float32) / 3.6, min=1e-6)
    dt = float(dx_m) / speed_mps
    anchor_pos = int(max(0, min(n_ch - 1, anchor_pos)))
    t_path[anchor_pos] = float(anchor_time)
    for pos in range(anchor_pos, n_ch - 1):
        t_path[pos + 1] = t_path[pos] + dt[pos]
    for pos in range(anchor_pos - 1, -1, -1):
        t_path[pos] = t_path[pos + 1] - dt[pos]
    return t_path


def _effective_speed_kmh(t_center: torch.Tensor, dx_m: float, fallback_speed_kmh: float) -> float:
    if int(t_center.numel()) <= 1:
        return float(fallback_speed_kmh)
    travel_s = float(torch.max(t_center).item() - torch.min(t_center).item())
    if travel_s <= 1e-9:
        return float(fallback_speed_kmh)
    distance_m = float(int(t_center.numel()) - 1) * float(dx_m)
    return float(3.6 * distance_m / travel_s)


def _sample_track_times(
    args: argparse.Namespace,
    gen: torch.Generator,
    n_ch: int,
    is_primary: bool,
    speed_kmh: float,
    anchor_ch: int,
    anchor_time: float,
) -> tuple[torch.Tensor, torch.Tensor, float, str, torch.Tensor]:
    n_seg = max(0, int(n_ch) - 1)
    motion_model = _choose_motion_model(args, gen)
    segment_speed = torch.full((n_seg,), float(speed_kmh), dtype=torch.float32)
    stop_path_mask = torch.zeros((int(n_ch),), dtype=torch.bool)

    if motion_model == "constant_sparse":
        segment_speed = _apply_constant_sparse_perturbations(segment_speed, args, gen)
    elif motion_model == "smooth_random":
        segment_speed = _apply_smooth_random_speed(segment_speed, args, gen)
    elif motion_model == "stop_go":
        restart_ratio = _rand_uniform(gen, float(args.restart_speed_ratio_min), float(args.restart_speed_ratio_max))
        if n_seg > 0:
            stop_pos = int(torch.randint(0, int(n_ch) - 1, (1,), generator=gen).item())
            segment_speed[stop_pos:] *= float(restart_ratio)
        else:
            stop_pos = 0
    else:
        raise ValueError(f"Unknown motion model: {motion_model}")

    segment_speed = torch.clamp(
        segment_speed,
        min=float(args.speed_min_kmh),
        max=float(args.speed_max_kmh),
    )
    anchor_pos = int(anchor_ch) if bool(is_primary) else int(n_ch) - 1 - int(anchor_ch)
    t_path = _integrate_segment_times(
        segment_speed,
        n_ch=int(n_ch),
        dx_m=float(args.dx_m),
        anchor_pos=anchor_pos,
        anchor_time=float(anchor_time),
    )

    if motion_model == "stop_go" and int(n_ch) > 1:
        stop_duration = _rand_uniform(gen, float(args.stop_duration_min_s), float(args.stop_duration_max_s))
        stop_pos = int(max(0, min(int(n_ch) - 1, stop_pos)))
        t_path[stop_pos + 1 :] += float(stop_duration)
        width_min = int(max(1, args.stop_channel_width_min))
        width_max = int(max(width_min, args.stop_channel_width_max))
        width = int(torch.randint(width_min, width_max + 1, (1,), generator=gen).item())
        half_left = width // 2
        start = max(0, stop_pos - half_left)
        end = min(int(n_ch), start + width)
        start = max(0, end - width)
        stop_path_mask[start:end] = True

    if bool(is_primary):
        t_center = t_path
        stop_mask = stop_path_mask
    else:
        order = torch.arange(int(n_ch) - 1, -1, -1, dtype=torch.long)
        t_center = torch.empty((int(n_ch),), dtype=torch.float32)
        stop_mask = torch.zeros((int(n_ch),), dtype=torch.bool)
        t_center[order] = t_path
        stop_mask[order] = stop_path_mask

    effective_speed = _effective_speed_kmh(t_center, float(args.dx_m), float(speed_kmh))
    return t_center, segment_speed, effective_speed, motion_model, stop_mask


def _add_track_to_sample(
    args: argparse.Namespace,
    *,
    data_ds: torch.Tensor,
    t_axis_s: torch.Tensor,
    time_label: torch.Tensor,
    visibility: torch.Tensor,
    direction: torch.Tensor,
    speed: torch.Tensor,
    gt_valid: torch.Tensor,
    track_id: int,
    t_center: torch.Tensor,
    direction_label: int,
    speed_kmh: float,
    amp: float,
    sigma_s: float,
) -> bool:
    n_ch = int(data_ds.shape[0])
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    visible = (t_center >= 0.0) & (t_center < float(args.window_seconds))
    if int(visible.sum().item()) < int(args.min_visible_channels):
        return False
    for ch in torch.where(visible)[0].tolist():
        pulse = float(amp) * torch.exp(-0.5 * ((t_axis_s - float(t_center[ch].item())) / max(1e-6, float(sigma_s))) ** 2)
        data_ds[int(ch)] += pulse
    center_idx = torch.round(t_center * float(args.fs)).to(torch.long).clamp(0, window_samples - 1)
    time_label[track_id, visible] = (
        center_idx[visible].to(torch.float32) / float(max(1, window_samples - 1))
    ).clamp(0.0, 1.0)
    visibility[track_id] = visible.to(torch.float32)
    direction[track_id] = int(direction_label)
    speed[track_id] = float(speed_kmh / max(1e-6, float(args.speed_norm_kmh)))
    gt_valid[track_id] = True
    return True


def _interaction_time_pair(
    args: argparse.Namespace,
    gen: torch.Generator,
    n_ch: int,
    interaction_type: str,
) -> tuple[torch.Tensor, int, float, torch.Tensor, int, float]:
    cross_ch = int(torch.randint(0, int(n_ch), (1,), generator=gen).item())
    lo = max(0.0, min(1.0, float(args.interaction_time_min_frac)))
    hi = max(lo, min(1.0, float(args.interaction_time_max_frac)))
    cross_time = _rand_uniform(gen, lo * float(args.window_seconds), hi * float(args.window_seconds))
    speed_min = float(args.speed_min_kmh)
    speed_max = float(args.speed_max_kmh)
    speed_span = max(1e-6, speed_max - speed_min)
    slow = _rand_uniform(gen, speed_min, speed_min + 0.35 * speed_span)
    fast = _rand_uniform(gen, speed_min + 0.65 * speed_span, speed_max)
    ch_axis = torch.arange(int(n_ch), dtype=torch.float32)
    dx = float(args.dx_m)
    if interaction_type == "crossing":
        t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
        t_b = float(cross_time) - (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
        return t_a, 0, slow, t_b, 1, fast
    if interaction_type == "near_parallel":
        offset = _rand_uniform(gen, 0.4, 2.0) * (-1.0 if torch.rand((), generator=gen).item() < 0.5 else 1.0)
        t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
        t_b = float(cross_time + offset) + (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
        return t_a, 0, slow, t_b, 0, fast
    t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
    t_b = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
    return t_a, 0, slow, t_b, 0, fast


def _add_isolated_noise(args: argparse.Namespace, gen: torch.Generator, data_ds: torch.Tensor, t_axis_s: torch.Tensor) -> None:
    rate = max(0.0, float(args.isolated_noise_rate))
    count = int(rate)
    if torch.rand((), generator=gen).item() < rate - count:
        count += 1
    n_ch = int(data_ds.shape[0])
    for _ in range(count):
        ch = int(torch.randint(0, n_ch, (1,), generator=gen).item())
        center = _rand_uniform(gen, 0.0, float(args.window_seconds))
        sigma = _rand_uniform(gen, float(args.isolated_noise_sigma_min_s), float(args.isolated_noise_sigma_max_s))
        amp = _rand_uniform(gen, float(args.isolated_noise_amp_min), float(args.isolated_noise_amp_max))
        data_ds[ch] += float(amp) * torch.exp(-0.5 * ((t_axis_s - float(center)) / max(1e-6, float(sigma))) ** 2)


def _smooth_time_noise(noise: torch.Tensor, *, corr_samples: int) -> torch.Tensor:
    width = int(max(1, corr_samples))
    if width <= 1 or int(noise.shape[-1]) <= 1:
        return noise
    width = min(width, int(noise.shape[-1]))
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = torch.nn.functional.pad(noise, (pad_left, pad_right), mode="replicate")
    cumsum = torch.nn.functional.pad(torch.cumsum(padded, dim=-1), (1, 0))
    return (cumsum[:, width:] - cumsum[:, :-width]) / float(width)


def _add_background_noise(args: argparse.Namespace, gen: torch.Generator, data_ds: torch.Tensor) -> None:
    n_ch, t_down = int(data_ds.shape[0]), int(data_ds.shape[1])
    if float(args.noise_std) > 0.0:
        data_ds += torch.normal(
            mean=0.0,
            std=float(args.noise_std),
            size=(n_ch, t_down),
            generator=gen,
            dtype=torch.float32,
        )
    if float(args.colored_noise_std) > 0.0:
        corr = int(round(float(args.colored_noise_corr_s) * float(args.fs) / float(max(1, args.time_downsample))))
        colored = torch.randn((n_ch, t_down), generator=gen, dtype=torch.float32)
        colored = _smooth_time_noise(colored, corr_samples=corr)
        colored = colored / torch.clamp(torch.std(colored, dim=1, keepdim=True, unbiased=False), min=1e-6)
        data_ds += float(args.colored_noise_std) * colored
    if float(args.channel_bias_std) > 0.0:
        bias = torch.normal(
            mean=0.0,
            std=float(args.channel_bias_std),
            size=(n_ch, 1),
            generator=gen,
            dtype=torch.float32,
        )
        data_ds += bias
    if float(args.baseline_drift_std) > 0.0:
        corr = int(round(float(args.baseline_drift_corr_s) * float(args.fs) / float(max(1, args.time_downsample))))
        drift = torch.randn((n_ch, t_down), generator=gen, dtype=torch.float32)
        drift = _smooth_time_noise(drift, corr_samples=corr)
        drift = drift / torch.clamp(torch.std(drift, dim=1, keepdim=True, unbiased=False), min=1e-6)
        data_ds += float(args.baseline_drift_std) * drift


def _apply_channel_gain(args: argparse.Namespace, gen: torch.Generator, data_ds: torch.Tensor) -> None:
    if float(args.channel_gain_std) <= 0.0:
        return
    gain = 1.0 + torch.normal(
        mean=0.0,
        std=float(args.channel_gain_std),
        size=(int(data_ds.shape[0]), 1),
        generator=gen,
        dtype=torch.float32,
    )
    data_ds *= torch.clamp(gain, min=0.2, max=3.0)


def _sample_dead_channels(args: argparse.Namespace, gen: torch.Generator, n_ch: int) -> list[int]:
    dead = {idx for idx in _parse_int_csv(str(args.dead_channel_indices)) if 0 <= idx < int(n_ch)}
    if (
        float(args.random_dead_channel_ratio) > 0.0
        and torch.rand((), generator=gen).item() < float(args.random_dead_channel_ratio)
    ):
        count_min = int(max(0, args.random_dead_channel_min))
        count_max = int(max(count_min, args.random_dead_channel_max))
        if count_max > 0:
            count = int(torch.randint(count_min, count_max + 1, (1,), generator=gen).item())
            count = min(count, int(n_ch))
            if count > 0:
                order = torch.randperm(int(n_ch), generator=gen).tolist()
                dead.update(int(idx) for idx in order[:count])
    return sorted(dead)


def _apply_dead_channels(
    data_ds: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    dead_channels: list[int],
    *,
    min_visible_channels: int,
) -> None:
    if not dead_channels:
        return
    data_ds[dead_channels, :] = 0.0
    if visibility.numel() <= 0:
        return
    visibility[:, dead_channels] = 0.0
    visible_counts = visibility.sum(dim=1)
    gt_valid &= visible_counts >= int(min_visible_channels)


def _add_zero_background_blocks(args: argparse.Namespace, gen: torch.Generator, data_ds: torch.Tensor) -> None:
    if float(args.zero_background_ratio) <= 0.0:
        return
    if torch.rand((), generator=gen).item() >= float(args.zero_background_ratio):
        return
    rate = max(0.0, float(args.zero_background_rate))
    count = int(rate)
    if torch.rand((), generator=gen).item() < rate - count:
        count += 1
    if count <= 0:
        return
    n_ch, t_down = int(data_ds.shape[0]), int(data_ds.shape[1])
    ch_min = int(max(1, args.zero_background_channel_min))
    ch_max = int(max(ch_min, args.zero_background_channel_max))
    dur_min = float(max(0.0, args.zero_background_duration_min_s))
    dur_max = float(max(dur_min, args.zero_background_duration_max_s))
    for _ in range(count):
        width_ch = int(torch.randint(ch_min, ch_max + 1, (1,), generator=gen).item())
        width_ch = max(1, min(width_ch, n_ch))
        start_ch = int(torch.randint(0, max(1, n_ch - width_ch + 1), (1,), generator=gen).item())
        duration_s = _rand_uniform(gen, dur_min, dur_max)
        width_t = int(max(1, round(duration_s * float(args.fs) / float(max(1, args.time_downsample)))))
        width_t = max(1, min(width_t, t_down))
        start_t = int(torch.randint(0, max(1, t_down - width_t + 1), (1,), generator=gen).item())
        data_ds[start_ch : start_ch + width_ch, start_t : start_t + width_t] = 0.0


def _generate_one(args: argparse.Namespace, index: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + int(index) * 1_000_003)
    n_ch = int(args.n_ch)
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    time_downsample = int(max(1, args.time_downsample))
    t_down = int(max(1, len(range(0, window_samples, time_downsample))))
    t_axis_s = torch.arange(t_down, dtype=torch.float32) * (float(time_downsample) / float(args.fs))
    data_ds = torch.zeros((n_ch, t_down), dtype=torch.float32)
    _add_background_noise(args, gen, data_ds)
    _add_zero_background_blocks(args, gen, data_ds)

    max_gt = int(max(args.vehicles_min, args.vehicles_max))
    time_label = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    visibility = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    direction = torch.zeros((max_gt,), dtype=torch.long)
    speed = torch.zeros((max_gt,), dtype=torch.float32)
    gt_valid = torch.zeros((max_gt,), dtype=torch.bool)

    n_vehicles = int(torch.randint(int(args.vehicles_min), int(args.vehicles_max) + 1, (1,), generator=gen).item())
    track_id = 0
    if n_vehicles >= 2 and float(args.interaction_ratio) > 0.0 and torch.rand((), generator=gen).item() < float(args.interaction_ratio):
        interaction_types = _split_csv(str(args.interaction_types)) or ["crossing"]
        interaction_type = interaction_types[int(torch.randint(0, len(interaction_types), (1,), generator=gen).item())]
        t_a, dir_a, speed_a, t_b, dir_b, speed_b = _interaction_time_pair(args, gen, n_ch, interaction_type)
        for t_center, dir_label, speed_kmh in [(t_a, dir_a, speed_a), (t_b, dir_b, speed_b)]:
            if track_id >= n_vehicles:
                break
            added = _add_track_to_sample(
                args,
                data_ds=data_ds,
                t_axis_s=t_axis_s,
                time_label=time_label,
                visibility=visibility,
                direction=direction,
                speed=speed,
                gt_valid=gt_valid,
                track_id=track_id,
                t_center=t_center,
                direction_label=int(dir_label),
                speed_kmh=float(speed_kmh),
                amp=_rand_uniform(gen, float(args.amp_min), float(args.amp_max)),
                sigma_s=_rand_uniform(gen, float(args.sigma_min_s), float(args.sigma_max_s)),
            )
            if added:
                track_id += 1
    attempts = 0
    max_attempts = max(64, n_vehicles * 64)
    while track_id < n_vehicles and attempts < max_attempts:
        attempts += 1
        is_primary = bool(torch.rand((), generator=gen).item() < float(args.primary_ratio))
        direction_label = 0 if is_primary else 1
        speed_kmh = _rand_uniform(gen, float(args.speed_min_kmh), float(args.speed_max_kmh))
        sigma_s = _rand_uniform(gen, float(args.sigma_min_s), float(args.sigma_max_s))
        amp = _rand_uniform(gen, float(args.amp_min), float(args.amp_max))
        anchor_ch = int(torch.randint(0, n_ch, (1,), generator=gen).item())
        anchor_time = float(torch.rand((), generator=gen).item() * float(args.window_seconds))
        t_center, _, effective_speed_kmh, motion_model, stop_mask = _sample_track_times(
            args,
            gen,
            n_ch,
            is_primary,
            speed_kmh,
            anchor_ch,
            anchor_time,
        )
        visible = (t_center >= 0.0) & (t_center < float(args.window_seconds))
        if int(visible.sum().item()) < int(args.min_visible_channels):
            continue

        for ch in torch.where(visible)[0].tolist():
            sigma_scale = float(args.stop_response_sigma_scale) if motion_model == "stop_go" and bool(stop_mask[ch]) else 1.0
            amp_scale = float(args.stop_response_amp_scale) if motion_model == "stop_go" and bool(stop_mask[ch]) else 1.0
            pulse_sigma = max(1e-6, sigma_s * sigma_scale)
            pulse = amp * amp_scale * torch.exp(-0.5 * ((t_axis_s - float(t_center[ch].item())) / pulse_sigma) ** 2)
            data_ds[int(ch)] += pulse
        center_idx = torch.round(t_center * float(args.fs)).to(torch.long).clamp(0, window_samples - 1)
        time_label[track_id, visible] = (
            center_idx[visible].to(torch.float32) / float(max(1, window_samples - 1))
        ).clamp(0.0, 1.0)
        visibility[track_id] = visible.to(torch.float32)
        direction[track_id] = int(direction_label)
        speed[track_id] = float(effective_speed_kmh / max(1e-6, float(args.speed_norm_kmh)))
        gt_valid[track_id] = True
        track_id += 1

    if float(args.isolated_noise_ratio) > 0.0 and torch.rand((), generator=gen).item() < float(args.isolated_noise_ratio):
        _add_isolated_noise(args, gen, data_ds, t_axis_s)
    _apply_channel_gain(args, gen, data_ds)
    _apply_dead_channels(
        data_ds,
        visibility,
        gt_valid,
        _sample_dead_channels(args, gen, n_ch),
        min_visible_channels=int(args.min_visible_channels),
    )
    x = _prepare_input(data_ds, clip_ratio=float(args.clip_ratio), input_mode=str(args.input_mode))
    if str(args.x_dtype) == "float16":
        x = x.to(torch.float16)
    return {
        "x": x.contiguous(),
        "time": time_label,
        "visibility": visibility,
        "direction": direction,
        "speed": speed,
        "gt_valid": gt_valid,
    }


def _write_shard(out_path: Path, items: list[dict[str, torch.Tensor]]) -> None:
    payload = {
        "x": torch.stack([item["x"] for item in items], dim=0).contiguous(),
        "time": torch.stack([item["time"] for item in items], dim=0).contiguous(),
        "visibility": torch.stack([item["visibility"] for item in items], dim=0).contiguous(),
        "direction": torch.stack([item["direction"] for item in items], dim=0).contiguous(),
        "speed": torch.stack([item["speed"] for item in items], dim=0).contiguous(),
        "gt_valid": torch.stack([item["gt_valid"] for item in items], dim=0).contiguous(),
    }
    torch.save(payload, str(out_path))


def _args_payload(args: argparse.Namespace) -> dict[str, object]:
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}


def _generate_shard_worker(payload: tuple[dict[str, object], int, int, int]) -> tuple[int, str, int, float]:
    raw_args, shard_idx, start, end = payload
    torch.set_num_threads(1)
    args = argparse.Namespace(**raw_args)
    shard_name = f"shard_{int(shard_idx):06d}.pt"
    out_path = Path(str(raw_args["out_dir"])).expanduser() / shard_name
    t0 = time.perf_counter()
    items = [_generate_one(args, index) for index in range(int(start), int(end))]
    _write_shard(out_path, items)
    return int(shard_idx), shard_name, int(end) - int(start), float(time.perf_counter() - t0)


def main() -> int:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0")
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size must be > 0")
    if int(args.n_ch) <= 0:
        raise ValueError("--n-ch must be > 0")
    if int(args.vehicles_min) < 0 or int(args.vehicles_max) < int(args.vehicles_min):
        raise ValueError("--vehicles-max must be >= --vehicles-min >= 0")
    if float(args.speed_min_kmh) <= 0.0 or float(args.speed_max_kmh) < float(args.speed_min_kmh):
        raise ValueError("speed range must be positive and ordered")
    motion_models, motion_weights = _motion_models_and_weights(args)
    if not (0.0 <= float(args.constant_perturb_prob) <= 1.0):
        raise ValueError("--constant-perturb-prob must be in [0, 1]")
    if float(args.constant_perturb_max_frac) < 0.0 or float(args.smooth_speed_max_frac) < 0.0:
        raise ValueError("speed variation fractions must be >= 0")
    if int(args.constant_perturb_width_min) <= 0 or int(args.constant_perturb_width_max) < int(args.constant_perturb_width_min):
        raise ValueError("constant perturb widths must be positive and ordered")
    if int(args.smooth_speed_corr_channels) <= 0:
        raise ValueError("--smooth-speed-corr-channels must be > 0")
    if float(args.stop_duration_min_s) < 0.0 or float(args.stop_duration_max_s) < float(args.stop_duration_min_s):
        raise ValueError("stop duration range must be non-negative and ordered")
    if int(args.stop_channel_width_min) <= 0 or int(args.stop_channel_width_max) < int(args.stop_channel_width_min):
        raise ValueError("stop channel widths must be positive and ordered")
    if float(args.stop_response_sigma_scale) <= 0.0 or float(args.stop_response_amp_scale) <= 0.0:
        raise ValueError("stop response scales must be > 0")
    if float(args.restart_speed_ratio_min) <= 0.0 or float(args.restart_speed_ratio_max) < float(args.restart_speed_ratio_min):
        raise ValueError("restart speed ratio range must be positive and ordered")
    if not (0.0 <= float(args.primary_ratio) <= 1.0):
        raise ValueError("--primary-ratio must be in [0, 1]")
    if any(
        float(value) < 0.0
        for value in [
            args.noise_std,
            args.colored_noise_std,
            args.channel_bias_std,
            args.channel_gain_std,
            args.baseline_drift_std,
        ]
    ):
        raise ValueError("noise std parameters must be >= 0")
    if float(args.colored_noise_corr_s) <= 0.0 or float(args.baseline_drift_corr_s) <= 0.0:
        raise ValueError("noise correlation lengths must be > 0")
    dead_channels = _parse_int_csv(str(args.dead_channel_indices))
    invalid_dead = [idx for idx in dead_channels if idx < 0 or idx >= int(args.n_ch)]
    if invalid_dead:
        raise ValueError(f"dead channel indices outside [0, {int(args.n_ch) - 1}]: {invalid_dead}")
    if not (0.0 <= float(args.random_dead_channel_ratio) <= 1.0):
        raise ValueError("--random-dead-channel-ratio must be in [0, 1]")
    if int(args.random_dead_channel_min) < 0 or int(args.random_dead_channel_max) < int(args.random_dead_channel_min):
        raise ValueError("random dead channel counts must be non-negative and ordered")
    if not (0.0 <= float(args.zero_background_ratio) <= 1.0):
        raise ValueError("--zero-background-ratio must be in [0, 1]")
    if float(args.zero_background_rate) < 0.0:
        raise ValueError("--zero-background-rate must be >= 0")
    if int(args.zero_background_channel_min) <= 0 or int(args.zero_background_channel_max) < int(args.zero_background_channel_min):
        raise ValueError("zero background channel widths must be positive and ordered")
    if float(args.zero_background_duration_min_s) <= 0.0 or float(args.zero_background_duration_max_s) < float(args.zero_background_duration_min_s):
        raise ValueError("zero background durations must be positive and ordered")
    if not (0.0 <= float(args.interaction_ratio) <= 1.0):
        raise ValueError("--interaction-ratio must be in [0, 1]")
    allowed_interactions = {"crossing", "overtake", "near_parallel"}
    unknown_interactions = sorted(set(_split_csv(str(args.interaction_types))) - allowed_interactions)
    if unknown_interactions:
        raise ValueError(f"Unknown interaction type(s): {', '.join(unknown_interactions)}")
    if float(args.interaction_time_min_frac) < 0.0 or float(args.interaction_time_max_frac) < float(args.interaction_time_min_frac):
        raise ValueError("interaction time fractions must be ordered and non-negative")
    if float(args.isolated_noise_rate) < 0.0:
        raise ValueError("--isolated-noise-rate must be >= 0")
    if not (0.0 <= float(args.isolated_noise_ratio) <= 1.0):
        raise ValueError("--isolated-noise-ratio must be in [0, 1]")
    if float(args.isolated_noise_amp_min) < 0.0 or float(args.isolated_noise_amp_max) < float(args.isolated_noise_amp_min):
        raise ValueError("isolated noise amplitude range must be non-negative and ordered")
    if float(args.isolated_noise_sigma_min_s) <= 0.0 or float(args.isolated_noise_sigma_max_s) < float(args.isolated_noise_sigma_min_s):
        raise ValueError("isolated noise sigma range must be positive and ordered")

    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace metadata/shards.")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("shard_*.pt"):
        old.unlink()

    t0 = time.perf_counter()
    total = int(args.num_samples)
    shard_size = int(args.shard_size)
    shard_count = int(math.ceil(total / shard_size))
    jobs = [(idx, idx * shard_size, min(total, (idx + 1) * shard_size)) for idx in range(shard_count)]
    shard_files: list[Optional[str]] = [None] * shard_count
    done_samples = 0
    workers = int(max(0, args.workers))
    raw_args = _args_payload(args)
    if workers <= 1 or shard_count <= 1:
        for shard_idx, start, end in jobs:
            done_idx, shard_name, sample_count, shard_seconds = _generate_shard_worker((raw_args, shard_idx, start, end))
            shard_files[done_idx] = shard_name
            done_samples += sample_count
            elapsed = time.perf_counter() - t0
            print(
                f"wrote {shard_name}: samples={sample_count}, total={done_samples}/{total}, "
                f"shard_elapsed={shard_seconds:.1f}s, elapsed={elapsed:.1f}s",
                flush=True,
            )
    else:
        max_workers = int(min(workers, shard_count))
        print(f"parallel generation: workers={max_workers}, shards={shard_count}", flush=True)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_generate_shard_worker, (raw_args, shard_idx, start, end))
                for shard_idx, start, end in jobs
            ]
            for future in as_completed(futures):
                done_idx, shard_name, sample_count, shard_seconds = future.result()
                shard_files[done_idx] = shard_name
                done_samples += sample_count
                elapsed = time.perf_counter() - t0
                print(
                    f"wrote {shard_name}: samples={sample_count}, total={done_samples}/{total}, "
                    f"shard_elapsed={shard_seconds:.1f}s, elapsed={elapsed:.1f}s",
                    flush=True,
                )
    shard_names = [name for name in shard_files if name is not None]

    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    meta = {
        "format": "track_slot_shards_v1",
        "realism_preset": str(args.realism_preset),
        "created_at_unix": time.time(),
        "num_samples": total,
        "shard_size": shard_size,
        "shards": shard_names,
        "n_channels": int(args.n_ch),
        "in_channels": 1 if str(args.input_mode) == "raw" else 2,
        "max_gt": int(args.vehicles_max),
        "fs": float(args.fs),
        "window_seconds": float(args.window_seconds),
        "window_samples": int(window_samples),
        "time_downsample": int(args.time_downsample),
        "downsampled_time": int(max(1, len(range(0, window_samples, int(max(1, args.time_downsample)))))),
        "dx_m": float(args.dx_m),
        "speed_norm_kmh": float(args.speed_norm_kmh),
        "motion_models": motion_models,
        "motion_weights": motion_weights,
        "clip_ratio": float(args.clip_ratio),
        "input_mode": str(args.input_mode),
        "generator_args": _args_payload(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={total}, shards={len(shard_names)}, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
