"""Generate sparse-point to continuous-line tensor shards for PeakLineNet.

Purpose:
    This script creates a semantic trajectory-line segmentation dataset. Each
    sample contains a sparse peak-point image as input and a thin trajectory
    line image as the label. The model trained on this dataset does not
    distinguish vehicle identities; it only learns where any vehicle trajectory
    line exists.

Example:
    uv run python -m autotrack.dl.generate_peak_line_dataset \
        --out-dir datasets/peak_line/train \
        --num-samples 20000 \
        --shard-size 256 \
        --window-seconds 240 \
        --time-downsample 10 \
        --vehicles-min 32 \
        --vehicles-max 48 \
        --workers 8

Arguments:
    --line-sigma-ch and --line-sigma-t control the continuous label width.
    --point-drop-prob randomly removes true peak points from the input only.
    --nearby-distractor-* adds shifted Gaussian peaks around true vehicle
    points in the input only.
    --false-peak-prob-per-channel adds isolated false peak points in the input
    only.

Outputs:
    <out-dir>/meta.json
        Dataset configuration and shard inventory.
    <out-dir>/shard_000000.pt, ...
        Each shard is a dictionary containing:
        - x:            [N, 1, H, W] sparse peak-point image
        - line_mask:    [N, 1, H, W] per-vehicle trajectory polylines
        - point_target: [N, 1, H, W] clean true peak points

Notes:
    The trajectory motion is sampled with the same motion model helper used by
    TrackSlotNet: mostly constant_sparse, some smooth_random, and rare stop_go.
    Dropout and distractor points are applied after the clean label is
    generated, so the label remains the true physical trajectory-line image.
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

from autotrack.dl.generate_track_slot_dataset import (
    _motion_models_and_weights,
    _rand_uniform,
    _sample_track_times,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PeakLineNet point-image to line-mask tensor shards.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for meta.json and .pt shards.")
    parser.add_argument("--num-samples", type=int, default=20000, help="Total generated windows.")
    parser.add_argument("--shard-size", type=int, default=256, help="Samples per .pt shard.")
    parser.add_argument("--n-ch", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Original sampling rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=240.0, help="Window length in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Compatibility metadata for older tensor datasets.")
    parser.add_argument("--image-height", type=int, default=256, help="Rendered image height in pixels.")
    parser.add_argument("--image-width", type=int, default=1024, help="Rendered image width in pixels.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--vehicles-min", type=int, default=32, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=48, help="Maximum vehicles per window.")
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
    parser.add_argument("--stop-channel-width-min", type=int, default=1, help="Minimum path positions with widened stop response.")
    parser.add_argument("--stop-channel-width-max", type=int, default=3, help="Maximum path positions with widened stop response.")
    parser.add_argument("--stop-response-sigma-scale", type=float, default=3.0, help="Line/point sigma multiplier near a stop event.")
    parser.add_argument("--stop-response-amp-scale", type=float, default=1.2, help="Point amplitude multiplier near a stop event.")
    parser.add_argument("--restart-speed-ratio-min", type=float, default=0.95, help="Minimum restart speed ratio after a stop.")
    parser.add_argument("--restart-speed-ratio-max", type=float, default=1.05, help="Maximum restart speed ratio after a stop.")
    parser.add_argument("--primary-ratio", type=float, default=0.8333333333, help="Fraction of forward-direction vehicles.")
    parser.add_argument("--min-visible-channels", type=int, default=2, help="Reject vehicles with fewer visible channels.")
    parser.add_argument("--line-width", type=int, default=1, help="Label polyline width in rendered pixels.")
    parser.add_argument("--point-width", type=int, default=2, help="Binary input/target point width in rendered pixels.")
    parser.add_argument("--point-drop-prob", type=float, default=0.10, help="Probability of dropping each true point from x only.")
    parser.add_argument("--nearby-distractor-prob", type=float, default=0.20, help="Probability of adding nearby distractors around a true point.")
    parser.add_argument("--nearby-distractor-count-max", type=int, default=2, help="Maximum distractor peaks per selected true point.")
    parser.add_argument("--nearby-distractor-offset-min-s", type=float, default=0.2, help="Minimum distractor time offset in seconds.")
    parser.add_argument("--nearby-distractor-offset-max-s", type=float, default=1.5, help="Maximum distractor time offset in seconds.")
    parser.add_argument("--nearby-distractor-amp-min", type=float, default=0.35, help="Minimum nearby distractor amplitude.")
    parser.add_argument("--nearby-distractor-amp-max", type=float, default=0.90, help="Maximum nearby distractor amplitude.")
    parser.add_argument("--false-peak-prob-per-channel", type=float, default=0.08, help="Probability of one isolated false peak per channel.")
    parser.add_argument("--false-peak-amp-min", type=float, default=0.25, help="Minimum isolated false peak amplitude.")
    parser.add_argument("--false-peak-amp-max", type=float, default=0.85, help="Maximum isolated false peak amplitude.")
    parser.add_argument("--input-noise-std", type=float, default=0.0, help="Optional weak background noise std added to x. Default keeps x as points only.")
    parser.add_argument("--x-dtype", default="float16", choices=["float16", "float32"], help="Stored dtype for x tensor.")
    parser.add_argument("--workers", type=int, default=0, help="Parallel shard workers. 0 or 1 runs sequentially.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing shard files in the output directory.")
    return parser.parse_args()


def _add_gaussian_spot(
    image: torch.Tensor,
    *,
    ch_center: float,
    t_center: float,
    sigma_ch: float,
    sigma_t: float,
    amp: float,
    mode: str = "max",
) -> None:
    n_ch, t_down = int(image.shape[0]), int(image.shape[1])
    if not math.isfinite(ch_center) or not math.isfinite(t_center):
        return
    if ch_center < -3.0 or ch_center > n_ch + 2.0 or t_center < -6.0 or t_center > t_down + 5.0:
        return
    sigma_ch = float(max(1e-6, sigma_ch))
    sigma_t = float(max(1e-6, sigma_t))
    ch_radius = int(max(0, math.ceil(3.0 * sigma_ch)))
    t_radius = int(max(0, math.ceil(3.0 * sigma_t)))
    ch0 = max(0, int(math.floor(ch_center)) - ch_radius)
    ch1 = min(n_ch, int(math.floor(ch_center)) + ch_radius + 2)
    t0 = max(0, int(math.floor(t_center)) - t_radius)
    t1 = min(t_down, int(math.floor(t_center)) + t_radius + 2)
    if ch1 <= ch0 or t1 <= t0:
        return
    ch_axis = torch.arange(ch0, ch1, dtype=torch.float32).view(-1, 1)
    t_axis = torch.arange(t0, t1, dtype=torch.float32).view(1, -1)
    patch = float(amp) * torch.exp(
        -0.5 * ((ch_axis - float(ch_center)) / sigma_ch) ** 2
        - 0.5 * ((t_axis - float(t_center)) / sigma_t) ** 2
    )
    view = image[ch0:ch1, t0:t1]
    if mode == "add":
        view += patch
    else:
        torch.maximum(view, patch, out=view)


def _set_point_pixel(
    image: torch.Tensor,
    *,
    ch: int,
    t_bin: float,
    width: int,
    amp: float,
) -> None:
    n_ch, t_down = int(image.shape[0]), int(image.shape[1])
    ch_i = int(round(float(ch)))
    t_i = int(round(float(t_bin)))
    radius = int(max(0, width // 2))
    ch0 = max(0, ch_i - radius)
    ch1 = min(n_ch, ch_i + radius + 1)
    t0 = max(0, t_i - radius)
    t1 = min(t_down, t_i + radius + 1)
    if ch1 <= ch0 or t1 <= t0:
        return
    image[ch0:ch1, t0:t1] = torch.maximum(
        image[ch0:ch1, t0:t1],
        torch.full((ch1 - ch0, t1 - t0), float(amp), dtype=image.dtype),
    )


def _add_gaussian_segment(
    image: torch.Tensor,
    *,
    ch0: float,
    t0: float,
    ch1: float,
    t1: float,
    sigma_ch: float,
    sigma_t: float,
    amp: float,
) -> None:
    n_ch, t_down = int(image.shape[0]), int(image.shape[1])
    sigma_ch = float(max(1e-6, sigma_ch))
    sigma_t = float(max(1e-6, sigma_t))
    ch_radius = int(max(0, math.ceil(3.0 * sigma_ch)))
    t_radius = int(max(0, math.ceil(3.0 * sigma_t)))
    ch_min = min(float(ch0), float(ch1))
    ch_max = max(float(ch0), float(ch1))
    t_min = min(float(t0), float(t1))
    t_max = max(float(t0), float(t1))
    ch_start = max(0, int(math.floor(ch_min)) - ch_radius)
    ch_end = min(n_ch, int(math.ceil(ch_max)) + ch_radius + 1)
    t_start = max(0, int(math.floor(t_min)) - t_radius)
    t_end = min(t_down, int(math.ceil(t_max)) + t_radius + 1)
    if ch_end <= ch_start or t_end <= t_start:
        return
    ch_axis = torch.arange(ch_start, ch_end, dtype=torch.float32).view(-1, 1)
    t_axis = torch.arange(t_start, t_end, dtype=torch.float32).view(1, -1)
    px = ch_axis / sigma_ch
    py = t_axis / sigma_t
    ax = float(ch0) / sigma_ch
    ay = float(t0) / sigma_t
    bx = float(ch1) / sigma_ch
    by = float(t1) / sigma_t
    abx = bx - ax
    aby = by - ay
    denom = max(1e-12, abx * abx + aby * aby)
    u = ((px - ax) * abx + (py - ay) * aby) / denom
    u = torch.clamp(u, 0.0, 1.0)
    closest_x = ax + u * abx
    closest_y = ay + u * aby
    dist2 = (px - closest_x) ** 2 + (py - closest_y) ** 2
    patch = float(amp) * torch.exp(-0.5 * dist2)
    view = image[ch_start:ch_end, t_start:t_end]
    torch.maximum(view, patch, out=view)


def _draw_line_pixels(
    image: torch.Tensor,
    *,
    ch0: float,
    t0: float,
    ch1: float,
    t1: float,
    width: int,
    amp: float = 1.0,
) -> None:
    n_ch, t_down = int(image.shape[0]), int(image.shape[1])
    steps = int(max(abs(float(ch1) - float(ch0)), abs(float(t1) - float(t0)), 1.0))
    frac = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)
    ch_idx = torch.round(float(ch0) + (float(ch1) - float(ch0)) * frac).to(torch.long)
    t_idx = torch.round(float(t0) + (float(t1) - float(t0)) * frac).to(torch.long)
    valid = (ch_idx >= 0) & (ch_idx < n_ch) & (t_idx >= 0) & (t_idx < t_down)
    if not bool(valid.any()):
        return
    ch_idx = ch_idx[valid]
    t_idx = t_idx[valid]
    radius = int(max(0, int(width) // 2))
    if radius <= 0:
        image[ch_idx, t_idx] = torch.maximum(
            image[ch_idx, t_idx],
            torch.full_like(image[ch_idx, t_idx], float(amp)),
        )
        return
    for dc in range(-radius, radius + 1):
        for dt in range(-radius, radius + 1):
            cc = ch_idx + int(dc)
            tt = t_idx + int(dt)
            keep = (cc >= 0) & (cc < n_ch) & (tt >= 0) & (tt < t_down)
            if bool(keep.any()):
                image[cc[keep], tt[keep]] = torch.maximum(
                    image[cc[keep], tt[keep]],
                    torch.full_like(image[cc[keep], tt[keep]], float(amp)),
                )


def _render_line(line_mask: torch.Tensor, channels: list[float], t_bins: list[float], *, width: int) -> None:
    if not channels:
        return
    if len(channels) == 1:
        _set_point_pixel(line_mask, ch=int(round(float(channels[0]))), t_bin=float(t_bins[0]), width=int(width), amp=1.0)
        return
    for idx in range(len(channels) - 1):
        ch0, ch1 = float(channels[idx]), float(channels[idx + 1])
        t0, t1 = float(t_bins[idx]), float(t_bins[idx + 1])
        _draw_line_pixels(
            line_mask,
            ch0=ch0,
            t0=t0,
            ch1=ch1,
            t1=t1,
            width=int(width),
        )


def _generate_one(args: argparse.Namespace, index: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + int(index) * 1_000_003)
    n_ch = int(args.n_ch)
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    image_height = int(args.image_height)
    image_width = int(args.image_width)
    seconds_to_cols = float(max(1, image_width - 1)) / float(max(1e-6, args.window_seconds))
    channel_to_rows = float(max(1, image_height - 1)) / float(max(1, n_ch - 1))
    line_mask = torch.zeros((image_height, image_width), dtype=torch.float32)
    point_target = torch.zeros((image_height, image_width), dtype=torch.float32)
    x_image = torch.zeros((image_height, image_width), dtype=torch.float32)

    n_vehicles = int(torch.randint(int(args.vehicles_min), int(args.vehicles_max) + 1, (1,), generator=gen).item())
    track_id = 0
    attempts = 0
    max_attempts = max(64, n_vehicles * 64)
    while track_id < n_vehicles and attempts < max_attempts:
        attempts += 1
        is_primary = bool(torch.rand((), generator=gen).item() < float(args.primary_ratio))
        speed_kmh = _rand_uniform(gen, float(args.speed_min_kmh), float(args.speed_max_kmh))
        anchor_ch = int(torch.randint(0, n_ch, (1,), generator=gen).item())
        anchor_time = float(torch.rand((), generator=gen).item() * float(args.window_seconds))
        t_center, _, _, motion_model, stop_mask = _sample_track_times(
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
        physical_channels = [int(ch) for ch in torch.where(visible)[0].tolist()]
        channels = [float(ch) * channel_to_rows for ch in physical_channels]
        t_bins = [float(t_center[ch].item()) * seconds_to_cols for ch in physical_channels]
        _render_line(
            line_mask,
            channels,
            t_bins,
            width=int(args.line_width),
        )
        for physical_ch, ch, t_bin in zip(physical_channels, channels, t_bins):
            _set_point_pixel(
                point_target,
                ch=int(round(ch)),
                t_bin=float(t_bin),
                width=int(args.point_width),
                amp=1.0,
            )
            if float(torch.rand((), generator=gen).item()) >= float(args.point_drop_prob):
                _set_point_pixel(
                    x_image,
                    ch=int(round(ch)),
                    t_bin=float(t_bin),
                    width=int(args.point_width),
                    amp=1.0,
                )
            if float(torch.rand((), generator=gen).item()) < float(args.nearby_distractor_prob):
                max_count = int(max(1, args.nearby_distractor_count_max))
                count = int(torch.randint(1, max_count + 1, (1,), generator=gen).item())
                for _ in range(count):
                    sign = -1.0 if float(torch.rand((), generator=gen).item()) < 0.5 else 1.0
                    offset_s = _rand_uniform(gen, float(args.nearby_distractor_offset_min_s), float(args.nearby_distractor_offset_max_s))
                    distractor_amp = _rand_uniform(gen, float(args.nearby_distractor_amp_min), float(args.nearby_distractor_amp_max))
                    _set_point_pixel(
                        x_image,
                        ch=int(round(ch)),
                        t_bin=float(t_bin) + sign * offset_s * seconds_to_cols,
                        width=int(args.point_width),
                        amp=float(distractor_amp),
                    )
        track_id += 1

    for ch in range(n_ch):
        if float(torch.rand((), generator=gen).item()) >= float(args.false_peak_prob_per_channel):
            continue
        t_bin = _rand_uniform(gen, 0.0, float(max(0, image_width - 1)))
        amp = _rand_uniform(gen, float(args.false_peak_amp_min), float(args.false_peak_amp_max))
        _set_point_pixel(
            x_image,
            ch=int(round(float(ch) * channel_to_rows)),
            t_bin=t_bin,
            width=int(args.point_width),
            amp=float(amp),
        )
    if float(args.input_noise_std) > 0.0:
        x_image += torch.normal(mean=0.0, std=float(args.input_noise_std), size=x_image.shape, generator=gen, dtype=torch.float32)
    x = torch.clamp(x_image, 0.0, 1.0).unsqueeze(0).contiguous()
    line_mask = torch.clamp(line_mask, 0.0, 1.0).unsqueeze(0).contiguous()
    point_target = torch.clamp(point_target, 0.0, 1.0).unsqueeze(0).contiguous()
    if str(args.x_dtype) == "float16":
        x = x.to(torch.float16)
    return {"x": x, "line_mask": line_mask, "point_target": point_target}


def _write_shard(out_path: Path, items: list[dict[str, torch.Tensor]]) -> None:
    payload = {
        "x": torch.stack([item["x"] for item in items], dim=0).contiguous(),
        "line_mask": torch.stack([item["line_mask"] for item in items], dim=0).contiguous(),
        "point_target": torch.stack([item["point_target"] for item in items], dim=0).contiguous(),
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


def _validate_args(args: argparse.Namespace) -> tuple[list[str], list[float]]:
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0")
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size must be > 0")
    if int(args.n_ch) <= 0:
        raise ValueError("--n-ch must be > 0")
    if int(args.image_height) <= 1 or int(args.image_width) <= 1:
        raise ValueError("--image-height and --image-width must be > 1")
    if int(args.vehicles_min) < 0 or int(args.vehicles_max) < int(args.vehicles_min):
        raise ValueError("--vehicles-max must be >= --vehicles-min >= 0")
    if float(args.speed_min_kmh) <= 0.0 or float(args.speed_max_kmh) < float(args.speed_min_kmh):
        raise ValueError("speed range must be positive and ordered")
    for name in ("point_drop_prob", "nearby_distractor_prob", "false_peak_prob_per_channel"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if float(args.nearby_distractor_offset_min_s) < 0.0 or float(args.nearby_distractor_offset_max_s) < float(args.nearby_distractor_offset_min_s):
        raise ValueError("nearby distractor offset range must be non-negative and ordered")
    if int(args.nearby_distractor_count_max) < 1:
        raise ValueError("--nearby-distractor-count-max must be >= 1")
    if int(args.line_width) <= 0:
        raise ValueError("--line-width must be > 0")
    if int(args.point_width) <= 0:
        raise ValueError("--point-width must be > 0")
    return _motion_models_and_weights(args)


def main() -> int:
    args = parse_args()
    motion_models, motion_weights = _validate_args(args)
    out_dir = Path(args.out_dir).expanduser()
    args.out_dir = out_dir
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
            print(
                f"wrote {shard_name}: samples={sample_count}, total={done_samples}/{total}, "
                f"shard_elapsed={shard_seconds:.1f}s, elapsed={time.perf_counter() - t0:.1f}s",
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
                print(
                    f"wrote {shard_name}: samples={sample_count}, total={done_samples}/{total}, "
                    f"shard_elapsed={shard_seconds:.1f}s, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

    shard_names = [name for name in shard_files if name is not None]
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    meta = {
        "format": "peak_line_shards_v1",
        "created_at_unix": time.time(),
        "num_samples": total,
        "shard_size": shard_size,
        "shards": shard_names,
        "n_channels": int(args.image_height),
        "physical_channels": int(args.n_ch),
        "in_channels": 1,
        "out_channels": 1,
        "fs": float(args.fs),
        "window_seconds": float(args.window_seconds),
        "window_samples": int(window_samples),
        "time_downsample": int(args.time_downsample),
        "downsampled_time": int(args.image_width),
        "image_height": int(args.image_height),
        "image_width": int(args.image_width),
        "dx_m": float(args.dx_m),
        "motion_models": motion_models,
        "motion_weights": motion_weights,
        "generator_args": _args_payload(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Wrote {len(shard_names)} shards to {out_dir} in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
