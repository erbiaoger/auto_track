"""Generate tensor shards for TrackSlotNet training.

Purpose:
    This script creates a disk dataset tailored for the track-slot model. It
    directly writes PyTorch `.pt` shards and does not create SAC files. Each
    sample is a clean Gaussian DAS heatmap window plus instance trajectory
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
    once and reuse the shards across runs.
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
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise std added to the downsampled heatmap.")
    parser.add_argument("--amp-min", type=float, default=6.0, help="Minimum Gaussian pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=6.0, help="Maximum Gaussian pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.25, help="Minimum Gaussian sigma in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.25, help="Maximum Gaussian sigma in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=0.8333333333, help="Fraction of forward-direction vehicles.")
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


def _generate_one(args: argparse.Namespace, index: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + int(index) * 1_000_003)
    n_ch = int(args.n_ch)
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    time_downsample = int(max(1, args.time_downsample))
    t_down = int(max(1, len(range(0, window_samples, time_downsample))))
    t_axis_s = torch.arange(t_down, dtype=torch.float32) * (float(time_downsample) / float(args.fs))
    if float(args.noise_std) > 0.0:
        data_ds = torch.normal(
            mean=0.0,
            std=float(args.noise_std),
            size=(n_ch, t_down),
            generator=gen,
            dtype=torch.float32,
        )
    else:
        data_ds = torch.zeros((n_ch, t_down), dtype=torch.float32)

    max_gt = int(max(args.vehicles_min, args.vehicles_max))
    time_label = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    visibility = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    direction = torch.zeros((max_gt,), dtype=torch.long)
    speed = torch.zeros((max_gt,), dtype=torch.float32)
    gt_valid = torch.zeros((max_gt,), dtype=torch.bool)

    channel_index = torch.arange(n_ch, dtype=torch.float32)
    n_vehicles = int(torch.randint(int(args.vehicles_min), int(args.vehicles_max) + 1, (1,), generator=gen).item())
    track_id = 0
    attempts = 0
    max_attempts = max(64, n_vehicles * 64)
    while track_id < n_vehicles and attempts < max_attempts:
        attempts += 1
        is_primary = bool(torch.rand((), generator=gen).item() < float(args.primary_ratio))
        direction_label = 0 if is_primary else 1
        speed_kmh = _rand_uniform(gen, float(args.speed_min_kmh), float(args.speed_max_kmh))
        speed_mps = speed_kmh / 3.6
        sigma_s = _rand_uniform(gen, float(args.sigma_min_s), float(args.sigma_max_s))
        amp = _rand_uniform(gen, float(args.amp_min), float(args.amp_max))
        dist_m = channel_index * float(args.dx_m) if is_primary else (n_ch - 1 - channel_index) * float(args.dx_m)
        anchor_ch = int(torch.randint(0, n_ch, (1,), generator=gen).item())
        anchor_time = float(torch.rand((), generator=gen).item() * float(args.window_seconds))
        t_entry = anchor_time - float(dist_m[anchor_ch].item()) / max(1e-6, speed_mps)
        t_center = t_entry + dist_m / max(1e-6, speed_mps)
        visible = (t_center >= 0.0) & (t_center < float(args.window_seconds))
        if int(visible.sum().item()) < int(args.min_visible_channels):
            continue

        for ch in torch.where(visible)[0].tolist():
            pulse = amp * torch.exp(-0.5 * ((t_axis_s - float(t_center[ch].item())) / max(1e-6, sigma_s)) ** 2)
            data_ds[int(ch)] += pulse
        center_idx = torch.round(t_center * float(args.fs)).to(torch.long).clamp(0, window_samples - 1)
        time_label[track_id, visible] = (
            center_idx[visible].to(torch.float32) / float(max(1, window_samples - 1))
        ).clamp(0.0, 1.0)
        visibility[track_id] = visible.to(torch.float32)
        direction[track_id] = int(direction_label)
        speed[track_id] = float(speed_kmh / max(1e-6, float(args.speed_norm_kmh)))
        gt_valid[track_id] = True
        track_id += 1

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
    if int(args.vehicles_min) < 0 or int(args.vehicles_max) < int(args.vehicles_min):
        raise ValueError("--vehicles-max must be >= --vehicles-min >= 0")
    if float(args.speed_min_kmh) <= 0.0 or float(args.speed_max_kmh) < float(args.speed_min_kmh):
        raise ValueError("speed range must be positive and ordered")

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
        "clip_ratio": float(args.clip_ratio),
        "input_mode": str(args.input_mode),
        "generator_args": _args_payload(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={total}, shards={len(shard_names)}, out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
