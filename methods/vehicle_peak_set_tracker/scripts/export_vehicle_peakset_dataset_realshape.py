from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.simple_vehicle_peak_dataset import (
    SimpleLinearVehiclePeakDataset,
    SimplePeakSetDatasetConfig,
    dataset_config_to_dict,
    peakset_collate,
)


FWHM_FACTOR = 2.3548200450309493


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export real-shape synthetic vehicle peak-set dataset shards.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--num-samples", type=int, default=16384, help="Total number of samples.")
    parser.add_argument("--shard-size", type=int, default=128, help="Samples per shard.")
    parser.add_argument("--seed", type=int, default=20260630, help="Dataset seed.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window length in seconds.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--n-channels", type=int, default=50, help="Channel count.")
    parser.add_argument("--vehicles-min", type=int, default=4, help="Minimum vehicles per scene.")
    parser.add_argument("--vehicles-max", type=int, default=10, help="Maximum vehicles per scene.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--amp-min", type=float, default=3.0, help="Vehicle amplitude minimum.")
    parser.add_argument("--amp-max", type=float, default=8.0, help="Vehicle amplitude maximum.")
    parser.add_argument("--sigma-min-s", type=float, default=0.42, help="Gaussian sigma minimum in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.60, help="Gaussian sigma maximum in seconds.")
    parser.add_argument("--motion-mix", default="constant_sparse,smooth_random,stop_go", help="Comma-separated motion model mix.")
    parser.add_argument("--motion-weights", default="0.84,0.15,0.01", help="Comma-separated motion model weights.")
    parser.add_argument("--same-direction-ratio", type=float, default=0.82, help="Probability of keeping vehicles in the same direction.")
    parser.add_argument("--crossing-ratio", type=float, default=0.18, help="Probability of generating crossing-style scenes.")
    parser.add_argument("--scene-cluster-ratio", type=float, default=0.35, help="Probability of sampling vehicles from a shared scene cluster.")
    parser.add_argument("--interaction-ratio", type=float, default=0.0, help="Probability of injecting explicit vehicle interaction pairs.")
    parser.add_argument(
        "--interaction-types",
        default="parallel_crossing,overtake,crossing,near_parallel",
        help="Comma-separated interaction types used when interaction_ratio > 0.",
    )
    parser.add_argument("--stop-response-sigma-scale", type=float, default=3.0, help="Sigma scale for stop-go response.")
    parser.add_argument("--stop-response-amp-scale", type=float, default=1.2, help="Amplitude scale for stop-go response.")
    parser.add_argument("--dead-channel-indices", default="5,6,15,16,22,36,38,45", help="Fixed dead channels.")
    parser.add_argument("--intermittent-dead-channel-rates", default="11:0.69,17:0.37,42:0.75", help="Per-window intermittent dead channel rates.")
    parser.add_argument("--random-dead-channel-ratio", type=float, default=0.0, help="Optional random dead channel probability.")
    parser.add_argument("--per-vehicle-drop-channel-ratio", type=float, default=0.0, help="Probability of dropping per-vehicle channels.")
    parser.add_argument("--per-vehicle-drop-channel-min", type=int, default=5, help="Minimum per-vehicle dropped channels.")
    parser.add_argument("--per-vehicle-drop-channel-max", type=int, default=14, help="Maximum per-vehicle dropped channels.")
    parser.add_argument("--missing-random-ratio-min", type=float, default=0.0, help="Lower bound for per-vehicle random missing ratio.")
    parser.add_argument("--missing-random-ratio-max", type=float, default=0.0, help="Upper bound for per-vehicle random missing ratio.")
    parser.add_argument("--missing-segment-count-max", type=int, default=0, help="Maximum count of per-vehicle missing segments.")
    parser.add_argument("--missing-segment-min-len", type=int, default=2, help="Minimum per-vehicle missing segment length.")
    parser.add_argument("--missing-segment-max-len", type=int, default=6, help="Maximum per-vehicle missing segment length.")
    parser.add_argument("--isolated-noise-ratio", type=float, default=0.30, help="Probability of adding isolated background peaks.")
    parser.add_argument("--isolated-noise-rate", type=float, default=14.0, help="Mean number of isolated background peaks per window.")
    parser.add_argument("--isolated-noise-amp-min", type=float, default=0.6, help="Isolated peak amplitude minimum.")
    parser.add_argument("--isolated-noise-amp-max", type=float, default=4.0, help="Isolated peak amplitude maximum.")
    parser.add_argument("--isolated-noise-sigma-min-s", type=float, default=0.42, help="Isolated peak sigma minimum in seconds.")
    parser.add_argument("--isolated-noise-sigma-max-s", type=float, default=0.60, help="Isolated peak sigma maximum in seconds.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--input-scale", type=float, default=0.0, help="Fixed input scale for x normalization; <=0 uses robust per-window scale.")
    parser.add_argument("--include-raw-window", action="store_true", help="Store full-resolution raw_window in each shard.")
    parser.add_argument("--stats-samples", type=int, default=512, help="Samples used for dataset statistics.")
    parser.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 1))), help="Parallel shard workers.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting an existing directory.")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _quantiles(values: np.ndarray, probs: tuple[float, ...]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {f"q{int(round(p * 100)):02d}": None for p in probs}
    result: dict[str, float] = {}
    for p in probs:
        result[f"q{int(round(p * 100)):02d}"] = float(np.quantile(arr, float(p)))
    return result


def _build_dataset(args: argparse.Namespace, *, length: int, seed: int) -> SimpleLinearVehiclePeakDataset:
    cfg = SimplePeakSetDatasetConfig(
        length=int(length),
        n_channels=int(args.n_channels),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=10,
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        noise_std=0.0,
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        min_visible_channels=5,
        primary_ratio=0.82,
        same_direction_ratio=float(args.same_direction_ratio),
        crossing_ratio=float(args.crossing_ratio),
        scene_cluster_ratio=float(args.scene_cluster_ratio),
        motion_mix=str(args.motion_mix),
        motion_weights=str(args.motion_weights),
        constant_perturb_prob=0.0,
        constant_perturb_max_frac=0.0,
        smooth_speed_max_frac=0.0,
        track_time_jitter_max_s=0.0,
        stop_duration_min_s=1.0,
        stop_duration_max_s=1.0,
        stop_response_sigma_scale=float(args.stop_response_sigma_scale),
        stop_response_amp_scale=float(args.stop_response_amp_scale),
        dead_channel_indices=str(args.dead_channel_indices),
        intermittent_dead_channel_rates=str(args.intermittent_dead_channel_rates),
        random_dead_channel_ratio=float(args.random_dead_channel_ratio),
        zero_background_ratio=0.0,
        zero_background_rate=0.0,
        per_vehicle_drop_channel_ratio=float(args.per_vehicle_drop_channel_ratio),
        per_vehicle_drop_channel_min=int(args.per_vehicle_drop_channel_min),
        per_vehicle_drop_channel_max=int(args.per_vehicle_drop_channel_max),
        missing_random_ratio_min=float(args.missing_random_ratio_min),
        missing_random_ratio_max=float(args.missing_random_ratio_max),
        missing_segment_count_max=int(args.missing_segment_count_max),
        missing_segment_min_len=int(args.missing_segment_min_len),
        missing_segment_max_len=int(args.missing_segment_max_len),
        interaction_ratio=float(args.interaction_ratio),
        interaction_types=str(args.interaction_types),
        isolated_noise_ratio=float(args.isolated_noise_ratio),
        isolated_noise_rate=float(args.isolated_noise_rate),
        isolated_noise_amp_min=float(args.isolated_noise_amp_min),
        isolated_noise_amp_max=float(args.isolated_noise_amp_max),
        isolated_noise_sigma_min_s=float(args.isolated_noise_sigma_min_s),
        isolated_noise_sigma_max_s=float(args.isolated_noise_sigma_max_s),
        clip_ratio=float(args.clip_ratio),
        input_scale=float(args.input_scale),
        input_mode="raw",
        seed=int(seed),
        return_raw_window=bool(args.include_raw_window),
    )
    return SimpleLinearVehiclePeakDataset(config=cfg)


def _sample_isolated_ratio(peak_time: torch.Tensor, peak_valid: torch.Tensor, window_seconds: float) -> tuple[int, int]:
    total = 0
    isolated = 0
    peak_time = peak_time.detach().cpu()
    peak_valid = peak_valid.detach().cpu()
    bsz, n_ch, k_count = peak_valid.shape
    for b in range(bsz):
        mask = peak_valid[b].reshape(-1).numpy().astype(bool, copy=False)
        if not mask.any():
            continue
        times = peak_time[b].reshape(-1).numpy()[mask] * float(window_seconds)
        channels = np.repeat(np.arange(n_ch, dtype=np.int32), k_count)[mask]
        total += int(times.size)
        if times.size <= 1:
            isolated += int(times.size)
            continue
        dt = np.abs(times[:, None] - times[None, :])
        dc = np.abs(channels[:, None] - channels[None, :])
        neighbor = (dt <= 2.0) & (dc <= 2) & (~np.eye(times.size, dtype=bool))
        isolated += int((~neighbor.any(axis=1)).sum())
    return isolated, total


def _collect_stats(dataset: SimpleLinearVehiclePeakDataset, sample_count: int) -> dict[str, Any]:
    batch = [dataset[i] for i in range(int(sample_count))]
    xs, targets = peakset_collate(batch)
    peak_valid = targets["peak_valid"].to(torch.bool)
    peak_amp = targets["peak_amp"].to(torch.float32)
    peak_time = targets["peak_time"].to(torch.float32)

    total_peaks_per_window = peak_valid.sum(dim=(1, 2)).to(torch.float32).cpu().numpy()
    per_channel_peaks = peak_valid.sum(dim=2).to(torch.float32).reshape(-1).cpu().numpy()
    detected_amplitudes = peak_amp[peak_valid].to(torch.float32).cpu().numpy()

    sigma_s = targets["sigma_s"].to(torch.float32)
    target_amp = targets["amp"].to(torch.float32)
    gt_valid = targets["gt_valid"].to(torch.bool)
    rendered_sigmas = sigma_s[gt_valid].cpu().numpy()
    rendered_fwhm = rendered_sigmas * FWHM_FACTOR
    rendered_amps = target_amp[gt_valid].cpu().numpy()

    observed_visibility = targets["observed_visibility"].to(torch.float32)
    observed_zero_ratio = 1.0 - observed_visibility.mean(dim=(0, 1)).cpu().numpy()

    isolated, total = _sample_isolated_ratio(peak_time[: int(sample_count)], peak_valid[: int(sample_count)], dataset.window_seconds)

    return {
        "stats_sample_count": int(sample_count),
        "fixed_dead_channels": [int(x) for x in dataset._parse_int_csv(dataset.dead_channel_indices)],
        "intermittent_dead_channel_rates": {str(k): float(v) for k, v in dataset._intermittent_dead_channel_rate_map.items()},
        "random_dead_channel_ratio": float(dataset.random_dead_channel_ratio),
        "peak_count_per_window_quantiles": _quantiles(total_peaks_per_window, (0.0, 0.5, 0.9, 1.0)),
        "peak_count_per_channel_quantiles": _quantiles(per_channel_peaks, (0.5, 0.9, 0.95, 1.0)),
        "peak_amplitude_quantiles": _quantiles(detected_amplitudes, (0.1, 0.5, 0.9, 1.0)),
        "target_amplitude_quantiles": _quantiles(rendered_amps, (0.1, 0.5, 0.9, 1.0)),
        "target_sigma_seconds_quantiles": _quantiles(rendered_sigmas, (0.1, 0.5, 0.9, 1.0)),
        "peak_width_fwhm_seconds_quantiles": _quantiles(rendered_fwhm, (0.1, 0.5, 0.9, 1.0)),
        "isolated_peak_ratio": float(isolated / max(1, total)),
        "observed_zero_ratio_by_channel": [float(v) for v in np.asarray(observed_zero_ratio, dtype=np.float32)],
    }


def _export_single_shard(
    *,
    out_dir: Path,
    args_dict: dict[str, Any],
    start: int,
    end: int,
    shard_idx: int,
) -> tuple[str, int]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    dataset = _build_dataset(argparse.Namespace(**args_dict), length=int(end), seed=int(args_dict["seed"]))
    batch = [dataset[i] for i in range(int(start), int(end))]
    xs, targets = peakset_collate(batch)
    shard_name = f"shard_{shard_idx:06d}.pt"
    payload = {
        "x": xs.to(torch.float16).contiguous(),
        "targets": {key: value.contiguous() if torch.is_tensor(value) else value for key, value in targets.items()},
        "sample_range": [int(start), int(end)],
    }
    if bool(args_dict.get("include_raw_window", False)):
        raw_windows = torch.stack([item[1]["raw_window"].to(torch.float32) for item in batch], dim=0)
        payload["raw_window"] = raw_windows.to(torch.float16).contiguous()
    torch.save(payload, str(out_dir / shard_name))
    return shard_name, int(end - start)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
        raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_size = int(max(1, args.shard_size))
    shard_count = int(math.ceil(int(args.num_samples) / shard_size))
    shard_jobs = [
        (shard_idx, shard_idx * shard_size, min(int(args.num_samples), (shard_idx + 1) * shard_size))
        for shard_idx in range(shard_count)
    ]
    args_dict = vars(args).copy()
    worker_count = int(max(1, args.workers))
    shards: list[str] = []
    shard_sizes: list[int] = []

    if worker_count == 1 or len(shard_jobs) == 1:
        for shard_idx, start, end in shard_jobs:
            shard_name, shard_n = _export_single_shard(out_dir=out_dir, args_dict=args_dict, start=start, end=end, shard_idx=shard_idx)
            shards.append(shard_name)
            shard_sizes.append(shard_n)
            print(f"wrote {shard_name}: samples={shard_n}", flush=True)
    else:
        with futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
            future_map = {
                pool.submit(
                    _export_single_shard,
                    out_dir=out_dir,
                    args_dict=args_dict,
                    start=start,
                    end=end,
                    shard_idx=shard_idx,
                ): (shard_idx, start, end)
                for shard_idx, start, end in shard_jobs
            }
            for fut in futures.as_completed(future_map):
                shard_idx, start, end = future_map[fut]
                shard_name, shard_n = fut.result()
                shards.append(shard_name)
                shard_sizes.append(shard_n)
                print(f"wrote {shard_name}: samples={shard_n}", flush=True)
        shards.sort()
        shard_sizes = [int(end - start) for _, start, end in shard_jobs]

    stats_sample_count = int(max(1, min(int(args.stats_samples), int(args.num_samples))))
    stats_dataset = _build_dataset(args, length=int(stats_sample_count), seed=int(args.seed))
    stats = _collect_stats(stats_dataset, stats_sample_count)
    meta_dataset = _build_dataset(args, length=int(args.num_samples), seed=int(args.seed))

    meta = {
        "format": "vehicle_peakset_shards_realshape_gauss_v1",
        "num_samples": int(args.num_samples),
        "shard_size": shard_size,
        "shards": shards,
        "shard_sizes": shard_sizes,
        "dataset_config": dataset_config_to_dict(meta_dataset.config),
    }
    (out_dir / "meta.json").write_text(json.dumps(_json_ready(meta), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "dataset_stats.json").write_text(json.dumps(_json_ready(stats), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_ready(meta), ensure_ascii=False), flush=True)
    print(json.dumps(_json_ready(stats), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
