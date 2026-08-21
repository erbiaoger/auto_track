"""Build a fixed single-vehicle benchmark from synthetic or real backgrounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed benchmark for single-vehicle tracking.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output .pt benchmark file.")
    parser.add_argument("--samples", type=int, default=256, help="Number of benchmark windows.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--vehicles-min", type=int, default=1, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=1, help="Maximum vehicles per window.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=100.0, help="Maximum vehicle speed.")
    parser.add_argument("--noise-std", type=float, default=0.12, help="Background noise standard deviation.")
    parser.add_argument("--amp-min", type=float, default=0.8, help="Minimum pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=2.0, help="Maximum pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.03, help="Minimum pulse width in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.08, help="Maximum pulse width in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=1.0, help="Probability of forward-direction samples.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels per sample.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by synthetic labels.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Layout of the real background .npy.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="First channel index to slice from the real background.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to the real background window.")
    parser.add_argument("--artifact-dropout-ratio", type=float, default=0.0, help="Chance to remove a contiguous block of visible channels.")
    parser.add_argument("--artifact-dropout-min-channels", type=int, default=2, help="Minimum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-dropout-max-channels", type=int, default=6, help="Maximum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-decoy-ratio", type=float, default=0.0, help="Chance to inject a decoy branch or spike cluster.")
    parser.add_argument("--artifact-decoy-min-points", type=int, default=1, help="Minimum decoy points per sample.")
    parser.add_argument("--artifact-decoy-max-points", type=int, default=3, help="Maximum decoy points per sample.")
    parser.add_argument("--artifact-decoy-amp-scale-min", type=float, default=1.1, help="Minimum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-amp-scale-max", type=float, default=2.2, help="Maximum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-time-jitter-s", type=float, default=0.18, help="Random decoy time jitter in seconds.")
    parser.add_argument("--artifact-competing-ratio", type=float, default=0.0, help="Chance to inject an unlabeled competing vehicle track.")
    parser.add_argument("--artifact-competing-time-jitter-s", type=float, default=0.8, help="Random time jitter for the competing vehicle anchor.")
    parser.add_argument("--artifact-competing-amp-scale-min", type=float, default=0.8, help="Minimum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-amp-scale-max", type=float, default=1.6, help="Maximum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-speed-ratio-min", type=float, default=0.88, help="Minimum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-speed-ratio-max", type=float, default=1.12, help="Maximum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-channel-offset-max", type=int, default=5, help="Maximum channel offset for the competing vehicle anchor relative to the target anchor.")
    parser.add_argument("--artifact-competing-opposite-direction-ratio", type=float, default=0.0, help="Probability of assigning the competing vehicle the opposite direction.")
    parser.add_argument("--raw-window-dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Storage dtype for raw_window in the benchmark file.")
    parser.add_argument("--input-mode", default="raw", choices=["raw"], help="Synthetic input mode.")
    parser.add_argument("--mask-sigma-ch", type=float, default=0.8, help="Target heatmap sigma in channel units.")
    parser.add_argument("--mask-sigma-t", type=float, default=2.0, help="Target heatmap sigma in downsampled time bins.")
    return parser.parse_args()


def _build_dataset(args: argparse.Namespace) -> OnlineSyntheticTrajectoryDataset:
    return OnlineSyntheticTrajectoryDataset(
        length=int(max(1, args.samples)),
        n_channels=int(args.n_channels),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=0.0,
        slow_speed_min_kmh=float(args.speed_min_kmh),
        slow_speed_max_kmh=float(args.speed_max_kmh),
        fast_speed_min_kmh=float(args.speed_min_kmh),
        fast_speed_max_kmh=float(args.speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        speed_norm_kmh=float(args.speed_norm_kmh),
        clip_ratio=float(args.clip_ratio),
        input_mode=str(args.input_mode),
        seed=int(args.seed),
        mask_sigma_ch=float(args.mask_sigma_ch),
        mask_sigma_t=float(args.mask_sigma_t),
        cache_dataset=False,
        return_raw_window=True,
        background_npy=args.background_npy,
        background_layout=str(args.background_layout),
        background_channel_start=int(args.background_channel_start),
        background_scale=float(args.background_scale),
        artifact_dropout_ratio=float(args.artifact_dropout_ratio),
        artifact_dropout_min_channels=int(args.artifact_dropout_min_channels),
        artifact_dropout_max_channels=int(args.artifact_dropout_max_channels),
        artifact_decoy_ratio=float(args.artifact_decoy_ratio),
        artifact_decoy_min_points=int(args.artifact_decoy_min_points),
        artifact_decoy_max_points=int(args.artifact_decoy_max_points),
        artifact_decoy_amp_scale_min=float(args.artifact_decoy_amp_scale_min),
        artifact_decoy_amp_scale_max=float(args.artifact_decoy_amp_scale_max),
        artifact_decoy_time_jitter_s=float(args.artifact_decoy_time_jitter_s),
        artifact_competing_ratio=float(args.artifact_competing_ratio),
        artifact_competing_time_jitter_s=float(args.artifact_competing_time_jitter_s),
        artifact_competing_amp_scale_min=float(args.artifact_competing_amp_scale_min),
        artifact_competing_amp_scale_max=float(args.artifact_competing_amp_scale_max),
        artifact_competing_speed_ratio_min=float(args.artifact_competing_speed_ratio_min),
        artifact_competing_speed_ratio_max=float(args.artifact_competing_speed_ratio_max),
        artifact_competing_channel_offset_max=int(args.artifact_competing_channel_offset_max),
        artifact_competing_opposite_direction_ratio=float(args.artifact_competing_opposite_direction_ratio),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def main() -> int:
    args = parse_args()
    ds = _build_dataset(args)
    raw_window_dtype = str(args.raw_window_dtype)
    samples: list[dict[str, Any]] = []
    for idx in range(len(ds)):
        x, target = ds[idx]
        if "raw_window" in target:
            if raw_window_dtype == "float16":
                target["raw_window"] = target["raw_window"].to(torch.float16)
            elif raw_window_dtype == "bfloat16":
                target["raw_window"] = target["raw_window"].to(torch.bfloat16)
            else:
                target["raw_window"] = target["raw_window"].to(torch.float32)
        samples.append(
            {
                "x": x.to(torch.float32).cpu(),
                "target": {key: value.cpu() for key, value in target.items()},
            }
        )
    payload = {
        "format": "single_vehicle_benchmark_v1",
        "meta": _json_ready(vars(args)),
        "length": int(len(samples)),
        "samples": samples,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out_file))
    meta_path = args.out_file.with_suffix(".json")
    meta_path.write_text(json.dumps(_json_ready(payload["meta"]), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
