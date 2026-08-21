"""Generate TrackSlotNet shards from real `.npy` background windows plus synthetic vehicles.

Purpose:
    Reduce the train/real domain gap by sampling background directly from a real
    DAS `.npy` array, then overlaying simulated vehicle trajectories with exact
    labels. The output format matches `generate_track_slot_dataset.py`, so the
    resulting directory can be used directly by TrackSlotNet and then converted
    to PeakSlotNet shards.

Example:
    uv run python -m autotrack.dl.generate_track_slot_dataset_from_real_npy \
        --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
        --profile /tmp/real_profile/realism_profile.json \
        --out-dir datasets/track_slot_realbg_120s_profile/train \
        --num-samples 2000 \
        --window-seconds 120 \
        --window-sampler profile_weighted \
        --artifact-policy hybrid \
        --overwrite

How it works:
    1. Sample a background window from the real `.npy` array, uniformly or with
       profile-driven weights.
    2. Optionally apply defaults inferred from `realism_profile.json` unless the
       user explicitly overrides them through CLI parameters.
    3. Overlay synthetic vehicle trajectories and optional extra artifacts.
    4. Write TrackSlotNet-compatible `shard_*.pt` plus `meta.json`.

Outputs:
    - `meta.json`
    - `shard_000000.pt`, ...
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.generate_track_slot_dataset import (
    _apply_channel_gain,
    _apply_dead_channels,
    _args_payload,
    _prepare_input,
    _sample_track_times,
    _split_csv,
    _write_shard,
)


ARG_DEFAULTS: dict[str, Any] = {
    "array_layout": "time_channel",
    "num_samples": 2000,
    "shard_size": 128,
    "seed": 42,
    "fs": 1000.0,
    "dx_m": 100.0,
    "window_seconds": 120.0,
    "window_stride_seconds": 60.0,
    "time_downsample": 10,
    "channel_start": 0,
    "channel_count": 50,
    "background_scale_min": 0.95,
    "background_scale_max": 1.05,
    "background_offset_std": 0.0,
    "vehicles_min": 6,
    "vehicles_max": 24,
    "speed_min_kmh": 70.0,
    "speed_max_kmh": 86.0,
    "speed_norm_kmh": 150.0,
    "fixed_amp": 6.0,
    "sigma_seconds": 0.25,
    "primary_ratio": 0.83,
    "min_visible_channels": 4,
    "motion_mix": "constant_sparse,smooth_random,stop_go",
    "motion_weights": "0.84,0.15,0.01",
    "constant_perturb_prob": 0.05,
    "constant_perturb_max_frac": 0.01,
    "constant_perturb_width_min": 1,
    "constant_perturb_width_max": 2,
    "smooth_speed_max_frac": 0.05,
    "smooth_speed_corr_channels": 8.0,
    "stop_duration_min_s": 1.0,
    "stop_duration_max_s": 8.0,
    "stop_channel_width_min": 1,
    "stop_channel_width_max": 3,
    "stop_response_sigma_scale": 3.0,
    "stop_response_amp_scale": 1.2,
    "restart_speed_ratio_min": 0.95,
    "restart_speed_ratio_max": 1.05,
    "isolated_noise_ratio": 0.85,
    "isolated_noise_rate": 220.0,
    "isolated_noise_amp_min": 0.6,
    "isolated_noise_amp_max": 5.5,
    "isolated_noise_sigma_min_s": 0.04,
    "isolated_noise_sigma_max_s": 0.22,
    "random_dead_channel_ratio": 0.85,
    "random_dead_channel_min": 4,
    "random_dead_channel_max": 12,
    "dead_channel_indices": "",
    "per_vehicle_drop_channel_ratio": 1.0,
    "per_vehicle_drop_channel_min": 6,
    "per_vehicle_drop_channel_max": 10,
    "zero_background_ratio": 0.9,
    "zero_background_rate": 32.0,
    "zero_background_channel_min": 1,
    "zero_background_channel_max": 3,
    "zero_background_duration_min_s": 0.6,
    "zero_background_duration_max_s": 4.0,
    "clip_ratio": 1.35,
    "input_mode": "raw",
    "x_dtype": "float16",
    "profile_strength": 1.0,
    "window_sampler": "uniform",
    "artifact_policy": "manual",
}

PROFILE_NUMERIC_KEYS = {
    "background_scale_min",
    "background_scale_max",
    "background_offset_std",
    "vehicles_min",
    "vehicles_max",
    "speed_min_kmh",
    "speed_max_kmh",
    "speed_norm_kmh",
    "fixed_amp",
    "sigma_seconds",
    "primary_ratio",
    "min_visible_channels",
    "isolated_noise_ratio",
    "isolated_noise_rate",
    "isolated_noise_amp_min",
    "isolated_noise_amp_max",
    "isolated_noise_sigma_min_s",
    "isolated_noise_sigma_max_s",
    "per_vehicle_drop_channel_ratio",
    "per_vehicle_drop_channel_min",
    "per_vehicle_drop_channel_max",
    "random_dead_channel_ratio",
    "random_dead_channel_min",
    "random_dead_channel_max",
    "zero_background_ratio",
    "zero_background_rate",
    "zero_background_channel_min",
    "zero_background_channel_max",
    "zero_background_duration_min_s",
    "zero_background_duration_max_s",
}

PROFILE_ARTEFACT_KEYS = {
    "dead_channel_indices",
    "isolated_noise_ratio",
    "isolated_noise_rate",
    "isolated_noise_amp_min",
    "isolated_noise_amp_max",
    "isolated_noise_sigma_min_s",
    "isolated_noise_sigma_max_s",
    "per_vehicle_drop_channel_ratio",
    "per_vehicle_drop_channel_min",
    "per_vehicle_drop_channel_max",
    "random_dead_channel_ratio",
    "random_dead_channel_min",
    "random_dead_channel_max",
    "zero_background_ratio",
    "zero_background_rate",
    "zero_background_channel_min",
    "zero_background_channel_max",
    "zero_background_duration_min_s",
    "zero_background_duration_max_s",
}

PROFILE_VEHICLE_KEYS = {
    "vehicles_min",
    "vehicles_max",
    "speed_min_kmh",
    "speed_max_kmh",
    "speed_norm_kmh",
    "fixed_amp",
    "sigma_seconds",
    "primary_ratio",
    "min_visible_channels",
}

PROFILE_BACKGROUND_KEYS = {
    "background_scale_min",
    "background_scale_max",
    "background_offset_std",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TrackSlotNet shards from real `.npy` background windows.")
    parser.add_argument("--input", required=True, type=Path, help="Real DAS `.npy` background file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output track_slot shard directory.")
    parser.add_argument("--profile", type=Path, default=None, help="Optional realism_profile.json produced by profile_real_npy_background.py.")
    parser.add_argument("--profile-strength", type=float, default=1.0, help="Blend factor in [0, 1] when applying profile-derived defaults.")
    parser.add_argument("--window-sampler", choices=["uniform", "profile_weighted"], default="uniform", help="Background window sampler.")
    parser.add_argument("--artifact-policy", choices=["manual", "profile_matched", "hybrid"], default="manual", help="How strongly profile defaults influence vehicle/artifact sampling.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--num-samples", type=int, default=2000, help="Number of generated windows.")
    parser.add_argument("--shard-size", type=int, default=128, help="Samples per shard.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")

    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window length in seconds.")
    parser.add_argument("--window-stride-seconds", type=float, default=60.0, help="Stride used when sampling candidate background windows.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample factor.")
    parser.add_argument("--channel-start", type=int, default=0, help="First background channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--background-scale-min", type=float, default=0.95, help="Minimum multiplicative scale applied to each sampled real background window.")
    parser.add_argument("--background-scale-max", type=float, default=1.05, help="Maximum multiplicative scale applied to each sampled real background window.")
    parser.add_argument("--background-offset-std", type=float, default=0.0, help="Optional additive offset std applied per sampled window before vehicle overlay.")

    parser.add_argument("--vehicles-min", type=int, default=6, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=24, help="Maximum vehicles per window.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=86.0, help="Maximum vehicle speed.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization constant stored in labels.")
    parser.add_argument("--fixed-amp", type=float, default=6.0, help="Vehicle pulse amplitude.")
    parser.add_argument("--sigma-seconds", type=float, default=0.25, help="Vehicle pulse sigma in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=0.83, help="Forward-direction ratio.")
    parser.add_argument("--min-visible-channels", type=int, default=4, help="Minimum visible channels per vehicle.")
    parser.add_argument("--motion-mix", default="constant_sparse,smooth_random,stop_go", help="Vehicle motion model mix.")
    parser.add_argument("--motion-weights", default="0.84,0.15,0.01", help="Comma-separated motion weights.")
    parser.add_argument("--constant-perturb-prob", type=float, default=0.05, help="Sparse local speed perturbation probability.")
    parser.add_argument("--constant-perturb-max-frac", type=float, default=0.01, help="Sparse perturbation magnitude fraction.")
    parser.add_argument("--constant-perturb-width-min", type=int, default=1, help="Sparse perturbation minimum channel width.")
    parser.add_argument("--constant-perturb-width-max", type=int, default=2, help="Sparse perturbation maximum channel width.")
    parser.add_argument("--smooth-speed-max-frac", type=float, default=0.05, help="Smooth speed variation maximum fraction.")
    parser.add_argument("--smooth-speed-corr-channels", type=float, default=8.0, help="Smooth variation correlation width in channels.")
    parser.add_argument("--stop-duration-min-s", type=float, default=1.0, help="Minimum stop-go stop duration.")
    parser.add_argument("--stop-duration-max-s", type=float, default=8.0, help="Maximum stop-go stop duration.")
    parser.add_argument("--stop-channel-width-min", type=int, default=1, help="Minimum stop-go affected width.")
    parser.add_argument("--stop-channel-width-max", type=int, default=3, help="Maximum stop-go affected width.")
    parser.add_argument("--stop-response-sigma-scale", type=float, default=3.0, help="Stop-go sigma scale near stop area.")
    parser.add_argument("--stop-response-amp-scale", type=float, default=1.2, help="Stop-go amplitude scale near stop area.")
    parser.add_argument("--restart-speed-ratio-min", type=float, default=0.95, help="Minimum restart speed ratio.")
    parser.add_argument("--restart-speed-ratio-max", type=float, default=1.05, help="Maximum restart speed ratio.")

    parser.add_argument("--isolated-noise-ratio", type=float, default=0.85, help="Probability of adding extra isolated Gaussian artifacts after overlay.")
    parser.add_argument("--isolated-noise-rate", type=float, default=220.0, help="Expected isolated artifact count per window when enabled.")
    parser.add_argument("--isolated-noise-amp-min", type=float, default=0.6, help="Minimum isolated artifact amplitude.")
    parser.add_argument("--isolated-noise-amp-max", type=float, default=5.5, help="Maximum isolated artifact amplitude.")
    parser.add_argument("--isolated-noise-sigma-min-s", type=float, default=0.04, help="Minimum isolated artifact sigma (s).")
    parser.add_argument("--isolated-noise-sigma-max-s", type=float, default=0.22, help="Maximum isolated artifact sigma (s).")

    parser.add_argument("--random-dead-channel-ratio", type=float, default=0.85, help="Extra random dead-channel probability on top of real background.")
    parser.add_argument("--random-dead-channel-min", type=int, default=4, help="Minimum extra dead channels.")
    parser.add_argument("--random-dead-channel-max", type=int, default=12, help="Maximum extra dead channels.")
    parser.add_argument("--dead-channel-indices", default="", help="Optional fixed extra dead channels, comma-separated.")
    parser.add_argument("--per-vehicle-drop-channel-ratio", type=float, default=1.0, help="Probability of dropping channels independently for each vehicle track.")
    parser.add_argument("--per-vehicle-drop-channel-min", type=int, default=6, help="Minimum dropped channels per vehicle when enabled.")
    parser.add_argument("--per-vehicle-drop-channel-max", type=int, default=10, help="Maximum dropped channels per vehicle when enabled.")
    parser.add_argument("--zero-background-ratio", type=float, default=0.9, help="Probability of adding extra missing blocks on top of real background.")
    parser.add_argument("--zero-background-rate", type=float, default=32.0, help="Expected extra missing block count when enabled.")
    parser.add_argument("--zero-background-channel-min", type=int, default=1, help="Minimum extra missing-block channel width.")
    parser.add_argument("--zero-background-channel-max", type=int, default=3, help="Maximum extra missing-block channel width.")
    parser.add_argument("--zero-background-duration-min-s", type=float, default=0.6, help="Minimum extra missing-block duration.")
    parser.add_argument("--zero-background-duration-max-s", type=float, default=4.0, help="Maximum extra missing-block duration.")

    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--input-mode", choices=["raw", "raw_abs"], default="raw", help="Model input mode.")
    parser.add_argument("--x-dtype", choices=["float16", "float32"], default="float16", help="Stored x tensor dtype.")
    return parser.parse_args()


def _load_background(data_path: Path, *, layout: str, channel_start: int, channel_count: int) -> np.ndarray:
    arr = np.load(str(data_path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D input array, got shape {arr.shape}")
    end = int(channel_start + channel_count)
    if layout == "time_channel":
        if channel_start < 0 or end > int(arr.shape[1]):
            raise ValueError(f"Channel slice [{channel_start}, {end}) outside source shape {arr.shape}")
        view = np.array(arr[:, channel_start:end], dtype=np.float32, copy=True)
    else:
        if channel_start < 0 or end > int(arr.shape[0]):
            raise ValueError(f"Channel slice [{channel_start}, {end}) outside source shape {arr.shape}")
        view = np.array(arr[channel_start:end, :].T, dtype=np.float32, copy=True)
    return np.nan_to_num(view, copy=False)


def _candidate_window_starts(n_time: int, window_samples: int, stride_samples: int) -> list[int]:
    if window_samples <= 0 or window_samples > n_time:
        raise ValueError(f"window_samples={window_samples} outside n_time={n_time}")
    starts = list(range(0, max(1, n_time - window_samples + 1), max(1, stride_samples)))
    last = int(n_time - window_samples)
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(item) for item in starts))


def _rand_uniform(gen: torch.Generator, lo: float, hi: float) -> float:
    if hi <= lo:
        return float(lo)
    return float(lo + torch.rand((), generator=gen).item() * (hi - lo))


def _parse_motion_weights(text: str, motion_models: list[str]) -> list[float]:
    raw = [item.strip() for item in str(text).split(",") if item.strip()]
    if len(raw) != len(motion_models):
        raise ValueError("motion weights count must match motion-mix count")
    weights = [float(item) for item in raw]
    total = sum(max(0.0, value) for value in weights)
    if total <= 0:
        raise ValueError("motion weights sum must be > 0")
    return [float(max(0.0, value) / total) for value in weights]


def _is_default_arg(args: argparse.Namespace, name: str) -> bool:
    if name not in ARG_DEFAULTS:
        return False
    current = getattr(args, name)
    default = ARG_DEFAULTS[name]
    if isinstance(default, float):
        return abs(float(current) - float(default)) <= 1e-9
    return current == default


def _blend_value(default_value: Any, profile_value: Any, strength: float) -> Any:
    if isinstance(default_value, str):
        return profile_value
    if isinstance(default_value, int) and not isinstance(default_value, bool):
        blended = float(default_value) + float(strength) * (float(profile_value) - float(default_value))
        return int(round(blended))
    if isinstance(default_value, float):
        return float(default_value) + float(strength) * (float(profile_value) - float(default_value))
    return profile_value


def _load_realism_profile(profile_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(profile_path).expanduser().read_text(encoding="utf-8"))
    if str(payload.get("format", "")) != "realism_profile_v1":
        raise ValueError(f"Unsupported realism profile format in {profile_path}")
    return payload


def _apply_profile_defaults(args: argparse.Namespace, profile: dict[str, Any]) -> dict[str, Any]:
    if profile is None:
        return {}
    generator_defaults = dict(profile.get("generator_defaults", {}))
    applied: dict[str, Any] = {}
    strength = float(min(1.0, max(0.0, args.profile_strength)))
    for key, profile_value in generator_defaults.items():
        if key not in ARG_DEFAULTS:
            continue
        in_vehicle = key in PROFILE_VEHICLE_KEYS
        in_artifact = key in PROFILE_ARTEFACT_KEYS
        in_background = key in PROFILE_BACKGROUND_KEYS
        allow = False
        if in_background:
            allow = True
        elif args.artifact_policy == "profile_matched":
            allow = in_vehicle or in_artifact
        elif args.artifact_policy == "hybrid":
            allow = in_vehicle or in_artifact
        elif args.artifact_policy == "manual":
            allow = False
        if not allow or not _is_default_arg(args, key):
            continue
        base_default = ARG_DEFAULTS[key]
        if key in PROFILE_NUMERIC_KEYS:
            new_value = _blend_value(base_default, profile_value, strength)
        else:
            new_value = profile_value
        setattr(args, key, new_value)
        applied[key] = new_value
    return applied


def _resolve_profile_window_weights(
    profile: dict[str, Any] | None,
    *,
    window_starts: list[int],
    fs: float,
) -> tuple[list[float] | None, dict[int, dict[str, Any]]]:
    if profile is None:
        return None, {}
    entries = [dict(item) for item in profile.get("window_catalog", {}).get("entries", [])]
    if not entries:
        return None, {}
    by_start: dict[int, dict[str, Any]] = {}
    for item in entries:
        start_sample = int(item.get("start_sample", -1))
        if start_sample >= 0:
            by_start[start_sample] = item
            continue
        start_seconds = float(item.get("start_seconds", -1.0))
        if start_seconds >= 0.0:
            by_start[int(round(start_seconds * float(fs)))] = item
    weights: list[float] = []
    for start in window_starts:
        entry = by_start.get(int(start))
        weight = float(entry.get("sampling_weight", 1.0)) if entry is not None else 1.0
        if not math.isfinite(weight) or weight <= 0.0:
            weight = 1.0
        weights.append(weight)
    if not weights or max(weights) == min(weights) == 1.0:
        return None, by_start
    return weights, by_start


def _parse_dead_channel_indices(text: str, n_ch: int) -> list[int]:
    values: list[int] = []
    for item in _split_csv(text):
        try:
            idx = int(item)
        except ValueError:
            continue
        if 0 <= idx < int(n_ch):
            values.append(idx)
    return sorted(set(values))


def _resolve_profile_structures(profile: dict[str, Any] | None) -> dict[str, Any]:
    if profile is None:
        return {
            "stable_dead_channels": [],
            "probabilistic_dead_channels": [],
            "target_peak_total_q50": float("nan"),
            "target_peaks_per_channel_q50": float("nan"),
        }
    zero = profile.get("zero_components", {})
    peaks = profile.get("peak_density", {})
    return {
        "stable_dead_channels": [int(v) for v in zero.get("stable_dead_channel_indices", zero.get("dead_channel_indices", []))],
        "probabilistic_dead_channels": [
            {
                "channel": int(item.get("channel", -1)),
                "probability": float(item.get("probability", 0.0)),
            }
            for item in zero.get("probabilistic_dead_channels", [])
            if isinstance(item, dict)
        ],
        "target_peak_total_q50": float(peaks.get("total_peaks_per_window", {}).get("q50", float("nan"))),
        "target_peaks_per_channel_q50": float(peaks.get("peaks_per_channel_per_window", {}).get("q50", float("nan"))),
    }


def _profile_density_scale(window_meta: dict[str, Any] | None, args: argparse.Namespace) -> float:
    if not window_meta:
        return 1.0
    base_total = float(window_meta.get("total_peaks", float("nan")))
    target_total = float(getattr(args, "profile_target_peak_total_q50", float("nan")))
    if not (math.isfinite(base_total) and math.isfinite(target_total) and target_total > 0.0):
        return 1.0
    ratio = base_total / max(1e-6, target_total)
    if ratio >= 1.10:
        return max(0.10, min(1.0, 1.0 / ratio))
    if ratio <= 0.75:
        return min(1.0, 0.85 + 0.2 * (1.0 - ratio))
    return 1.0


def _effective_vehicle_range(args: argparse.Namespace, window_meta: dict[str, Any] | None) -> tuple[int, int]:
    base_min = int(args.vehicles_min)
    base_max = int(args.vehicles_max)
    if base_max < base_min:
        base_max = base_min
    if not window_meta:
        return base_min, base_max
    density_scale = _profile_density_scale(window_meta, args)
    if density_scale >= 0.95:
        return base_min, base_max
    if density_scale <= 0.35:
        return max(0, base_min - 1), max(0, base_max - 1)
    return base_min, max(base_min, base_max - 1)


def _channel_activity_weights(data_ds: torch.Tensor, stable_dead_channels: list[int]) -> torch.Tensor:
    weights = torch.mean(torch.abs(data_ds), dim=1)
    weights = torch.clamp(weights, min=0.0)
    if stable_dead_channels:
        weights[stable_dead_channels] = 0.0
    if float(torch.sum(weights).item()) <= 1e-9:
        weights = torch.ones((int(data_ds.shape[0]),), dtype=torch.float32)
        if stable_dead_channels:
            weights[stable_dead_channels] = 0.0
    if float(torch.sum(weights).item()) <= 1e-9:
        weights = torch.ones((int(data_ds.shape[0]),), dtype=torch.float32)
    return weights


def _time_activity_weights(data_ds: torch.Tensor) -> torch.Tensor:
    weights = torch.mean(torch.abs(data_ds), dim=0)
    if int(weights.numel()) <= 1:
        return torch.ones_like(weights)
    kernel_width = min(401, int(weights.numel()))
    if kernel_width % 2 == 0:
        kernel_width = max(1, kernel_width - 1)
    if kernel_width > 1:
        kernel = torch.ones((1, 1, kernel_width), dtype=torch.float32) / float(kernel_width)
        padded = torch.nn.functional.pad(weights.view(1, 1, -1), (kernel_width // 2, kernel_width // 2), mode="replicate")
        weights = torch.nn.functional.conv1d(padded, kernel).view(-1)[: int(data_ds.shape[1])]
    if float(torch.sum(weights).item()) <= 1e-9:
        weights = torch.ones_like(weights)
    return weights


def _sample_from_weights(gen: torch.Generator, weights: torch.Tensor) -> int:
    positive = torch.clamp(weights.to(torch.float32), min=0.0)
    total = float(torch.sum(positive).item())
    if not math.isfinite(total) or total <= 0.0:
        return int(torch.randint(0, int(weights.numel()), (1,), generator=gen).item())
    probs = positive / total
    return int(torch.multinomial(probs, 1, generator=gen).item())


def _add_isolated_noise_structured(
    args: argparse.Namespace,
    gen: torch.Generator,
    data_ds: torch.Tensor,
    t_axis_s: torch.Tensor,
    *,
    stable_dead_channels: list[int],
    window_meta: dict[str, Any] | None,
) -> None:
    density_scale = _profile_density_scale(window_meta, args)
    rate = max(0.0, float(args.isolated_noise_rate) * float(density_scale))
    count = int(rate)
    if torch.rand((), generator=gen).item() < rate - count:
        count += 1
    if count <= 0:
        return
    channel_weights = _channel_activity_weights(data_ds, stable_dead_channels)
    time_weights = _time_activity_weights(data_ds)
    n_ch = int(data_ds.shape[0])
    for _ in range(count):
        if torch.rand((), generator=gen).item() < 0.85:
            ch = _sample_from_weights(gen, channel_weights)
        else:
            candidates = [idx for idx in range(n_ch) if idx not in set(stable_dead_channels)]
            if not candidates:
                candidates = list(range(n_ch))
            ch = int(candidates[int(torch.randint(0, len(candidates), (1,), generator=gen).item())])
        if torch.rand((), generator=gen).item() < 0.80:
            center_idx = _sample_from_weights(gen, time_weights)
            center = float(t_axis_s[max(0, min(int(center_idx), int(t_axis_s.numel()) - 1))].item())
            center += _rand_uniform(gen, -0.35, 0.35)
            center = max(0.0, min(float(args.window_seconds), center))
        else:
            center = _rand_uniform(gen, 0.0, float(args.window_seconds))
        sigma = _rand_uniform(gen, float(args.isolated_noise_sigma_min_s), float(args.isolated_noise_sigma_max_s))
        amp = _rand_uniform(gen, float(args.isolated_noise_amp_min), float(args.isolated_noise_amp_max))
        data_ds[ch] += float(amp) * torch.exp(-0.5 * ((t_axis_s - float(center)) / max(1e-6, float(sigma))) ** 2)


def _sample_dead_channels_structured(
    args: argparse.Namespace,
    gen: torch.Generator,
    n_ch: int,
    *,
    stable_dead_channels: list[int],
    probabilistic_dead_channels: list[dict[str, float]],
) -> list[int]:
    dead = set(int(idx) for idx in stable_dead_channels if 0 <= int(idx) < int(n_ch))
    dead.update(_parse_dead_channel_indices(str(args.dead_channel_indices), n_ch))
    for item in probabilistic_dead_channels:
        ch = int(item.get("channel", -1))
        prob = float(item.get("probability", 0.0))
        if 0 <= ch < int(n_ch) and prob > 0.0 and torch.rand((), generator=gen).item() < min(1.0, max(0.0, prob)):
            dead.add(ch)
    if float(args.random_dead_channel_ratio) <= 0.0 or torch.rand((), generator=gen).item() >= float(args.random_dead_channel_ratio):
        return sorted(dead)
    count_min = int(max(0, args.random_dead_channel_min))
    count_max = int(max(count_min, args.random_dead_channel_max))
    if count_max <= 0:
        return sorted(dead)
    count = int(torch.randint(count_min, count_max + 1, (1,), generator=gen).item())
    count = min(count, int(n_ch) - len(dead))
    if count <= 0:
        return sorted(dead)

    stable_sorted = sorted(dead)
    candidate_pool: list[int] = []
    for idx in stable_sorted:
        for delta in (-1, 1, -2, 2):
            cand = int(idx + delta)
            if 0 <= cand < int(n_ch) and cand not in dead and cand not in candidate_pool:
                candidate_pool.append(cand)
    fallback_pool = [idx for idx in range(int(n_ch)) if idx not in dead and idx not in candidate_pool]
    ordered: list[int] = []
    if candidate_pool:
        perm = torch.randperm(len(candidate_pool), generator=gen).tolist()
        ordered.extend(candidate_pool[idx] for idx in perm)
    if fallback_pool:
        perm = torch.randperm(len(fallback_pool), generator=gen).tolist()
        ordered.extend(fallback_pool[idx] for idx in perm)
    for idx in ordered[:count]:
        dead.add(int(idx))
    return sorted(dead)


def _apply_missing_blocks_structured(
    args: argparse.Namespace,
    gen: torch.Generator,
    data_ds: torch.Tensor,
    time_label: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    stable_dead_channels: list[int],
    window_meta: dict[str, Any] | None,
) -> None:
    if float(args.zero_background_ratio) <= 0.0:
        return
    if torch.rand((), generator=gen).item() >= float(args.zero_background_ratio):
        return
    density_scale = _profile_density_scale(window_meta, args)
    rate = max(0.0, float(args.zero_background_rate) * float(density_scale))
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
    stable_set = sorted(set(int(v) for v in stable_dead_channels if 0 <= int(v) < int(n_ch)))
    time_weights = _time_activity_weights(data_ds)
    for _ in range(count):
        width_ch = int(torch.randint(ch_min, ch_max + 1, (1,), generator=gen).item())
        width_ch = max(1, min(width_ch, n_ch))
        if stable_set and torch.rand((), generator=gen).item() < 0.75:
            anchor = int(stable_set[int(torch.randint(0, len(stable_set), (1,), generator=gen).item())])
            start_ch = max(0, min(int(n_ch - width_ch), anchor - width_ch // 2))
        else:
            start_ch = int(torch.randint(0, max(1, n_ch - width_ch + 1), (1,), generator=gen).item())
        duration_s = _rand_uniform(gen, dur_min, dur_max)
        width_t = int(max(1, round(duration_s * float(args.fs) / float(max(1, args.time_downsample)))))
        width_t = max(1, min(width_t, t_down))
        if torch.rand((), generator=gen).item() < 0.80:
            center_idx = _sample_from_weights(gen, time_weights)
            start_t = max(0, min(int(t_down - width_t), int(center_idx) - width_t // 2))
        else:
            start_t = int(torch.randint(0, max(1, t_down - width_t + 1), (1,), generator=gen).item())
        end_ch = start_ch + width_ch
        end_t = start_t + width_t
        data_ds[start_ch:end_ch, start_t:end_t] = 0.0
        channels = torch.arange(start_ch, end_ch, dtype=torch.long)
        if channels.numel() <= 0 or visibility.numel() <= 0:
            continue
        t_down_idx = torch.round(
            time_label[:, channels] * float(max(1, int(data_ds.shape[1]) - 1))
        ).to(torch.long)
        missing = (t_down_idx >= int(start_t)) & (t_down_idx < int(end_t)) & (visibility[:, channels] > 0.5)
        if bool(missing.any().item()):
            visibility[:, channels] = torch.where(missing, torch.zeros_like(visibility[:, channels]), visibility[:, channels])
    if visibility.numel() > 0:
        gt_valid &= visibility.sum(dim=1) >= int(args.min_visible_channels)


def _sample_weighted_index(gen: torch.Generator, weights: list[float] | None, count: int) -> int:
    if count <= 0:
        raise ValueError("count must be > 0")
    if weights is None:
        return int(torch.randint(0, count, (1,), generator=gen).item())
    tensor = torch.tensor(weights, dtype=torch.float32)
    total = float(torch.sum(tensor).item())
    if not math.isfinite(total) or total <= 0.0:
        return int(torch.randint(0, count, (1,), generator=gen).item())
    probs = tensor / total
    return int(torch.multinomial(probs, 1, generator=gen).item())


def _drop_channels_for_one_vehicle(
    args: argparse.Namespace,
    gen: torch.Generator,
    visible: torch.Tensor,
) -> torch.Tensor:
    kept = visible.clone()
    if float(args.per_vehicle_drop_channel_ratio) <= 0.0:
        return kept
    if torch.rand((), generator=gen).item() >= float(args.per_vehicle_drop_channel_ratio):
        return kept
    visible_idx = torch.where(visible)[0]
    n_visible = int(visible_idx.numel())
    if n_visible <= int(args.min_visible_channels):
        return kept
    drop_min = int(max(0, args.per_vehicle_drop_channel_min))
    drop_max = int(max(drop_min, args.per_vehicle_drop_channel_max))
    max_drop = max(0, n_visible - int(args.min_visible_channels))
    if max_drop <= 0:
        return kept
    drop_count = min(int(torch.randint(drop_min, drop_max + 1, (1,), generator=gen).item()), max_drop)
    if drop_count <= 0:
        return kept
    order = torch.randperm(n_visible, generator=gen)
    drop_idx = visible_idx[order[:drop_count]]
    kept[drop_idx] = False
    return kept


def _add_track_to_sample(
    args: argparse.Namespace,
    gen: torch.Generator,
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
    visible = (t_center >= 0.0) & (t_center < float(args.window_seconds))
    visible = _drop_channels_for_one_vehicle(args, gen, visible)
    if int(visible.sum().item()) < int(args.min_visible_channels):
        return False
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    center_idx = torch.round(t_center * float(args.fs)).to(torch.long).clamp(0, window_samples - 1)
    for ch in torch.where(visible)[0].tolist():
        pulse = amp * torch.exp(-0.5 * ((t_axis_s - float(t_center[ch].item())) / max(1e-6, sigma_s)) ** 2)
        data_ds[int(ch)] += pulse
    time_label[track_id, visible] = (
        center_idx[visible].to(torch.float32) / float(max(1, window_samples - 1))
    ).clamp(0.0, 1.0)
    visibility[track_id] = visible.to(torch.float32)
    direction[track_id] = int(direction_label)
    speed[track_id] = float(speed_kmh / max(1e-6, float(args.speed_norm_kmh)))
    gt_valid[track_id] = True
    return True


def _generate_one(
    args: argparse.Namespace,
    *,
    index: int,
    real_bg: np.ndarray,
    window_starts: list[int],
    window_weights: list[float] | None,
    profile_windows: dict[int, dict[str, Any]],
) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(args.seed) + int(index) * 1_000_003)
    bg_rng = np.random.default_rng(int(args.seed) + int(index) * 2_000_003)
    n_ch = int(args.channel_count)
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    time_downsample = int(max(1, args.time_downsample))
    sampled_idx = _sample_weighted_index(gen, window_weights if args.window_sampler == "profile_weighted" else None, len(window_starts))
    start = int(window_starts[sampled_idx])
    _window_meta = profile_windows.get(start)
    stable_dead_channels = [int(v) for v in getattr(args, "profile_stable_dead_channels", [])]
    probabilistic_dead_channels = [dict(item) for item in getattr(args, "profile_probabilistic_dead_channels", [])]
    window = np.array(real_bg[start : start + window_samples, :], dtype=np.float32, copy=True).T
    scale = _rand_uniform(gen, float(args.background_scale_min), float(args.background_scale_max))
    window *= float(scale)
    if float(args.background_offset_std) > 0.0:
        window += float(bg_rng.normal(loc=0.0, scale=float(args.background_offset_std)))
    window = np.nan_to_num(window, copy=False)
    data_ds = torch.from_numpy(window[:, ::time_downsample].copy()).to(torch.float32)
    t_axis_s = torch.arange(int(data_ds.shape[1]), dtype=torch.float32) * (float(time_downsample) / float(args.fs))

    max_gt = int(max(args.vehicles_min, args.vehicles_max))
    time_label = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    visibility = torch.zeros((max_gt, n_ch), dtype=torch.float32)
    direction = torch.zeros((max_gt,), dtype=torch.long)
    speed = torch.zeros((max_gt,), dtype=torch.float32)
    gt_valid = torch.zeros((max_gt,), dtype=torch.bool)

    vehicle_min, vehicle_max = _effective_vehicle_range(args, _window_meta)
    if vehicle_max < vehicle_min:
        vehicle_max = vehicle_min
    n_vehicles = int(torch.randint(int(vehicle_min), int(vehicle_max) + 1, (1,), generator=gen).item())
    track_id = 0
    attempts = 0
    max_attempts = max(64, n_vehicles * 64)
    while track_id < n_vehicles and attempts < max_attempts:
        attempts += 1
        is_primary = bool(torch.rand((), generator=gen).item() < float(args.primary_ratio))
        direction_label = 0 if is_primary else 1
        speed_kmh = _rand_uniform(gen, float(args.speed_min_kmh), float(args.speed_max_kmh))
        anchor_ch = int(torch.randint(0, n_ch, (1,), generator=gen).item())
        anchor_time = float(torch.rand((), generator=gen).item() * float(args.window_seconds))
        t_center, _, effective_speed_kmh, _, _ = _sample_track_times(
            args,
            gen,
            n_ch,
            is_primary,
            speed_kmh,
            anchor_ch,
            anchor_time,
        )
        added = _add_track_to_sample(
            args,
            gen,
            data_ds=data_ds,
            t_axis_s=t_axis_s,
            time_label=time_label,
            visibility=visibility,
            direction=direction,
            speed=speed,
            gt_valid=gt_valid,
            track_id=track_id,
            t_center=t_center,
            direction_label=int(direction_label),
            speed_kmh=float(effective_speed_kmh),
            amp=float(args.fixed_amp),
            sigma_s=float(args.sigma_seconds),
        )
        if added:
            track_id += 1

    if float(args.isolated_noise_ratio) > 0.0 and torch.rand((), generator=gen).item() < float(args.isolated_noise_ratio):
        _add_isolated_noise_structured(
            args,
            gen,
            data_ds,
            t_axis_s,
            stable_dead_channels=stable_dead_channels,
            window_meta=_window_meta,
        )
    if float(args.zero_background_ratio) > 0.0:
        _apply_missing_blocks_structured(
            args,
            gen,
            data_ds,
            time_label,
            visibility,
            gt_valid,
            stable_dead_channels=stable_dead_channels,
            window_meta=_window_meta,
        )
    _apply_channel_gain(args, gen, data_ds)
    _apply_dead_channels(
        data_ds,
        visibility,
        gt_valid,
        _sample_dead_channels_structured(
            args,
            gen,
            n_ch,
            stable_dead_channels=stable_dead_channels,
            probabilistic_dead_channels=probabilistic_dead_channels,
        ),
        min_visible_channels=int(args.min_visible_channels),
    )
    x = _prepare_input(data_ds, clip_ratio=float(args.clip_ratio), input_mode=str(args.input_mode))
    if str(args.x_dtype) == "float16":
        x = x.to(torch.float16)
    result = {
        "x": x.contiguous(),
        "time": time_label.contiguous(),
        "visibility": visibility.contiguous(),
        "direction": direction.contiguous(),
        "speed": speed.contiguous(),
        "gt_valid": gt_valid.contiguous(),
    }
    if _window_meta is not None:
        result["profile_window_start_sample"] = torch.tensor(int(start), dtype=torch.long)
    return result


def main() -> int:
    args = parse_args()
    if int(args.num_samples) <= 0:
        raise ValueError("--num-samples must be > 0")
    if int(args.shard_size) <= 0:
        raise ValueError("--shard-size must be > 0")
    if int(args.channel_count) <= 0:
        raise ValueError("--channel-count must be > 0")
    if int(args.vehicles_min) < 0 or int(args.vehicles_max) < int(args.vehicles_min):
        raise ValueError("--vehicles-max must be >= --vehicles-min >= 0")
    if float(args.background_scale_min) <= 0.0 or float(args.background_scale_max) < float(args.background_scale_min):
        raise ValueError("background scale range must satisfy 0 < min <= max")
    if not 0.0 <= float(args.per_vehicle_drop_channel_ratio) <= 1.0:
        raise ValueError("--per-vehicle-drop-channel-ratio must be in [0, 1]")
    if int(args.per_vehicle_drop_channel_min) < 0 or int(args.per_vehicle_drop_channel_max) < int(args.per_vehicle_drop_channel_min):
        raise ValueError("per-vehicle drop channel counts must be non-negative and ordered")
    if not 0.0 <= float(args.profile_strength) <= 1.0:
        raise ValueError("--profile-strength must be in [0, 1]")
    args.n_ch = int(args.channel_count)
    args.amp_min = float(args.fixed_amp)
    args.amp_max = float(args.fixed_amp)
    args.sigma_min_s = float(args.sigma_seconds)
    args.sigma_max_s = float(args.sigma_seconds)
    args.noise_std = 0.0
    args.colored_noise_std = 0.0
    args.colored_noise_corr_s = 0.8
    args.channel_bias_std = 0.0
    args.channel_gain_std = 0.0
    args.baseline_drift_std = 0.0
    args.baseline_drift_corr_s = 6.0
    args.interaction_ratio = 0.0
    args.interaction_types = "crossing,overtake,near_parallel"
    args.interaction_time_min_frac = 0.05
    args.interaction_time_max_frac = 0.95

    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not bool(args.overwrite):
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = _load_realism_profile(Path(args.profile).expanduser()) if args.profile is not None else None
    applied_profile_defaults = _apply_profile_defaults(args, profile)
    profile_structures = _resolve_profile_structures(profile)
    args.profile_stable_dead_channels = list(profile_structures["stable_dead_channels"])
    args.profile_probabilistic_dead_channels = list(profile_structures["probabilistic_dead_channels"])
    args.profile_target_peak_total_q50 = float(profile_structures["target_peak_total_q50"])
    args.profile_target_peaks_per_channel_q50 = float(profile_structures["target_peaks_per_channel_q50"])

    real_bg = _load_background(
        input_path,
        layout=str(args.array_layout),
        channel_start=int(args.channel_start),
        channel_count=int(args.channel_count),
    )
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    stride_samples = int(round(float(args.window_stride_seconds) * float(args.fs)))
    window_starts = _candidate_window_starts(int(real_bg.shape[0]), window_samples, stride_samples)
    motion_models = _split_csv(str(args.motion_mix))
    _ = _parse_motion_weights(str(args.motion_weights), motion_models)
    window_weights, profile_windows = _resolve_profile_window_weights(profile, window_starts=window_starts, fs=float(args.fs))

    shard_size = int(args.shard_size)
    total = int(args.num_samples)
    shard_count = int(math.ceil(total / shard_size))
    shard_names: list[str] = []
    t0 = time.perf_counter()
    done = 0
    for shard_idx in range(shard_count):
        start = shard_idx * shard_size
        end = min(total, (shard_idx + 1) * shard_size)
        items = [
            _generate_one(
                args,
                index=index,
                real_bg=real_bg,
                window_starts=window_starts,
                window_weights=window_weights,
                profile_windows=profile_windows,
            )
            for index in range(start, end)
        ]
        shard_name = f"shard_{int(shard_idx):06d}.pt"
        _write_shard(out_dir / shard_name, items)
        shard_names.append(shard_name)
        done += len(items)
        print(f"wrote {shard_name}: total={done}/{total}, elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    meta = {
        "format": "track_slot_shards_v1",
        "realism_preset": "real_npy_background_mix",
        "created_at_unix": time.time(),
        "num_samples": total,
        "shard_size": shard_size,
        "shards": shard_names,
        "n_channels": int(args.channel_count),
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
        "background_source": {
            "input": str(input_path),
            "array_layout": str(args.array_layout),
            "channel_start": int(args.channel_start),
            "channel_count": int(args.channel_count),
            "candidate_window_count": int(len(window_starts)),
            "window_stride_seconds": float(args.window_stride_seconds),
        },
        "profile_mode": {
            "profile": str(Path(args.profile).expanduser()) if args.profile is not None else None,
            "profile_strength": float(args.profile_strength),
            "window_sampler": str(args.window_sampler),
            "artifact_policy": str(args.artifact_policy),
            "applied_profile_defaults": applied_profile_defaults,
            "profile_weighted_window_count": int(len(profile_windows)),
            "stable_dead_channels": list(args.profile_stable_dead_channels),
            "probabilistic_dead_channels": list(args.profile_probabilistic_dead_channels),
            "target_peak_total_q50": args.profile_target_peak_total_q50,
            "target_peaks_per_channel_q50": args.profile_target_peaks_per_channel_q50,
        },
        "generator_args": _args_payload(args),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={total}, shards={len(shard_names)}, out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
