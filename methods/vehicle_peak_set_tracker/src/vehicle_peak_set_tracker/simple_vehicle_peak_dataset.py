from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.signal import find_peaks
from torch.utils.data import Dataset


@dataclass
class SimplePeakSetDatasetConfig:
    length: int = 1024
    n_channels: int = 50
    fs: float = 1000.0
    window_seconds: float = 60.0
    time_downsample: int = 10
    dx_m: float = 20.0
    vehicles_min: int = 4
    vehicles_max: int = 12
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 90.0
    noise_std: float = 0.05
    amp_min: float = 4.0
    amp_max: float = 8.0
    sigma_min_s: float = 0.05
    sigma_max_s: float = 0.09
    min_visible_channels: int = 5
    primary_ratio: float = 0.82
    same_direction_ratio: float = 0.82
    crossing_ratio: float = 0.18
    speed_jitter_ratio: float = 0.06
    scene_cluster_ratio: float = 0.35
    scene_cluster_time_jitter_s: float = 4.0
    scene_cluster_channel_jitter: int = 7
    motion_mix: str = "constant_sparse,smooth_random,stop_go"
    motion_weights: str = "0.84,0.15,0.01"
    constant_perturb_prob: float = 0.05
    constant_perturb_max_frac: float = 0.01
    constant_perturb_width_min: int = 1
    constant_perturb_width_max: int = 2
    smooth_speed_max_frac: float = 0.05
    smooth_speed_corr_channels: int = 8
    track_time_jitter_max_s: float = 0.0
    track_time_jitter_corr_channels: int = 6
    track_time_jitter_min_gap_ratio: float = 0.35
    stop_duration_min_s: float = 1.0
    stop_duration_max_s: float = 8.0
    stop_channel_width_min: int = 1
    stop_channel_width_max: int = 3
    stop_response_sigma_scale: float = 3.0
    stop_response_amp_scale: float = 1.2
    restart_speed_ratio_min: float = 0.95
    restart_speed_ratio_max: float = 1.05
    dead_channel_indices: str = ""
    intermittent_dead_channel_rates: str = ""
    random_dead_channel_ratio: float = 0.18
    random_dead_channel_min: int = 2
    random_dead_channel_max: int = 6
    zero_background_ratio: float = 0.65
    zero_background_rate: float = 6.0
    zero_background_channel_min: int = 1
    zero_background_channel_max: int = 4
    zero_background_duration_min_s: float = 0.4
    zero_background_duration_max_s: float = 3.5
    per_vehicle_drop_channel_ratio: float = 1.00
    per_vehicle_drop_channel_min: int = 5
    per_vehicle_drop_channel_max: int = 14
    missing_random_ratio_min: float = 0.20
    missing_random_ratio_max: float = 0.45
    missing_segment_count_max: int = 4
    missing_segment_min_len: int = 2
    missing_segment_max_len: int = 8
    interaction_ratio: float = 0.55
    interaction_types: str = "parallel_crossing,overtake,crossing,near_parallel"
    interaction_time_min_frac: float = 0.05
    interaction_time_max_frac: float = 0.95
    isolated_noise_ratio: float = 0.80
    isolated_noise_rate: float = 24.0
    isolated_noise_amp_min: float = 0.6
    isolated_noise_amp_max: float = 4.0
    isolated_noise_sigma_min_s: float = 0.04
    isolated_noise_sigma_max_s: float = 0.18
    clip_ratio: float = 1.35
    input_scale: float = 0.0
    input_mode: str = "raw"
    speed_norm_kmh: float = 150.0
    seed: int = 42
    return_raw_window: bool = False


def _robust_scale(data: np.ndarray) -> float:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite.size == 0:
        return 1.0
    abs_vals = np.abs(finite)
    q995 = float(np.quantile(abs_vals, 0.995))
    rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
    return max(q995, 3.0 * rms, 1e-6)


def prepare_peakset_input(
    data_window: np.ndarray,
    time_downsample: int,
    clip_ratio: float = 1.35,
    input_mode: str = "raw",
    input_scale: float = 0.0,
) -> torch.Tensor:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = float(input_scale) if float(input_scale) > 0.0 else _robust_scale(arr_ds)
    scale = max(scale, 1e-6)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    mode = str(input_mode).lower()
    if mode == "raw":
        features = raw[None, :, :].astype(np.float32, copy=False)
    elif mode == "raw_abs":
        abs_feat = np.clip(np.abs(arr_ds) / scale, 0.0, clip) / clip
        features = np.stack([raw, abs_feat], axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}; expected raw or raw_abs")
    return torch.from_numpy(features)


def _select_peak_source(data_window: np.ndarray) -> np.ndarray:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] >= 2 and np.all(arr[1] >= 0.0):
            return arr[0]
        return arr[0]
    raise ValueError("data_window must have shape [n_channel, n_sample] or [in_channels, n_channel, n_sample]")


def detect_peak_candidates(
    data_window: np.ndarray,
    *,
    fs: float,
    time_downsample: int,
    candidates_per_channel: int = 64,
    min_distance_s: float = 0.15,
    min_height: float = 0.02,
    prominence: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    arr = np.asarray(_select_peak_source(data_window), dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("peak candidate detection expects shape [n_channel, n_sample]")
    n_ch, n_t = int(arr.shape[0]), int(arr.shape[1])
    k_count = int(max(1, candidates_per_channel))
    peak_time = torch.zeros((n_ch, k_count), dtype=torch.float32)
    peak_amp = torch.zeros((n_ch, k_count), dtype=torch.float32)
    peak_valid = torch.zeros((n_ch, k_count), dtype=torch.bool)
    peak_index = torch.full((n_ch, k_count), -1, dtype=torch.long)
    distance = int(max(1, round(float(min_distance_s) * float(fs) / float(max(1, time_downsample)))))
    for ch in range(n_ch):
        row = np.abs(arr[ch]).astype(np.float32, copy=False)
        peaks, props = find_peaks(
            row,
            height=float(min_height),
            prominence=float(prominence),
            distance=distance,
        )
        if peaks.size == 0:
            peaks, props = find_peaks(row, distance=distance)
        if peaks.size == 0:
            continue
        amps = row[peaks].astype(np.float32, copy=False)
        prominences = props.get("prominences", amps).astype(np.float32, copy=False)
        score = amps + 0.1 * prominences
        if peaks.size > k_count:
            keep = np.argsort(score)[-k_count:]
            peaks = peaks[keep]
            amps = amps[keep]
        order = np.argsort(peaks)
        peaks = peaks[order]
        amps = amps[order]
        take = min(k_count, int(peaks.size))
        idx = torch.as_tensor(peaks[:take], dtype=torch.long)
        peak_index[ch, :take] = idx
        peak_time[ch, :take] = idx.to(torch.float32) * float(max(1, time_downsample)) / float(max(1, n_t * int(time_downsample) - 1))
        peak_amp[ch, :take] = torch.as_tensor(amps[:take], dtype=torch.float32)
        peak_valid[ch, :take] = True
    return peak_time, peak_amp, peak_valid, peak_index


def _batch_peak_candidates(
    xs: torch.Tensor,
    *,
    fs: float,
    time_downsample: int,
    candidates_per_channel: int = 64,
    min_distance_s: float = 0.15,
    min_height: float = 0.02,
    prominence: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    peak_time_list: list[torch.Tensor] = []
    peak_amp_list: list[torch.Tensor] = []
    peak_valid_list: list[torch.Tensor] = []
    peak_index_list: list[torch.Tensor] = []
    for item in xs:
        peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates(
            item.detach().cpu().numpy(),
            fs=float(fs),
            time_downsample=int(time_downsample),
            candidates_per_channel=int(candidates_per_channel),
            min_distance_s=float(min_distance_s),
            min_height=float(min_height),
            prominence=float(prominence),
        )
        peak_time_list.append(peak_time)
        peak_amp_list.append(peak_amp)
        peak_valid_list.append(peak_valid)
        peak_index_list.append(peak_index)
    return (
        torch.stack(peak_time_list, dim=0),
        torch.stack(peak_amp_list, dim=0),
        torch.stack(peak_valid_list, dim=0),
        torch.stack(peak_index_list, dim=0),
    )


def match_gt_peak_indices(
    targets: dict[str, torch.Tensor],
    peak_time: torch.Tensor,
    peak_valid: torch.Tensor,
    peak_index: torch.Tensor,
    *,
    match_tolerance_s: float,
    fs: float,
    time_downsample: int,
    window_seconds: float,
) -> torch.Tensor:
    if "full_time" not in targets or "observed_visibility" not in targets:
        raise KeyError("full_time and observed_visibility are required to build gt_peak_index")
    full_time = targets["full_time"].to(torch.float32)
    observed = targets["observed_visibility"].to(torch.float32) > 0.5
    if full_time.ndim != 3:
        raise ValueError("full_time must have shape [B, G, C]")
    bsz, max_gt, n_ch = int(full_time.shape[0]), int(full_time.shape[1]), int(full_time.shape[2])
    k_count = int(peak_valid.shape[-1])
    none_index = int(k_count)
    gt_peak_index = torch.full((bsz, max_gt, n_ch), none_index, dtype=torch.long)
    if int(max_gt) == 0:
        return gt_peak_index
    tol_norm = float(match_tolerance_s) / float(max(1e-6, window_seconds))
    for b in range(bsz):
        for g in range(max_gt):
            for ch in range(n_ch):
                if not bool(observed[b, g, ch]):
                    continue
                valid = torch.where(peak_valid[b, ch])[0]
                if valid.numel() == 0:
                    continue
                t_norm = float(full_time[b, g, ch].item())
                cand_norm = peak_time[b, ch, valid].to(torch.float32)
                diffs = torch.abs(cand_norm - float(t_norm))
                best = int(torch.argmin(diffs).item())
                if float(diffs[best]) <= tol_norm:
                    gt_peak_index[b, g, ch] = int(valid[best].item())
    return gt_peak_index


def _time_index_to_norm(t_idx: np.ndarray, window_samples: int) -> np.ndarray:
    denom = float(max(1, window_samples - 1))
    return np.asarray(t_idx, dtype=np.float32) / denom


class SimpleLinearVehiclePeakDataset(Dataset):
    def __init__(
        self,
        *,
        config: Optional[SimplePeakSetDatasetConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = SimplePeakSetDatasetConfig(**kwargs)
        elif kwargs:
            raise TypeError("pass either config or keyword args, not both")
        self.config = config
        self.length = int(config.length)
        self.n_channels = int(config.n_channels)
        self.fs = float(config.fs)
        self.window_seconds = float(config.window_seconds)
        self.window_samples = int(round(self.window_seconds * self.fs))
        self.time_downsample = int(max(1, config.time_downsample))
        self.dx_m = float(config.dx_m)
        self.vehicles_min = int(config.vehicles_min)
        self.vehicles_max = int(max(config.vehicles_min, config.vehicles_max))
        self.speed_min_kmh = float(config.speed_min_kmh)
        self.speed_max_kmh = float(max(config.speed_min_kmh, config.speed_max_kmh))
        self.noise_std = float(config.noise_std)
        self.amp_min = float(config.amp_min)
        self.amp_max = float(max(config.amp_min, config.amp_max))
        self.sigma_min_s = float(config.sigma_min_s)
        self.sigma_max_s = float(max(config.sigma_min_s, config.sigma_max_s))
        self.min_visible_channels = int(config.min_visible_channels)
        self.primary_ratio = float(config.primary_ratio)
        self.same_direction_ratio = float(config.same_direction_ratio)
        self.crossing_ratio = float(config.crossing_ratio)
        self.speed_jitter_ratio = float(config.speed_jitter_ratio)
        self.scene_cluster_ratio = float(config.scene_cluster_ratio)
        self.scene_cluster_time_jitter_s = float(config.scene_cluster_time_jitter_s)
        self.scene_cluster_channel_jitter = int(max(0, config.scene_cluster_channel_jitter))
        self.motion_mix = str(config.motion_mix)
        self.motion_weights = str(config.motion_weights)
        self.constant_perturb_prob = float(config.constant_perturb_prob)
        self.constant_perturb_max_frac = float(config.constant_perturb_max_frac)
        self.constant_perturb_width_min = int(max(1, config.constant_perturb_width_min))
        self.constant_perturb_width_max = int(max(self.constant_perturb_width_min, config.constant_perturb_width_max))
        self.smooth_speed_max_frac = float(config.smooth_speed_max_frac)
        self.smooth_speed_corr_channels = int(max(1, config.smooth_speed_corr_channels))
        self.track_time_jitter_max_s = float(config.track_time_jitter_max_s)
        self.track_time_jitter_corr_channels = int(max(1, config.track_time_jitter_corr_channels))
        self.track_time_jitter_min_gap_ratio = float(config.track_time_jitter_min_gap_ratio)
        self.stop_duration_min_s = float(config.stop_duration_min_s)
        self.stop_duration_max_s = float(max(config.stop_duration_min_s, config.stop_duration_max_s))
        self.stop_channel_width_min = int(max(1, config.stop_channel_width_min))
        self.stop_channel_width_max = int(max(self.stop_channel_width_min, config.stop_channel_width_max))
        self.stop_response_sigma_scale = float(config.stop_response_sigma_scale)
        self.stop_response_amp_scale = float(config.stop_response_amp_scale)
        self.restart_speed_ratio_min = float(config.restart_speed_ratio_min)
        self.restart_speed_ratio_max = float(max(config.restart_speed_ratio_min, config.restart_speed_ratio_max))
        self.dead_channel_indices = str(config.dead_channel_indices)
        self.intermittent_dead_channel_rates = str(config.intermittent_dead_channel_rates)
        self._intermittent_dead_channel_rate_map = self._parse_channel_rate_csv(self.intermittent_dead_channel_rates)
        self.random_dead_channel_ratio = float(config.random_dead_channel_ratio)
        self.random_dead_channel_min = int(max(0, config.random_dead_channel_min))
        self.random_dead_channel_max = int(max(self.random_dead_channel_min, config.random_dead_channel_max))
        self.zero_background_ratio = float(config.zero_background_ratio)
        self.zero_background_rate = float(config.zero_background_rate)
        self.zero_background_channel_min = int(max(1, config.zero_background_channel_min))
        self.zero_background_channel_max = int(max(self.zero_background_channel_min, config.zero_background_channel_max))
        self.zero_background_duration_min_s = float(config.zero_background_duration_min_s)
        self.zero_background_duration_max_s = float(max(config.zero_background_duration_min_s, config.zero_background_duration_max_s))
        self.per_vehicle_drop_channel_ratio = float(config.per_vehicle_drop_channel_ratio)
        self.per_vehicle_drop_channel_min = int(max(0, config.per_vehicle_drop_channel_min))
        self.per_vehicle_drop_channel_max = int(max(self.per_vehicle_drop_channel_min, config.per_vehicle_drop_channel_max))
        self.missing_random_ratio_min = float(max(0.0, config.missing_random_ratio_min))
        self.missing_random_ratio_max = float(max(self.missing_random_ratio_min, config.missing_random_ratio_max))
        self.missing_segment_count_max = int(max(0, config.missing_segment_count_max))
        self.missing_segment_min_len = int(max(1, config.missing_segment_min_len))
        self.missing_segment_max_len = int(max(self.missing_segment_min_len, config.missing_segment_max_len))
        self.interaction_ratio = float(config.interaction_ratio)
        self.interaction_types = str(config.interaction_types)
        self.interaction_time_min_frac = float(config.interaction_time_min_frac)
        self.interaction_time_max_frac = float(config.interaction_time_max_frac)
        self.isolated_noise_ratio = float(config.isolated_noise_ratio)
        self.isolated_noise_rate = float(config.isolated_noise_rate)
        self.isolated_noise_amp_min = float(config.isolated_noise_amp_min)
        self.isolated_noise_amp_max = float(config.isolated_noise_amp_max)
        self.isolated_noise_sigma_min_s = float(config.isolated_noise_sigma_min_s)
        self.isolated_noise_sigma_max_s = float(config.isolated_noise_sigma_max_s)
        self.clip_ratio = float(config.clip_ratio)
        self.input_scale = float(config.input_scale)
        self.input_mode = str(config.input_mode).lower()
        self.speed_norm_kmh = float(config.speed_norm_kmh)
        self.seed = int(config.seed)
        self.return_raw_window = bool(config.return_raw_window)

    def __len__(self) -> int:
        return self.length

    def _sample_vehicle_count(self, gen: torch.Generator) -> int:
        lo = int(max(1, self.vehicles_min))
        hi = int(max(lo, self.vehicles_max))
        if hi <= lo:
            return lo
        r = float(torch.rand((), generator=gen).item())
        if r < 0.35:
            return int(torch.randint(lo, min(hi, lo + max(1, (hi - lo) // 3)) + 1, (1,), generator=gen).item())
        if r < 0.75:
            return int(torch.randint(min(hi, lo + 1), min(hi, lo + max(2, (hi - lo) * 2 // 3)) + 1, (1,), generator=gen).item())
        return int(torch.randint(max(lo, hi - max(2, (hi - lo) // 3)), hi + 1, (1,), generator=gen).item())

    def _speed_kmh(self, gen: torch.Generator, base_speed: Optional[float] = None) -> float:
        if base_speed is None:
            return float(self.speed_min_kmh + torch.rand((), generator=gen).item() * (self.speed_max_kmh - self.speed_min_kmh))
        jitter = 1.0 + float((2.0 * torch.rand((), generator=gen).item() - 1.0) * self.speed_jitter_ratio)
        speed = float(base_speed) * jitter
        return float(np.clip(speed, self.speed_min_kmh, self.speed_max_kmh))

    def _sample_anchor_time_ch(
        self,
        gen: torch.Generator,
        *,
        cluster_time: Optional[float] = None,
        cluster_ch: Optional[int] = None,
    ) -> tuple[float, int]:
        if cluster_time is not None and cluster_ch is not None and torch.rand((), generator=gen).item() < self.scene_cluster_ratio:
            anchor_time = float(cluster_time) + float(self.scene_cluster_time_jitter_s * torch.randn((), generator=gen).item())
            anchor_ch = int(
                np.clip(
                    int(cluster_ch) + int(torch.randint(-self.scene_cluster_channel_jitter, self.scene_cluster_channel_jitter + 1, (1,), generator=gen).item()),
                    0,
                    self.n_channels - 1,
                )
            )
            return anchor_time, anchor_ch
        anchor_time = self._rand_uniform(gen, 0.0, self.window_seconds)
        anchor_ch = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
        return anchor_time, anchor_ch

    def _sample_direction_plan(self, gen: torch.Generator, n_veh: int) -> list[int]:
        if n_veh <= 0:
            return []
        reverse_count = 0
        if n_veh == 1:
            reverse_count = 0
        else:
            r = float(torch.rand((), generator=gen).item())
            if r < 0.70:
                reverse_count = 0
            elif r < 0.95:
                reverse_count = 1
            else:
                reverse_count = 2
        return self._make_direction_plan(gen, n_veh, reverse_count)

    def _make_direction_plan(self, gen: torch.Generator, n_veh: int, reverse_count: int) -> list[int]:
        if n_veh <= 0:
            return []
        reverse_count = int(min(max(0, reverse_count), n_veh))
        plan = [0] * (n_veh - reverse_count) + [1] * reverse_count
        perm = torch.randperm(n_veh, generator=gen).tolist()
        return [int(plan[idx]) for idx in perm]

    def _split_csv(self, text: str) -> list[str]:
        return [item.strip() for item in str(text).split(",") if item.strip()]

    def _parse_int_csv(self, text: str) -> list[int]:
        return [int(item) for item in self._split_csv(text)]

    def _parse_channel_rate_csv(self, text: str) -> dict[int, float]:
        rates: dict[int, float] = {}
        for item in self._split_csv(text):
            if ":" not in item:
                continue
            key_text, value_text = item.split(":", 1)
            try:
                key = int(key_text.strip())
                value = float(value_text.strip())
            except ValueError:
                continue
            if 0 <= key < int(self.n_channels):
                rates[key] = float(np.clip(value, 0.0, 1.0))
        return rates

    def _rand_uniform(self, gen: torch.Generator, lo: float, hi: float) -> float:
        return float(lo + torch.rand((), generator=gen).item() * (hi - lo))

    def _motion_models_and_weights(self) -> tuple[list[str], list[float]]:
        allowed = {"constant_sparse", "smooth_random", "stop_go"}
        models = self._split_csv(self.motion_mix)
        weights = [float(item) for item in self._split_csv(self.motion_weights)]
        if not models:
            return ["constant_sparse"], [1.0]
        models = [model for model in models if model in allowed]
        if not models:
            return ["constant_sparse"], [1.0]
        if len(weights) != len(models):
            weights = [1.0 / float(len(models))] * len(models)
        total = float(sum(max(0.0, item) for item in weights))
        if total <= 0.0:
            return models, [1.0 / float(len(models))] * len(models)
        weights = [float(max(0.0, item) / total) for item in weights]
        return models, weights

    def _sample_interaction_count(self, gen: torch.Generator, n_veh: int) -> int:
        if n_veh < 2 or self.interaction_ratio <= 0.0:
            return 0
        r = float(torch.rand((), generator=gen).item())
        if n_veh >= 6 and r < 0.20:
            return 2
        if r < 0.85:
            return 1
        return 0

    def _choose_motion_model(self, gen: torch.Generator) -> str:
        models, weights = self._motion_models_and_weights()
        if len(models) == 1:
            return models[0]
        idx = int(torch.multinomial(torch.tensor(weights, dtype=torch.float32), 1, generator=gen).item())
        return models[idx]

    def _apply_constant_sparse_perturbations(self, speed_kmh: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        n_seg = int(speed_kmh.numel())
        out = speed_kmh.clone()
        if n_seg <= 0 or self.constant_perturb_prob <= 0.0 or self.constant_perturb_max_frac <= 0.0:
            return out
        for start in range(n_seg):
            if float(torch.rand((), generator=gen).item()) >= self.constant_perturb_prob:
                continue
            width = int(torch.randint(self.constant_perturb_width_min, self.constant_perturb_width_max + 1, (1,), generator=gen).item())
            end = min(n_seg, start + width)
            delta = self._rand_uniform(gen, -self.constant_perturb_max_frac, self.constant_perturb_max_frac)
            out[start:end] *= float(1.0 + delta)
        return out

    def _apply_smooth_random_speed(self, speed_kmh: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        n_seg = int(speed_kmh.numel())
        if n_seg <= 0 or self.smooth_speed_max_frac <= 0.0:
            return speed_kmh
        noise = torch.randn((n_seg,), generator=gen, dtype=torch.float32)
        if n_seg > 1 and self.smooth_speed_corr_channels > 1:
            kernel_width = min(self.smooth_speed_corr_channels, n_seg)
            kernel = torch.ones((1, 1, kernel_width), dtype=torch.float32) / float(kernel_width)
            pad_left = kernel_width // 2
            pad_right = kernel_width - 1 - pad_left
            padded = torch.nn.functional.pad(noise.view(1, 1, -1), (pad_left, pad_right), mode="replicate")
            noise = torch.nn.functional.conv1d(padded, kernel).view(-1)
        max_abs = float(torch.max(torch.abs(noise)).item())
        if max_abs <= 1e-9:
            return speed_kmh
        amplitude = self._rand_uniform(gen, 0.0, self.smooth_speed_max_frac)
        factor = 1.0 + noise / max_abs * float(amplitude)
        return speed_kmh * factor

    def _integrate_segment_times(
        self,
        segment_speed_kmh: torch.Tensor,
        *,
        anchor_pos: int,
        anchor_time: float,
    ) -> torch.Tensor:
        n_ch = int(self.n_channels)
        t_path = torch.empty((n_ch,), dtype=torch.float32)
        if n_ch <= 1:
            t_path.fill_(float(anchor_time))
            return t_path
        speed_mps = torch.clamp(segment_speed_kmh.to(torch.float32) / 3.6, min=1e-6)
        dt = float(self.dx_m) / speed_mps
        anchor_pos = int(max(0, min(n_ch - 1, anchor_pos)))
        t_path[anchor_pos] = float(anchor_time)
        for pos in range(anchor_pos, n_ch - 1):
            t_path[pos + 1] = t_path[pos] + dt[pos]
        for pos in range(anchor_pos - 1, -1, -1):
            t_path[pos] = t_path[pos + 1] - dt[pos]
        return t_path

    def _apply_track_time_jitter(self, t_path: torch.Tensor, *, anchor_pos: int, gen: torch.Generator) -> torch.Tensor:
        n_ch = int(t_path.numel())
        if n_ch <= 1 or self.track_time_jitter_max_s <= 0.0:
            return t_path
        noise = torch.randn((n_ch,), generator=gen, dtype=torch.float32)
        if self.track_time_jitter_corr_channels > 1:
            width = min(self.track_time_jitter_corr_channels, n_ch)
            kernel = torch.ones((1, 1, width), dtype=torch.float32) / float(width)
            pad_left = width // 2
            pad_right = width - 1 - pad_left
            padded = torch.nn.functional.pad(noise.view(1, 1, -1), (pad_left, pad_right), mode="replicate")
            noise = torch.nn.functional.conv1d(padded, kernel).view(-1)
        anchor_pos = int(max(0, min(n_ch - 1, anchor_pos)))
        noise = noise - noise[anchor_pos]
        max_abs = float(torch.max(torch.abs(noise)).item())
        if max_abs <= 1e-9:
            return t_path
        amplitude = self._rand_uniform(gen, 0.0, self.track_time_jitter_max_s)
        jitter = noise / max_abs * float(amplitude)
        candidate = t_path + jitter
        out = candidate.clone()
        out[anchor_pos] = t_path[anchor_pos]
        base_dt = torch.diff(t_path)
        min_gap_ratio = float(min(0.95, max(0.05, self.track_time_jitter_min_gap_ratio)))
        min_dt = torch.clamp(base_dt * float(min_gap_ratio), min=1e-4)
        for pos in range(anchor_pos + 1, n_ch):
            out[pos] = max(float(candidate[pos].item()), float(out[pos - 1].item() + min_dt[pos - 1].item()))
        for pos in range(anchor_pos - 1, -1, -1):
            out[pos] = min(float(candidate[pos].item()), float(out[pos + 1].item() - min_dt[pos].item()))
        return out

    def _effective_speed_kmh(self, t_center: torch.Tensor, fallback_speed_kmh: float) -> float:
        if int(t_center.numel()) <= 1:
            return float(fallback_speed_kmh)
        travel_s = float(torch.max(t_center).item() - torch.min(t_center).item())
        if travel_s <= 1e-9:
            return float(fallback_speed_kmh)
        distance_m = float(int(t_center.numel()) - 1) * float(self.dx_m)
        return float(3.6 * distance_m / travel_s)

    def _sample_track_times(
        self,
        gen: torch.Generator,
        *,
        is_primary: bool,
        speed_kmh: float,
        anchor_ch: int,
        anchor_time: float,
    ) -> tuple[torch.Tensor, float, str, torch.Tensor]:
        n_ch = int(self.n_channels)
        n_seg = max(0, n_ch - 1)
        motion_model = self._choose_motion_model(gen)
        segment_speed = torch.full((n_seg,), float(speed_kmh), dtype=torch.float32)
        stop_path_mask = torch.zeros((n_ch,), dtype=torch.bool)
        stop_pos = 0

        if motion_model == "constant_sparse":
            segment_speed = self._apply_constant_sparse_perturbations(segment_speed, gen)
        elif motion_model == "smooth_random":
            segment_speed = self._apply_smooth_random_speed(segment_speed, gen)
        elif motion_model == "stop_go":
            restart_ratio = self._rand_uniform(gen, float(self.restart_speed_ratio_min), float(self.restart_speed_ratio_max))
            if n_seg > 0:
                stop_pos = int(torch.randint(0, n_ch - 1, (1,), generator=gen).item())
                segment_speed[stop_pos:] *= float(restart_ratio)
        else:
            raise ValueError(f"Unknown motion model: {motion_model}")

        segment_speed = torch.clamp(segment_speed, min=float(self.speed_min_kmh), max=float(self.speed_max_kmh))
        anchor_pos = int(anchor_ch) if bool(is_primary) else n_ch - 1 - int(anchor_ch)
        t_path = self._integrate_segment_times(segment_speed, anchor_pos=anchor_pos, anchor_time=float(anchor_time))
        t_path = self._apply_track_time_jitter(t_path, anchor_pos=anchor_pos, gen=gen)

        if motion_model == "stop_go" and n_ch > 1:
            stop_duration = self._rand_uniform(gen, float(self.stop_duration_min_s), float(self.stop_duration_max_s))
            stop_pos = int(max(0, min(n_ch - 1, stop_pos)))
            t_path[stop_pos + 1 :] += float(stop_duration)
            width = int(
                torch.randint(self.stop_channel_width_min, self.stop_channel_width_max + 1, (1,), generator=gen).item()
            )
            half_left = width // 2
            start = max(0, stop_pos - half_left)
            end = min(n_ch, start + width)
            start = max(0, end - width)
            stop_path_mask[start:end] = True

        if bool(is_primary):
            t_center = t_path
            stop_mask = stop_path_mask
        else:
            order = torch.arange(n_ch - 1, -1, -1, dtype=torch.long)
            t_center = torch.empty((n_ch,), dtype=torch.float32)
            stop_mask = torch.zeros((n_ch,), dtype=torch.bool)
            t_center[order] = t_path
            stop_mask[order] = stop_path_mask
        effective_speed = self._effective_speed_kmh(t_center, float(speed_kmh))
        return t_center, effective_speed, motion_model, stop_mask

    def _interaction_time_pair(
        self,
        gen: torch.Generator,
        interaction_type: str,
    ) -> tuple[torch.Tensor, int, float, torch.Tensor, int, float]:
        cross_ch = int(torch.randint(0, int(self.n_channels), (1,), generator=gen).item())
        lo = max(0.0, min(1.0, float(self.interaction_time_min_frac)))
        hi = max(lo, min(1.0, float(self.interaction_time_max_frac)))
        cross_time = self._rand_uniform(gen, lo * float(self.window_seconds), hi * float(self.window_seconds))
        speed_span = max(1e-6, float(self.speed_max_kmh) - float(self.speed_min_kmh))
        slow = self._rand_uniform(gen, float(self.speed_min_kmh), float(self.speed_min_kmh) + 0.35 * speed_span)
        fast = self._rand_uniform(gen, float(self.speed_min_kmh) + 0.65 * speed_span, float(self.speed_max_kmh))
        ch_axis = torch.arange(int(self.n_channels), dtype=torch.float32)
        dx = float(self.dx_m)
        if interaction_type == "crossing":
            t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
            t_b = float(cross_time) - (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
            return t_a, 0, slow, t_b, 1, fast
        if interaction_type == "parallel_crossing":
            leader_is_a = bool(torch.rand((), generator=gen).item() < 0.5)
            if leader_is_a:
                leader_speed = fast
                follower_speed = slow
            else:
                leader_speed = slow
                follower_speed = fast
            lead_bias = self._rand_uniform(gen, -1.8, -0.4)
            lag_bias = self._rand_uniform(gen, 0.4, 1.8)
            t_a = float(cross_time) + lead_bias + (ch_axis - float(cross_ch)) * dx / (leader_speed / 3.6)
            t_b = float(cross_time) + lag_bias + (ch_axis - float(cross_ch)) * dx / (follower_speed / 3.6)
            return t_a, 0, leader_speed, t_b, 0, follower_speed
        if interaction_type == "near_parallel":
            offset = self._rand_uniform(gen, 0.1, 1.0) * (-1.0 if torch.rand((), generator=gen).item() < 0.5 else 1.0)
            t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
            t_b = float(cross_time + offset) + (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
            return t_a, 0, slow, t_b, 0, fast
        t_a = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (slow / 3.6)
        t_b = float(cross_time) + (ch_axis - float(cross_ch)) * dx / (fast / 3.6)
        return t_a, 0, slow, t_b, 0, fast

    def _sample_dead_channels(self, gen: torch.Generator) -> list[int]:
        dead = {idx for idx in self._parse_int_csv(self.dead_channel_indices) if 0 <= idx < int(self.n_channels)}
        for idx, rate in self._intermittent_dead_channel_rate_map.items():
            if rate > 0.0 and torch.rand((), generator=gen).item() < rate:
                dead.add(int(idx))
        if self.random_dead_channel_ratio > 0.0 and torch.rand((), generator=gen).item() < self.random_dead_channel_ratio:
            count = int(torch.randint(self.random_dead_channel_min, self.random_dead_channel_max + 1, (1,), generator=gen).item())
            count = min(count, int(self.n_channels))
            if count > 0:
                order = torch.randperm(int(self.n_channels), generator=gen).tolist()
                dead.update(int(idx) for idx in order[:count])
        return sorted(dead)

    def _sample_occupied_blocks(self, gen: torch.Generator) -> list[tuple[int, int, int, int]]:
        if self.zero_background_ratio <= 0.0 or torch.rand((), generator=gen).item() >= self.zero_background_ratio:
            return []
        count = int(self.zero_background_rate)
        if torch.rand((), generator=gen).item() < self.zero_background_rate - count:
            count += 1
        if count <= 0:
            return []
        blocks: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            width_ch = int(torch.randint(self.zero_background_channel_min, self.zero_background_channel_max + 1, (1,), generator=gen).item())
            width_ch = max(1, min(width_ch, int(self.n_channels)))
            start_ch = int(torch.randint(0, max(1, int(self.n_channels) - width_ch + 1), (1,), generator=gen).item())
            duration_s = self._rand_uniform(gen, float(self.zero_background_duration_min_s), float(self.zero_background_duration_max_s))
            width_t = int(max(1, round(duration_s * float(self.fs))))
            width_t = max(1, min(width_t, int(self.window_samples)))
            start_t = int(torch.randint(0, max(1, int(self.window_samples) - width_t + 1), (1,), generator=gen).item())
            blocks.append((start_ch, start_ch + width_ch, start_t, start_t + width_t))
        return blocks

    def _apply_zero_blocks(
        self,
        data: np.ndarray,
        target: dict[str, np.ndarray],
        blocks: list[tuple[int, int, int, int]],
    ) -> None:
        if not blocks:
            return
        for ch0, ch1, t0, t1 in blocks:
            data[ch0:ch1, t0:t1] = 0.0
            visibility = target.get("observed_visibility")
            full_time = target.get("full_time")
            if visibility is None or full_time is None or visibility.size == 0:
                continue
            t0_norm = float(t0) / float(max(1, self.window_samples - 1))
            t1_norm = float(t1) / float(max(1, self.window_samples - 1))
            affected = (full_time[:, ch0:ch1] >= t0_norm) & (full_time[:, ch0:ch1] <= t1_norm)
            if bool(affected.any()):
                visibility[:, ch0:ch1] = np.where(affected, 0.0, visibility[:, ch0:ch1])
        visibility = target.get("observed_visibility")
        if visibility is not None and visibility.size > 0:
            target["observed_visibility"] = visibility

    def _apply_isolated_noise(self, gen: torch.Generator, data: np.ndarray) -> None:
        if self.isolated_noise_ratio <= 0.0 or torch.rand((), generator=gen).item() >= self.isolated_noise_ratio:
            return
        count = int(self.isolated_noise_rate)
        if torch.rand((), generator=gen).item() < self.isolated_noise_rate - count:
            count += 1
        count = max(0, count)
        if count <= 0:
            return
        t_axis = np.arange(self.window_samples, dtype=np.float32) / float(self.fs)
        for _ in range(count):
            ch = int(torch.randint(0, int(self.n_channels), (1,), generator=gen).item())
            center = self._rand_uniform(gen, 0.0, float(self.window_seconds))
            sigma = self._rand_uniform(gen, float(self.isolated_noise_sigma_min_s), float(self.isolated_noise_sigma_max_s))
            amp = self._rand_uniform(gen, float(self.isolated_noise_amp_min), float(self.isolated_noise_amp_max))
            data[ch] += float(amp) * np.exp(-0.5 * ((t_axis - float(center)) / max(1e-6, float(sigma))) ** 2).astype(np.float32)

    def _render_vehicle(
        self,
        *,
        data: np.ndarray,
        gen: torch.Generator,
        direction_label: int,
        speed_kmh: float,
        anchor_ch: int,
        anchor_time: float,
        amp: float,
        sigma_s: float,
        t_center_override: Optional[torch.Tensor] = None,
        speed_override: Optional[float] = None,
        motion_model_override: Optional[str] = None,
        stop_mask_override: Optional[torch.Tensor] = None,
    ) -> dict[str, np.ndarray] | None:
        if t_center_override is None:
            t_center, effective_speed_kmh, motion_model, stop_mask = self._sample_track_times(
                gen,
                is_primary=bool(direction_label == 0),
                speed_kmh=speed_kmh,
                anchor_ch=anchor_ch,
                anchor_time=anchor_time,
            )
        else:
            t_center = torch.as_tensor(t_center_override, dtype=torch.float32)
            effective_speed_kmh = float(speed_override if speed_override is not None else speed_kmh)
            motion_model = str(motion_model_override or "constant_sparse")
            stop_mask = torch.as_tensor(stop_mask_override, dtype=torch.bool) if stop_mask_override is not None else torch.zeros_like(t_center, dtype=torch.bool)
        t_center_np = np.asarray(t_center.detach().cpu().numpy(), dtype=np.float32)
        full_valid = (t_center_np >= 0.0) & (t_center_np < self.window_seconds)
        if int(full_valid.sum()) < self.min_visible_channels:
            return None

        observed = full_valid.copy()
        if int(observed.sum()) >= self.min_visible_channels:
            missing_ratio = float(
                self.missing_random_ratio_min
                + torch.rand((), generator=gen).item() * (self.missing_random_ratio_max - self.missing_random_ratio_min)
            )
            miss_count = int(round(int(observed.sum()) * missing_ratio))
            miss_count = min(miss_count, int(observed.sum()) - self.min_visible_channels)
            if miss_count > 0:
                missing_choices = np.flatnonzero(observed)
                chosen = np.asarray(
                    torch.randperm(int(missing_choices.size), generator=gen)[:miss_count].cpu().numpy(),
                    dtype=np.int64,
                )
                observed[missing_choices[chosen]] = False

            if self.missing_segment_count_max > 0 and int(observed.sum()) > self.min_visible_channels + 1:
                seg_count = int(torch.randint(0, self.missing_segment_count_max + 1, (1,), generator=gen).item())
                for _ in range(seg_count):
                    visible_idx = np.flatnonzero(observed)
                    if visible_idx.size <= self.min_visible_channels + self.missing_segment_min_len:
                        break
                    seg_len = int(
                        torch.randint(
                            self.missing_segment_min_len,
                            min(self.missing_segment_max_len, int(visible_idx.size) - self.min_visible_channels) + 1,
                            (1,),
                            generator=gen,
                        ).item()
                    )
                    if seg_len <= 0 or seg_len >= visible_idx.size:
                        continue
                    start = int(torch.randint(0, visible_idx.size - seg_len + 1, (1,), generator=gen).item())
                    observed[visible_idx[start : start + seg_len]] = False

        if self.per_vehicle_drop_channel_ratio > 0.0 and torch.rand((), generator=gen).item() < self.per_vehicle_drop_channel_ratio:
            drop_count = int(
                torch.randint(self.per_vehicle_drop_channel_min, self.per_vehicle_drop_channel_max + 1, (1,), generator=gen).item()
            )
            drop_count = min(drop_count, int(observed.sum()) - self.min_visible_channels)
            if drop_count > 0:
                drop_idx = np.flatnonzero(observed)
                if drop_idx.size > 0:
                    order = torch.randperm(int(drop_idx.size), generator=gen)[:drop_count].cpu().numpy()
                    observed[drop_idx[order]] = False

        if int(observed.sum()) < self.min_visible_channels:
            return None

        center_idx = np.clip(np.round(t_center_np * self.fs).astype(np.int64), 0, self.window_samples - 1)
        half_width = int(max(1, round(4.0 * sigma_s * self.fs)))
        for ch in np.flatnonzero(observed):
            center = int(center_idx[ch])
            left = max(0, center - half_width)
            right = min(self.window_samples - 1, center + half_width)
            idx = np.arange(left, right + 1, dtype=np.float32)
            dt = idx / self.fs - float(t_center_np[ch])
            local_amp = float(amp)
            local_sigma = float(sigma_s)
            if motion_model == "stop_go" and bool(stop_mask[ch]):
                local_amp *= float(self.stop_response_amp_scale)
                local_sigma *= float(self.stop_response_sigma_scale)
            pulse = local_amp * np.exp(-0.5 * (dt / max(1e-6, local_sigma)) ** 2)
            data[ch, left : right + 1] += pulse.astype(np.float32, copy=False)

        full_time = np.zeros((self.n_channels,), dtype=np.float32)
        full_valid = full_valid.astype(np.float32)
        observed_visibility = observed.astype(np.float32)
        missing_visibility = (full_valid > 0.5) & (observed_visibility < 0.5)
        full_time[full_valid > 0.5] = _time_index_to_norm(center_idx[full_valid > 0.5], self.window_samples)
        return {
            "full_time": full_time,
            "full_valid": full_valid,
            "observed_visibility": observed_visibility,
            "missing_channel_mask": missing_visibility.astype(np.float32),
            "direction": np.asarray([int(direction_label)], dtype=np.int64),
            "speed": np.asarray([float(effective_speed_kmh) / max(1e-6, self.speed_norm_kmh)], dtype=np.float32),
            "amp": np.asarray([float(amp)], dtype=np.float32),
            "sigma_s": np.asarray([float(sigma_s)], dtype=np.float32),
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + int(index) * 1000003)
        data = np.zeros((self.n_channels, self.window_samples), dtype=np.float32)
        n_veh = int(self._sample_vehicle_count(gen))
        targets: list[dict[str, np.ndarray]] = []
        dead_channels = self._sample_dead_channels(gen)
        zero_blocks = self._sample_occupied_blocks(gen)

        scene_cluster_time = self._rand_uniform(gen, 0.10 * self.window_seconds, 0.90 * self.window_seconds)
        scene_cluster_ch = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
        reverse_budget = 0
        if n_veh >= 2:
            r = float(torch.rand((), generator=gen).item())
            if r < 0.70:
                reverse_budget = 0
            elif r < 0.95:
                reverse_budget = 1
            else:
                reverse_budget = 2
        interaction_pairs: list[tuple[torch.Tensor, int, float, torch.Tensor, int, float]] = []
        interaction_reverse_count = 0
        interaction_count = self._sample_interaction_count(gen, n_veh)
        if interaction_count > 0:
            interaction_types = [
                item
                for item in self._split_csv(self.interaction_types)
                if item in {"parallel_crossing", "crossing", "overtake", "near_parallel"}
            ]
            if not interaction_types:
                interaction_types = ["parallel_crossing"]
            for _ in range(interaction_count):
                interaction_type = interaction_types[int(torch.randint(0, len(interaction_types), (1,), generator=gen).item())]
                if interaction_type == "crossing":
                    interaction_reverse_count += 1
                interaction_pairs.append(self._interaction_time_pair(gen, interaction_type))
        remaining_veh = max(0, n_veh - 2 * len(interaction_pairs))
        direction_plan = self._make_direction_plan(gen, remaining_veh, max(0, reverse_budget - interaction_reverse_count))

        track_idx = 0
        for pair in interaction_pairs:
            for t_center, dir_label, speed_a in [(pair[0], pair[1], pair[2]), (pair[3], pair[4], pair[5])]:
                if track_idx >= n_veh:
                    break
                rendered = self._render_vehicle(
                    data=data,
                    gen=gen,
                    direction_label=int(dir_label),
                    speed_kmh=float(speed_a),
                    anchor_ch=int(torch.randint(0, self.n_channels, (1,), generator=gen).item()),
                    anchor_time=float(self._rand_uniform(gen, 0.0, self.window_seconds)),
                    amp=float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min)),
                    sigma_s=float(self.sigma_min_s + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s)),
                    t_center_override=t_center,
                    speed_override=float(speed_a),
                    motion_model_override="interaction",
                )
                if rendered is None:
                    continue
                rendered["track_id"] = np.asarray([track_idx], dtype=np.int64)
                targets.append(rendered)
                track_idx += 1

        attempts = 0
        max_attempts = max(64, n_veh * 64)
        direction_cursor = 0
        while track_idx < n_veh and attempts < max_attempts:
            attempts += 1
            is_primary = bool(torch.rand((), generator=gen).item() < self.primary_ratio)
            if direction_cursor < len(direction_plan):
                direction_label = int(direction_plan[direction_cursor])
                direction_cursor += 1
            else:
                direction_label = 0
            speed_kmh = self._speed_kmh(gen)
            sigma_s = float(self.sigma_min_s + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s))
            amp = float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min))
            use_cluster = bool(torch.rand((), generator=gen).item() < self.scene_cluster_ratio)
            if use_cluster:
                anchor_time, anchor_ch = self._sample_anchor_time_ch(
                    gen,
                    cluster_time=scene_cluster_time,
                    cluster_ch=scene_cluster_ch,
                )
            else:
                anchor_time, anchor_ch = self._sample_anchor_time_ch(gen)
            rendered = self._render_vehicle(
                data=data,
                gen=gen,
                direction_label=direction_label,
                speed_kmh=speed_kmh,
                anchor_ch=anchor_ch,
                anchor_time=anchor_time,
                amp=amp,
                sigma_s=sigma_s,
            )
            if rendered is None:
                continue
            rendered["track_id"] = np.asarray([track_idx], dtype=np.int64)
            targets.append(rendered)
            track_idx += 1

        self._apply_isolated_noise(gen, data)
        if self.noise_std > 0.0:
            data += np.random.default_rng(self.seed + int(index) * 7).normal(0.0, self.noise_std, size=data.shape).astype(np.float32)

        if not targets:
            target_np = {
                "full_time": np.zeros((0, self.n_channels), dtype=np.float32),
                "full_valid": np.zeros((0, self.n_channels), dtype=np.float32),
                "observed_visibility": np.zeros((0, self.n_channels), dtype=np.float32),
                "missing_channel_mask": np.zeros((0, self.n_channels), dtype=np.float32),
                "dead_channel_mask": np.zeros((self.n_channels,), dtype=np.float32),
                "direction": np.zeros((0,), dtype=np.int64),
                "speed": np.zeros((0,), dtype=np.float32),
                "track_id": np.zeros((0,), dtype=np.int64),
                "amp": np.zeros((0,), dtype=np.float32),
                "sigma_s": np.zeros((0,), dtype=np.float32),
            }
        else:
            dead_channel_mask = np.zeros((self.n_channels,), dtype=np.float32)
            if dead_channels:
                dead_channel_mask[np.asarray(dead_channels, dtype=np.int64)] = 1.0
            target_np = {
                "full_time": np.stack([item["full_time"] for item in targets], axis=0),
                "full_valid": np.stack([item["full_valid"] for item in targets], axis=0),
                "observed_visibility": np.stack([item["observed_visibility"] for item in targets], axis=0),
                "missing_channel_mask": np.stack([item["missing_channel_mask"] for item in targets], axis=0),
                "dead_channel_mask": dead_channel_mask,
                "direction": np.concatenate([item["direction"] for item in targets], axis=0),
                "speed": np.concatenate([item["speed"] for item in targets], axis=0),
                "track_id": np.concatenate([item["track_id"] for item in targets], axis=0),
                "amp": np.concatenate([item["amp"] for item in targets], axis=0),
                "sigma_s": np.concatenate([item["sigma_s"] for item in targets], axis=0),
            }
        if dead_channels and target_np["observed_visibility"].size > 0:
            data[dead_channels, :] = 0.0
            target_np["observed_visibility"][:, dead_channels] = 0.0
        if zero_blocks:
            self._apply_zero_blocks(data, target_np, zero_blocks)

        x = prepare_peakset_input(
            data,
            time_downsample=self.time_downsample,
            clip_ratio=self.clip_ratio,
            input_mode=self.input_mode,
            input_scale=self.input_scale,
        )
        if not targets:
            target = {
                "full_time": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "full_valid": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "observed_visibility": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "missing_channel_mask": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "dead_channel_mask": torch.zeros((self.n_channels,), dtype=torch.float32),
                "direction": torch.zeros((0,), dtype=torch.long),
                "speed": torch.zeros((0,), dtype=torch.float32),
                "track_id": torch.zeros((0,), dtype=torch.long),
                "amp": torch.zeros((0,), dtype=torch.float32),
                "sigma_s": torch.zeros((0,), dtype=torch.float32),
            }
        else:
            target = {
                "full_time": torch.from_numpy(np.asarray(target_np["full_time"], dtype=np.float32)),
                "full_valid": torch.from_numpy(np.asarray(target_np["full_valid"], dtype=np.float32)),
                "observed_visibility": torch.from_numpy(np.asarray(target_np["observed_visibility"], dtype=np.float32)),
                "missing_channel_mask": torch.from_numpy(np.asarray(target_np["missing_channel_mask"], dtype=np.float32)),
                "dead_channel_mask": torch.from_numpy(np.asarray(target_np["dead_channel_mask"], dtype=np.float32)),
                "direction": torch.from_numpy(np.asarray(target_np["direction"], dtype=np.int64)).long(),
                "speed": torch.from_numpy(np.asarray(target_np["speed"], dtype=np.float32)).float(),
                "track_id": torch.from_numpy(np.asarray(target_np["track_id"], dtype=np.int64)).long(),
                "amp": torch.from_numpy(np.asarray(target_np["amp"], dtype=np.float32)).float(),
                "sigma_s": torch.from_numpy(np.asarray(target_np["sigma_s"], dtype=np.float32)).float(),
            }
        if self.return_raw_window:
            target["raw_window"] = torch.from_numpy(data.copy())
        return x, target


def targets_to_batched_peakset(
    targets: Sequence[dict[str, torch.Tensor]],
    *,
    n_channels: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    batch_size = len(targets)
    max_gt = max((int(target["full_time"].shape[0]) for target in targets), default=0)
    if n_channels is None:
        n_channels = max((int(target["full_time"].shape[1]) for target in targets if target["full_time"].ndim == 2), default=0)
    n_channels = int(n_channels or 0)

    full_time = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    full_valid = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    observed_visibility = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    missing_channel_mask = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    dead_channel_mask = torch.zeros((batch_size, n_channels), dtype=torch.float32)
    direction = torch.zeros((batch_size, max_gt), dtype=torch.long)
    speed = torch.zeros((batch_size, max_gt), dtype=torch.float32)
    amp = torch.zeros((batch_size, max_gt), dtype=torch.float32)
    sigma_s = torch.zeros((batch_size, max_gt), dtype=torch.float32)
    track_id = torch.full((batch_size, max_gt), -1, dtype=torch.long)
    gt_valid = torch.zeros((batch_size, max_gt), dtype=torch.bool)

    for b, target in enumerate(targets):
        n_gt = int(target["full_time"].shape[0])
        if n_gt <= 0:
            continue
        copy_ch = min(n_channels, int(target["full_time"].shape[1]))
        full_time[b, :n_gt, :copy_ch] = target["full_time"][:, :copy_ch].to(torch.float32)
        full_valid[b, :n_gt, :copy_ch] = target["full_valid"][:, :copy_ch].to(torch.float32)
        observed_visibility[b, :n_gt, :copy_ch] = target["observed_visibility"][:, :copy_ch].to(torch.float32)
        if "missing_channel_mask" in target:
            missing_channel_mask[b, :n_gt, :copy_ch] = target["missing_channel_mask"][:, :copy_ch].to(torch.float32)
        if "dead_channel_mask" in target:
            dead_channel_mask[b, :copy_ch] = target["dead_channel_mask"][:copy_ch].to(torch.float32)
        direction[b, :n_gt] = target["direction"].to(torch.long)
        speed[b, :n_gt] = target["speed"].to(torch.float32)
        if "amp" in target:
            amp[b, :n_gt] = target["amp"].to(torch.float32)
        if "sigma_s" in target:
            sigma_s[b, :n_gt] = target["sigma_s"].to(torch.float32)
        if "track_id" in target:
            track_id[b, :n_gt] = target["track_id"].to(torch.long)
        gt_valid[b, :n_gt] = True

    return {
        "full_time": full_time,
        "full_valid": full_valid,
        "observed_visibility": observed_visibility,
        "missing_channel_mask": missing_channel_mask,
        "dead_channel_mask": dead_channel_mask,
        "direction": direction,
        "speed": speed,
        "amp": amp,
        "sigma_s": sigma_s,
        "track_id": track_id,
        "gt_valid": gt_valid,
        "gt_count": gt_valid.sum(dim=1).to(torch.long),
    }


def peakset_collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in batch], dim=0)
    raw_targets = [item[1] for item in batch]
    n_channels = int(raw_targets[0]["full_time"].shape[1]) if raw_targets and raw_targets[0]["full_time"].ndim == 2 else 0
    targets = targets_to_batched_peakset(raw_targets, n_channels=n_channels)
    time_downsample = 10
    fs = 1000.0
    peak_time, peak_amp, peak_valid, peak_index = _batch_peak_candidates(
        xs,
        fs=float(fs),
        time_downsample=int(time_downsample),
    )
    targets["peak_time"] = peak_time
    targets["peak_amp"] = peak_amp
    targets["peak_valid"] = peak_valid
    targets["peak_index"] = peak_index
    if targets["full_time"].ndim == 3:
        window_seconds = float(xs.shape[-1] * max(1, time_downsample)) / float(max(1e-6, fs))
        targets["gt_peak_index"] = match_gt_peak_indices(
            targets,
            peak_time,
            peak_valid,
            peak_index,
            match_tolerance_s=0.25,
            fs=float(fs),
            time_downsample=int(time_downsample),
            window_seconds=float(window_seconds),
        )
    else:
        targets["gt_peak_index"] = torch.zeros((xs.shape[0], 0, n_channels), dtype=torch.long)
    return xs, targets


def dataset_config_to_dict(config: SimplePeakSetDatasetConfig) -> dict[str, object]:
    return asdict(config)
