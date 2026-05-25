"""Profile sparse-background statistics from a real DAS `.npy` array.

用途：
    从真实 DAS `.npy` 数据中提取背景分布特征，并生成一个可直接驱动
    `generate_track_slot_dataset_from_real_npy.py` 的 `realism_profile.json`。
    重点面向：

    - 通道稀疏度 / 缺道比例
    - 正值幅度分布
    - 120 s 窗口内逐道峰候选密度
    - 零值缺块的宽度和持续时间
    - 窗口级训练相似度权重
    - 无标签车辆代理统计：速度、方向比例、可见段长度、交汇/近平行代理频率

用法：
    uv run python -m autotrack.simulation.profile_real_npy_background \
        --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
        --out-dir /tmp/real_profile \
        --channel-count 50 \
        --window-seconds 120 \
        --window-stride-seconds 600

输出：
    - `profile.json`
    - `realism_profile.json`
    - `report.md`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks, peak_widths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile sparse-background statistics from a real DAS .npy array.")
    parser.add_argument("--input", required=True, type=Path, help="Input real DAS .npy path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for profile.json, realism_profile.json, and report.md.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window length used for profiling.")
    parser.add_argument("--window-stride-seconds", type=float, default=600.0, help="Stride between sampled windows used for profiling.")
    parser.add_argument("--peak-height", type=float, default=0.02, help="Peak height threshold used for profiling.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Peak prominence threshold used for profiling.")
    parser.add_argument("--peak-distance-seconds", type=float, default=0.15, help="Minimum distance between peaks used for profiling.")
    parser.add_argument("--zero-threshold", type=float, default=1e-8, help="Values <= this threshold are treated as missing / zero.")
    parser.add_argument("--component-time-downsample", type=int, default=100, help="Time downsample used for 2-D zero-component profiling.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Clip ratio used to emulate training normalization during profiling.")
    parser.add_argument("--proxy-speed-min-kmh", type=float, default=55.0, help="Minimum proxy vehicle speed used in adjacent-channel matching.")
    parser.add_argument("--proxy-speed-max-kmh", type=float, default=110.0, help="Maximum proxy vehicle speed used in adjacent-channel matching.")
    parser.add_argument("--proxy-match-slack-s", type=float, default=0.08, help="Extra time slack added to adjacent-channel proxy matching.")
    parser.add_argument("--window-catalog-limit", type=int, default=4096, help="Maximum window summaries stored in realism_profile.json.")
    return parser.parse_args()


def _safe_stats(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _finite_value(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(parsed):
        return float(fallback)
    return float(parsed)


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _odd_kernel(width: int) -> int:
    base = max(1, int(width))
    return base if base % 2 == 1 else base + 1


def _load_array(path: Path, *, layout: str, channel_start: int, channel_count: int) -> np.ndarray:
    arr = np.load(str(path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D .npy array, got shape {arr.shape}")
    end = int(channel_start + channel_count)
    if layout == "time_channel":
        if channel_start < 0 or end > int(arr.shape[1]):
            raise ValueError(f"Channel slice [{channel_start}, {end}) outside input shape {arr.shape}")
        view = np.array(arr[:, channel_start:end], dtype=np.float32, copy=True)
    else:
        if channel_start < 0 or end > int(arr.shape[0]):
            raise ValueError(f"Channel slice [{channel_start}, {end}) outside input shape {arr.shape}")
        view = np.array(arr[channel_start:end, :].T, dtype=np.float32, copy=True)
    return np.nan_to_num(view, copy=False)


def _candidate_window_starts(n_time: int, window_samples: int, stride_samples: int) -> list[int]:
    if window_samples <= 0 or window_samples > n_time:
        return [0]
    starts = list(range(0, max(1, n_time - window_samples + 1), max(1, stride_samples)))
    last = int(n_time - window_samples)
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(item) for item in starts))


def _prepare_window(window_tc: np.ndarray, *, clip_ratio: float, zero_threshold: float) -> np.ndarray:
    window_ct = np.asarray(window_tc.T, dtype=np.float32)
    abs_vals = np.abs(window_ct)
    finite = abs_vals[np.isfinite(abs_vals)]
    if finite.size <= 0:
        return np.zeros_like(window_ct, dtype=np.float32)
    q995 = float(np.quantile(finite, 0.995))
    rms = float(np.sqrt(np.mean(finite * finite)))
    scale = max(1e-6, max(q995, 3.0 * rms))
    clip = max(1e-6, float(clip_ratio))
    raw = np.clip(window_ct / scale, -clip, clip) / clip
    raw[np.abs(window_ct) <= float(zero_threshold)] = 0.0
    return raw.astype(np.float32, copy=False)


def _profile_zero_components(
    x: np.ndarray,
    *,
    fs: float,
    zero_threshold: float,
    component_time_downsample: int,
) -> dict[str, Any]:
    mask = x <= float(zero_threshold)
    stride = int(max(1, component_time_downsample))
    coarse = mask[::stride]
    structure = ndimage.generate_binary_structure(2, 2)
    labels, _ = ndimage.label(coarse, structure=structure)
    objects = ndimage.find_objects(labels)
    widths: list[float] = []
    durations_s: list[float] = []
    dead_channels: list[int] = []
    zero_ratio = coarse.mean(axis=0)
    for ch in np.where(zero_ratio > 0.95)[0].tolist():
        dead_channels.append(int(ch))
    for sl in objects:
        if sl is None:
            continue
        time_sl, ch_sl = sl
        duration_samples = int((time_sl.stop - time_sl.start) * stride)
        channel_width = int(ch_sl.stop - ch_sl.start)
        if channel_width <= 0 or duration_samples <= 0:
            continue
        if duration_samples >= int(0.95 * x.shape[0]):
            continue
        widths.append(float(channel_width))
        durations_s.append(float(duration_samples) / float(fs))
    return {
        "zero_ratio_per_channel": _safe_stats(zero_ratio.astype(np.float64).tolist()),
        "dead_channel_indices": dead_channels,
        "stable_dead_channel_indices": dead_channels,
        "dead_channel_count": int(len(dead_channels)),
        "drop_block_channel_width": _safe_stats(widths),
        "drop_block_duration_s": _safe_stats(durations_s),
    }


def _profile_probabilistic_dead_channels(
    window_zero_ratios: list[np.ndarray],
) -> dict[str, Any]:
    if not window_zero_ratios:
        return {
            "stable_dead_channel_indices": [],
            "probabilistic_dead_channels": [],
            "channel_zero_ratio_summary": [],
        }
    ratios = np.stack(window_zero_ratios, axis=0).astype(np.float64, copy=False)
    mean_ratio = np.mean(ratios, axis=0)
    q50_ratio = np.quantile(ratios, 0.50, axis=0)
    q90_ratio = np.quantile(ratios, 0.90, axis=0)
    freq_gt_080 = np.mean(ratios > 0.80, axis=0)
    freq_gt_095 = np.mean(ratios > 0.95, axis=0)

    stable_dead_channels: list[int] = []
    probabilistic_dead_channels: list[dict[str, float]] = []
    channel_zero_ratio_summary: list[dict[str, float]] = []
    for ch in range(int(ratios.shape[1])):
        mean_value = float(mean_ratio[ch])
        q50_value = float(q50_ratio[ch])
        q90_value = float(q90_ratio[ch])
        freq80_value = float(freq_gt_080[ch])
        freq95_value = float(freq_gt_095[ch])
        channel_zero_ratio_summary.append(
            {
                "channel": int(ch),
                "mean_zero_ratio": mean_value,
                "median_zero_ratio": q50_value,
                "q90_zero_ratio": q90_value,
                "freq_zero_ratio_gt_0_80": freq80_value,
                "freq_zero_ratio_gt_0_95": freq95_value,
            }
        )
        if mean_value >= 0.95 or freq95_value >= 0.80:
            stable_dead_channels.append(int(ch))
            continue
        if mean_value < 0.87:
            continue
        probability = max(0.25, min(0.95, 0.60 * mean_value + 0.40 * freq80_value))
        probabilistic_dead_channels.append(
            {
                "channel": int(ch),
                "probability": float(probability),
                "mean_zero_ratio": mean_value,
                "median_zero_ratio": q50_value,
                "q90_zero_ratio": q90_value,
                "freq_zero_ratio_gt_0_80": freq80_value,
                "freq_zero_ratio_gt_0_95": freq95_value,
            }
        )
    probabilistic_dead_channels.sort(key=lambda item: (-float(item["probability"]), int(item["channel"])))
    channel_zero_ratio_summary.sort(key=lambda item: (-float(item["mean_zero_ratio"]), int(item["channel"])))
    return {
        "stable_dead_channel_indices": stable_dead_channels,
        "probabilistic_dead_channels": probabilistic_dead_channels,
        "channel_zero_ratio_summary": channel_zero_ratio_summary,
    }


def _extract_window_peak_rows(
    window_tc: np.ndarray,
    *,
    fs: float,
    peak_height: float,
    peak_prominence: float,
    peak_distance_seconds: float,
) -> list[dict[str, Any]]:
    distance = int(max(1, round(float(peak_distance_seconds) * float(fs))))
    rows: list[dict[str, Any]] = []
    for ch in range(int(window_tc.shape[1])):
        row = np.asarray(window_tc[:, ch], dtype=np.float32)
        peaks, _ = find_peaks(row, height=float(peak_height), prominence=float(peak_prominence), distance=distance)
        if peaks.size <= 0:
            rows.append({"times_s": [], "amps": [], "sigmas_s": []})
            continue
        widths = peak_widths(row, peaks, rel_height=0.5)[0]
        sigma_est = (widths / 2.355) / float(fs)
        rows.append(
            {
                "times_s": [float(idx) / float(fs) for idx in peaks.tolist()],
                "amps": row[peaks].astype(np.float64, copy=False).tolist(),
                "sigmas_s": np.asarray(sigma_est, dtype=np.float64).tolist(),
            }
        )
    return rows


def _window_stats_from_peak_rows(
    peak_rows: list[dict[str, Any]],
    *,
    normalized_window_ct: np.ndarray,
    zero_threshold: float,
) -> dict[str, Any]:
    total_peaks = 0
    per_channel_counts: list[float] = []
    peak_values: list[float] = []
    sigma_t_s: list[float] = []
    for row in peak_rows:
        count = len(row["times_s"])
        total_peaks += count
        per_channel_counts.append(float(count))
        peak_values.extend(float(v) for v in row["amps"])
        sigma_t_s.extend(float(v) for v in row["sigmas_s"])
    positive_ratio = float(np.mean(normalized_window_ct > 0.0))
    zero_ratio_mean = float(np.mean(np.abs(normalized_window_ct) <= float(zero_threshold)))
    abs_vals = np.abs(normalized_window_ct).reshape(-1)
    abs_q95 = float(np.quantile(abs_vals, 0.95)) if abs_vals.size > 0 else float("nan")
    return {
        "total_peaks": int(total_peaks),
        "peaks_per_channel_mean": float(np.mean(per_channel_counts)) if per_channel_counts else 0.0,
        "positive_ratio": positive_ratio,
        "zero_ratio_mean": zero_ratio_mean,
        "normalized_abs_q95": abs_q95,
        "peak_values": peak_values,
        "peak_sigma_t_s": sigma_t_s,
    }


def _estimate_vehicle_proxy(
    peak_rows: list[dict[str, Any]],
    *,
    dx_m: float,
    proxy_speed_min_kmh: float,
    proxy_speed_max_kmh: float,
    proxy_match_slack_s: float,
) -> dict[str, Any]:
    dt_min = float(dx_m) / max(1e-6, float(proxy_speed_max_kmh) / 3.6)
    dt_max = float(dx_m) / max(1e-6, float(proxy_speed_min_kmh) / 3.6) + float(proxy_match_slack_s)
    segment_speeds: list[float] = []
    segment_direction: list[int] = []
    support_lengths: list[float] = []
    crossing_proxy = 0.0
    near_parallel_proxy = 0.0
    active_peak_channels = 0

    for ch in range(len(peak_rows) - 1):
        current = np.asarray(peak_rows[ch]["times_s"], dtype=np.float64)
        nxt = np.asarray(peak_rows[ch + 1]["times_s"], dtype=np.float64)
        if current.size > 0:
            active_peak_channels += 1
        if current.size <= 0 or nxt.size <= 0:
            continue
        diffs = nxt[:, None] - current[None, :]
        diffs_abs = np.abs(diffs)
        best_idx = np.argmin(diffs_abs, axis=0)
        best_dt = diffs[best_idx, np.arange(current.size)]
        mask = (np.abs(best_dt) >= dt_min) & (np.abs(best_dt) <= dt_max)
        if not bool(np.any(mask)):
            continue
        matched_dt = best_dt[mask]
        speeds = 3.6 * float(dx_m) / np.maximum(np.abs(matched_dt), 1e-6)
        segment_speeds.extend(np.asarray(speeds, dtype=np.float64).tolist())
        segment_direction.extend([0 if float(dt) >= 0.0 else 1 for dt in matched_dt.tolist()])
        support_lengths.extend([2.0] * int(matched_dt.size))
        n_forward = int(np.sum(matched_dt >= 0.0))
        n_reverse = int(np.sum(matched_dt < 0.0))
        if n_forward > 0 and n_reverse > 0:
            crossing_proxy += 1.0
        if matched_dt.size >= 2:
            speed_spread = float(np.std(speeds) / max(1e-6, np.mean(speeds)))
            if speed_spread <= 0.12:
                near_parallel_proxy += 1.0

    direction_forward_ratio = (
        float(sum(1 for item in segment_direction if int(item) == 0)) / float(len(segment_direction))
        if segment_direction
        else float("nan")
    )
    segment_count = len(segment_speeds)
    vehicle_proxy_count = float(segment_count) / float(max(1, max(1, len(peak_rows) - 1)))
    return {
        "segment_speed_values": [float(v) for v in segment_speeds],
        "support_length_values": [float(v) for v in support_lengths],
        "segment_speed_kmh": _safe_stats(segment_speeds),
        "direction_forward_ratio": direction_forward_ratio,
        "support_length_channels": _safe_stats(support_lengths),
        "crossing_event_proxy": float(crossing_proxy),
        "near_parallel_event_proxy": float(near_parallel_proxy),
        "segment_count": int(segment_count),
        "vehicle_count_proxy": vehicle_proxy_count,
        "active_peak_channel_count": int(active_peak_channels),
    }


def _window_similarity_score(window_summary: dict[str, Any], global_summary: dict[str, Any]) -> float:
    keys = [
        ("total_peaks", "peak_density", "total_peaks_per_window", "q50"),
        ("peaks_per_channel_mean", "peak_density", "peaks_per_channel_per_window", "q50"),
        ("zero_ratio_mean", "zero_components", "zero_ratio_per_channel", "mean"),
        ("positive_ratio", "normalized_window_stats", "positive_ratio", "q50"),
        ("normalized_abs_q95", "normalized_window_stats", "abs_q95", "q50"),
    ]
    penalties: list[float] = []
    for local_key, group_key, stat_key, stat_field in keys:
        target = _finite_value(global_summary[group_key][stat_key].get(stat_field), float("nan"))
        if not math.isfinite(target):
            continue
        value = _finite_value(window_summary.get(local_key), target)
        denom = max(1e-6, abs(target))
        penalties.append(abs(value - target) / denom)
    if not penalties:
        return 1.0
    return float(1.0 / (1.0 + float(np.mean(penalties))))


def _build_generator_defaults(profile: dict[str, Any]) -> dict[str, Any]:
    peak_density = profile["peak_density"]
    zero_components = profile["zero_components"]
    proxy = profile["vehicle_proxy"]
    speed_stats = proxy["segment_speed_kmh"]
    support_stats = proxy["support_length_channels"]
    vehicle_proxy = proxy["vehicle_count_proxy_per_window"]
    dead_stats = zero_components["dead_channel_count_per_window"]
    drop_width = zero_components["drop_block_channel_width"]
    drop_duration = zero_components["drop_block_duration_s"]
    stable_dead_channels = [int(v) for v in zero_components.get("stable_dead_channel_indices", [])]
    stable_dead_count = int(len(stable_dead_channels))
    probabilistic_dead_channels = [
        item
        for item in zero_components.get("probabilistic_dead_channels", [])
        if isinstance(item, dict)
    ]
    expected_prob_dead = float(sum(_finite_value(item.get("probability"), 0.0) for item in probabilistic_dead_channels))

    speed_min = max(40.0, _finite_value(speed_stats.get("q10"), 68.0))
    speed_max = max(speed_min + 1.0, _finite_value(speed_stats.get("q90"), 88.0))
    vehicle_q10 = _finite_value(vehicle_proxy.get("q10"), 0.0)
    vehicle_mean = max(0.0, _finite_value(vehicle_proxy.get("q50"), 3.0))
    vehicle_q90 = max(vehicle_mean, _finite_value(vehicle_proxy.get("q90"), max(1.0, vehicle_mean + 1.0)))
    visible_mean = max(4.0, _finite_value(support_stats.get("q50"), 6.0))
    peak_q50 = max(1.0, _finite_value(peak_density["total_peaks_per_window"].get("q50"), 120.0))
    clutter_top_up_rate = min(18.0, max(0.0, 0.05 * peak_q50))
    dead_q50 = max(0.0, _finite_value(dead_stats.get("q50"), 4.0))
    dead_q90 = max(dead_q50, _finite_value(dead_stats.get("q90"), dead_q50 + 2.0))
    residual_dead_q50 = max(0.0, dead_q50 - float(stable_dead_count) - expected_prob_dead)
    residual_dead_q90 = max(residual_dead_q50, dead_q90 - float(stable_dead_count) - expected_prob_dead)
    drop_count_q50 = max(0.0, _finite_value(zero_components["drop_block_count_per_window"].get("q50"), 4.0))
    residual_drop_rate = 0.0 if stable_dead_count > 0 else min(1.0, max(0.0, 0.1 * drop_count_q50))
    duration_q10 = max(0.05, _finite_value(drop_duration.get("q10"), 0.4))
    duration_q50 = max(duration_q10, _finite_value(drop_duration.get("q50"), 1.2))
    duration_q90 = max(duration_q50, _finite_value(drop_duration.get("q90"), duration_q50 + 0.5))

    canonical_peak_amp = max(0.05, _finite_value(peak_density["peak_value"].get("q90"), 0.60))
    canonical_sigma_s = max(0.02, _finite_value(peak_density["peak_sigma_t_s"].get("q50"), 0.15))

    defaults = {
        "background_scale_min": 0.98,
        "background_scale_max": 1.02,
        "background_offset_std": 0.0,
        "vehicles_min": int(max(0, math.floor(max(0.0, vehicle_q10 - 0.25)))),
        "vehicles_max": int(max(1, math.ceil(max(1.0, vehicle_q90)))),
        "speed_min_kmh": float(speed_min),
        "speed_max_kmh": float(speed_max),
        "speed_norm_kmh": 150.0,
        "fixed_amp": float(canonical_peak_amp),
        "sigma_seconds": float(canonical_sigma_s),
        "primary_ratio": _clip01(_finite_value(proxy.get("direction_forward_ratio"), 0.83)),
        "min_visible_channels": int(max(4, round(min(16.0, visible_mean)))),
        "isolated_noise_ratio": 1.0 if clutter_top_up_rate >= 1.0 else 0.0,
        "isolated_noise_rate": float(clutter_top_up_rate),
        "isolated_noise_amp_min": float(canonical_peak_amp),
        "isolated_noise_amp_max": float(canonical_peak_amp),
        "isolated_noise_sigma_min_s": float(canonical_sigma_s),
        "isolated_noise_sigma_max_s": float(canonical_sigma_s),
        "per_vehicle_drop_channel_ratio": 1.0 if _finite_value(dead_stats.get("q50"), 0.0) >= 1.0 else 0.5,
        "per_vehicle_drop_channel_min": int(max(0, round(max(1.0, _finite_value(support_stats.get("q10"), 4.0) * 0.3)))),
        "per_vehicle_drop_channel_max": int(max(1, round(max(2.0, _finite_value(support_stats.get("q50"), 6.0) * 0.6)))),
        "dead_channel_indices": ",".join(str(idx) for idx in stable_dead_channels),
        "probabilistic_dead_channel_indices": ",".join(str(int(item.get("channel", -1))) for item in probabilistic_dead_channels),
        "random_dead_channel_ratio": 1.0 if residual_dead_q50 >= 1.0 else 0.35,
        "random_dead_channel_min": int(round(residual_dead_q50)),
        "random_dead_channel_max": int(max(round(residual_dead_q50), round(residual_dead_q90))),
        "zero_background_ratio": 1.0 if residual_drop_rate > 0.0 else 0.0,
        "zero_background_rate": float(residual_drop_rate),
        "zero_background_channel_min": int(max(1, round(_finite_value(drop_width.get("q10"), 1.0)))),
        "zero_background_channel_max": int(max(1, round(min(_finite_value(drop_width.get("q90"), 3.0), _finite_value(drop_width.get("q50"), 2.0) + 1.0)))),
        "zero_background_duration_min_s": float(duration_q10),
        "zero_background_duration_max_s": float(min(duration_q90, duration_q50 + 0.25 * max(0.0, duration_q90 - duration_q50))),
    }
    defaults["vehicles_max"] = max(defaults["vehicles_min"], defaults["vehicles_max"])
    defaults["speed_max_kmh"] = max(defaults["speed_min_kmh"] + 1.0, defaults["speed_max_kmh"])
    defaults["background_scale_max"] = max(defaults["background_scale_min"], defaults["background_scale_max"])
    defaults["isolated_noise_amp_max"] = max(defaults["isolated_noise_amp_min"], defaults["isolated_noise_amp_max"])
    defaults["isolated_noise_sigma_max_s"] = max(defaults["isolated_noise_sigma_min_s"], defaults["isolated_noise_sigma_max_s"])
    defaults["per_vehicle_drop_channel_max"] = max(defaults["per_vehicle_drop_channel_min"], defaults["per_vehicle_drop_channel_max"])
    defaults["random_dead_channel_max"] = max(defaults["random_dead_channel_min"], defaults["random_dead_channel_max"])
    defaults["zero_background_channel_max"] = max(defaults["zero_background_channel_min"], defaults["zero_background_channel_max"])
    defaults["zero_background_duration_max_s"] = max(defaults["zero_background_duration_min_s"], defaults["zero_background_duration_max_s"])
    return defaults


def _report_markdown(profile: dict[str, Any], realism_profile: dict[str, Any]) -> str:
    zero = profile["zero_components"]
    peaks = profile["peak_density"]
    proxy = realism_profile["vehicle_proxy"]
    defaults = realism_profile["generator_defaults"]
    lines = [
        "# Real Background Profile",
        "",
        f"- Input: `{profile['input']}`",
        f"- Shape used for profiling: `{tuple(profile['shape_time_channel'])}`",
        f"- FS / DX: `{profile['fs']}` Hz / `{profile['dx_m']}` m",
        f"- Sampled windows: `{realism_profile['window_catalog']['sampled_window_count']}`",
        "",
        "## Core Statistics",
        "",
        f"- Positive-value mean: `{profile['positive_value_stats']['mean']:.6f}`",
        f"- Positive-value q90 / q99: `{profile['positive_value_stats']['q90']:.6f}` / `{profile['positive_value_stats']['q99']:.6f}`",
        f"- Dead channel count: `{zero['dead_channel_count']}`",
        f"- Zero ratio mean across channels: `{zero['zero_ratio_per_channel']['mean']:.4f}`",
        f"- Drop-block width mean: `{zero['drop_block_channel_width']['mean']:.2f}` channels",
        f"- Drop-block duration mean: `{zero['drop_block_duration_s']['mean']:.2f}` s",
        f"- Peaks per window mean: `{peaks['total_peaks_per_window']['mean']:.2f}`",
        f"- Peaks per channel per window mean: `{peaks['peaks_per_channel_per_window']['mean']:.2f}`",
        f"- Peak sigma_t mean: `{peaks['peak_sigma_t_s']['mean']:.4f}` s",
        f"- Stable dead channels: `{zero.get('stable_dead_channel_indices', [])}`",
        f"- Probabilistic dead channels: `{[int(item.get('channel', -1)) for item in zero.get('probabilistic_dead_channels', [])]}`",
        "",
        "## Proxy Vehicle Statistics",
        "",
        f"- Proxy vehicle count per window q10 / q50 / q90: `{proxy['vehicle_count_proxy_per_window']['q10']:.2f}` / `{proxy['vehicle_count_proxy_per_window']['q50']:.2f}` / `{proxy['vehicle_count_proxy_per_window']['q90']:.2f}`",
        f"- Proxy speed q10 / q50 / q90: `{proxy['segment_speed_kmh']['q10']:.2f}` / `{proxy['segment_speed_kmh']['q50']:.2f}` / `{proxy['segment_speed_kmh']['q90']:.2f}` km/h",
        f"- Forward proxy ratio: `{proxy['direction_forward_ratio']:.3f}`",
        f"- Crossing proxy mean per window: `{proxy['crossing_event_proxy_per_window']['mean']:.3f}`",
        f"- Near-parallel proxy mean per window: `{proxy['near_parallel_event_proxy_per_window']['mean']:.3f}`",
        "",
        "## Suggested Generator Defaults",
        "",
        f"- vehicles_min / vehicles_max: `{defaults['vehicles_min']}` / `{defaults['vehicles_max']}`",
        f"- speed_min_kmh / speed_max_kmh: `{defaults['speed_min_kmh']:.2f}` / `{defaults['speed_max_kmh']:.2f}`",
        f"- fixed_amp / sigma_seconds: `{defaults['fixed_amp']:.3f}` / `{defaults['sigma_seconds']:.4f}`",
        f"- isolated_noise_rate: `{defaults['isolated_noise_rate']:.2f}`",
        f"- probabilistic_dead_channel_indices: `{defaults.get('probabilistic_dead_channel_indices', '')}`",
        f"- random_dead_channel_min / max: `{defaults['random_dead_channel_min']}` / `{defaults['random_dead_channel_max']}`",
        f"- zero_background_rate: `{defaults['zero_background_rate']:.2f}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    x = _load_array(
        Path(args.input).expanduser(),
        layout=str(args.array_layout),
        channel_start=int(args.channel_start),
        channel_count=int(args.channel_count),
    )

    positive = x[x > float(args.zero_threshold)]
    positive_q = np.quantile(positive, [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]).tolist() if positive.size else []
    zero_components = _profile_zero_components(
        x,
        fs=float(args.fs),
        zero_threshold=float(args.zero_threshold),
        component_time_downsample=int(args.component_time_downsample),
    )

    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    stride_samples = int(round(float(args.window_stride_seconds) * float(args.fs)))
    window_starts = _candidate_window_starts(int(x.shape[0]), window_samples, stride_samples)

    total_peaks_per_window: list[float] = []
    per_channel_peaks: list[float] = []
    peak_values: list[float] = []
    sigma_t_s: list[float] = []
    norm_pos_ratio: list[float] = []
    norm_zero_ratio: list[float] = []
    norm_abs_q95: list[float] = []
    dead_channel_counts: list[float] = []
    drop_block_count_per_window: list[float] = []
    window_zero_ratios: list[np.ndarray] = []
    vehicle_proxy_counts: list[float] = []
    crossing_proxy_per_window: list[float] = []
    near_parallel_proxy_per_window: list[float] = []
    proxy_speeds_all: list[float] = []
    proxy_support_all: list[float] = []
    direction_forward_hits = 0
    direction_total_hits = 0
    window_summaries: list[dict[str, Any]] = []

    for start in window_starts:
        window_tc = np.asarray(x[start : start + window_samples], dtype=np.float32)
        normalized_window_ct = _prepare_window(
            window_tc,
            clip_ratio=float(args.clip_ratio),
            zero_threshold=float(args.zero_threshold),
        )
        peak_rows = _extract_window_peak_rows(
            window_tc,
            fs=float(args.fs),
            peak_height=float(args.peak_height),
            peak_prominence=float(args.peak_prominence),
            peak_distance_seconds=float(args.peak_distance_seconds),
        )
        window_peak_stats = _window_stats_from_peak_rows(
            peak_rows,
            normalized_window_ct=normalized_window_ct,
            zero_threshold=float(args.zero_threshold),
        )
        total_peaks_per_window.append(float(window_peak_stats["total_peaks"]))
        per_channel_peaks.append(float(window_peak_stats["peaks_per_channel_mean"]))
        peak_values.extend(float(v) for v in window_peak_stats["peak_values"])
        sigma_t_s.extend(float(v) for v in window_peak_stats["peak_sigma_t_s"])
        norm_pos_ratio.append(float(window_peak_stats["positive_ratio"]))
        norm_zero_ratio.append(float(window_peak_stats["zero_ratio_mean"]))
        norm_abs_q95.append(float(window_peak_stats["normalized_abs_q95"]))

        zero_local = _profile_zero_components(
            window_tc,
            fs=float(args.fs),
            zero_threshold=float(args.zero_threshold),
            component_time_downsample=int(args.component_time_downsample),
        )
        window_zero_ratios.append(np.mean(np.abs(window_tc) <= float(args.zero_threshold), axis=0).astype(np.float32, copy=False))
        dead_channel_counts.append(float(zero_local["dead_channel_count"]))
        block_count = float(zero_local["drop_block_channel_width"].get("count", 0.0))
        drop_block_count_per_window.append(block_count)

        proxy = _estimate_vehicle_proxy(
            peak_rows,
            dx_m=float(args.dx_m),
            proxy_speed_min_kmh=float(args.proxy_speed_min_kmh),
            proxy_speed_max_kmh=float(args.proxy_speed_max_kmh),
            proxy_match_slack_s=float(args.proxy_match_slack_s),
        )
        proxy_speeds_all.extend(float(v) for v in proxy["segment_speed_values"])
        proxy_support_all.extend(float(v) for v in proxy["support_length_values"])
        if math.isfinite(float(proxy["direction_forward_ratio"])):
            direction_total_hits += int(proxy["segment_speed_kmh"].get("count", 0))
            direction_forward_hits += int(round(float(proxy["direction_forward_ratio"]) * int(proxy["segment_speed_kmh"].get("count", 0))))
        vehicle_proxy_counts.append(float(proxy["vehicle_count_proxy"]))
        crossing_proxy_per_window.append(float(proxy["crossing_event_proxy"]))
        near_parallel_proxy_per_window.append(float(proxy["near_parallel_event_proxy"]))

        window_summaries.append(
            {
                "start_sample": int(start),
                "start_seconds": float(start) / float(args.fs),
                "positive_ratio": float(window_peak_stats["positive_ratio"]),
                "zero_ratio_mean": float(window_peak_stats["zero_ratio_mean"]),
                "normalized_abs_q95": float(window_peak_stats["normalized_abs_q95"]),
                "total_peaks": int(window_peak_stats["total_peaks"]),
                "peaks_per_channel_mean": float(window_peak_stats["peaks_per_channel_mean"]),
                "dead_channel_count": int(zero_local["dead_channel_count"]),
                "drop_block_count_proxy": int(block_count),
                "vehicle_count_proxy": float(proxy["vehicle_count_proxy"]),
                "crossing_event_proxy": float(proxy["crossing_event_proxy"]),
                "near_parallel_event_proxy": float(proxy["near_parallel_event_proxy"]),
            }
        )

    peak_density = {
        "sampled_window_count": int(len(window_starts)),
        "window_seconds": float(args.window_seconds),
        "window_stride_seconds": float(args.window_stride_seconds),
        "total_peaks_per_window": _safe_stats(total_peaks_per_window),
        "peaks_per_channel_per_window": _safe_stats(per_channel_peaks),
        "peak_value": _safe_stats(peak_values),
        "peak_sigma_t_s": _safe_stats(sigma_t_s),
    }
    normalized_window_stats = {
        "positive_ratio": _safe_stats(norm_pos_ratio),
        "zero_ratio_mean": _safe_stats(norm_zero_ratio),
        "abs_q95": _safe_stats(norm_abs_q95),
    }
    vehicle_proxy = {
        "segment_speed_kmh": _safe_stats([value for value in proxy_speeds_all if math.isfinite(value) and value > 0.0]),
        "support_length_channels": _safe_stats([value for value in proxy_support_all if math.isfinite(value) and value > 0.0]),
        "vehicle_count_proxy_per_window": _safe_stats(vehicle_proxy_counts),
        "crossing_event_proxy_per_window": _safe_stats(crossing_proxy_per_window),
        "near_parallel_event_proxy_per_window": _safe_stats(near_parallel_proxy_per_window),
        "direction_forward_ratio": (
            float(direction_forward_hits) / float(direction_total_hits)
            if direction_total_hits > 0
            else float("nan")
        ),
    }
    channel_dead_profile = _profile_probabilistic_dead_channels(window_zero_ratios)
    zero_components["stable_dead_channel_indices"] = list(channel_dead_profile["stable_dead_channel_indices"])
    zero_components["dead_channel_indices"] = list(channel_dead_profile["stable_dead_channel_indices"])
    zero_components["dead_channel_count"] = int(len(channel_dead_profile["stable_dead_channel_indices"]))

    base_profile = {
        "input": str(Path(args.input).expanduser()),
        "array_layout": str(args.array_layout),
        "shape_time_channel": [int(x.shape[0]), int(x.shape[1])],
        "fs": float(args.fs),
        "dx_m": float(args.dx_m),
        "channel_start": int(args.channel_start),
        "channel_count": int(args.channel_count),
        "window_seconds": float(args.window_seconds),
        "window_stride_seconds": float(args.window_stride_seconds),
        "zero_threshold": float(args.zero_threshold),
        "positive_value_stats": {
            **_safe_stats(positive.astype(np.float64).tolist() if positive.size else []),
            "q95": float(positive_q[2]) if len(positive_q) >= 3 else float("nan"),
            "q99": float(positive_q[3]) if len(positive_q) >= 4 else float("nan"),
            "q999": float(positive_q[5]) if len(positive_q) >= 6 else float("nan"),
        },
        "zero_components": zero_components,
        "peak_density": peak_density,
        "normalized_window_stats": normalized_window_stats,
        "vehicle_proxy": vehicle_proxy,
    }

    for window_summary in window_summaries:
        window_summary["training_like_score"] = _window_similarity_score(window_summary, base_profile)
    training_scores = [float(item["training_like_score"]) for item in window_summaries]
    mean_score = float(np.mean(training_scores)) if training_scores else 1.0
    for window_summary in window_summaries:
        window_summary["sampling_weight"] = float(max(1e-3, window_summary["training_like_score"] / max(1e-6, mean_score)))
    if int(args.window_catalog_limit) > 0 and len(window_summaries) > int(args.window_catalog_limit):
        window_summaries = sorted(window_summaries, key=lambda item: float(item["sampling_weight"]), reverse=True)[: int(args.window_catalog_limit)]

    realism_profile = {
        "format": "realism_profile_v1",
        "input": str(Path(args.input).expanduser()),
        "array_layout": str(args.array_layout),
        "shape_time_channel": [int(x.shape[0]), int(x.shape[1])],
        "fs": float(args.fs),
        "dx_m": float(args.dx_m),
        "channel_start": int(args.channel_start),
        "channel_count": int(args.channel_count),
        "window_seconds": float(args.window_seconds),
        "window_stride_seconds": float(args.window_stride_seconds),
        "peak_detection": {
            "peak_height": float(args.peak_height),
            "peak_prominence": float(args.peak_prominence),
            "peak_distance_seconds": float(args.peak_distance_seconds),
        },
        "zero_threshold": float(args.zero_threshold),
        "clip_ratio": float(args.clip_ratio),
        "positive_value_stats": base_profile["positive_value_stats"],
        "peak_density": peak_density,
        "normalized_window_stats": normalized_window_stats,
        "zero_components": {
            **zero_components,
            "dead_channel_count_per_window": _safe_stats(dead_channel_counts),
            "drop_block_count_per_window": _safe_stats(drop_block_count_per_window),
            "probabilistic_dead_channels": channel_dead_profile["probabilistic_dead_channels"],
            "channel_zero_ratio_summary": channel_dead_profile["channel_zero_ratio_summary"],
        },
        "vehicle_proxy": vehicle_proxy,
        "window_catalog": {
            "sampled_window_count": int(len(window_starts)),
            "stored_window_count": int(len(window_summaries)),
            "entries": window_summaries,
        },
    }
    realism_profile["generator_defaults"] = _build_generator_defaults(realism_profile)

    (out_dir / "profile.json").write_text(json.dumps(base_profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "realism_profile.json").write_text(json.dumps(realism_profile, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(_report_markdown(base_profile, realism_profile), encoding="utf-8")
    print(f"wrote profile: {out_dir / 'profile.json'}", flush=True)
    print(f"wrote realism profile: {out_dir / 'realism_profile.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
