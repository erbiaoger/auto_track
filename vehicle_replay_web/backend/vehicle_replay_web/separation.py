"""Large-vehicle signal separation adapted from the three-way script.

The original script is an offline diagnostic program.  This module keeps the
small, window-safe part needed by the replay workers: pick candidate peaks,
classify them by local RMS, and synthesize the large-vehicle Gaussian plane.
"""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_SEPARATION_CONFIG: dict[str, Any] = {
    "enabled": True,
    "pick_threshold": 0.5,
    "min_gap_s": 3.0,
    "energy_window_s": 1.0,
    # Match the original three-way separation script used for the reference
    # figure: classify the top 25% of picked RMS values as large vehicles.
    "split_method": "percentile",
    "split_percentile": 75.0,
    "split_threshold": 0.35,
    "gaussian_width_s": 0.5,
    "amplitude_min": 0.25,
    "amplitude_max": 1.0,
}


def normalize_separation_config(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate and fill the public web configuration."""
    config = dict(DEFAULT_SEPARATION_CONFIG)
    if payload:
        config.update(payload)
    config["enabled"] = bool(config["enabled"])
    for key in ("pick_threshold", "split_threshold"):
        config[key] = float(config[key])
        if not 0.0 <= config[key] <= 1.0:
            raise ValueError(f"{key} must be between 0 and 1")
    config["split_percentile"] = float(config["split_percentile"])
    if not 0.0 <= config["split_percentile"] <= 100.0:
        raise ValueError("split_percentile must be between 0 and 100")
    for key in ("min_gap_s", "energy_window_s", "gaussian_width_s"):
        config[key] = float(config[key])
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    config["split_method"] = str(config["split_method"])
    if config["split_method"] not in {"threshold", "percentile", "kmeans"}:
        raise ValueError("split_method must be threshold, percentile, or kmeans")
    config["amplitude_min"] = float(config["amplitude_min"])
    config["amplitude_max"] = float(config["amplitude_max"])
    if config["amplitude_min"] < 0 or config["amplitude_max"] <= config["amplitude_min"]:
        raise ValueError("amplitude range is invalid")
    return config


def _local_rms(signal: np.ndarray, sample_rate_hz: float, window_s: float) -> np.ndarray:
    half = max(1, int(window_s * sample_rate_hz / 2.0))
    values = np.asarray(signal, dtype=np.float32)
    squared = values.astype(np.float64) ** 2
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    left = np.maximum(0, np.arange(values.size) - half)
    right = np.minimum(values.size, np.arange(values.size) + half + 1)
    return np.sqrt((cumulative[right] - cumulative[left]) / (right - left) + 1e-12).astype(np.float32)


def _pick_peaks(probability: np.ndarray, sample_rate_hz: float, threshold: float, min_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    above = np.asarray(probability) > threshold
    gap = max(1, int(min_gap_s * sample_rate_hz))
    indices: list[int] = []
    probabilities: list[float] = []
    index = 0
    while index < above.size:
        if not above[index]:
            index += 1
            continue
        end = index
        while end < above.size and above[end]:
            end += 1
        peak = index + int(np.argmax(probability[index:end]))
        indices.append(peak)
        probabilities.append(float(probability[peak]))
        index = end
    if len(indices) <= 1:
        return np.asarray(indices, dtype=np.int64), np.asarray(probabilities, dtype=np.float32)
    merged_i = [indices[0]]
    merged_p = [probabilities[0]]
    for current_i, current_p in zip(indices[1:], probabilities[1:]):
        if current_i - merged_i[-1] < gap:
            if current_p > merged_p[-1]:
                merged_i[-1], merged_p[-1] = current_i, current_p
        else:
            merged_i.append(current_i)
            merged_p.append(current_p)
    return np.asarray(merged_i, dtype=np.int64), np.asarray(merged_p, dtype=np.float32)


def _classify(rms_values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if rms_values.size == 0:
        return np.zeros(0, dtype=bool)
    normalized = rms_values / (float(rms_values.max()) + 1e-12)
    method = config["split_method"]
    if method == "threshold":
        return normalized >= float(config["split_threshold"])
    if method == "percentile":
        return normalized >= np.percentile(normalized, float(config["split_percentile"]))
    values = normalized.astype(np.float64)
    small_center, large_center = float(values.min()), float(values.max())
    for _ in range(60):
        large = np.abs(values - large_center) < np.abs(values - small_center)
        new_small = float(values[~large].mean()) if (~large).any() else small_center
        new_large = float(values[large].mean()) if large.any() else large_center
        if abs(new_small - small_center) < 1e-7 and abs(new_large - large_center) < 1e-7:
            break
        small_center, large_center = new_small, new_large
    return large


def separate_large_vehicle_signal(
    raw: np.ndarray | None,
    prediction: np.ndarray | None,
    gauss: np.ndarray,
    sample_rate_hz: float,
    payload: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the script-compatible large-vehicle Gaussian plane.

    ``raw`` is used for vehicle-size energy when available.  For Gauss-only
    archives, Gauss is used for both energy and candidate probability so the
    web path remains usable without inventing missing Raw/Pre files.
    """
    config = normalize_separation_config(payload)
    source = np.asarray(raw if raw is not None else gauss, dtype=np.float32)
    probability = np.asarray(prediction if prediction is not None else gauss, dtype=np.float32)
    base_gauss = np.asarray(gauss, dtype=np.float32)
    if source.ndim != 2 or probability.ndim != 2 or base_gauss.ndim != 2:
        raise ValueError("separation inputs must be 2-D [time, station] arrays")
    if source.shape != probability.shape or source.shape != base_gauss.shape:
        raise ValueError(f"separation inputs are not aligned: {source.shape}, {probability.shape}, {base_gauss.shape}")
    if not config["enabled"]:
        return base_gauss.copy(), {"enabled": False, "pick_count": 0, "large_count": 0, "small_count": 0}

    sample_rate_hz = float(sample_rate_hz)
    rms_by_station = np.stack(
        [_local_rms(source[:, station], sample_rate_hz, float(config["energy_window_s"])) for station in range(source.shape[1])],
        axis=1,
    )
    picks: list[tuple[int, int, float, float]] = []
    for station in range(probability.shape[1]):
        indices, probabilities = _pick_peaks(
            probability[:, station], sample_rate_hz,
            float(config["pick_threshold"]), float(config["min_gap_s"]),
        )
        for index, probability_value in zip(indices.tolist(), probabilities.tolist()):
            picks.append((station, int(index), float(probability_value), float(rms_by_station[index, station])))
    if not picks:
        return np.zeros_like(base_gauss), {"enabled": True, "pick_count": 0, "large_count": 0, "small_count": 0}

    rms_values = np.asarray([item[3] for item in picks], dtype=np.float32)
    is_large = _classify(rms_values, config)
    large_picks = [item for item, large in zip(picks, is_large.tolist()) if large]
    rms_min = float(rms_values.min())
    rms_max = float(rms_values.max())
    width = float(config["gaussian_width_s"])
    output = np.zeros_like(base_gauss, dtype=np.float32)
    time = np.arange(base_gauss.shape[0], dtype=np.float32) / sample_rate_hz
    for station, index, _, rms in large_picks:
        center = float(index) / sample_rate_hz
        normalized_rms = (rms - rms_min) / (rms_max - rms_min + 1e-12)
        amplitude = float(config["amplitude_min"]) + (float(config["amplitude_max"]) - float(config["amplitude_min"])) * normalized_rms
        output[:, station] += amplitude * np.exp(-0.5 * ((time - center) / width) ** 2).astype(np.float32)
    return output, {
        "enabled": True,
        "pick_count": len(picks),
        "large_count": len(large_picks),
        "small_count": len(picks) - len(large_picks),
        "pick_threshold": float(config["pick_threshold"]),
        "split_method": config["split_method"],
        "split_threshold": float(config["split_threshold"]),
    }
