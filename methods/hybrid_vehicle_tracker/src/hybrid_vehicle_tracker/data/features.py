from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_vehicle_tracker.types import StationGeometry


@dataclass
class FeatureBatch:
    tensor: np.ndarray
    raw_score: np.ndarray
    pre_score: np.ndarray
    gauss_score: np.ndarray
    quality: np.ndarray
    coordinates: np.ndarray
    feature_rate_hz: float
    sample_rate_hz: float
    duration_s: float

    @property
    def station_count(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def time_bins(self) -> int:
        return int(self.tensor.shape[2])


def _robust_z(values: np.ndarray, axis: int = 0) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True)
    scale = np.maximum(1.4826 * mad, 1e-6)
    return (values - median) / scale


def _percentile_scale(values: np.ndarray) -> np.ndarray:
    low = np.quantile(values, 0.01, axis=0, keepdims=True)
    high = np.quantile(values, 0.995, axis=0, keepdims=True)
    width = np.maximum(high - low, 1e-6)
    return np.clip((values - low) / width, 0.0, 1.0)


def _pool_time(values: np.ndarray, bin_samples: int, mode: str) -> np.ndarray:
    usable = values.shape[0] - values.shape[0] % bin_samples
    if usable <= 0:
        raise ValueError("window is shorter than one feature bin")
    reshaped = values[:usable].reshape(-1, bin_samples, values.shape[1])
    if mode == "max":
        return np.max(reshaped, axis=1)
    if mode == "mean":
        return np.mean(reshaped, axis=1)
    if mode == "rms":
        return np.sqrt(np.mean(np.square(reshaped, dtype=np.float64), axis=1))
    raise ValueError(f"unknown pooling mode {mode!r}")


def build_feature_batch(
    raw: np.ndarray,
    pre: np.ndarray,
    gauss: np.ndarray,
    geometry: StationGeometry,
    *,
    sample_rate_hz: float = 1000.0,
    feature_rate_hz: float = 20.0,
) -> FeatureBatch:
    """Create the five-plane network input from aligned modal arrays."""
    if raw.shape != pre.shape or raw.shape != gauss.shape:
        raise ValueError(f"modal shapes differ: {raw.shape}, {pre.shape}, {gauss.shape}")
    if raw.ndim != 2 or raw.shape[1] != len(geometry):
        raise ValueError(
            f"expected [time, {len(geometry)}] modal arrays, got {raw.shape}"
        )
    ratio = sample_rate_hz / feature_rate_hz
    bin_samples = int(round(ratio))
    if not np.isclose(ratio, bin_samples):
        raise ValueError("sample_rate_hz must be an integer multiple of feature_rate_hz")

    raw_rms = _pool_time(raw, bin_samples, "rms")
    raw_log = np.log1p(raw_rms)
    raw_z = _robust_z(raw_log)
    raw_plane = np.clip(raw_z / 6.0, -1.0, 1.0)
    raw_score = 1.0 / (1.0 + np.exp(-np.clip(raw_z, -20.0, 20.0)))

    pre_max = _pool_time(pre, bin_samples, "max")
    pre_score = _percentile_scale(pre_max)
    gauss_max = _pool_time(gauss, bin_samples, "max")
    gauss_score = np.clip(gauss_max, 0.0, 1.0)

    finite_ratio = np.mean(np.isfinite(raw), axis=0)
    variability = np.std(raw_log, axis=0)
    positive = variability[variability > 0]
    reference = float(np.median(positive)) if positive.size else 1.0
    quality = np.clip(finite_ratio * variability / max(reference, 1e-6), 0.0, 1.0)

    positions = geometry.relative_positions_m.astype(np.float32)
    span = max(float(positions[-1]), 1.0)
    coordinates = positions / span
    time_bins = raw_plane.shape[0]
    quality_plane = np.broadcast_to(quality[None, :], (time_bins, len(geometry)))
    coordinate_plane = np.broadcast_to(coordinates[None, :], (time_bins, len(geometry)))

    tensor = np.stack(
        [raw_plane, pre_score, gauss_score, quality_plane, coordinate_plane], axis=0
    ).transpose(0, 2, 1)
    return FeatureBatch(
        tensor=np.asarray(tensor, dtype=np.float32),
        raw_score=np.asarray(raw_score.T, dtype=np.float32),
        pre_score=np.asarray(pre_score.T, dtype=np.float32),
        gauss_score=np.asarray(gauss_score.T, dtype=np.float32),
        quality=np.asarray(quality, dtype=np.float32),
        coordinates=np.asarray(coordinates, dtype=np.float32),
        feature_rate_hz=float(feature_rate_hz),
        sample_rate_hz=float(sample_rate_hz),
        duration_s=float(raw.shape[0] / sample_rate_hz),
    )

