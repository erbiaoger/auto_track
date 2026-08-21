from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks

from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.data.features import FeatureBatch
from hybrid_vehicle_tracker.types import StationGeometry, VehicleObservation


def _calibrate_full_rate(values: np.ndarray) -> np.ndarray:
    low = np.quantile(values, 0.01, axis=0, keepdims=True)
    high = np.quantile(values, 0.995, axis=0, keepdims=True)
    return np.clip((values - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)


def _sample_grid(grid: np.ndarray | None, channel: int, time_s: float, rate: float) -> float:
    if grid is None:
        return 0.0
    index = int(np.clip(round(time_s * rate), 0, grid.shape[-1] - 1))
    return float(grid[channel, index])


def extract_observations(
    gauss: np.ndarray,
    pre: np.ndarray,
    features: FeatureBatch,
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    network_probability: np.ndarray | None = None,
    crossing_probability: np.ndarray | None = None,
    embedding: np.ndarray | None = None,
) -> list[VehicleObservation]:
    """Extract strong and weak exact-time observations and merge local duplicates."""
    if gauss.shape != pre.shape or gauss.ndim != 2:
        raise ValueError("gauss and pre must be aligned [time, station] arrays")
    sample_rate = features.sample_rate_hz
    min_distance = max(1, int(config.candidate_min_distance_s * sample_rate))
    calibrated_pre = _calibrate_full_rate(pre)
    pre_event_score = 1.0 - calibrated_pre
    candidates: dict[int, list[tuple[int, bool]]] = defaultdict(list)

    for channel in range(gauss.shape[1]):
        strong_indices, _ = find_peaks(
            gauss[:, channel],
            height=config.strong_gauss_threshold,
            distance=min_distance,
        )
        candidates[channel].extend((int(index), True) for index in strong_indices)

        threshold = float(np.quantile(pre_event_score[:, channel], config.weak_pre_quantile))
        weak_indices, _ = find_peaks(
            pre_event_score[:, channel],
            height=max(threshold, 0.65),
            distance=min_distance,
        )
        candidates[channel].extend((int(index), False) for index in weak_indices)

        if network_probability is not None:
            net_indices, _ = find_peaks(
                network_probability[channel], height=0.55, distance=max(1, int(0.75 * features.feature_rate_hz))
            )
            candidates[channel].extend(
                (int(round(index * sample_rate / features.feature_rate_hz)), False)
                for index in net_indices
            )

    observations: list[VehicleObservation] = []
    merge_samples = int(round(0.20 * sample_rate))
    for channel, rows in candidates.items():
        rows.sort(key=lambda item: item[0])
        groups: list[list[tuple[int, bool]]] = []
        for row in rows:
            if not groups or row[0] - groups[-1][-1][0] > merge_samples:
                groups.append([row])
            else:
                groups[-1].append(row)

        station = geometry.stations[channel]
        for group in groups:
            index, strong = max(
                group,
                key=lambda item: (
                    bool(item[1]),
                    float(gauss[min(item[0], gauss.shape[0] - 1), channel]),
                    float(pre_event_score[min(item[0], pre.shape[0] - 1), channel]),
                ),
            )
            index = int(np.clip(index, 0, gauss.shape[0] - 1))
            time_s = index / sample_rate
            bin_index = int(
                np.clip(round(time_s * features.feature_rate_hz), 0, features.time_bins - 1)
            )
            emb: tuple[float, ...] = ()
            if embedding is not None:
                emb = tuple(float(value) for value in embedding[:, channel, bin_index])
            net_score = _sample_grid(
                network_probability, channel, time_s, features.feature_rate_hz
            )
            raw_score = float(features.raw_score[channel, bin_index])
            pre_score = float(pre_event_score[index, channel])
            gauss_score = float(np.clip(gauss[index, channel], 0.0, 1.0))
            is_strong = bool(strong and gauss_score >= config.strong_gauss_threshold)
            if not is_strong and max(pre_score, net_score, raw_score) < 0.55:
                continue
            observations.append(
                VehicleObservation(
                    observation_id=-1,
                    channel_index=channel,
                    station_id=station.station_id,
                    position_m=station.position_m,
                    time_s=float(time_s),
                    gauss_score=gauss_score,
                    pre_score=pre_score,
                    raw_energy=raw_score,
                    network_score=net_score,
                    crossing_score=_sample_grid(
                        crossing_probability, channel, time_s, features.feature_rate_hz
                    ),
                    strong=is_strong,
                    ambiguous=False,
                    embedding=emb,
                )
            )

    observations.sort(key=lambda item: item.evidence_score, reverse=True)
    observations = observations[: config.max_candidates]
    observations.sort(key=lambda item: (item.position_m, item.time_s))
    for observation_id, observation in enumerate(observations):
        observation.observation_id = observation_id
    return observations
