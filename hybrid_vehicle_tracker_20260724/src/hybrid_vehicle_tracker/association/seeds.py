from __future__ import annotations

from collections import defaultdict

import numpy as np

from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.types import HoughSeed, StationGeometry, VehicleObservation


def observation_pair_seeds(
    observations: list[VehicleObservation],
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    top_k: int,
    intercept_step_s: float,
) -> list[HoughSeed]:
    """Sparse physical Hough voting from strong observation pairs."""
    direction = int(config.motion_direction)
    x0 = geometry.positions_m[0]
    slope_step = (config.max_slowness_s_per_m - config.min_slowness_s_per_m) / 16.0
    votes: dict[tuple[int, int], list[float]] = defaultdict(list)
    supports: dict[tuple[int, int], set[int]] = defaultdict(set)
    strong = [item for item in observations if item.strong or item.evidence_score >= 0.75]
    for left_index, left in enumerate(strong):
        for right in strong[left_index + 1 :]:
            dx = direction * (right.position_m - left.position_m)
            if dx < min(config.min_span_m, config.max_gap_m) or dx > 3.0 * config.max_gap_m:
                continue
            dt = right.time_s - left.time_s
            if dt <= 0.0:
                continue
            magnitude = dt / dx
            if not config.min_slowness_s_per_m <= magnitude <= config.max_slowness_s_per_m:
                continue
            slope = direction * magnitude
            intercept = left.time_s - slope * (left.position_m - x0)
            key = (
                int(round((magnitude - config.min_slowness_s_per_m) / max(slope_step, 1e-6))),
                int(round(intercept / intercept_step_s)),
            )
            votes[key].append(np.sqrt(left.evidence_score * right.evidence_score))
            supports[key].update((left.observation_id, right.observation_id))

    ranked = sorted(
        votes,
        key=lambda key: (len(supports[key]), float(np.sum(votes[key]))),
        reverse=True,
    )
    seeds = []
    for slope_bin, intercept_bin in ranked[:top_k]:
        values = votes[(slope_bin, intercept_bin)]
        seeds.append(
            HoughSeed(
                slope_s_per_m=direction * (config.min_slowness_s_per_m + slope_bin * slope_step),
                intercept_s=intercept_bin * intercept_step_s,
                score=float(np.mean(values) * np.sqrt(len(values))),
                support=len(supports[(slope_bin, intercept_bin)]),
                source="observation_pair",
            )
        )
    return seeds


def merge_seeds(seeds: list[HoughSeed], *, top_k: int) -> list[HoughSeed]:
    selected: list[HoughSeed] = []
    for seed in sorted(seeds, key=lambda item: item.score, reverse=True):
        if any(
            abs(seed.slope_s_per_m - old.slope_s_per_m) <= 0.0015
            and abs(seed.intercept_s - old.intercept_s) <= 1.0
            for old in selected
        ):
            continue
        selected.append(seed)
        if len(selected) >= top_k:
            break
    return selected


def rescore_seeds_on_dense_evidence(
    seeds: list[HoughSeed],
    evidence: np.ndarray,
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    feature_rate_hz: float,
    duration_s: float,
) -> list[HoughSeed]:
    """Require every seed to be supported by the dense multimodal probability map."""
    if evidence.ndim != 2 or evidence.shape[0] != len(geometry):
        raise ValueError("dense evidence must have shape [station, time]")
    x = geometry.relative_positions_m
    radius = max(1, int(round(0.25 * feature_rate_hz)))
    rescored = []
    for seed in seeds:
        predicted = seed.intercept_s + seed.slope_s_per_m * x
        values = []
        for channel, time_s in enumerate(predicted):
            if not 0.0 <= time_s < duration_s:
                continue
            center = int(np.clip(round(time_s * feature_rate_hz), 0, evidence.shape[1] - 1))
            lower = max(0, center - radius)
            upper = min(evidence.shape[1], center + radius + 1)
            values.append(float(np.max(evidence[channel, lower:upper])))
        if len(values) < config.min_observations:
            continue
        array = np.asarray(values, dtype=np.float64)
        support = int(np.count_nonzero(array >= config.dense_support_threshold))
        top_count = max(config.min_observations, int(np.ceil(0.30 * len(array))))
        top_mean = float(np.mean(np.partition(array, -top_count)[-top_count:]))
        support_fraction = support / len(array)
        dense_score = float(
            np.clip(0.45 * np.mean(array) + 0.35 * top_mean + 0.20 * support_fraction, 1e-4, 1 - 1e-4)
        )
        if seed.source == "deep_hough":
            score = float(np.clip(0.70 * dense_score + 0.30 * seed.score, 1e-4, 1 - 1e-4))
        else:
            score = dense_score
        if support < config.min_observations or dense_score < config.min_dense_seed_score:
            continue
        rescored.append(
            HoughSeed(
                slope_s_per_m=seed.slope_s_per_m,
                intercept_s=seed.intercept_s,
                score=score,
                support=support,
                source=seed.source,
            )
        )
    return sorted(rescored, key=lambda item: (item.score, item.support), reverse=True)
