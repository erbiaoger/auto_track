from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.types import (
    CandidatePath,
    StationGeometry,
    TrackPoint,
    VehicleObservation,
    VehicleTrack,
)


def _fit_bounded_slowness(
    station_positions: np.ndarray,
    observed_indices: np.ndarray,
    observed_times: np.ndarray,
    config: AssociationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dx = np.diff(station_positions)
    variable_count = 1 + len(dx)
    observation_design = np.zeros((len(observed_indices), variable_count), dtype=np.float64)
    observation_design[:, 0] = 1.0
    for row, station_index in enumerate(observed_indices):
        observation_design[row, 1 : 1 + station_index] = dx[:station_index]
    if len(dx) > 1:
        smooth = np.zeros((len(dx) - 1, variable_count), dtype=np.float64)
        for row in range(len(dx) - 1):
            smooth[row, 1 + row] = -0.35
            smooth[row, 2 + row] = 0.35
    else:
        smooth = np.empty((0, variable_count), dtype=np.float64)

    weights = np.ones(len(observed_times), dtype=np.float64)
    solution = None
    for _ in range(4):
        design = np.vstack([observation_design * np.sqrt(weights)[:, None], smooth])
        target = np.concatenate([observed_times * np.sqrt(weights), np.zeros(len(smooth))])
        lower = np.concatenate(([-600.0], np.full(len(dx), config.min_slowness_s_per_m)))
        upper = np.concatenate(([600.0], np.full(len(dx), config.max_slowness_s_per_m)))
        solution = lsq_linear(design, target, bounds=(lower, upper), lsmr_tol="auto")
        residuals = observed_times - observation_design @ solution.x
        scale = max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 0.05)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.minimum(1.0, 1.0 / np.maximum(normalized, 1e-12))
    assert solution is not None
    fitted_times = solution.x[0] + np.concatenate(([0.0], np.cumsum(dx * solution.x[1:])))
    return fitted_times, solution.x[1:]


def _robust_linear_diagnostic(
    positions: np.ndarray,
    times: np.ndarray,
    config: AssociationConfig,
) -> tuple[float, float, np.ndarray]:
    relative = positions - positions[0]
    design = np.column_stack([np.ones(len(relative)), relative])
    weights = np.ones(len(times), dtype=np.float64)
    intercept = float(np.median(times))
    slope = 0.5 * (config.min_slowness_s_per_m + config.max_slowness_s_per_m)
    for _ in range(5):
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = times * np.sqrt(weights)
        solution, *_ = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)
        slope = float(
            np.clip(solution[1], config.min_slowness_s_per_m, config.max_slowness_s_per_m)
        )
        intercept = float(np.average(times - slope * relative, weights=weights))
        residuals = times - (intercept + slope * relative)
        scale = max(1.4826 * np.median(np.abs(residuals - np.median(residuals))), 0.05)
        normalized = np.abs(residuals) / (1.5 * scale)
        weights = np.minimum(1.0, 1.0 / np.maximum(normalized, 1e-12))
    residuals = times - (intercept + slope * relative)
    return intercept, slope, residuals


def refine_path(
    path: CandidatePath,
    observations: list[VehicleObservation],
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    duration_s: float,
    track_index: int,
) -> VehicleTrack | None:
    selected = [observations[index] for index in path.observation_ids]
    direction = int(config.motion_direction)
    # Fit in the travel coordinate (increasing along the vehicle motion), but
    # keep the public TrackPoint order in the original channel order.
    selected.sort(key=lambda item: direction * item.position_m)
    # Fit against the complete mapped array, not only the first/last observed
    # nodes.  With identical nuisance and vehicle peaks, a detector can miss a
    # legitimate head or tail; limiting points to selected nodes then makes a
    # long vehicle look like a short fragment.  Evidence remains selected-only
    # while every in-window mapped station receives an expected point.
    physical_rows = geometry.stations
    travel_rows = physical_rows if direction > 0 else tuple(reversed(physical_rows))
    reference_position_m = float(geometry.positions_m[0])
    station_positions = np.asarray(
        [direction * (station.position_m - reference_position_m) for station in travel_rows],
        dtype=np.float64,
    )
    travel_index = {
        station.channel_index: index for index, station in enumerate(travel_rows)
    }
    observed_indices = np.asarray(
        [travel_index[observation.channel_index] for observation in selected], dtype=np.int64
    )
    observed_times = np.asarray([observation.time_s for observation in selected], dtype=np.float64)
    _, diagnostic_slope, diagnostic_residuals = _robust_linear_diagnostic(
        np.asarray(
            [direction * (observation.position_m - reference_position_m) for observation in selected],
            dtype=np.float64,
        ),
        observed_times,
        config,
    )
    fitted_times, slowness = _fit_bounded_slowness(
        station_positions, observed_indices, observed_times, config
    )
    residuals = observed_times - fitted_times[observed_indices]
    median_residual = float(np.median(np.abs(diagnostic_residuals)))
    observed_span_m = float(
        abs(
            max(item.position_m for item in selected)
            - min(item.position_m for item in selected)
        )
    )
    max_gap_m = float(
        np.max(np.abs(np.diff([item.position_m for item in selected])))
        if len(selected) > 1
        else 0.0
    )
    if (
        len(selected) < config.min_observations
        or observed_span_m < config.min_span_m
        or median_residual > config.max_median_residual_s
        or max_gap_m > config.max_gap_m + 1e-6
    ):
        return None

    observation_by_channel = {item.channel_index: item for item in selected}
    residual_by_channel = {
        item.channel_index: float(residual)
        for item, residual in zip(selected, diagnostic_residuals)
    }
    fitted_time_by_channel = {
        station.channel_index: float(fitted_times[index])
        for index, station in enumerate(travel_rows)
    }
    points = []
    for station in physical_rows:
        observation = observation_by_channel.get(station.channel_index)
        fitted_time = fitted_time_by_channel[station.channel_index]
        # A vehicle may enter or leave the requested time window.  Do not
        # emit stations outside the window unless they carry a real observed
        # node, but keep all in-window mapped stations for a continuous head
        # and tail through missing channels.
        if not 0.0 <= fitted_time < duration_s and observation is None:
            continue
        points.append(
            TrackPoint(
                channel_index=station.channel_index,
                station_id=station.station_id,
                position_m=station.position_m,
                time_s=float(
                    observation.time_s
                    if observation
                    else fitted_time
                ),
                observed=observation is not None,
                observation_id=observation.observation_id if observation else None,
                residual_s=residual_by_channel.get(station.channel_index),
                ambiguous=bool(observation and observation.ambiguous),
            )
        )
    local_speeds = (3.6 / np.clip(slowness, 1e-6, None)).tolist()
    median_speed = float(3.6 / diagnostic_slope)
    if not config.speed_min_kmh - 1e-6 <= median_speed <= config.speed_max_kmh + 1e-6:
        return None
    mean_evidence = float(np.mean([item.evidence_score for item in selected]))
    confidence = float(
        np.clip(
            0.50 * mean_evidence
            + 0.25 * np.exp(-median_residual / max(config.max_median_residual_s, 1e-6))
            + 0.25 * (1.0 - np.exp(-len(selected) / 6.0)),
            0.0,
            1.0,
        )
    )
    # ``travel_rows`` spans the complete geometry, so these are the actual
    # entry/exit times at the physical array ends rather than an extrapolation
    # from a short selected fragment.
    left_time = float(fitted_times[0])
    right_time = float(fitted_times[-1])
    output_span_m = float(
        abs(points[-1].position_m - points[0].position_m) if len(points) > 1 else 0.0
    )
    return VehicleTrack(
        track_id=f"vehicle_{track_index:04d}",
        points=points,
        median_speed_kmh=median_speed,
        local_speeds_kmh=[float(value) for value in local_speeds],
        confidence=confidence,
        score=float(path.score),
        observed_count=len(selected),
        span_m=output_span_m,
        median_residual_s=median_residual,
        max_gap_m=max_gap_m,
        enters_window=bool(left_time < 0.0),
        exits_window=bool(right_time > duration_s),
        ambiguous_crossing=any(item.ambiguous for item in selected),
    )
