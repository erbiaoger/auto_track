from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from auto_track_common import (
    PredictionBatch,
    StationGeometry,
    Track,
    TrackPoint,
    as_track,
    deduplicate_tracks,
    detect_peak_candidates,
    travel_positions,
)
from auto_track_common.physical import PeakCandidate


@dataclass
class HungarianAssignmentConfig:
    prominence: float = 0.4
    min_distance_s: float = 0.5
    wlen_s: float = 2.0
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 90.0
    nominal_speed_kmh: float = 75.0
    residual_gate_s_per_100m: float = 0.6
    max_gap_m: float = 800.0
    min_observed_points: int = 12
    edge_min_observed_points: int = 4
    edge_time_margin_s: float = 8.0
    unmatched_cost: float = 1.0
    min_overlap_ratio: float = 0.5
    dedup_residual_s: float = 0.3

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "HungarianAssignmentConfig":
        if not value:
            return cls()
        fields = {key: raw for key, raw in value.items() if key in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class _TrackState:
    track_id: int
    points: list[TrackPoint] = field(default_factory=list)
    last_travel_m: float = 0.0
    last_time_s: float = 0.0

    def predict(self, travel_m: float, nominal_speed_kmh: float) -> float:
        distance = float(travel_m - self.last_travel_m)
        if len(self.points) < 2:
            return self.last_time_s + distance / (float(nominal_speed_kmh) / 3.6)
        slopes: list[float] = []
        for left, right in zip(self.points[-5:-1], self.points[-4:]):
            d = abs(float(right.position_m) - float(left.position_m))
            dt = float(right.time_s) - float(left.time_s)
            if d > 0 and dt > 0:
                slopes.append(dt / d)
        slope = float(np.median(slopes)) if slopes else 1.0 / (float(nominal_speed_kmh) / 3.6)
        return self.last_time_s + distance * slope


class HungarianAssignmentTracker:
    """Multi-target peak association using a global Hungarian assignment."""

    method_id = "hungarian_assignment"
    preset_id = "hungarian_day11"

    def __init__(self, config: HungarianAssignmentConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, HungarianAssignmentConfig) else HungarianAssignmentConfig.from_mapping(config)

    def _pair_cost(self, state: _TrackState, candidate: PeakCandidate, travel_m: float) -> float:
        cfg = self.config
        distance = float(travel_m - state.last_travel_m)
        if distance <= 0:
            return 1e6
        dt = float(candidate.time_s - state.last_time_s)
        if dt <= 0:
            return 1e6
        min_dt = distance / (float(cfg.speed_max_kmh) / 3.6)
        max_dt = distance / (float(cfg.speed_min_kmh) / 3.6)
        gate = max(0.08, float(cfg.residual_gate_s_per_100m) * max(1.0, distance / 100.0))
        if dt < min_dt - gate or dt > max_dt + gate:
            return 1e6
        speed = 3.6 * distance / dt
        predicted = state.predict(travel_m, cfg.nominal_speed_kmh)
        residual = abs(float(candidate.time_s) - predicted)
        if residual > gate:
            return 1e6
        speed_penalty = abs(speed - float(cfg.nominal_speed_kmh)) / max(1e-6, cfg.speed_max_kmh - cfg.speed_min_kmh)
        return float(residual / gate + 0.2 * speed_penalty - 0.02 * candidate.score)

    def _assign(self, states: list[_TrackState], candidates: list[PeakCandidate], travel_m: float) -> dict[int, int]:
        if not states or not candidates:
            return {}
        unmatched = float(self.config.unmatched_cost)
        n_state, n_candidate = len(states), len(candidates)
        size = n_state + n_candidate
        matrix = np.full((size, size), unmatched, dtype=np.float64)
        matrix[n_state:, n_candidate:] = 0.0
        for row, state in enumerate(states):
            for col, candidate in enumerate(candidates):
                matrix[row, col] = self._pair_cost(state, candidate, travel_m)
        rows, cols = linear_sum_assignment(matrix)
        return {
            int(row): int(col)
            for row, col in zip(rows, cols)
            if row < n_state and col < n_candidate and matrix[row, col] < unmatched
        }

    def predict_window(
        self,
        data_time_channel: np.ndarray,
        *,
        sample_rate_hz: float,
        geometry: StationGeometry,
        start_s: float = 0.0,
        duration_s: float | None = None,
        request_id: str = "",
    ) -> PredictionBatch:
        data = np.asarray(data_time_channel, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != len(geometry):
            raise ValueError("data_time_channel must have shape [time, station]")
        fs = float(sample_rate_hz)
        duration = float(duration_s if duration_s is not None else data.shape[0] / fs)
        candidates_by_channel = detect_peak_candidates(
            data.T,
            sample_rate_hz=fs,
            prominence=self.config.prominence,
            min_distance_s=self.config.min_distance_s,
            wlen_s=self.config.wlen_s,
        )
        travel = travel_positions(geometry, direction="reverse")
        order = np.argsort(-np.asarray(geometry.positions_m, dtype=np.float64))
        states: list[_TrackState] = []
        finished: list[_TrackState] = []
        next_id = 0
        association_count = 0

        for channel in order.tolist():
            channel = int(channel)
            candidates = candidates_by_channel[channel]
            active = list(states)
            assigned = self._assign(active, candidates, float(travel[channel]))
            used_candidates: set[int] = set()
            updated: list[_TrackState] = []
            for state_index, state in enumerate(active):
                candidate_index = assigned.get(state_index)
                distance = float(travel[channel] - state.last_travel_m)
                if candidate_index is not None:
                    candidate = candidates[candidate_index]
                    state.points.append(
                        TrackPoint(
                            channel_index=channel,
                            station_id=geometry.stations[channel].station_id,
                            position_m=float(geometry.positions_m[channel]),
                            time_s=float(candidate.time_s),
                            observed=True,
                            score=float(candidate.score),
                        )
                    )
                    state.last_travel_m = float(travel[channel])
                    state.last_time_s = float(candidate.time_s)
                    updated.append(state)
                    used_candidates.add(int(candidate_index))
                    association_count += 1
                elif distance <= float(self.config.max_gap_m):
                    updated.append(state)
                else:
                    finished.append(state)
            births: list[_TrackState] = []
            for index, candidate in enumerate(candidates):
                if index in used_candidates:
                    continue
                births.append(
                    _TrackState(
                        track_id=next_id,
                        points=[
                            TrackPoint(
                                channel_index=channel,
                                station_id=geometry.stations[channel].station_id,
                                position_m=float(geometry.positions_m[channel]),
                                time_s=float(candidate.time_s),
                                observed=True,
                                score=float(candidate.score),
                            )
                        ],
                        last_travel_m=float(travel[channel]),
                        last_time_s=float(candidate.time_s),
                    )
                )
                next_id += 1
            states = updated + births

        finished.extend(states)
        tracks: list[Track] = []
        for state in finished:
            observed = [point for point in state.points if point.observed]
            if len(observed) < 2:
                continue
            times = [float(point.time_s) for point in observed]
            speed = as_track(str(state.track_id), observed, geometry).median_speed_kmh
            edge = min(times) <= self.config.edge_time_margin_s or max(times) >= duration - self.config.edge_time_margin_s
            minimum = self.config.edge_min_observed_points if edge else self.config.min_observed_points
            if len(observed) < int(minimum) or not np.isfinite(speed):
                continue
            if speed < self.config.speed_min_kmh or speed > self.config.speed_max_kmh:
                continue
            tracks.append(as_track(str(state.track_id), observed, geometry))
        tracks = deduplicate_tracks(
            tracks,
            overlap_ratio=self.config.min_overlap_ratio,
            residual_s=self.config.dedup_residual_s,
        )
        return PredictionBatch(
            method_id=self.method_id,
            preset_id=self.preset_id,
            request_id=str(request_id),
            start_s=float(start_s),
            duration_s=duration,
            tracks=tracks,
            diagnostics={
                "engine": "scipy_cpu",
                "extractor": "classic_peak_hungarian",
                "candidate_count": int(sum(len(items) for items in candidates_by_channel)),
                "raw_track_count": int(len(finished)),
                "association_count": int(association_count),
                "track_count": int(len(tracks)),
                "direction": "reverse",
            },
        )
