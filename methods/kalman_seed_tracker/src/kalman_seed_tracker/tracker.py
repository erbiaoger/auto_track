from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
class KalmanFilterConfig:
    prominence: float = 0.4
    min_distance_s: float = 0.5
    wlen_s: float = 2.0
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 90.0
    nominal_speed_kmh: float = 75.0
    measurement_std_s: float = 0.25
    process_time_std_s_per_100m: float = 0.15
    process_slowness_std_s_per_m_per_100m: float = 0.001
    initial_slowness_std_s_per_m: float = 0.01
    gate_sigma: float = 3.0
    max_gap_m: float = 800.0
    min_observed_points: int = 12
    edge_min_observed_points: int = 4
    edge_time_margin_s: float = 8.0
    min_overlap_ratio: float = 0.5
    dedup_residual_s: float = 0.3

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "KalmanFilterConfig":
        if not value:
            return cls()
        fields = {key: raw for key, raw in value.items() if key in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass
class _KalmanState:
    track_id: int
    state: np.ndarray
    covariance: np.ndarray
    last_travel_m: float
    last_time_s: float
    points: list[TrackPoint] = field(default_factory=list)
    pending: list[tuple[int, float, float]] = field(default_factory=list)


class KalmanVehicleTracker:
    """Automatic multi-target constant-velocity filter in physical units."""

    method_id = "kalman_seed"
    preset_id = "kalman_day11"

    def __init__(self, config: KalmanFilterConfig | dict[str, Any] | None = None) -> None:
        self.config = config if isinstance(config, KalmanFilterConfig) else KalmanFilterConfig.from_mapping(config)

    @property
    def nominal_slowness(self) -> float:
        return 3.6 / float(self.config.nominal_speed_kmh)

    def _predict(self, state: _KalmanState, travel_m: float) -> tuple[np.ndarray, np.ndarray, float]:
        distance = float(travel_m - state.last_travel_m)
        ratio = max(1.0, distance / 100.0)
        transition = np.array([[1.0, distance], [0.0, 1.0]], dtype=np.float64)
        q = np.diag(
            [
                (float(self.config.process_time_std_s_per_100m) * ratio) ** 2,
                (float(self.config.process_slowness_std_s_per_m_per_100m) * ratio) ** 2,
            ]
        )
        predicted = transition @ state.state
        covariance = transition @ state.covariance @ transition.T + q
        return predicted, covariance, distance

    def _pair_score(
        self,
        state: _KalmanState,
        candidate: PeakCandidate,
        travel_m: float,
    ) -> tuple[float, np.ndarray, np.ndarray] | None:
        cfg = self.config
        predicted, covariance, distance = self._predict(state, travel_m)
        if distance <= 0:
            return None
        dt = float(candidate.time_s - state.last_time_s)
        if dt <= 0:
            return None
        min_dt = distance / (float(cfg.speed_max_kmh) / 3.6)
        max_dt = distance / (float(cfg.speed_min_kmh) / 3.6)
        innovation = float(candidate.time_s - predicted[0])
        innovation_var = max(1e-8, float(covariance[0, 0]) + float(cfg.measurement_std_s) ** 2)
        gate = float(cfg.gate_sigma) * np.sqrt(innovation_var)
        if abs(innovation) > gate:
            return None
        if dt < min_dt - gate or dt > max_dt + gate:
            return None
        speed = 3.6 * distance / dt
        if speed < cfg.speed_min_kmh or speed > cfg.speed_max_kmh:
            return None
        normalized = abs(innovation) / np.sqrt(innovation_var) - 0.02 * candidate.score
        return float(normalized), predicted, covariance

    def _update(self, predicted: np.ndarray, covariance: np.ndarray, observation_s: float) -> tuple[np.ndarray, np.ndarray]:
        measurement_var = float(self.config.measurement_std_s) ** 2
        innovation = float(observation_s - predicted[0])
        innovation_var = max(1e-8, float(covariance[0, 0]) + measurement_var)
        gain = covariance[:, 0] / innovation_var
        state = predicted + gain * innovation
        state[1] = np.clip(
            state[1],
            3.6 / float(self.config.speed_max_kmh),
            3.6 / float(self.config.speed_min_kmh),
        )
        covariance = covariance - np.outer(gain, covariance[0, :])
        covariance = 0.5 * (covariance + covariance.T)
        return state, covariance

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
        states: list[_KalmanState] = []
        finished: list[_KalmanState] = []
        next_id = 0
        predicted_count = 0

        for channel in order.tolist():
            channel = int(channel)
            candidates = candidates_by_channel[channel]
            pairs: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
            for state_index, state in enumerate(states):
                for candidate_index, candidate in enumerate(candidates):
                    pair = self._pair_score(state, candidate, float(travel[channel]))
                    if pair is not None:
                        score, predicted, covariance = pair
                        pairs.append((score, state_index, candidate_index, predicted, covariance))
            pairs.sort(key=lambda item: item[0])
            assigned_states: dict[int, tuple[int, np.ndarray, np.ndarray]] = {}
            used_candidates: set[int] = set()
            for _, state_index, candidate_index, predicted, covariance in pairs:
                if state_index in assigned_states or candidate_index in used_candidates:
                    continue
                assigned_states[state_index] = (candidate_index, predicted, covariance)
                used_candidates.add(candidate_index)

            updated: list[_KalmanState] = []
            for state_index, state in enumerate(states):
                assignment = assigned_states.get(state_index)
                predicted, predicted_covariance, distance = self._predict(state, float(travel[channel]))
                if assignment is not None:
                    candidate_index, _, _ = assignment
                    candidate = candidates[candidate_index]
                    state.state, state.covariance = self._update(predicted, predicted_covariance, candidate.time_s)
                    for pending_channel, pending_time, _ in state.pending:
                        state.points.append(
                            TrackPoint(
                                channel_index=int(pending_channel),
                                station_id=geometry.stations[int(pending_channel)].station_id,
                                position_m=float(geometry.positions_m[int(pending_channel)]),
                                time_s=float(pending_time),
                                observed=False,
                                score=None,
                            )
                        )
                    state.pending.clear()
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
                elif distance <= float(self.config.max_gap_m):
                    state.state = predicted
                    state.covariance = predicted_covariance
                    state.pending.append((channel, float(predicted[0]), 0.0))
                    state.last_travel_m = float(travel[channel])
                    state.last_time_s = float(predicted[0])
                    predicted_count += 1
                    updated.append(state)
                else:
                    finished.append(state)
            births: list[_KalmanState] = []
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in used_candidates:
                    continue
                births.append(
                    _KalmanState(
                        track_id=next_id,
                        state=np.array([candidate.time_s, self.nominal_slowness], dtype=np.float64),
                        covariance=np.diag(
                            [
                                float(self.config.measurement_std_s) ** 2,
                                float(self.config.initial_slowness_std_s_per_m) ** 2,
                            ]
                        ),
                        last_travel_m=float(travel[channel]),
                        last_time_s=float(candidate.time_s),
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
            track = as_track(str(state.track_id), state.points, geometry)
            speed = as_track(str(state.track_id), observed, geometry).median_speed_kmh
            edge = min(times) <= self.config.edge_time_margin_s or max(times) >= duration - self.config.edge_time_margin_s
            minimum = self.config.edge_min_observed_points if edge else self.config.min_observed_points
            if len(observed) < int(minimum) or not np.isfinite(speed):
                continue
            if speed < self.config.speed_min_kmh or speed > self.config.speed_max_kmh:
                continue
            tracks.append(track)
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
                "engine": "numpy_cpu",
                "extractor": "classic_peak_kalman",
                "candidate_count": int(sum(len(items) for items in candidates_by_channel)),
                "predicted_point_count": int(predicted_count),
                "track_count": int(len(tracks)),
                "direction": "reverse",
            },
        )
