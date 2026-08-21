from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .methods import WorkerBatch, WorkerPoint, WorkerTrack


@dataclass(frozen=True)
class StitchConfig:
    """Physical-unit defaults shared by the replay UI and offline reports."""

    duplicate_time_gate_s: float = 0.75
    duplicate_line_gate_s: float = 0.75
    duplicate_speed_gate_kmh: float = 8.0
    match_overlap_gate_s: float = 1.5
    match_disjoint_gate_s: float = 3.0
    match_speed_gate_kmh: float = 12.0
    max_extrapolation_gap_m: float = 800.0
    max_missing_windows: int = 2
    # Normalize reconnecting to physical time as well as source-window count.
    reconnect_horizon_s: float = 10.0
    final_disjoint_gate_s: float = 1.5
    confirmation_batches: int = 3
    confirmation_hits: int = 2
    merge_evidence_batches: int = 3
    measurement_std_s: float = 0.25
    process_time_std_s_per_100m: float = 0.15
    process_slowness_std_s_per_100m: float = 0.001
    max_station_samples: int = 64

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "StitchConfig":
        if not payload:
            return cls()
        names = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in payload.items() if key in names}
        return cls(**values)


@dataclass
class StreamingTrack:
    global_vehicle_id: str
    source_window_start_s: float
    source_track_id: str
    direction: str
    points: list[dict[str, Any]]
    median_speed_kmh: float
    confidence: float | None
    observed_count: int
    span_m: float
    max_gap_m: float
    enters_window: bool
    exits_window: bool
    ambiguous_crossing: bool
    last_seen_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_vehicle_id": self.global_vehicle_id,
            "source_window_start_s": self.source_window_start_s,
            "source_track_id": self.source_track_id,
            "direction": self.direction,
            "points": self.points,
            "median_speed_kmh": self.median_speed_kmh,
            "confidence": self.confidence,
            "observed_count": self.observed_count,
            "span_m": self.span_m,
            "max_gap_m": self.max_gap_m,
            "enters_window": self.enters_window,
            "exits_window": self.exits_window,
            "ambiguous_crossing": self.ambiguous_crossing,
            "last_seen_s": self.last_seen_s,
        }


@dataclass(frozen=True)
class StitchUpdate:
    """An incremental canonical-track update.

    ``__iter__``/``__getitem__`` deliberately keep the old list-like API so
    existing workers and scripts can adopt alias handling incrementally.
    """

    tracks: list[StreamingTrack]
    removed_track_ids: list[str] = field(default_factory=list)
    id_aliases: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, index: int) -> StreamingTrack:
        return self.tracks[index]


@dataclass(frozen=True)
class VehicleCounters:
    window_candidates: int
    current_unique: int
    cumulative_unique: int
    mean_speed_kmh: float | None
    median_speed_kmh: float | None
    speed_range_kmh: tuple[float | None, float | None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_candidates": self.window_candidates,
            "current_unique": self.current_unique,
            "cumulative_unique": self.cumulative_unique,
            "mean_speed_kmh": self.mean_speed_kmh,
            "median_speed_kmh": self.median_speed_kmh,
            "speed_range_kmh": list(self.speed_range_kmh),
        }


@dataclass
class _Descriptor:
    line: np.ndarray | None
    speed_kmh: float
    direction_sign: int
    positions: list[float]
    times: dict[str, float]

    @property
    def min_position(self) -> float:
        return min(self.positions, default=0.0)

    @property
    def max_position(self) -> float:
        return max(self.positions, default=0.0)


@dataclass
class _Candidate:
    track: WorkerTrack
    points: list[dict[str, Any]]
    descriptor: _Descriptor | None


@dataclass
class _TrackState:
    global_vehicle_id: str
    first_window_index: int
    last_window_index: int
    first_window_start_s: float
    last_source_track_id: str
    direction: str
    confidence: float | None
    status: str = "tentative"
    misses: int = 0
    hit_windows: list[int] = field(default_factory=list)
    descriptors_by_window: dict[int, _Descriptor] = field(default_factory=dict)
    samples: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    enters_window: bool = False
    exits_window: bool = False
    ambiguous_crossing: bool = False

    def add(self, candidate: _Candidate, window_index: int, start_s: float, config: StitchConfig) -> None:
        self.last_window_index = window_index
        self.last_source_track_id = str(candidate.track.track_id)
        self.direction = str(getattr(candidate.track, "direction", self.direction))
        if candidate.track.confidence is not None:
            self.confidence = max(float(candidate.track.confidence), self.confidence or 0.0)
        self.misses = 0
        self.hit_windows.append(window_index)
        cutoff = window_index - max(1, config.confirmation_batches) + 1
        self.hit_windows = sorted(set(item for item in self.hit_windows if item >= cutoff))
        if candidate.descriptor is not None:
            self.descriptors_by_window[window_index] = candidate.descriptor
            for old_index in list(self.descriptors_by_window):
                if old_index < cutoff - 2:
                    self.descriptors_by_window.pop(old_index, None)
        self.enters_window = self.enters_window or bool(candidate.track.enters_window)
        self.exits_window = self.exits_window or bool(candidate.track.exits_window)
        self.ambiguous_crossing = self.ambiguous_crossing or bool(candidate.track.ambiguous_crossing)
        for point in candidate.points:
            key = int(point.get("channel_index", -1))
            row = dict(point)
            row["_window_index"] = window_index
            row["_weight"] = 1.0 if bool(point.get("observed", True)) else 0.2
            bucket = self.samples.setdefault(key, [])
            # A station contributes at most once per recognition window.
            bucket[:] = [item for item in bucket if int(item.get("_window_index", -1)) != window_index]
            bucket.append(row)
            if len(bucket) > config.max_station_samples:
                del bucket[:-config.max_station_samples]


def _absolute_points(track: WorkerTrack, start_s: float) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for point in track.points:
        row = dict(vars(point))
        row["time_s"] = float(point.time_s + start_s)
        points.append(row)
    return points


def _descriptor(points: Iterable[dict[str, Any]]) -> _Descriptor | None:
    rows = [point for point in points if bool(point.get("observed", True))]
    by_channel: dict[int, dict[str, Any]] = {}
    for point in rows:
        channel = int(point.get("channel_index", -1))
        if channel >= 0:
            by_channel[channel] = point
    rows = list(by_channel.values())
    if len(rows) < 2:
        return None
    positions = np.asarray([float(point.get("position_m", 0.0)) for point in rows], dtype=np.float64)
    times = np.asarray([float(point.get("time_s", 0.0)) for point in rows], dtype=np.float64)
    if np.unique(positions).size < 2:
        return None
    line = np.polyfit(positions, times, 1)
    slope = float(line[0])
    speed = 3.6 / abs(slope) if abs(slope) > 1e-9 else float("inf")
    return _Descriptor(
        line=np.asarray(line, dtype=np.float64),
        speed_kmh=float(speed),
        direction_sign=1 if slope >= 0 else -1,
        positions=[float(value) for value in positions],
        times={_station_key(point): float(point.get("time_s", 0.0)) for point in rows},
    )


def _station_key(point: dict[str, Any]) -> str:
    station_id = str(point.get("station_id", "")).strip()
    return station_id or f"CH{int(point.get('channel_index', -1))}"


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 1e-6)
    cutoff = 0.5 * float(np.sum(weights))
    return float(values[int(np.searchsorted(np.cumsum(weights), cutoff, side="left"))])


def _consensus_samples(samples: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for _channel, rows in samples.items():
        if not rows:
            continue
        times = np.asarray([float(row.get("time_s", 0.0)) for row in rows], dtype=np.float64)
        weights = np.asarray([float(row.get("_weight", 1.0)) for row in rows], dtype=np.float64)
        median_time = _weighted_median(times, weights)
        representative = min(rows, key=lambda row: abs(float(row.get("time_s", 0.0)) - median_time))
        point = {key: value for key, value in representative.items() if not str(key).startswith("_")}
        point["time_s"] = median_time
        point["observed"] = any(bool(row.get("observed", True)) for row in rows)
        points.append(point)
    return sorted(points, key=lambda item: (int(item.get("channel_index", -1)), float(item.get("position_m", 0.0))))


def _line_points(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) < 3:
        return points
    positions = np.asarray([float(point.get("position_m", 0.0)) for point in points], dtype=np.float64)
    times = np.asarray([float(point.get("time_s", 0.0)) for point in points], dtype=np.float64)
    if np.unique(positions).size < 3:
        return points
    keep = np.ones(len(points), dtype=bool)
    for _ in range(2):
        if int(keep.sum()) < 3:
            break
        line = np.polyfit(positions[keep], times[keep], 1)
        residual = np.abs(times - np.polyval(line, positions))
        mad = float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        gate = max(0.75, min(1.5, 3.0 * 1.4826 * mad))
        updated = residual <= gate
        if np.array_equal(updated, keep):
            break
        keep = updated
    if int(keep.sum()) < 3:
        return [point for point, accepted in zip(points, keep) if accepted]
    line = np.polyfit(positions[keep], times[keep], 1)
    return [{**point, "time_s": float(np.polyval(line, float(point.get("position_m", 0.0))))} for point in points]


def _smooth_points(samples: dict[int, list[dict[str, Any]]], config: StitchConfig) -> list[dict[str, Any]]:
    """Robust arrival-time consensus followed by a variable-speed Kalman smoother."""
    points = _consensus_samples(samples)
    if len(points) < 3:
        return points
    ordered = sorted(points, key=lambda item: float(item.get("position_m", 0.0)))
    positions = np.asarray([float(point["position_m"]) for point in ordered], dtype=np.float64)
    measurements = np.asarray([float(point["time_s"]) for point in ordered], dtype=np.float64)
    initial_line = np.polyfit(positions, measurements, 1)
    if not np.isfinite(initial_line).all() or abs(float(initial_line[0])) < 1e-9:
        return _line_points(points)

    variances: list[float] = []
    for point in ordered:
        rows = samples.get(int(point.get("channel_index", -1)), [])
        values = np.asarray([float(row.get("time_s", 0.0)) for row in rows], dtype=np.float64)
        mad = float(np.median(np.abs(values - np.median(values)))) if values.size else 0.0
        variances.append(max(config.measurement_std_s**2, (1.4826 * mad) ** 2))

    def run(accepted: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        filtered = np.zeros((len(ordered), 2), dtype=np.float64)
        covariance = np.zeros((len(ordered), 2, 2), dtype=np.float64)
        predicted = np.zeros_like(filtered)
        predicted_covariance = np.zeros_like(covariance)
        filtered[0] = [measurements[0], float(initial_line[0])]
        covariance[0] = np.diag([variances[0], 1e-5])
        for index in range(1, len(ordered)):
            dx = float(positions[index] - positions[index - 1])
            transition = np.asarray([[1.0, dx], [0.0, 1.0]], dtype=np.float64)
            scale = max(abs(dx) / 100.0, 1e-3)
            process = np.diag([
                (config.process_time_std_s_per_100m * scale) ** 2,
                (config.process_slowness_std_s_per_100m * scale) ** 2,
            ])
            predicted[index] = transition @ filtered[index - 1]
            predicted_covariance[index] = transition @ covariance[index - 1] @ transition.T + process
            if not accepted[index]:
                filtered[index] = predicted[index]
                covariance[index] = predicted_covariance[index]
                continue
            innovation = measurements[index] - predicted[index, 0]
            innovation_variance = predicted_covariance[index, 0, 0] + variances[index]
            gain = predicted_covariance[index][:, 0] / max(innovation_variance, 1e-9)
            filtered[index] = predicted[index] + gain * innovation
            covariance[index] = predicted_covariance[index] - np.outer(gain, predicted_covariance[index][0])
        smoothed = filtered.copy()
        smoothed_covariance = covariance.copy()
        for index in range(len(ordered) - 2, -1, -1):
            dx = float(positions[index + 1] - positions[index])
            transition = np.asarray([[1.0, dx], [0.0, 1.0]], dtype=np.float64)
            scale = max(abs(dx) / 100.0, 1e-3)
            process = np.diag([
                (config.process_time_std_s_per_100m * scale) ** 2,
                (config.process_slowness_std_s_per_100m * scale) ** 2,
            ])
            next_pred_cov = transition @ covariance[index] @ transition.T + process
            smoother_gain = covariance[index] @ transition.T @ np.linalg.pinv(next_pred_cov)
            smoothed[index] = filtered[index] + smoother_gain @ (smoothed[index + 1] - predicted[index + 1])
            smoothed_covariance[index] = covariance[index] + smoother_gain @ (smoothed_covariance[index + 1] - next_pred_cov) @ smoother_gain.T
        return smoothed, smoothed_covariance, predicted, predicted_covariance

    accepted = np.ones(len(ordered), dtype=bool)
    smoothed, _, predicted, predicted_covariance = run(accepted)
    residual = np.abs(measurements - smoothed[:, 0])
    gates = np.maximum(0.75, 3.0 * np.sqrt(np.maximum(predicted_covariance[:, 0, 0], 1e-9) + np.asarray(variances)))
    accepted = residual <= gates
    accepted[0] = True
    smoothed, _, _, _ = run(accepted)
    direction = 1 if float(initial_line[0]) >= 0 else -1
    delta = np.diff(smoothed[:, 0])
    if np.any(direction * delta <= 0.0):
        return _line_points(points)
    by_channel = {int(point.get("channel_index", -1)): point for point in ordered}
    result = []
    for point, value in zip(ordered, smoothed[:, 0]):
        result.append({**by_channel[int(point.get("channel_index", -1))], "time_s": float(value)})
    return sorted(result, key=lambda item: (float(item.get("time_s", 0.0)), int(item.get("channel_index", -1))))


def _descriptor_distance(left: _Descriptor | None, right: _Descriptor | None, config: StitchConfig) -> tuple[bool, float, bool, float]:
    if left is None or right is None or left.line is None or right.line is None:
        return False, float("inf"), False, float("inf")
    if left.direction_sign != right.direction_sign:
        return False, float("inf"), False, float("inf")
    speed_diff = abs(left.speed_kmh - right.speed_kmh)
    if not np.isfinite(speed_diff) or speed_diff > config.match_speed_gate_kmh:
        return False, float("inf"), False, speed_diff
    common = sorted(set(left.times).intersection(right.times))
    point_residual = float(np.median([abs(left.times[channel] - right.times[channel]) for channel in common])) if common else None
    low = max(left.min_position, right.min_position)
    high = min(left.max_position, right.max_position)
    overlap = high >= low
    if overlap:
        probes = np.asarray([low, (low + high) / 2.0, high], dtype=np.float64)
        line_gate = config.match_overlap_gate_s
    else:
        gap = min(abs(right.min_position - left.max_position), abs(left.min_position - right.max_position))
        if gap > config.max_extrapolation_gap_m:
            return False, float("inf"), False, speed_diff
        anchor = (left.max_position + right.min_position) / 2.0 if left.max_position < right.min_position else (right.max_position + left.min_position) / 2.0
        probes = np.asarray([anchor], dtype=np.float64)
        line_gate = config.match_disjoint_gate_s
    line_residual = float(np.median(np.abs(np.polyval(left.line, probes) - np.polyval(right.line, probes))))
    # Keep the extrapolation boundary exclusive: vehicles separated by the
    # documented 3 s gate must not collapse into one identity.
    if line_residual > line_gate or (not overlap and line_residual >= line_gate - 1e-6):
        return False, line_residual, overlap, speed_diff
    if point_residual is not None and len(common) >= 3 and point_residual > line_gate:
        return False, line_residual, overlap, speed_diff
    overlap_ratio = len(common) / max(1, min(len(left.times), len(right.times)))
    station_term = (point_residual if point_residual is not None else line_residual) / max(line_gate, 1e-6)
    cost = 0.45 * station_term + 0.30 * (line_residual / max(line_gate, 1e-6)) + 0.15 * (speed_diff / config.match_speed_gate_kmh) + 0.10 * (1.0 - min(1.0, overlap_ratio))
    return True, float(cost), overlap, speed_diff


def _same_window_duplicate(left: _Candidate, right: _Candidate, config: StitchConfig) -> bool:
    if left.descriptor is None or right.descriptor is None:
        return False
    left_ids = {int(point["observation_id"]) for point in left.points if point.get("observation_id") is not None}
    right_ids = {int(point["observation_id"]) for point in right.points if point.get("observation_id") is not None}
    if left_ids and right_ids and len(left_ids & right_ids) / max(1, min(len(left_ids), len(right_ids))) >= 0.8:
        return True
    common = sorted(set(left.descriptor.times).intersection(right.descriptor.times))
    if len(common) < 3:
        return False
    residual = float(np.median([abs(left.descriptor.times[channel] - right.descriptor.times[channel]) for channel in common]))
    valid, line_cost, _overlap, speed_diff = _descriptor_distance(left.descriptor, right.descriptor, config)
    return valid and residual <= config.duplicate_time_gate_s and speed_diff <= config.duplicate_speed_gate_kmh and line_cost <= 1.0


def _quality(candidate: _Candidate) -> tuple[int, float, float, float]:
    track = candidate.track
    confidence = float(track.confidence) if track.confidence is not None and np.isfinite(track.confidence) else 0.0
    score = float(track.score) if track.score is not None and np.isfinite(track.score) else 0.0
    return int(track.observed_count), float(track.span_m), confidence, score


class TrackStitcher:
    """Window-safe vehicle identity tracker with robust station consensus."""

    def __init__(self, *, stride_s: float = 60.0, window_s: float = 120.0, retention_s: float | None = None, config: StitchConfig | None = None) -> None:
        self.stride_s = float(stride_s)
        self.window_s = float(window_s)
        self.retention_s = float(retention_s if retention_s is not None else max(2.0 * self.stride_s, self.window_s))
        self.config = config or StitchConfig()
        self.reset()

    def reset(self) -> None:
        self._next_id = 1
        self._window_index = -1
        self._states: dict[str, _TrackState] = {}
        self._aliases: dict[str, str] = {}
        self._all_ids: set[str] = set()
        self.last_window_tracks: list[StreamingTrack] = []
        self.last_update = StitchUpdate([])

    def _missing_limit(self) -> int:
        """Stride-normalized empty-window budget (at least two windows)."""
        physical = int(np.ceil(max(0.0, self.config.reconnect_horizon_s) / max(self.stride_s, 1e-6))) - 1
        return max(int(self.config.max_missing_windows), physical)

    def _canonical(self, identifier: str) -> str:
        seen: set[str] = set()
        while identifier in self._aliases and identifier not in seen:
            seen.add(identifier)
            identifier = self._aliases[identifier]
        return identifier

    @property
    def cumulative_unique_count(self) -> int:
        return len(self._all_ids)

    @property
    def active_tracks(self) -> list[StreamingTrack]:
        limit = self._missing_limit()
        return [self._state_to_track(state) for state in self._states.values() if state.status == "confirmed" and state.misses <= limit]

    def _window_candidates(self, batch: WorkerBatch) -> list[_Candidate]:
        raw = [_Candidate(track, _absolute_points(track, batch.start_s), None) for track in batch.tracks]
        for candidate in raw:
            candidate.descriptor = _descriptor(candidate.points)
        raw.sort(key=_quality, reverse=True)
        representatives: list[_Candidate] = []
        for candidate in raw:
            duplicate = next((item for item in representatives if _same_window_duplicate(item, candidate, self.config)), None)
            if duplicate is None:
                representatives.append(candidate)
                continue
            duplicate.points = _merge_point_lists(duplicate.points, candidate.points)
            duplicate.descriptor = _descriptor(duplicate.points)
        return representatives

    def _new_state(self, candidate: _Candidate, start_s: float) -> _TrackState:
        self._next_id += 1
        identifier = f"V{self._next_id - 1:04d}"
        state = _TrackState(
            global_vehicle_id=identifier,
            first_window_index=self._window_index,
            last_window_index=self._window_index,
            first_window_start_s=float(start_s),
            last_source_track_id=str(candidate.track.track_id),
            direction=str(getattr(candidate.track, "direction", "unknown")),
            confidence=float(candidate.track.confidence) if candidate.track.confidence is not None else None,
        )
        state.add(candidate, self._window_index, start_s, self.config)
        return state

    def _state_candidate(self, state: _TrackState) -> _Candidate:
        points = _consensus_samples(state.samples)
        track = WorkerTrack(
            track_id=state.last_source_track_id,
            direction=state.direction,
            points=[WorkerPoint(
                channel_index=int(point.get("channel_index", -1)),
                station_id=str(point.get("station_id", "")),
                position_m=float(point.get("position_m", 0.0)),
                time_s=float(point.get("time_s", 0.0)),
                observed=bool(point.get("observed", True)),
            ) for point in points],
            median_speed_kmh=float("nan"), confidence=state.confidence, score=None,
            observed_count=sum(1 for point in points if point.get("observed", True)),
            span_m=float(max((float(point.get("position_m", 0.0)) for point in points), default=0.0) - min((float(point.get("position_m", 0.0)) for point in points), default=0.0)),
            max_gap_m=0.0, enters_window=state.enters_window, exits_window=state.exits_window, ambiguous_crossing=state.ambiguous_crossing,
        )
        return _Candidate(track, points, _descriptor(points))

    def _merge_states(self, *, include_retired: bool = False) -> tuple[list[str], dict[str, str]]:
        removed: list[str] = []
        aliases: dict[str, str] = {}
        identifiers = [
            identifier for identifier, state in self._states.items()
            if include_retired or state.misses <= self._missing_limit()
        ]
        for left_index, left_id in enumerate(identifiers):
            if left_id not in self._states:
                continue
            left = self._states[left_id]
            for right_id in identifiers[left_index + 1:]:
                if right_id not in self._states:
                    continue
                right = self._states[right_id]
                if left.status != right.status and "confirmed" in (left.status, right.status):
                    continue
                left_candidate = self._state_candidate(left)
                right_candidate = self._state_candidate(right)
                merge_config = self.config
                valid, _cost, _overlap, _speed = _descriptor_distance(left_candidate.descriptor, right_candidate.descriptor, merge_config)
                if not valid:
                    continue
                shared_windows = set(left.descriptors_by_window).intersection(right.descriptors_by_window)
                if left.status == "confirmed" and right.status == "confirmed" and len(shared_windows) < self.config.merge_evidence_batches:
                    if not include_retired:
                        continue
                    # Offline finalization can join fragments that never
                    # coexisted in one window, but only with substantially
                    # stricter gates than the online reconnect path.
                    merge_config = replace(
                        self.config,
                        match_overlap_gate_s=self.config.duplicate_line_gate_s,
                        match_disjoint_gate_s=self.config.final_disjoint_gate_s,
                        match_speed_gate_kmh=self.config.duplicate_speed_gate_kmh,
                    )
                    valid, _cost, _overlap, _speed = _descriptor_distance(left_candidate.descriptor, right_candidate.descriptor, merge_config)
                    if not valid:
                        continue
                # The oldest ID is canonical. This makes later alias events deterministic.
                canonical, duplicate = (left, right) if (left.first_window_index, left.global_vehicle_id) <= (right.first_window_index, right.global_vehicle_id) else (right, left)
                for key, rows in duplicate.samples.items():
                    canonical.samples.setdefault(key, []).extend(rows)
                canonical.hit_windows = sorted(set(canonical.hit_windows + duplicate.hit_windows))
                canonical.descriptors_by_window.update(duplicate.descriptors_by_window)
                canonical.last_window_index = max(canonical.last_window_index, duplicate.last_window_index)
                canonical.misses = min(canonical.misses, duplicate.misses)
                canonical.status = "confirmed" if "confirmed" in (canonical.status, duplicate.status) else "tentative"
                self._states[canonical.global_vehicle_id] = canonical
                self._states.pop(duplicate.global_vehicle_id, None)
                self._aliases[duplicate.global_vehicle_id] = canonical.global_vehicle_id
                self._all_ids.discard(duplicate.global_vehicle_id)
                removed.append(duplicate.global_vehicle_id)
                aliases[duplicate.global_vehicle_id] = canonical.global_vehicle_id
                left_id = canonical.global_vehicle_id
                left = canonical
        return removed, aliases

    def _state_to_track(self, state: _TrackState) -> StreamingTrack:
        points = _smooth_points(state.samples, self.config)
        descriptor = _descriptor(points)
        speed = descriptor.speed_kmh if descriptor is not None and np.isfinite(descriptor.speed_kmh) else float("nan")
        positions = sorted(float(point.get("position_m", 0.0)) for point in points)
        observed_count = sum(1 for point in points if bool(point.get("observed", True)))
        max_gap = max((right - left for left, right in zip(positions, positions[1:])), default=0.0)
        last_seen = max((float(point.get("time_s", 0.0)) for point in points), default=state.first_window_start_s)
        return StreamingTrack(
            global_vehicle_id=state.global_vehicle_id,
            source_window_start_s=state.first_window_start_s,
            source_track_id=state.last_source_track_id,
            direction=state.direction,
            points=points,
            median_speed_kmh=float(speed),
            confidence=state.confidence,
            observed_count=observed_count,
            span_m=float(max(positions, default=0.0) - min(positions, default=0.0)),
            max_gap_m=float(max_gap),
            enters_window=state.enters_window,
            exits_window=state.exits_window,
            ambiguous_crossing=state.ambiguous_crossing,
            last_seen_s=last_seen,
        )

    def update(self, batch: WorkerBatch) -> StitchUpdate:
        self._window_index += 1
        candidates = self._window_candidates(batch)
        missing_limit = self._missing_limit()
        states = [state for state in self._states.values() if state.misses <= missing_limit]
        matrix = np.full((len(states), len(candidates)), 1e6, dtype=np.float64)
        for row, state in enumerate(states):
            state_candidate = self._state_candidate(state)
            for column, candidate in enumerate(candidates):
                valid, cost, _overlap, _speed = _descriptor_distance(state_candidate.descriptor, candidate.descriptor, self.config)
                if valid:
                    # A skipped source window is allowed but slightly penalized.
                    cost += 0.15 * min(state.misses, missing_limit)
                    matrix[row, column] = cost
        matches: dict[int, int] = {}
        if matrix.size:
            rows, columns = linear_sum_assignment(matrix)
            matches = {int(column): int(row) for row, column in zip(rows, columns) if matrix[row, column] < 1.0}

        gap_reconnections = 0
        for column, row in matches.items():
            state = states[row]
            if state.misses:
                gap_reconnections += 1
            state.add(candidates[column], self._window_index, batch.start_s, self.config)
        matched_states = {states[row].global_vehicle_id for row in matches.values()}
        for state in states:
            if state.global_vehicle_id not in matched_states:
                state.misses += 1
        for column, candidate in enumerate(candidates):
            if column not in matches:
                state = self._new_state(candidate, batch.start_s)
                self._states[state.global_vehicle_id] = state

        for state in self._states.values():
            if state.misses > missing_limit and state.status == "tentative":
                # Keep confirmed histories for final export, but never revive
                # an unconfirmed one after it has expired.
                state.status = "retired"

        removed, aliases = self._merge_states()
        for state in self._states.values():
            if state.status == "tentative" and len(state.hit_windows) >= self.config.confirmation_hits and (self._window_index - state.first_window_index + 1) <= self.config.confirmation_batches:
                state.status = "confirmed"
            if state.status == "confirmed":
                self._all_ids.add(state.global_vehicle_id)
        tracks = [self._state_to_track(state) for state in self._states.values() if state.status == "confirmed" and state.misses <= missing_limit]
        tracks.sort(key=lambda item: (item.last_seen_s, item.global_vehicle_id))
        diagnostics = {
            "window_candidates": len(batch.tracks),
            "window_unique": len(candidates),
            "tentative_tracks": sum(1 for state in self._states.values() if state.status == "tentative"),
            "gap_reconnections": gap_reconnections,
            "id_merges": len(aliases),
        }
        self.last_window_tracks = tracks
        self.last_update = StitchUpdate(tracks, removed, aliases, diagnostics)
        return self.last_update

    def finalize(self) -> StitchUpdate:
        """Publish edge/boundary tracks and perform the final canonical merge."""
        for state in self._states.values():
            if state.status == "tentative" and sum(len(rows) for rows in state.samples.values()) >= 3:
                state.status = "confirmed"
                self._all_ids.add(state.global_vehicle_id)
        removed, aliases = self._merge_states(include_retired=True)
        for state in self._states.values():
            if state.status == "confirmed":
                self._all_ids.add(state.global_vehicle_id)
        tracks = [self._state_to_track(state) for state in self._states.values() if state.status == "confirmed"]
        tracks.sort(key=lambda item: (item.last_seen_s, item.global_vehicle_id))
        diagnostics = {
            "window_candidates": 0,
            "window_unique": 0,
            "tentative_tracks": 0,
            "gap_reconnections": 0,
            "id_merges": len(aliases),
            "finalized": True,
        }
        self.last_window_tracks = tracks
        self.last_update = StitchUpdate(tracks, removed, aliases, diagnostics)
        return self.last_update

    def counters(self, *, window_candidates: int, window_unique: int | None = None, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
        active = [state for state in self._states.values() if state.status == "confirmed" and state.misses <= self._missing_limit()]
        speeds = [self._state_to_track(state).median_speed_kmh for state in active]
        speeds = [speed for speed in speeds if np.isfinite(speed)]
        payload = {
            "window_candidates": int(window_candidates),
            "window_unique": int(window_unique if window_unique is not None else (diagnostics or {}).get("window_unique", window_candidates)),
            "tentative_tracks": sum(1 for state in self._states.values() if state.status == "tentative"),
            "current_unique": len(active),
            "cumulative_unique": self.cumulative_unique_count,
            "mean_speed_kmh": float(np.mean(speeds)) if speeds else None,
            "median_speed_kmh": float(np.median(speeds)) if speeds else None,
            "speed_range_kmh": [float(min(speeds)), float(max(speeds))] if speeds else [None, None],
        }
        if diagnostics:
            payload.update({key: value for key, value in diagnostics.items() if key not in payload})
        return payload


def _merge_point_lists(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for point in [*previous, *current]:
        key = int(point.get("channel_index", -1))
        grouped.setdefault(key, []).append(point)
    result: list[dict[str, Any]] = []
    for rows in grouped.values():
        times = np.asarray([float(row.get("time_s", 0.0)) for row in rows], dtype=np.float64)
        representative = min(rows, key=lambda row: abs(float(row.get("time_s", 0.0)) - float(np.median(times))))
        merged = dict(representative)
        merged["time_s"] = float(np.median(times))
        merged["observed"] = any(bool(row.get("observed", True)) for row in rows)
        result.append(merged)
    return sorted(result, key=lambda point: (int(point.get("channel_index", -1)), float(point.get("position_m", 0.0))))
