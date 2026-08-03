from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from hybrid_vehicle_tracker.types import TrackBatch, TrackPoint, VehicleTrack


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


def _absolute_points(track: VehicleTrack, start_s: float) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for point in track.points:
        row = dict(vars(point))
        row["time_s"] = float(point.time_s + start_s)
        points.append(row)
    return points


def _cost(previous: StreamingTrack, current: VehicleTrack, start_s: float) -> float:
    old = {str(p["station_id"]): float(p["time_s"]) for p in previous.points if p.get("observed", True)}
    new = {
        str(p.station_id): float(p.time_s + start_s)
        for p in current.points
        if p.observed
    }
    common = sorted(set(old).intersection(new))
    if common:
        residual = float(np.median(np.abs([new[k] - old[k] for k in common])))
    else:
        old_points = [(float(p["position_m"]), float(p["time_s"])) for p in previous.points if p.get("observed", True)]
        new_points = [(float(p.position_m), float(p.time_s + start_s)) for p in current.points if p.observed]
        if len(old_points) >= 2 and len(new_points) >= 2:
            old_fit = np.polyfit(*np.asarray(old_points, dtype=np.float64).T, 1)
            new_fit = np.polyfit(*np.asarray(new_points, dtype=np.float64).T, 1)
            low = max(min(item[0] for item in old_points), min(item[0] for item in new_points))
            high = min(max(item[0] for item in old_points), max(item[0] for item in new_points))
            probes = np.asarray([low, (low + high) / 2.0, high]) if high >= low else np.asarray([])
            residual = float(np.median(np.abs(np.polyval(old_fit, probes) - np.polyval(new_fit, probes)))) if probes.size else 999.0
        else:
            residual = 999.0
    span = min(previous.span_m, current.span_m)
    speed_diff = abs(previous.median_speed_kmh - current.median_speed_kmh)
    if (len(common) < 3 and span < 500.0) or residual > 0.6 or speed_diff > 5.0:
        return 1e6
    overlap_term = min(1.0, len(common) / 8.0)
    return float(residual + speed_diff / 10.0 + (1.0 - overlap_term))


class TrackStitcher:
    """Assign stable IDs to overlapping streaming windows."""

    def __init__(self, *, stride_s: float = 60.0, window_s: float = 120.0, retention_s: float | None = None) -> None:
        self.stride_s = float(stride_s)
        self.window_s = float(window_s)
        self.retention_s = float(retention_s if retention_s is not None else max(2.0 * self.stride_s, self.window_s))
        self.reset()

    def reset(self) -> None:
        self._next_id = 1
        self._active: dict[str, StreamingTrack] = {}
        self._all_ids: set[str] = set()
        self.last_window_tracks: list[StreamingTrack] = []

    @property
    def cumulative_unique_count(self) -> int:
        return len(self._all_ids)

    @property
    def active_tracks(self) -> list[StreamingTrack]:
        return list(self._active.values())

    def update(self, batch: TrackBatch) -> list[StreamingTrack]:
        current = batch.tracks
        previous = list(self._active.values())
        matrix = np.full((len(previous), len(current)), 1e6, dtype=np.float64)
        for i, old in enumerate(previous):
            for j, new in enumerate(current):
                matrix[i, j] = _cost(old, new, batch.start_s)
        matches: dict[int, int] = {}
        if matrix.size:
            rows, cols = linear_sum_assignment(matrix)
            matches = {int(c): int(r) for r, c in zip(rows, cols) if matrix[r, c] < 1e5}

        updated: list[StreamingTrack] = []
        used_old: set[str] = set()
        for index, track in enumerate(current):
            old = previous[matches[index]] if index in matches else None
            if old is None:
                global_id = f"V{self._next_id:04d}"
                self._next_id += 1
            else:
                global_id = old.global_vehicle_id
                used_old.add(global_id)
            points = _absolute_points(track, batch.start_s)
            last_seen = max((float(p["time_s"]) for p in points), default=batch.start_s)
            updated.append(
                StreamingTrack(
                    global_vehicle_id=global_id,
                    source_window_start_s=float(batch.start_s),
                    source_track_id=str(track.track_id),
                    direction=str(getattr(track, "direction", "unknown")),
                    points=points,
                    median_speed_kmh=float(track.median_speed_kmh),
                    confidence=float(track.confidence) if track.confidence is not None else None,
                    observed_count=int(track.observed_count),
                    span_m=float(track.span_m),
                    max_gap_m=float(track.max_gap_m),
                    enters_window=bool(track.enters_window),
                    exits_window=bool(track.exits_window),
                    ambiguous_crossing=bool(track.ambiguous_crossing),
                    last_seen_s=last_seen,
                )
            )
        cutoff = float(batch.start_s - self.retention_s)
        self._active = {
            item.global_vehicle_id: item
            for item in updated
        }
        for old in previous:
            if old.global_vehicle_id not in used_old and old.last_seen_s >= cutoff:
                self._active.setdefault(old.global_vehicle_id, old)
        self._all_ids.update(item.global_vehicle_id for item in updated)
        self.last_window_tracks = updated
        return updated

    def counters(self, *, window_candidates: int) -> dict[str, Any]:
        latest = max((item.last_seen_s for item in self._active.values()), default=0.0)
        active = [item for item in self._active.values() if item.last_seen_s >= latest - self.window_s]
        speeds = [item.median_speed_kmh for item in active if np.isfinite(item.median_speed_kmh)]
        return {
            "window_candidates": int(window_candidates),
            "current_unique": len(active),
            "cumulative_unique": self.cumulative_unique_count,
            "mean_speed_kmh": float(np.mean(speeds)) if speeds else None,
            "median_speed_kmh": float(np.median(speeds)) if speeds else None,
            "speed_range_kmh": [float(min(speeds)), float(max(speeds))] if speeds else [None, None],
        }
