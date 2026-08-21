from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Station:
    channel_index: int
    station_id: str
    position_m: float
    sequence: int | None = None
    location: str = ""


@dataclass(frozen=True)
class StationGeometry:
    stations: tuple[Station, ...]

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("station geometry cannot be empty")
        indices = [item.channel_index for item in self.stations]
        if indices != list(range(len(indices))):
            raise ValueError("channel indices must be contiguous and ordered")
        positions = self.positions_m
        if len(positions) > 1 and not np.all(np.diff(positions) > 0):
            raise ValueError("station positions must be strictly increasing")

    @property
    def positions_m(self) -> np.ndarray:
        return np.asarray([item.position_m for item in self.stations], dtype=np.float64)

    @property
    def station_ids(self) -> list[str]:
        return [item.station_id for item in self.stations]

    def __len__(self) -> int:
        return len(self.stations)


@dataclass
class TrackPoint:
    channel_index: int
    time_s: float
    position_m: float
    station_id: str = ""
    observed: bool = True
    score: float | None = None
    observation_id: int | None = None


@dataclass
class Track:
    track_id: str
    points: list[TrackPoint]
    direction: str = "unknown"
    median_speed_kmh: float | None = None
    confidence: float | None = None
    score: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionBatch:
    method_id: str
    preset_id: str
    request_id: str
    start_s: float
    duration_s: float
    tracks: list[Track]
    diagnostics: dict[str, Any] = field(default_factory=dict)
