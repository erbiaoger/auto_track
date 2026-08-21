from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Station:
    channel_index: int
    sequence: int
    station_id: str
    position_m: float
    location: str = ""


@dataclass(frozen=True)
class StationGeometry:
    stations: tuple[Station, ...]

    def __post_init__(self) -> None:
        if not self.stations:
            raise ValueError("station geometry cannot be empty")
        indices = [station.channel_index for station in self.stations]
        if indices != list(range(len(indices))):
            raise ValueError("channel indices must be contiguous and ordered")
        positions = self.positions_m
        if not np.all(np.diff(positions) > 0):
            raise ValueError("station positions must be strictly increasing")

    @property
    def positions_m(self) -> np.ndarray:
        return np.asarray([station.position_m for station in self.stations], dtype=np.float64)

    @property
    def relative_positions_m(self) -> np.ndarray:
        positions = self.positions_m
        return positions - positions[0]

    def __len__(self) -> int:
        return len(self.stations)


@dataclass
class VehicleObservation:
    observation_id: int
    channel_index: int
    station_id: str
    position_m: float
    time_s: float
    gauss_score: float
    pre_score: float
    raw_energy: float
    network_score: float = 0.0
    crossing_score: float = 0.0
    strong: bool = False
    ambiguous: bool = False
    embedding: tuple[float, ...] = ()

    @property
    def evidence_score(self) -> float:
        values = np.asarray(
            [self.gauss_score, self.pre_score, self.raw_energy, self.network_score],
            dtype=np.float64,
        )
        values = np.clip(values, 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - values))


@dataclass
class TrackPoint:
    channel_index: int
    station_id: str
    position_m: float
    time_s: float
    observed: bool
    observation_id: int | None = None
    residual_s: float | None = None
    ambiguous: bool = False


@dataclass
class VehicleTrack:
    track_id: str
    points: list[TrackPoint]
    median_speed_kmh: float
    local_speeds_kmh: list[float]
    confidence: float
    score: float
    observed_count: int
    span_m: float
    median_residual_s: float
    max_gap_m: float
    enters_window: bool
    exits_window: bool
    ambiguous_crossing: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HoughSeed:
    slope_s_per_m: float
    intercept_s: float
    score: float
    support: int = 0
    source: str = "unknown"


@dataclass
class CandidatePath:
    observation_ids: tuple[int, ...]
    score: float
    seed: HoughSeed
    edge_scores: tuple[float, ...] = ()


@dataclass
class TrackBatch:
    tracks: list[VehicleTrack]
    observations: list[VehicleObservation]
    start_s: float
    duration_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": self.start_s,
            "duration_s": self.duration_s,
            "tracks": [track.to_dict() for track in self.tracks],
            "diagnostics": self.diagnostics,
        }
