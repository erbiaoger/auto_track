from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    raw_path: str = ""
    pre_path: str = ""
    gauss_path: str = ""
    mapping_path: str = ""
    sample_rate_hz: float = 1000.0
    feature_rate_hz: float = 20.0
    start_s: float = 0.0
    duration_s: float = 120.0


@dataclass
class ModelConfig:
    base_channels: int = 12
    embedding_dim: int = 32
    hough_slopes: int = 17
    hough_intercept_step_s: float = 0.5
    hough_top_k: int = 64
    # Sign of dt/dx in the supplied station order.  DAY11 vehicles move from
    # the high-position end toward the low-position end, hence -1.  The
    # magnitude is still restricted to 60--90 km/h by the Hough head.
    motion_direction: int = 1
    checkpoint: str | None = None

    def __post_init__(self) -> None:
        if int(self.motion_direction) not in (-1, 1):
            raise ValueError("motion_direction must be +1 or -1")
        self.motion_direction = int(self.motion_direction)


@dataclass
class AssociationConfig:
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 90.0
    edge_time_tolerance_s: float = 0.3
    seed_time_tolerance_s: float = 0.4
    max_gap_m: float = 600.0
    min_observations: int = 5
    min_span_m: float = 500.0
    max_median_residual_s: float = 0.3
    beam_width: int = 24
    paths_per_seed: int = 4
    max_candidates: int = 1200
    strong_gauss_threshold: float = 0.5
    weak_pre_quantile: float = 0.997
    candidate_min_distance_s: float = 1.25
    min_track_score: float = 0.0
    null_quantile: float = 0.995
    min_dense_seed_score: float = 0.20
    dense_support_threshold: float = 0.45
    # +1 means time increases with increasing physical position, -1 means
    # time increases while position decreases.  Keep this explicit so that a
    # valid vehicle is not rejected merely because the station file is stored
    # in the opposite travel direction.
    motion_direction: int = 1

    @property
    def min_slowness_s_per_m(self) -> float:
        return 3.6 / self.speed_max_kmh

    @property
    def max_slowness_s_per_m(self) -> float:
        return 3.6 / self.speed_min_kmh

    def __post_init__(self) -> None:
        if int(self.motion_direction) not in (-1, 1):
            raise ValueError("motion_direction must be +1 or -1")
        self.motion_direction = int(self.motion_direction)


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    seed: int = 20260724
    output_dir: str = "runs/latest"


@dataclass
class TrackerConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_tracker_config(path: str | Path) -> TrackerConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return TrackerConfig(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        association=AssociationConfig(**raw.get("association", {})),
        runtime=RuntimeConfig(**raw.get("runtime", {})),
    )
