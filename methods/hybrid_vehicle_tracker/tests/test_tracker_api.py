from __future__ import annotations

import numpy as np

from hybrid_vehicle_tracker.config import (
    AssociationConfig,
    DataConfig,
    ModelConfig,
    RuntimeConfig,
    TrackerConfig,
)
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker
from hybrid_vehicle_tracker.types import Station, StationGeometry


def test_public_predict_api_runs_without_checkpoint():
    sample_rate = 100.0
    duration = 30.0
    stations = 8
    geometry = StationGeometry(
        tuple(Station(index, index + 1, f"S{index}", index * 100.0) for index in range(stations))
    )
    raw = np.random.default_rng(3).normal(0, 0.1, (int(sample_rate * duration), stations)).astype(np.float32)
    pre = np.zeros_like(raw)
    gauss = np.zeros_like(raw)
    for channel in range(6):
        time_s = 1.0 + 0.04 * geometry.positions_m[channel]
        index = int(round(time_s * sample_rate))
        if index < len(raw):
            raw[max(0, index - 2) : index + 3, channel] += 5.0
            pre[index, channel] = 5.0
            gauss[index, channel] = 0.7
    config = TrackerConfig(
        data=DataConfig(sample_rate_hz=sample_rate, feature_rate_hz=10.0),
        model=ModelConfig(base_channels=4, embedding_dim=8, hough_slopes=9, hough_top_k=12),
        association=AssociationConfig(
            min_observations=3,
            min_span_m=200.0,
            max_gap_m=300.0,
            candidate_min_distance_s=0.5,
            min_track_score=-100.0,
        ),
        runtime=RuntimeConfig(device="cpu"),
    )
    tracker = HybridVehicleTracker(config)
    batch = tracker.predict(raw, pre, gauss, geometry, duration_s=duration)
    assert batch.diagnostics["direction"] == "increasing_time_with_position"
    assert tracker.last_artifacts is not None
    assert len(batch.observations) >= 3
