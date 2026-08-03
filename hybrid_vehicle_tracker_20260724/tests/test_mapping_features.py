from __future__ import annotations

import json

import numpy as np

from hybrid_vehicle_tracker.data.features import build_feature_batch
from hybrid_vehicle_tracker.data.mapping import load_station_geometry


def test_mapping_uses_physical_positions_and_preserves_200m_gap(tmp_path):
    rows = []
    locations = ["北87.4", "北87.5", "北87.7", "北87.8"]
    for index, location in enumerate(locations):
        rows.append(
            {
                "channel_index": index,
                "sequence": index + 1,
                "station_id": f"S{index}",
                "location": location,
            }
        )
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"selected_channels": rows}), encoding="utf-8")
    geometry = load_station_geometry(path)
    assert np.allclose(np.diff(geometry.positions_m), [100.0, 200.0, 100.0])


def test_feature_batch_has_five_planes_and_robust_scores(tmp_path):
    rows = [
        {
            "channel_index": index,
            "sequence": index + 1,
            "station_id": f"S{index}",
            "location": 83.5 + 0.1 * index,
        }
        for index in range(4)
    ]
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps({"selected_channels": rows}), encoding="utf-8")
    geometry = load_station_geometry(path)
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(1000, 4)).astype(np.float32)
    pre = rng.normal(size=(1000, 4)).astype(np.float32)
    gauss = np.zeros((1000, 4), dtype=np.float32)
    gauss[250, 2] = 0.7
    batch = build_feature_batch(
        raw, pre, gauss, geometry, sample_rate_hz=100.0, feature_rate_hz=10.0
    )
    assert batch.tensor.shape == (5, 4, 100)
    assert batch.raw_score.shape == (4, 100)
    assert np.isfinite(batch.tensor).all()
    assert np.all((batch.pre_score >= 0) & (batch.pre_score <= 1))
