from __future__ import annotations

import numpy as np

from auto_track_common import Station, StationGeometry
from kalman_seed_tracker import KalmanVehicleTracker


def _scene() -> tuple[np.ndarray, StationGeometry, float]:
    fs = 20.0
    positions = np.asarray([0, 100, 200, 300, 400, 500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400], dtype=float)
    geometry = StationGeometry(tuple(Station(i, f"S{i}", float(position), i) for i, position in enumerate(positions)))
    times = np.arange(2400, dtype=np.float64) / fs
    data = np.zeros((len(times), len(positions)), dtype=np.float32)
    for base, speed in ((4.0, 75.0), (16.0, 84.0)):
        for channel, position in enumerate(positions):
            peak_time = base + (positions[-1] - position) / (speed / 3.6)
            data[:, channel] += 2.0 * np.exp(-0.5 * ((times - peak_time) / 0.12) ** 2)
    data[:, 10] = 0.0
    return data, geometry, fs


def test_kalman_tracks_multiple_vehicles_and_marks_reconnected_gap() -> None:
    data, geometry, fs = _scene()
    batch = KalmanVehicleTracker().predict_window(data, sample_rate_hz=fs, geometry=geometry, duration_s=120.0)
    assert len(batch.tracks) == 2
    assert batch.diagnostics["predicted_point_count"] > 0
    for track in batch.tracks:
        assert 60.0 <= float(track.median_speed_kmh) <= 90.0
        assert any(not point.observed for point in track.points)
        assert all(np.isfinite(point.time_s) for point in track.points)


def test_kalman_blank_window_is_empty() -> None:
    data, geometry, fs = _scene()
    batch = KalmanVehicleTracker().predict_window(np.zeros_like(data), sample_rate_hz=fs, geometry=geometry, duration_s=120.0)
    assert batch.tracks == []
