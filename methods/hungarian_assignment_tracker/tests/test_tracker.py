from __future__ import annotations

import numpy as np

from auto_track_common import Station, StationGeometry
from hungarian_assignment_tracker import HungarianAssignmentTracker


def _scene(*, missing_channel: int | None = None) -> tuple[np.ndarray, StationGeometry, float]:
    fs = 20.0
    positions = np.asarray([0, 100, 200, 300, 400, 500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400], dtype=float)
    geometry = StationGeometry(tuple(Station(i, f"S{i}", float(position), i) for i, position in enumerate(positions)))
    times = np.arange(2400, dtype=np.float64) / fs
    data = np.zeros((len(times), len(positions)), dtype=np.float32)
    for base, speed in ((4.0, 75.0), (16.0, 84.0)):
        for channel, position in enumerate(positions):
            peak_time = base + (positions[-1] - position) / (speed / 3.6)
            data[:, channel] += 2.0 * np.exp(-0.5 * ((times - peak_time) / 0.12) ** 2)
    if missing_channel is not None:
        data[:, int(missing_channel)] = 0.0
    return data, geometry, fs


def test_hungarian_tracks_two_reverse_vehicles_without_duplicate_observations() -> None:
    data, geometry, fs = _scene()
    batch = HungarianAssignmentTracker().predict_window(data, sample_rate_hz=fs, geometry=geometry, duration_s=120.0)
    assert len(batch.tracks) == 2
    assert all(60.0 <= float(track.median_speed_kmh) <= 90.0 for track in batch.tracks)
    for track in batch.tracks:
        observed = [point for point in track.points if point.observed]
        assert len({point.channel_index for point in observed}) == len(observed)
        assert len({(point.channel_index, round(point.time_s, 4)) for point in observed}) == len(observed)


def test_hungarian_uses_real_nonuniform_station_distance() -> None:
    data, geometry, fs = _scene()
    batch = HungarianAssignmentTracker().predict_window(data, sample_rate_hz=fs, geometry=geometry, duration_s=120.0)
    assert batch.tracks
    assert any(abs(float(track.median_speed_kmh) - 75.0) < 2.0 for track in batch.tracks)


def test_hungarian_blank_window_is_empty() -> None:
    data, geometry, fs = _scene()
    batch = HungarianAssignmentTracker().predict_window(np.zeros_like(data), sample_rate_hz=fs, geometry=geometry, duration_s=120.0)
    assert batch.tracks == []
    assert batch.diagnostics["candidate_count"] == 0
