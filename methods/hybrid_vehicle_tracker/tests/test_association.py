from __future__ import annotations

import numpy as np

from hybrid_vehicle_tracker.association.pipeline import associate_observations
from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.types import HoughSeed, Station, StationGeometry, VehicleObservation


def _geometry() -> StationGeometry:
    positions = [0, 100, 200, 300, 400, 600, 700, 800, 900, 1000]
    return StationGeometry(
        tuple(
            Station(index, index + 1, f"S{index}", float(position))
            for index, position in enumerate(positions)
        )
    )


def _observation(index, channel, time_s, geometry, ambiguous=False):
    station = geometry.stations[channel]
    return VehicleObservation(
        observation_id=index,
        channel_index=channel,
        station_id=station.station_id,
        position_m=station.position_m,
        time_s=time_s,
        gauss_score=0.70,
        pre_score=0.90,
        raw_energy=0.85,
        network_score=0.90,
        strong=True,
        ambiguous=ambiguous,
    )


def test_global_association_recovers_two_crossing_tracks():
    geometry = _geometry()
    rows = []
    for channel, position in enumerate(geometry.relative_positions_m):
        rows.append((channel, 5.0 + 0.05 * position, channel == 5))
        rows.append((channel, 11.0 + 0.04 * position, channel == 5))
    rows.sort(key=lambda item: (geometry.positions_m[item[0]], item[1]))
    observations = [
        _observation(index, channel, time_s, geometry, ambiguous)
        for index, (channel, time_s, ambiguous) in enumerate(rows)
    ]
    config = AssociationConfig(
        max_gap_m=300.0,
        min_observations=5,
        min_span_m=500.0,
        seed_time_tolerance_s=0.35,
        edge_time_tolerance_s=0.25,
        max_median_residual_s=0.25,
        min_track_score=-100.0,
        beam_width=16,
        paths_per_seed=3,
    )
    seeds = [
        HoughSeed(0.05, 5.0, 1.0, 10),
        HoughSeed(0.04, 11.0, 1.0, 10),
    ]
    result = associate_observations(
        observations,
        seeds,
        geometry,
        config,
        duration_s=60.0,
        embedding_dim=0,
        hough_intercept_step_s=0.5,
        hough_top_k=8,
    )
    assert len(result.tracks) == 2
    speeds = sorted(track.median_speed_kmh for track in result.tracks)
    assert np.allclose(speeds, [72.0, 90.0], atol=1.0)
    assert all(track.observed_count == 10 for track in result.tracks)


def test_missing_points_are_output_but_not_counted_as_observed():
    geometry = _geometry()
    # Deliberately omit the physical head channels.  Refinement should still
    # emit them as expected/observed=false points instead of shortening the
    # returned track to the first detected node.
    kept_channels = [2, 3, 4, 6, 7, 8, 9]
    observations = [
        _observation(index, channel, 2.0 + 0.05 * geometry.relative_positions_m[channel], geometry)
        for index, channel in enumerate(kept_channels)
    ]
    config = AssociationConfig(
        max_gap_m=400.0,
        min_observations=5,
        min_span_m=500.0,
        seed_time_tolerance_s=0.4,
        min_track_score=-100.0,
    )
    result = associate_observations(
        observations,
        [HoughSeed(0.05, 2.0, 1.0, 8)],
        geometry,
        config,
        duration_s=60.0,
        embedding_dim=0,
        hough_intercept_step_s=0.5,
        hough_top_k=4,
    )
    assert len(result.tracks) == 1
    track = result.tracks[0]
    assert track.observed_count == len(kept_channels)
    assert sum(point.observed for point in track.points) == len(kept_channels)
    assert any(not point.observed for point in track.points)
    assert track.points[0].channel_index == 0
    assert track.points[-1].channel_index == 9


def test_reverse_motion_direction_uses_travel_order_for_beam_and_fit():
    geometry = _geometry()
    observations = [
        _observation(index, channel, 42.0 - 0.05 * geometry.relative_positions_m[channel], geometry)
        for index, channel in enumerate(range(10))
    ]
    config = AssociationConfig(
        motion_direction=-1,
        max_gap_m=400.0,
        min_observations=5,
        min_span_m=500.0,
        seed_time_tolerance_s=0.25,
        edge_time_tolerance_s=0.25,
        min_track_score=-100.0,
    )
    result = associate_observations(
        observations,
        [HoughSeed(-0.05, 42.0, 1.0, 10)],
        geometry,
        config,
        duration_s=60.0,
        embedding_dim=0,
        hough_intercept_step_s=0.5,
        hough_top_k=4,
    )
    assert len(result.tracks) == 1
    assert np.isclose(result.tracks[0].median_speed_kmh, 72.0, atol=1.0)
