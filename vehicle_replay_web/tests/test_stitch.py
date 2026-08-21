from __future__ import annotations

from vehicle_replay_web.methods import WorkerBatch, WorkerPoint, WorkerTrack
from vehicle_replay_web.stitch import TrackStitcher


def _batch(start_s: float, track_id: str, positions: list[float], *, offset_s: float = 0.0) -> WorkerBatch:
    points = [
        WorkerPoint(
            channel_index=int(position / 100.0),
            station_id=f"S{int(position / 100.0)}",
            position_m=position,
            time_s=20.0 + position / 100.0 - start_s + offset_s,
            observed=True,
        )
        for index, position in enumerate(positions)
    ]
    track = WorkerTrack(
        track_id=track_id,
        direction="unknown",
        points=points,
        median_speed_kmh=72.0,
        confidence=0.9,
        score=1.0,
        observed_count=len(points),
        span_m=positions[-1] - positions[0],
        max_gap_m=100.0,
        enters_window=False,
        exits_window=False,
        ambiguous_crossing=False,
    )
    return WorkerBatch([track], [], start_s, 120.0, {})


def test_disjoint_station_ranges_match_by_line_extrapolation() -> None:
    stitcher = TrackStitcher()
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    # The next window sees only the next station segment. Its small offset
    # represents independent fitting error in the second inference.
    stitcher.update(_batch(60.0, "b", [300.0, 400.0, 500.0], offset_s=0.15))
    result = stitcher.finalize().tracks
    assert len(result) == 1
    assert result[0].global_vehicle_id == "V0001"
    assert len(result[0].points) == 6
    assert stitcher.cumulative_unique_count == 1


def test_incompatible_disjoint_lines_are_not_merged() -> None:
    stitcher = TrackStitcher()
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    stitcher.update(_batch(60.0, "b", [300.0, 400.0, 500.0], offset_s=8.0))
    result = stitcher.finalize().tracks
    assert len(result) == 2
    assert result[0].global_vehicle_id != result[1].global_vehicle_id
    assert stitcher.cumulative_unique_count == 2


def test_disjoint_parallel_vehicles_three_seconds_apart_stay_separate() -> None:
    stitcher = TrackStitcher()
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    stitcher.update(_batch(60.0, "b", [300.0, 400.0, 500.0], offset_s=3.0))
    assert len(stitcher.finalize().tracks) == 2


def test_repeated_window_points_collapse_to_station_median() -> None:
    stitcher = TrackStitcher()
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    stitcher.update(_batch(2.0, "b", [0.0, 100.0, 200.0], offset_s=0.2))
    result = stitcher.finalize().tracks
    assert len(result) == 1
    assert len(result[0].points) == 3
    assert len({int(point["channel_index"]) for point in result[0].points}) == 3


def test_same_window_duplicate_candidates_create_one_vehicle() -> None:
    stitcher = TrackStitcher(stride_s=2.0, window_s=120.0)
    first = _batch(0.0, "a", [0.0, 100.0, 200.0])
    duplicate = _batch(0.0, "a-duplicate", [0.0, 100.0, 200.0], offset_s=0.2).tracks[0]
    first.tracks.append(duplicate)
    stitcher.update(first)
    stitcher.update(_batch(2.0, "b", [0.0, 100.0, 200.0]))
    result = stitcher.finalize().tracks
    assert len(result) == 1
    assert stitcher.cumulative_unique_count == 1


def test_one_or_two_empty_windows_reconnect_the_same_id() -> None:
    stitcher = TrackStitcher(stride_s=2.0, window_s=120.0)
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    stitcher.update(WorkerBatch([], [], 2.0, 120.0, {}))
    stitcher.update(WorkerBatch([], [], 4.0, 120.0, {}))
    update = stitcher.update(_batch(6.0, "d", [0.0, 100.0, 200.0]))
    assert update.diagnostics["gap_reconnections"] == 1
    assert [item.global_vehicle_id for item in stitcher.finalize().tracks] == ["V0001"]


def test_reconnect_budget_is_stride_normalized() -> None:
    # Four empty 2-second windows still represent the same 10-second physical
    # horizon that a 5-second run gets with one empty window.
    stitcher = TrackStitcher(stride_s=2.0, window_s=120.0)
    stitcher.update(_batch(0.0, "a", [0.0, 100.0, 200.0]))
    for index in range(1, 5):
        stitcher.update(WorkerBatch([], [], 2.0 * index, 120.0, {}))
    update = stitcher.update(_batch(10.0, "b", [0.0, 100.0, 200.0]))
    assert update.diagnostics["gap_reconnections"] == 1
    assert [item.global_vehicle_id for item in stitcher.finalize().tracks] == ["V0001"]


def test_parallel_vehicle_three_seconds_apart_stays_separate() -> None:
    stitcher = TrackStitcher(stride_s=2.0, window_s=120.0)
    batch = _batch(0.0, "a", [0.0, 100.0, 200.0])
    batch.tracks.append(_batch(0.0, "b", [0.0, 100.0, 200.0], offset_s=3.0).tracks[0])
    stitcher.update(batch)
    stitcher.update(_batch(2.0, "a2", [0.0, 100.0, 200.0]))
    stitcher.update(_batch(4.0, "a3", [0.0, 100.0, 200.0]))
    assert len(stitcher.finalize().tracks) == 2


def test_tentative_duplicate_emits_canonical_alias() -> None:
    stitcher = TrackStitcher(stride_s=2.0, window_s=120.0)
    batch = _batch(0.0, "a", [0.0, 100.0, 200.0])
    batch.tracks.append(_batch(0.0, "b", [0.0, 100.0, 200.0], offset_s=1.0).tracks[0])
    update = stitcher.update(batch)
    assert update.id_aliases == {"V0002": "V0001"}
    assert update.removed_track_ids == ["V0002"]
