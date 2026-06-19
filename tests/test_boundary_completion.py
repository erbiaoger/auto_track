import unittest

import numpy as np

from autotrack.core.boundary_completion import complete_tracks_to_boundaries
from autotrack.core.track_extractor_graph import Track, TrackPoint


def _add_peak(data: np.ndarray, ch: int, t_idx: int, amp: float = 5.0) -> None:
    data[int(ch), int(t_idx)] = float(amp)


def _point(ch: int, t_idx: int, *, fs: float, dx_m: float, score: float = 1.0) -> TrackPoint:
    return TrackPoint(
        ch_idx=int(ch),
        t_idx=int(t_idx),
        time_s=float(t_idx) / float(fs),
        offset_m=float(ch) * float(dx_m),
        amp=5.0,
        score=float(score),
    )


def _track(track_id: int, channels: list[int], *, fs: float, dx_m: float, step_samples: int, start: int = 100) -> Track:
    points = [_point(ch, start + int(ch) * int(step_samples), fs=fs, dx_m=dx_m) for ch in channels]
    return Track(track_id=int(track_id), direction="forward", points=points, total_score=float(len(points)), mean_speed_kmh=80.0)


class BoundaryCompletionTest(unittest.TestCase):
    def test_links_two_fragments_into_boundary_complete_track(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 700), dtype=np.float32)
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * step)
        left = _track(1, [1, 2, 3], fs=fs, dx_m=dx_m, step_samples=step)
        right = _track(2, [5, 6, 7], fs=fs, dx_m=dx_m, step_samples=step)
        diagnostics: dict[str, object] = {}

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [left, right],
            "forward",
            60.0,
            100.0,
            {
                "boundary_completion_mode": "repair",
                "boundary_max_gap_channels": 4,
                "boundary_graph_prominence": 0.1,
                "boundary_graph_min_peak_distance": 10,
                "boundary_bridge_search_radius_samples": 80,
            },
            diagnostics,
        )

        self.assertEqual(len(completed), 1)
        self.assertGreaterEqual(diagnostics["boundary_linked_fragment_count"], 1)
        self.assertGreaterEqual(diagnostics["boundary_completed_count"], 1)
        self.assertEqual([point.ch_idx for point in completed[0].points], [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_rejects_speed_inconsistent_fragment_link(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 900), dtype=np.float32)
        left = _track(1, [1, 2, 3], fs=fs, dx_m=dx_m, step_samples=45)
        right = Track(
            track_id=2,
            direction="forward",
            points=[_point(ch, 700 + ch * 3, fs=fs, dx_m=dx_m) for ch in [5, 6, 7]],
            total_score=3.0,
            mean_speed_kmh=80.0,
        )

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [left, right],
            "forward",
            60.0,
            100.0,
            {"boundary_completion_mode": "repair", "boundary_max_gap_channels": 4, "boundary_min_seed_channels": 3},
        )

        self.assertEqual(len(completed), 2)

    def test_bridges_internal_gap_with_real_peak_only(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 700), dtype=np.float32)
        for ch in [2, 3, 4, 5, 6]:
            _add_peak(data, ch, 100 + ch * step)
        seed = _track(1, [2, 3, 5, 6], fs=fs, dx_m=dx_m, step_samples=step)

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "boundary_completion_mode": "repair",
                "boundary_graph_prominence": 0.1,
                "boundary_graph_min_peak_distance": 10,
                "boundary_bridge_search_radius_samples": 80,
            },
        )

        channels = [point.ch_idx for point in completed[0].points]
        self.assertIn(4, channels)
        self.assertEqual(len(channels), len(set(channels)))

    def test_time_boundary_to_right_boundary_counts_as_complete(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 700), dtype=np.float32)
        points = [_point(ch, (ch - 4) * step, fs=fs, dx_m=dx_m) for ch in [4, 5, 6, 7, 8]]
        seed = Track(track_id=1, direction="forward", points=points, total_score=5.0, mean_speed_kmh=80.0)
        diagnostics: dict[str, object] = {}

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {"boundary_completion_mode": "strict"},
            diagnostics,
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(diagnostics["boundary_completed_count"], 1)

    def test_keeps_near_parallel_tracks_with_separate_support(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 900), dtype=np.float32)
        first = _track(1, [0, 1, 2, 3], fs=fs, dx_m=dx_m, step_samples=45)
        second = Track(
            track_id=2,
            direction="forward",
            points=[_point(ch, 270 + ch * 45, fs=fs, dx_m=dx_m) for ch in [4, 5, 6, 7, 8]],
            total_score=5.0,
            mean_speed_kmh=80.0,
        )

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [first, second],
            "forward",
            60.0,
            100.0,
            {"boundary_completion_mode": "repair", "boundary_max_gap_channels": 6},
        )

        self.assertEqual(len(completed), 2)

    def test_strict_does_not_treat_max_gap_as_boundary_support(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 700), dtype=np.float32)
        seed = _track(1, [1, 2, 3, 4, 5, 6, 7], fs=fs, dx_m=dx_m, step_samples=45)
        diagnostics: dict[str, object] = {}

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "boundary_completion_mode": "strict",
                "boundary_validation_only": True,
                "boundary_max_gap_channels": 4,
                "boundary_margin_channels": 0,
            },
            diagnostics,
        )

        self.assertEqual(completed, [])
        self.assertEqual(diagnostics["boundary_completed_count"], 0)
        self.assertEqual(diagnostics["boundary_rejected_fragment_count"], 1)

    def test_validation_only_does_not_add_graph_edge_points(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 700), dtype=np.float32)
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * step)
        seed = _track(1, [1, 2, 3, 4, 5, 6, 7], fs=fs, dx_m=dx_m, step_samples=step)

        completed = complete_tracks_to_boundaries(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "boundary_completion_mode": "strict",
                "boundary_validation_only": True,
                "boundary_max_gap_channels": 4,
                "boundary_margin_channels": 0,
                "boundary_graph_prominence": 0.1,
                "boundary_graph_min_peak_distance": 10,
            },
        )

        self.assertEqual(completed, [])


if __name__ == "__main__":
    unittest.main()
