import unittest

import numpy as np

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.core.track_fusion import extend_peakslot_tracks_with_graph


def _add_peak(data: np.ndarray, ch: int, t_idx: int, amp: float = 5.0) -> None:
    data[int(ch), int(t_idx)] = float(amp)


def _track_from_channels(channels: list[int], *, fs: float, dx_m: float, step_samples: int) -> Track:
    points = []
    for ch in channels:
        t_idx = 100 + int(ch) * int(step_samples)
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(fs),
                offset_m=float(ch) * float(dx_m),
                amp=5.0,
                score=1.0,
            )
        )
    return Track(track_id=7, direction="forward", points=points, total_score=float(len(points)), mean_speed_kmh=80.0)


class TrackFusionTest(unittest.TestCase):
    def test_extends_peakslot_seed_at_both_ends(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 600), dtype=np.float32)
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * step)
        seed = _track_from_channels([2, 3, 4, 5, 6], fs=fs, dx_m=dx_m, step_samples=step)

        fused = extend_peakslot_tracks_with_graph(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "fusion_mode": "graph_extend",
                "fusion_graph_prominence": 0.1,
                "fusion_graph_min_peak_distance": 10,
                "fusion_graph_max_skip_channels": 4,
            },
        )

        channels = [point.ch_idx for point in fused[0].points]
        self.assertLessEqual(min(channels), 1)
        self.assertGreaterEqual(max(channels), 7)
        self.assertGreater(len(channels), len(seed.points))
        self.assertTrue(60.0 <= fused[0].mean_speed_kmh <= 100.0)

    def test_rejects_physics_inconsistent_extension(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 600), dtype=np.float32)
        for ch in [2, 3, 4, 5, 6]:
            _add_peak(data, ch, 100 + ch * step)
        _add_peak(data, 1, 20)
        _add_peak(data, 7, 580)
        seed = _track_from_channels([2, 3, 4, 5, 6], fs=fs, dx_m=dx_m, step_samples=step)

        fused = extend_peakslot_tracks_with_graph(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "fusion_mode": "graph_extend",
                "fusion_graph_prominence": 0.1,
                "fusion_graph_min_peak_distance": 10,
                "fusion_graph_max_skip_channels": 4,
            },
        )

        self.assertEqual([point.ch_idx for point in fused[0].points], [2, 3, 4, 5, 6])

    def test_bridges_internal_gap_without_duplicate_channels(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 600), dtype=np.float32)
        for ch in [2, 3, 4, 5, 6]:
            _add_peak(data, ch, 100 + ch * step)
        seed = _track_from_channels([2, 3, 5, 6], fs=fs, dx_m=dx_m, step_samples=step)

        fused = extend_peakslot_tracks_with_graph(
            data,
            fs,
            dx_m,
            [seed],
            "forward",
            60.0,
            100.0,
            {
                "fusion_mode": "graph_extend",
                "fusion_extend_left": False,
                "fusion_extend_right": False,
                "fusion_graph_prominence": 0.1,
                "fusion_graph_min_peak_distance": 10,
                "fusion_graph_max_skip_channels": 4,
                "fusion_bridge_search_radius_samples": 80,
            },
        )

        channels = [point.ch_idx for point in fused[0].points]
        self.assertIn(4, channels)
        self.assertEqual(len(channels), len(set(channels)))


if __name__ == "__main__":
    unittest.main()
