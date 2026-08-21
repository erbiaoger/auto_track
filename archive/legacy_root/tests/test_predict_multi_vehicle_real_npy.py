import unittest

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.multi_vehicle_pipeline import _merge_track_fragments, _prune_short_fragments
from autotrack.dl.predict_multi_vehicle_real_npy import _deduplicate_tracks


def _track(track_id: int, ch0: int, t0: int, slope: int, length: int, *, score: float) -> Track:
    points = [
        TrackPoint(
            ch_idx=ch0 + i,
            t_idx=t0 + slope * i,
            time_s=float(t0 + slope * i) / 1000.0,
            offset_m=float(ch0 + i) * 100.0,
            amp=1.0,
            score=1.0,
        )
        for i in range(length)
    ]
    return Track(track_id=track_id, direction="forward", points=points, total_score=score, mean_speed_kmh=80.0)


class PredictMultiVehicleRealNpyTest(unittest.TestCase):
    def test_deduplicate_tracks_merges_line_similar_fragments(self) -> None:
        left = _track(0, 2, 1000, 30, 8, score=12.0)
        right = _track(1, 2, 1030, 30, 8, score=10.0)
        kept = _deduplicate_tracks([left, right], tol_samples=180, min_overlap_channels=2, min_overlap_ratio=0.45)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].track_id, 0)

    def test_merge_track_fragments_stitches_disjoint_collinear_tracks(self) -> None:
        left = _track(0, 2, 1000, 30, 6, score=9.0)
        right = _track(1, 8, 1180, 30, 6, score=8.0)
        merged = _merge_track_fragments(
            [left, right],
            fs=1000.0,
            dx_m=100.0,
            max_gap_channels=8,
            max_gap_seconds=0.5,
            line_distance_threshold=0.8,
            speed_diff_kmh=20.0,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].points), 12)
        self.assertEqual(merged[0].track_id, 0)

    def test_prune_short_fragments_drops_short_low_score_tracks(self) -> None:
        short = _track(0, 2, 1000, 30, 5, score=7.0)
        long = _track(1, 2, 1000, 30, 6, score=9.0)
        kept = _prune_short_fragments([short, long], min_keep_points=6)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].track_id, 1)

    def test_prune_short_fragments_keeps_edge_fragments(self) -> None:
        edge = _track(0, 2, 10, 30, 5, score=6.0)
        kept = _prune_short_fragments([edge], min_keep_points=6, edge_margin_points=20, n_samples=1200)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].track_id, 0)


if __name__ == "__main__":
    unittest.main()
