import unittest

from autotrack.core.single_vehicle_tracker import merge_single_vehicle_track_fragments
from autotrack.core.track_extractor_graph import Track, TrackPoint


class SingleVehicleMergeTest(unittest.TestCase):
    def test_merge_fragments_into_one_track(self) -> None:
        left = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=0, t_idx=100, time_s=1.0, offset_m=0.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=1, t_idx=145, time_s=1.45, offset_m=10.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=2, t_idx=190, time_s=1.9, offset_m=20.0, amp=1.0, score=1.0),
            ],
            total_score=3.0,
            mean_speed_kmh=80.0,
        )
        right = Track(
            track_id=1,
            direction="forward",
            points=[
                TrackPoint(ch_idx=2, t_idx=191, time_s=1.91, offset_m=20.0, amp=1.0, score=0.9),
                TrackPoint(ch_idx=3, t_idx=236, time_s=2.36, offset_m=30.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=4, t_idx=281, time_s=2.81, offset_m=40.0, amp=1.0, score=1.0),
            ],
            total_score=2.9,
            mean_speed_kmh=79.0,
        )
        merged = merge_single_vehicle_track_fragments(
            [left, right],
            fs=100.0,
            dx_m=10.0,
            direction="forward",
            merge_tolerance_samples=3,
            max_gap_channels=4,
        )
        self.assertEqual(len(merged), 1)
        track = merged[0]
        self.assertEqual([p.ch_idx for p in track.points], [0, 1, 2, 3, 4])
        self.assertGreaterEqual(track.total_score, 4.0)


if __name__ == "__main__":
    unittest.main()
