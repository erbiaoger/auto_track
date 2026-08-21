from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.shard_vehicle_pipeline import (
    ShardBundle,
    ShardRunConfig,
    extract_tracks_from_shard_bundle,
    load_shard_bundle,
)


class ShardVehiclePipelineTest(unittest.TestCase):
    def test_load_shard_bundle_reads_expected_metadata(self) -> None:
        shard = Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/xi_gauss_50_120s_large/test/shard_000000.pt")
        meta = shard.with_name("meta.json")
        if not shard.exists() or not meta.exists():
            self.skipTest("real shard data is not available")

        bundle = load_shard_bundle(shard, meta)
        self.assertEqual(int(bundle.meta["num_samples"]), 59)
        self.assertEqual(bundle.sample_count, 59)
        sample = bundle.sample_at(0)
        self.assertIn("x", sample)
        self.assertEqual(tuple(sample["x"].shape), (1, 50, 12000))

    def test_shifted_window_tracks_merge_across_windows(self) -> None:
        raw = torch.zeros((2, 32), dtype=torch.float32)
        payload = {
            "samples": [
                {"target": {"raw_window": raw}},
                {"target": {"raw_window": raw}},
            ]
        }
        meta = {
            "fs": 100.0,
            "dx_m": 10.0,
            "num_samples": 2,
            "window_start_samples": [0, 100],
            "source_shape_time_channel": [200, 2],
        }
        bundle = ShardBundle(shard_path=Path("/tmp/test.pt"), meta_path=Path("/tmp/meta.json"), payload=payload, meta=meta)

        track_a = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=0, t_idx=80, time_s=0.80, offset_m=0.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=1, t_idx=90, time_s=0.90, offset_m=10.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=2, t_idx=100, time_s=1.00, offset_m=20.0, amp=1.0, score=1.0),
            ],
            total_score=3.0,
            mean_speed_kmh=360.0,
        )
        track_b = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=3, t_idx=0, time_s=0.00, offset_m=30.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=4, t_idx=10, time_s=0.10, offset_m=40.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=5, t_idx=20, time_s=0.20, offset_m=50.0, amp=1.0, score=1.0),
            ],
            total_score=3.0,
            mean_speed_kmh=360.0,
        )

        with mock.patch(
            "autotrack.dl.shard_vehicle_pipeline.extract_multi_vehicle_tracks_with_models",
            side_effect=[[track_a], [track_b]],
        ):
            result = extract_tracks_from_shard_bundle(
                bundle,
                ShardRunConfig(
                    model_path=None,
                    proposal_model_path=None,
                    device="cpu",
                    direction="forward",
                    max_samples=2,
                    refine_with_model=False,
                ),
            )

        self.assertEqual(result.window_track_counts, [1, 1])
        self.assertEqual(len(result.tracks), 1)
        self.assertEqual([int(p.ch_idx) for p in result.tracks[0].points], [0, 1, 2, 3, 4, 5])
        self.assertEqual([int(p.t_idx) for p in result.tracks[0].points], [80, 90, 100, 100, 110, 120])


if __name__ == "__main__":
    unittest.main()
