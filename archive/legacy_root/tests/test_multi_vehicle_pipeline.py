import unittest
from unittest import mock

import numpy as np
import torch

from autotrack.dl.multi_vehicle_pipeline import MultiVehiclePipelineConfig, extract_multi_vehicle_tracks, _proposal_peak_boxes
from autotrack.core.track_extractor_graph import ExtractorConfig
from autotrack.core.track_extractor_graph import Track, TrackPoint


class MultiVehiclePipelineTest(unittest.TestCase):
    def test_extracts_two_tracks_from_handmade_segment(self) -> None:
        n_channels = 20
        n_samples = 2000
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            for center, amp, slope in [(120 + 30 * ch, 2.0, 30), (350 + 35 * ch, 1.8, 35)]:
                idx = np.arange(max(0, center - 4), min(n_samples, center + 5), dtype=np.float32)
                pulse = amp * np.exp(-0.5 * ((idx - float(center)) / 1.5) ** 2)
                data[ch, max(0, center - 4) : min(n_samples, center + 5)] += pulse.astype(np.float32)

        cfg = MultiVehiclePipelineConfig(
            candidate_graph=ExtractorConfig(
                prominence=0.05,
                min_peak_distance=1,
                max_peaks_per_channel=64,
                max_skip_channels=8,
                min_track_channels=20,
                min_track_score=2.0,
                edge_relax_enabled=True,
                edge_min_track_channels=4,
                edge_time_margin_seconds=2.0,
                edge_min_score_scale=0.3,
            ),
            candidate_limit=8,
            candidate_min_score=1.0,
            dedup_tolerance_samples=40,
            dedup_min_overlap_channels=10,
            dedup_min_overlap_ratio=0.7,
            crop_channel_margin=2,
            crop_time_margin_s=1.0,
            refine_with_model=False,
        )
        tracks = extract_multi_vehicle_tracks(
            data,
            fs=100.0,
            dx_m=10.0,
            direction="forward",
            vmin_kmh=40.0,
            vmax_kmh=140.0,
            config=cfg,
        )
        self.assertGreaterEqual(len(tracks), 2)
        self.assertTrue(all(len(tr.points) >= 10 for tr in tracks[:2]))

    def test_filters_low_confidence_candidates_before_refinement(self) -> None:
        n_channels = 12
        n_samples = 1200
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            center = 100 + 25 * ch
            idx = np.arange(max(0, center - 3), min(n_samples, center + 4), dtype=np.float32)
            pulse = 2.0 * np.exp(-0.5 * ((idx - float(center)) / 1.2) ** 2)
            data[ch, max(0, center - 3) : min(n_samples, center + 4)] += pulse.astype(np.float32)

        cfg = MultiVehiclePipelineConfig(
            candidate_graph=ExtractorConfig(
                prominence=0.05,
                min_peak_distance=1,
                max_peaks_per_channel=32,
                max_skip_channels=6,
                min_track_channels=8,
                min_track_score=1.0,
                edge_relax_enabled=True,
                edge_min_track_channels=4,
                edge_time_margin_seconds=1.0,
                edge_min_score_scale=0.3,
            ),
            candidate_limit=4,
            candidate_min_score=0.5,
            dedup_tolerance_samples=30,
            dedup_min_overlap_channels=4,
            dedup_min_overlap_ratio=0.7,
            crop_channel_margin=2,
            crop_time_margin_s=0.8,
            refine_with_model=True,
            min_model_confidence=0.9,
        )

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                batch = int(x.shape[0])
                device = x.device
                return {
                    "heatmap_logits": torch.zeros((batch, x.shape[-2], x.shape[-1]), device=device),
                    "objectness_logits": torch.zeros((batch,), device=device),
                    "direction_logits": torch.zeros((batch, 2), device=device),
                    "speed": torch.zeros((batch,), device=device),
                    "line_endpoints": torch.zeros((batch, 2), device=device),
                    "trajectory_time": torch.zeros((batch, x.shape[-2]), device=device),
                }

        with mock.patch("autotrack.dl.multi_vehicle_pipeline.load_checkpoint_model", return_value=(DummyModel(), None)):
            tracks = extract_multi_vehicle_tracks(
                data,
                fs=100.0,
                dx_m=10.0,
                direction="forward",
                vmin_kmh=40.0,
                vmax_kmh=140.0,
                config=cfg,
                model_path="/tmp/does-not-matter.pt",
                device="cpu",
            )

        self.assertEqual(tracks, [])

    def test_refined_crop_candidate_stays_in_global_coordinates(self) -> None:
        n_channels = 10
        n_samples = 200
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        candidate = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=4, t_idx=40, time_s=0.4, offset_m=40.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=5, t_idx=60, time_s=0.6, offset_m=50.0, amp=1.0, score=1.0),
            ],
            total_score=10.0,
            mean_speed_kmh=80.0,
        )
        refined_local = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=0, t_idx=0, time_s=0.0, offset_m=0.0, amp=1.0, score=1.0),
                TrackPoint(ch_idx=1, t_idx=20, time_s=0.02, offset_m=10.0, amp=1.0, score=1.0),
            ],
            total_score=12.0,
            mean_speed_kmh=80.0,
        )

        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                batch = int(x.shape[0])
                device = x.device
                return {
                    "heatmap_logits": torch.zeros((batch, x.shape[-2], x.shape[-1]), device=device),
                    "objectness_logits": torch.zeros((batch,), device=device),
                    "direction_logits": torch.zeros((batch, 2), device=device),
                    "speed": torch.zeros((batch,), device=device),
                    "line_endpoints": torch.zeros((batch, 2), device=device),
                    "trajectory_time": torch.zeros((batch, x.shape[-2]), device=device),
                }

        cfg = MultiVehiclePipelineConfig(
            candidate_limit=1,
            candidate_min_score=1.0,
            dedup_tolerance_samples=30,
            dedup_min_overlap_channels=2,
            dedup_min_overlap_ratio=0.5,
            crop_channel_margin=2,
            crop_time_margin_s=0.5,
            refine_with_model=True,
            min_model_confidence=0.0,
            iterative_extraction=False,
        )
        with (
            mock.patch("autotrack.dl.multi_vehicle_pipeline.extract_all", return_value=[candidate]),
            mock.patch("autotrack.dl.multi_vehicle_pipeline.load_checkpoint_model", return_value=(DummyModel(), None)),
            mock.patch("autotrack.dl.multi_vehicle_pipeline._track_candidate_confidence", return_value=1.0),
            mock.patch("autotrack.dl.multi_vehicle_pipeline.predict_single_vehicle_track", return_value=[refined_local]),
        ):
            tracks = extract_multi_vehicle_tracks(
                data,
                fs=100.0,
                dx_m=10.0,
                direction="forward",
                vmin_kmh=40.0,
                vmax_kmh=140.0,
                config=cfg,
                model_path="/tmp/dummy.pt",
                device="cpu",
            )

        self.assertEqual(len(tracks), 1)
        self.assertTrue(all(0 <= int(p.ch_idx) < n_channels for p in tracks[0].points))

    def test_proposal_peak_boxes_falls_back_on_weak_heatmap(self) -> None:
        heatmap = np.zeros((8, 64), dtype=np.float32)
        heatmap[3, 18] = 0.12
        heatmap[6, 40] = 0.08
        boxes = _proposal_peak_boxes(
            heatmap,
            min_score=0.5,
            pad_channels=1,
            pad_time=2,
            max_peaks=4,
        )
        self.assertGreaterEqual(len(boxes), 2)
        self.assertTrue(all(len(box) == 4 for _, box in boxes))


if __name__ == "__main__":
    unittest.main()
