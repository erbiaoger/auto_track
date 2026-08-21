import unittest

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.vehicle_set_net import VehicleSetModelConfig, VehicleSetNet, decode_vehicle_set_tracks, refine_track_with_raw_window, vehicle_set_loss
from autotrack.dl.single_vehicle_focus_net import FocusModelConfig, SingleVehicleFocusNet, build_focus_targets, predict_single_vehicle_focus_track, single_vehicle_focus_loss


class VehicleSetNetTest(unittest.TestCase):
    def test_forward_loss_and_snap(self) -> None:
        model = VehicleSetNet(VehicleSetModelConfig(n_channels=12, base_dim=16, hidden_dim=32, query_dim=64, num_queries=4, decoder_layers=1, decoder_heads=4))
        x = torch.randn(2, 1, 12, 80)
        outputs = model(x)
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2, 4))
        self.assertEqual(tuple(outputs["visibility_logits"].shape), (2, 4, 12))
        self.assertEqual(tuple(outputs["time_logits"].shape), (2, 4, 12))

        batch = []
        for _ in range(2):
            n = 3
            batch.append(
                {
                    "time": torch.rand(n, 12),
                    "visibility": (torch.rand(n, 12) > 0.5).float(),
                    "direction": torch.randint(0, 2, (n,)),
                    "speed": torch.rand(n),
                    "gt_valid": torch.ones(n, dtype=torch.bool),
                }
            )
        loss, metrics = vehicle_set_loss(outputs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_time", metrics)

    def test_refine_track_with_raw_window_snaps_to_peak(self) -> None:
        raw = np.zeros((6, 200), dtype=np.float32)
        for ch in range(6):
            center = 20 + ch * 20
            raw[ch, center] = 1.0
        track = Track(
            track_id=0,
            direction="forward",
            points=[
                TrackPoint(ch_idx=0, t_idx=10, time_s=0.0, offset_m=0.0, amp=0.0, score=0.0),
                TrackPoint(ch_idx=1, t_idx=30, time_s=0.0, offset_m=0.0, amp=0.0, score=0.0),
                TrackPoint(ch_idx=2, t_idx=50, time_s=0.0, offset_m=0.0, amp=0.0, score=0.0),
            ],
            total_score=1.0,
            mean_speed_kmh=0.0,
        )
        refined = refine_track_with_raw_window(track, raw, fs=100.0, dx_m=100.0, search_radius=25)
        self.assertEqual([p.t_idx for p in refined.points], [20, 40, 60])

    def test_decode_vehicle_set_tracks_can_limit_to_single_track(self) -> None:
        raw = np.zeros((20, 2000), dtype=np.float32)
        for ch in range(20):
            t1 = 60 + ch * 40
            t2 = 900 + ch * 35
            raw[ch, t1] = 1.0
            raw[ch, t2] = 0.9
        outputs = {
            "objectness_logits": torch.tensor([[5.0, 4.5]], dtype=torch.float32),
            "direction_logits": torch.tensor([[[6.0, -6.0], [6.0, -6.0]]], dtype=torch.float32),
            "speed": torch.tensor([[0.55, 0.56]], dtype=torch.float32),
            "visibility_logits": torch.full((1, 2, 20), 8.0, dtype=torch.float32),
            "time_logits": torch.linspace(0.0, 1.0, 20, dtype=torch.float32).view(1, 1, 20).repeat(1, 2, 1),
        }
        tracks_all = decode_vehicle_set_tracks(
            {k: v[0] for k, v in outputs.items()},
            raw_time_bins=2000,
            fs=100.0,
            dx_m=100.0,
            raw_window=raw,
            config=VehicleSetModelConfig(n_channels=20, num_queries=2, max_output_tracks=None),
        )
        tracks_top1 = decode_vehicle_set_tracks(
            {k: v[0] for k, v in outputs.items()},
            raw_time_bins=2000,
            fs=100.0,
            dx_m=100.0,
            raw_window=raw,
            config=VehicleSetModelConfig(n_channels=20, num_queries=2, max_output_tracks=1),
        )
        self.assertGreaterEqual(len(tracks_all), 1)
        self.assertEqual(len(tracks_top1), 1)

    def test_single_vehicle_focus_net_forward_loss_and_decode(self) -> None:
        model = SingleVehicleFocusNet(FocusModelConfig(n_channels=12, hidden_dim=32, pooled_channels=8))
        x = torch.randn(2, 1, 12, 96)
        outputs = model(x)
        self.assertEqual(tuple(outputs["target_mask_logits"].shape), (2, 12, 96))
        self.assertEqual(tuple(outputs["competitor_mask_logits"].shape), (2, 12, 96))
        self.assertEqual(tuple(outputs["visibility_logits"].shape), (2, 12))

        time = torch.linspace(0.1, 0.9, 12)
        vis = torch.ones(12)
        target_mask, competitor_mask, line, traj = build_focus_targets(
            time,
            vis,
            n_channels=12,
            time_bins=96,
            competitor_time=time,
            competitor_visibility=torch.zeros(12),
        )
        loss, metrics = single_vehicle_focus_loss(
            {k: v[0] for k, v in outputs.items()},
            target_mask=target_mask,
            target_visibility=vis,
            target_time=time,
            target_objectness=torch.tensor(1.0),
            target_direction=torch.tensor(0),
            target_speed=torch.tensor(0.5),
            competitor_mask=competitor_mask,
            target_line=line,
            target_trajectory=traj,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_target_bce", metrics)

        raw = np.zeros((12, 480), dtype=np.float32)
        for ch in range(12):
            raw[ch, 30 + ch * 20] = 1.0
        tracks = predict_single_vehicle_focus_track(
            model,
            raw,
            fs=100.0,
            dx_m=100.0,
            direction="forward",
            vmin_kmh=60.0,
            vmax_kmh=100.0,
        )
        self.assertIsInstance(tracks, list)

    def test_focus_targets_and_loss_are_finite(self) -> None:
        time = torch.linspace(0.0, 1.0, 10)
        vis = torch.ones(10)
        target_mask, competitor_mask, line, traj = build_focus_targets(
            time,
            vis,
            n_channels=10,
            time_bins=80,
            competitor_time=time,
            competitor_visibility=torch.zeros(10),
        )
        outputs = {
            "target_mask_logits": torch.zeros(10, 80),
            "competitor_mask_logits": torch.zeros(10, 80),
            "target_prior_logits": torch.zeros(10, 80),
            "visibility_logits": torch.zeros(10),
            "trajectory_time": torch.zeros(10),
            "objectness_logits": torch.zeros(()),
            "direction_logits": torch.zeros(2),
            "speed": torch.zeros(()),
            "line_endpoints": torch.zeros(2),
        }
        loss, metrics = single_vehicle_focus_loss(
            outputs,
            target_mask=target_mask,
            target_visibility=vis,
            target_time=time,
            target_objectness=torch.tensor(1.0),
            target_direction=torch.tensor(0),
            target_speed=torch.tensor(0.5),
            competitor_mask=competitor_mask,
            target_line=line,
            target_trajectory=traj,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(metrics["loss_total"]))


if __name__ == "__main__":
    unittest.main()
