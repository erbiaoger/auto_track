import unittest

import torch
import numpy as np

from autotrack.dl.single_vehicle_net import (
    ModelConfig,
    SingleVehiclePeakNet,
    build_single_vehicle_heatmap_target,
    build_single_vehicle_line_target,
    build_single_vehicle_trajectory_target,
    estimate_direction_index_from_outputs,
    single_vehicle_heatmap_loss,
    single_vehicle_supervised_loss,
    predict_single_vehicle_track,
)


class SingleVehicleNetTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        model = SingleVehiclePeakNet(ModelConfig(in_channels=1, hidden_dim=64))
        x = torch.randn(2, 1, 50, 128)
        outputs = model(x)
        self.assertEqual(tuple(outputs["heatmap_logits"].shape), (2, 50, 128))
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2,))
        self.assertEqual(tuple(outputs["direction_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["speed"].shape), (2,))
        self.assertEqual(tuple(outputs["line_endpoints"].shape), (2, 2))
        self.assertEqual(tuple(outputs["trajectory_time"].shape), (2, 50))

    def test_heatmap_target_and_loss(self) -> None:
        time = torch.zeros((50,), dtype=torch.float32)
        visibility = torch.zeros((50,), dtype=torch.float32)
        visibility[10:15] = 1.0
        time[10:15] = torch.linspace(0.1, 0.3, 5)
        target = build_single_vehicle_heatmap_target(time, visibility, n_channels=50, time_bins=128)
        self.assertEqual(tuple(target.shape), (50, 128))
        model = SingleVehiclePeakNet(ModelConfig(in_channels=1, hidden_dim=64))
        outputs = model(torch.randn(1, 1, 50, 128))
        loss = single_vehicle_heatmap_loss(outputs, target.unsqueeze(0))
        self.assertTrue(torch.isfinite(loss))

    def test_line_target(self) -> None:
        time = torch.linspace(0.1, 0.9, 50)
        visibility = torch.zeros((50,), dtype=torch.float32)
        visibility[10:30] = 1.0
        line = build_single_vehicle_line_target(time, visibility)
        self.assertEqual(tuple(line.shape), (2,))
        self.assertTrue(torch.isfinite(line).all())

    def test_trajectory_target(self) -> None:
        time = torch.linspace(0.1, 0.9, 50)
        visibility = torch.zeros((50,), dtype=torch.float32)
        visibility[10:30] = 1.0
        traj = build_single_vehicle_trajectory_target(time, visibility)
        self.assertEqual(tuple(traj.shape), (50,))
        self.assertTrue(torch.isfinite(traj).all())

    def test_supervised_loss_uses_competitor_and_time_consistency(self) -> None:
        outputs = {
            "heatmap_logits": torch.zeros((1, 50, 64), dtype=torch.float32),
            "objectness_logits": torch.zeros((1,), dtype=torch.float32),
            "direction_logits": torch.zeros((1, 2), dtype=torch.float32),
            "speed": torch.zeros((1,), dtype=torch.float32),
        }
        time = torch.zeros((50,), dtype=torch.float32)
        visibility = torch.zeros((50,), dtype=torch.float32)
        visibility[5:12] = 1.0
        time[5:12] = torch.linspace(0.1, 0.4, 7)
        target_heatmap = build_single_vehicle_heatmap_target(time, visibility, n_channels=50, time_bins=64)
        target_line = build_single_vehicle_line_target(time, visibility)
        target_traj = build_single_vehicle_trajectory_target(time, visibility)
        competitor_time = torch.zeros((50,), dtype=torch.float32)
        competitor_visibility = torch.zeros((50,), dtype=torch.float32)
        competitor_visibility[7:14] = 1.0
        competitor_time[7:14] = torch.linspace(0.12, 0.42, 7)
        competitor_heatmap = build_single_vehicle_heatmap_target(competitor_time, competitor_visibility, n_channels=50, time_bins=64)
        loss, metrics = single_vehicle_supervised_loss(
            outputs,
            target_heatmap=target_heatmap.unsqueeze(0),
            target_competitor_heatmap=competitor_heatmap.unsqueeze(0),
            target_line=target_line.unsqueeze(0),
            target_trajectory=target_traj.unsqueeze(0),
            target_visibility=visibility.unsqueeze(0),
            target_time=time.unsqueeze(0),
            target_objectness=torch.ones((1,), dtype=torch.float32),
            target_direction=torch.zeros((1,), dtype=torch.long),
            target_speed=torch.ones((1,), dtype=torch.float32),
            competitor_weight=0.75,
            time_consistency_weight=0.5,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_competitor", metrics)
        self.assertIn("loss_time_consistency", metrics)
        self.assertIn("loss_line", metrics)
        self.assertIn("loss_trajectory", metrics)
        self.assertIn("loss_trajectory_smooth", metrics)
        self.assertGreater(float(metrics["loss_competitor"].item()), 0.0)

    def test_auto_direction_decoder_recovers_track_when_head_is_wrong(self) -> None:
        class DummyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1))
                self.has_trained_line_head = False
                self.has_trained_trajectory_head = False

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                batch = int(x.shape[0])
                device = x.device
                return {
                    "heatmap_logits": torch.zeros((batch, 50, 64), device=device),
                    "objectness_logits": torch.zeros((batch,), device=device),
                    "direction_logits": torch.tensor([[0.0, 8.0]], device=device).repeat(batch, 1),
                    "speed": torch.full((batch,), 0.55, device=device),
                }

        fs = 100.0
        dx_m = 10.0
        data = np.zeros((50, 6000), dtype=np.float32)
        for ch in range(50):
            t = 500 + 42 * ch
            data[ch, t] = 5.0
        tracks = predict_single_vehicle_track(
            DummyModel(),
            data,
            fs,
            dx_m,
            "auto",
            60.0,
            100.0,
            {
                "time_downsample": 1,
                "min_visible_channels": 3,
                "prior_weight": 0.0,
                "single_vehicle_tracker": {
                    "candidate_prominence": 0.08,
                    "candidate_min_distance": 12,
                    "candidate_max_peaks_per_channel": 32,
                    "max_skip_channels": 8,
                    "min_track_channels": 24,
                    "min_track_score": 10.0,
                    "kalman_bridge_gap_channels": 8,
                },
            },
            device="cpu",
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0].direction, "forward")
        self.assertGreaterEqual(len(tracks[0].points), 40)

    def test_direction_estimator_uses_trajectory_slope(self) -> None:
        outputs = {
            "trajectory_time": torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32),
            "line_endpoints": torch.tensor([[0.1, 0.9]], dtype=torch.float32),
            "direction_logits": torch.tensor([[9.0, -1.0]], dtype=torch.float32),
        }
        self.assertEqual(estimate_direction_index_from_outputs(outputs), 0)

        outputs["trajectory_time"] = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
        self.assertEqual(estimate_direction_index_from_outputs(outputs), 1)


if __name__ == "__main__":
    unittest.main()
