import unittest

import numpy as np
import torch

from autotrack.dl.trajectory_energy_model import (
    ModelConfig,
    InferenceConfig,
    TrajectoryEnergyNet,
    build_energy_target,
    extract_vehicle_tracks_from_energy,
    trajectory_energy_loss,
)


class TrajectoryEnergyModelTest(unittest.TestCase):
    def test_forward_and_loss_shapes(self) -> None:
        model = TrajectoryEnergyNet(ModelConfig(n_channels=12, base_dim=16, hidden_dim=32))
        x = torch.randn(2, 1, 12, 80)
        outputs = model(x)
        self.assertEqual(tuple(outputs["energy_logits"].shape), (2, 12, 80))
        self.assertEqual(tuple(outputs["visibility_logits"].shape), (2, 12))
        self.assertEqual(tuple(outputs["direction_logits"].shape), (2, 2))
        self.assertEqual(tuple(outputs["speed"].shape), (2,))

        target_time = torch.linspace(0.0, 1.0, 12).unsqueeze(0).repeat(2, 1)
        target_vis = torch.ones((2, 12), dtype=torch.float32)
        target_energy = torch.stack(
            [
                build_energy_target(target_time[i], target_vis[i], n_channels=12, time_bins=80)
                for i in range(2)
            ],
            dim=0,
        )
        loss, items = trajectory_energy_loss(
            outputs,
            target_energy=target_energy,
            target_visibility=target_vis,
            target_time=target_time,
            target_direction=torch.tensor([0, 1]),
            target_speed=torch.tensor([0.5, 0.6]),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_energy", items)

    def test_extracts_two_tracks_from_energy_map(self) -> None:
        n_channels = 20
        n_samples = 120000
        energy = np.zeros((n_channels, n_samples), dtype=np.float32)
        for ch in range(n_channels):
            for center, amp in [(2000 + 4500 * ch, 1.0), (10000 + 5000 * ch, 0.9)]:
                idx = np.arange(max(0, center - 3), min(n_samples, center + 4), dtype=np.float32)
                pulse = amp * np.exp(-0.5 * ((idx - float(center)) / 1.0) ** 2)
                energy[ch, max(0, center - 3) : min(n_samples, center + 4)] += pulse.astype(np.float32)

        tracks = extract_vehicle_tracks_from_energy(
            energy,
            fs=1000.0,
            dx_m=100.0,
            direction="forward",
            vmin_kmh=70.0,
            vmax_kmh=90.0,
            config={
                "time_downsample": 1,
                "seed_threshold": 0.1,
                "suppression_time_radius": 100,
                "suppression_channel_radius": 0,
            },
        )
        self.assertGreaterEqual(len(tracks), 2)
        self.assertTrue(all(len(tr.points) >= 8 for tr in tracks[:2]))

    def test_decoder_profile_selection_defaults_to_strict(self) -> None:
        from autotrack.dl.trajectory_energy_model import _select_decoder_profile

        cfg = InferenceConfig()
        energy = np.zeros((4, 8), dtype=np.float32)
        selected = _select_decoder_profile(cfg, energy)
        self.assertEqual(selected.decoder_profile, "strict")
        self.assertGreaterEqual(selected.min_track_channels, cfg.min_track_channels)


if __name__ == "__main__":
    unittest.main()
