import unittest

import torch

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.vehicle_proposal_net import (
    ProposalModelConfig,
    VehicleProposalNet,
    build_multi_vehicle_heatmap_target,
    build_multi_vehicle_objectness_target,
    vehicle_proposal_loss,
)


class VehicleProposalNetTest(unittest.TestCase):
    def test_forward_and_target_shapes(self) -> None:
        model = VehicleProposalNet(ProposalModelConfig(in_channels=1, hidden_dim=32))
        x = torch.randn(2, 1, 50, 1000)
        outputs = model(x)
        self.assertEqual(tuple(outputs["heatmap_logits"].shape), (2, 50, 1000))
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2,))

    def test_multi_vehicle_union_target(self) -> None:
        ds = OnlineSyntheticTrajectoryDataset(
            length=1,
            n_channels=50,
            fs=1000.0,
            window_seconds=10.0,
            time_downsample=10,
            dx_m=100.0,
            vehicles_min=6,
            vehicles_max=8,
            speed_min_kmh=70.0,
            speed_max_kmh=90.0,
            speed_outlier_ratio=0.0,
            slow_speed_min_kmh=70.0,
            slow_speed_max_kmh=90.0,
            fast_speed_min_kmh=70.0,
            fast_speed_max_kmh=90.0,
            noise_std=0.0,
            amp_min=6.0,
            amp_max=6.0,
            sigma_min_s=0.25,
            sigma_max_s=0.25,
            primary_ratio=0.5,
            min_visible_channels=3,
            speed_norm_kmh=150.0,
            clip_ratio=1.35,
            input_mode="raw",
            seed=7,
            return_raw_window=True,
            artifact_dropout_ratio=0.0,
            artifact_decoy_ratio=0.0,
            artifact_competing_ratio=0.0,
        )
        _, target = ds[0]
        heatmap = build_multi_vehicle_heatmap_target(target["gt_masks"])
        obj = build_multi_vehicle_objectness_target(target["gt_masks"])
        self.assertEqual(tuple(heatmap.shape), (50, 1000))
        self.assertEqual(tuple(obj.shape), ())
        self.assertGreater(float(heatmap.max().item()), 0.0)
        model = VehicleProposalNet(ProposalModelConfig(in_channels=1, hidden_dim=32))
        outputs = model(torch.randn(1, 1, 50, 1000))
        loss, metrics = vehicle_proposal_loss(outputs, target_heatmap=heatmap.unsqueeze(0), target_objectness=obj.unsqueeze(0))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_heatmap", metrics)


if __name__ == "__main__":
    unittest.main()
