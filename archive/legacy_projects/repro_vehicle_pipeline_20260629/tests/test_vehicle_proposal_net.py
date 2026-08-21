import unittest
from pathlib import Path

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
            dx_m=20.0,
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
            background_pt=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/xi_gauss_50_120s_large/test/shard_000000.pt"),
            artifact_dropout_ratio=1.0,
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

    def test_realistic_traffic_scene_stays_in_speed_band(self) -> None:
        ds = OnlineSyntheticTrajectoryDataset(
            length=32,
            n_channels=50,
            fs=1000.0,
            window_seconds=10.0,
            time_downsample=10,
            dx_m=20.0,
            vehicles_min=8,
            vehicles_max=12,
            speed_min_kmh=60.0,
            speed_max_kmh=90.0,
            speed_outlier_ratio=0.0,
            slow_speed_min_kmh=60.0,
            slow_speed_max_kmh=90.0,
            fast_speed_min_kmh=60.0,
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
            seed=11,
            scene_mode="realistic_traffic",
            vehicle_count_profile="mixed_density",
            speed_variation_ratio=0.08,
            same_direction_cluster_ratio=0.35,
            crossing_ratio=0.35,
            parallel_close_ratio=0.20,
            multi_gap_dropout_ratio=0.55,
            return_raw_window=True,
            background_pt=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/xi_gauss_50_120s_large/test/shard_000000.pt"),
            artifact_dropout_ratio=1.0,
            artifact_decoy_ratio=0.0,
            artifact_competing_ratio=0.35,
        )

        saw_gap = False
        max_tracks = 0
        saw_both_directions = False
        for idx in range(len(ds)):
            _, target = ds[idx]
            speeds_kmh = target["speed"] * 150.0
            self.assertTrue(torch.all(speeds_kmh >= 60.0 - 1e-4))
            self.assertTrue(torch.all(speeds_kmh <= 90.0 + 1e-4))
            max_tracks = max(max_tracks, int(target["time"].shape[0]))
            directions = target["direction"]
            saw_both_directions = saw_both_directions or (bool((directions == 0).any().item()) and bool((directions == 1).any().item()))
            for row in target["visibility"]:
                visible_idx = torch.where(row > 0.5)[0]
                if int(visible_idx.numel()) >= 3:
                    diffs = torch.diff(visible_idx)
                    if bool((diffs > 1).any().item()):
                        saw_gap = True
                        break

        self.assertGreater(max_tracks, 8)
        self.assertTrue(saw_gap)
        self.assertTrue(saw_both_directions)


if __name__ == "__main__":
    unittest.main()
