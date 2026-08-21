import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from autotrack.dl.trajectory_set_model import InferenceConfig, predict_tracks_from_window


class _DummyTrajectorySetModel(nn.Module):
    def __init__(self, n_channels: int, n_points: int = 6):
        super().__init__()
        self.config = SimpleNamespace(n_channels=int(n_channels), in_channels=1)
        self._param = nn.Parameter(torch.zeros(()))
        self.n_points = int(n_points)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        device = x.device
        points = torch.zeros((batch, 1, self.n_points, 2), dtype=torch.float32, device=device)
        valid = torch.full((batch, 1, self.n_points), 8.0, dtype=torch.float32, device=device)
        time = torch.linspace(0.08, 0.92, self.n_points, device=device, dtype=torch.float32)
        ch = torch.linspace(0.08, 0.92, self.n_points, device=device, dtype=torch.float32)
        points[:, 0, :, 0] = ch
        points[:, 0, :, 1] = time
        return {
            "objectness_logits": torch.tensor([[6.0]], device=device, dtype=torch.float32),
            "direction_logits": torch.tensor([[[8.0, -8.0]]], device=device, dtype=torch.float32),
            "speed": torch.tensor([[0.58]], device=device, dtype=torch.float32),
            "points": points,
            "point_valid_logits": valid,
            "visibility_logits": torch.full((batch, 1, self.config.n_channels), 8.0, dtype=torch.float32, device=device),
            "time": torch.linspace(0.08, 0.92, self.config.n_channels, device=device, dtype=torch.float32).view(1, 1, -1).repeat(batch, 1, 1),
        }


class TrajectorySetModelSmokeTest(unittest.TestCase):
    def test_predict_tracks_from_window_refines_one_track(self) -> None:
        raw = np.zeros((12, 600), dtype=np.float32)
        for ch in range(12):
            raw[ch, 30 + ch * 45] = 1.0
        model = _DummyTrajectorySetModel(n_channels=12, n_points=6)
        tracks = predict_tracks_from_window(
            model,
            raw,
            fs=100.0,
            x_axis_m=np.arange(12, dtype=np.float32) * 100.0,
            config=InferenceConfig(
                time_downsample=1,
                objectness_threshold=0.2,
                visibility_threshold=0.5,
                min_visible_channels=3,
                refine_radius_samples=40,
                max_tracks=4,
                dedup_tolerance_samples=12,
                graph_refine=True,
            ),
            device="cpu",
        )
        self.assertGreaterEqual(len(tracks), 1)
        self.assertGreaterEqual(len(tracks[0].points), 3)
        self.assertLess(tracks[0].points[0].t_idx, tracks[0].points[-1].t_idx)

    def test_predict_tracks_from_window_preserves_raw_time_scale(self) -> None:
        raw = np.zeros((12, 6000), dtype=np.float32)
        downsampled = raw[:, ::10]
        for ch in range(12):
            raw[ch, 80 + ch * 420] = 1.0
            downsampled[ch, 8 + ch * 42] = 1.0

        model = _DummyTrajectorySetModel(n_channels=12, n_points=6)
        cfg = InferenceConfig(
            time_downsample=10,
            objectness_threshold=0.2,
            visibility_threshold=0.5,
            min_visible_channels=3,
            refine_radius_samples=0,
            max_tracks=4,
            dedup_tolerance_samples=12,
            graph_refine=False,
        )

        raw_tracks = predict_tracks_from_window(
            model,
            raw,
            fs=1000.0,
            x_axis_m=np.arange(12, dtype=np.float32) * 100.0,
            config=cfg,
            device="cpu",
        )
        ds_tracks = predict_tracks_from_window(
            model,
            downsampled,
            fs=1000.0,
            x_axis_m=np.arange(12, dtype=np.float32) * 100.0,
            config=cfg,
            device="cpu",
        )

        self.assertGreaterEqual(len(raw_tracks), 1)
        self.assertGreaterEqual(len(ds_tracks), 1)
        raw_time_max = max(float(p.time_s) for p in raw_tracks[0].points)
        ds_time_max = max(float(p.time_s) for p in ds_tracks[0].points)
        self.assertGreater(raw_time_max, ds_time_max * 8.0)


if __name__ == "__main__":
    unittest.main()
