import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from autotrack.dl.simple_vehicle_peak_dataset import SimpleLinearVehiclePeakDataset, SimplePeakSetDatasetConfig
from autotrack.dl.simple_vehicle_peak_dataset import peakset_collate
from autotrack.dl.vehicle_peak_set_transformer import (
    PeakGuidedVehicleSetTransformer,
    PeakSetInferenceConfig,
    PeakSetModelConfig,
    decode_peak_guided_vehicle_tracks,
    peak_guided_set_loss,
)


class _DummyPeakGuidedModel(nn.Module):
    def __init__(self, n_channels: int, peak_candidates: int = 16):
        super().__init__()
        self.config = SimpleNamespace(n_channels=int(n_channels), in_channels=1, peak_candidates=int(peak_candidates))
        self._param = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        peak_time: torch.Tensor | None = None,
        peak_amp: torch.Tensor | None = None,
        peak_valid: torch.Tensor | None = None,
        peak_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        device = x.device
        n_channels = int(self.config.n_channels)
        k_count = int(peak_time.shape[-1] if peak_time is not None else int(self.config.peak_candidates))
        complete_time = torch.linspace(30.0, 30.0 + (n_channels - 1) * 20.0, n_channels, device=device, dtype=torch.float32)
        complete_time = (complete_time / 599.0).view(1, 1, -1).repeat(batch, 1, 1)
        anchor = torch.full((batch, 1, n_channels, k_count + 1), -8.0, dtype=torch.float32, device=device)
        anchor[..., :k_count] = 8.0
        if peak_valid is not None:
            anchor[..., :k_count] = anchor[..., :k_count].masked_fill(~peak_valid[:, None, :, :].to(device=device), -8.0)
            anchor[..., -1] = -2.0
        return {
            "num_regular_queries": 1,
            "objectness_logits": torch.full((batch, 1), 8.0, dtype=torch.float32, device=device),
            "direction_logits": torch.tensor([[[8.0, -8.0]]], dtype=torch.float32, device=device).repeat(batch, 1, 1),
            "speed": torch.full((batch, 1), 0.52, dtype=torch.float32, device=device),
            "complete_time": complete_time,
            "complete_valid_logits": torch.full((batch, 1, n_channels), 7.0, dtype=torch.float32, device=device),
            "anchor_peak_logits": anchor,
            "peak_time": peak_time if peak_time is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.float32, device=device),
            "peak_amp": peak_amp if peak_amp is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.float32, device=device),
            "peak_valid": peak_valid if peak_valid is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.bool, device=device),
            "peak_index": peak_index if peak_index is not None else torch.full((batch, n_channels, k_count), -1, dtype=torch.long, device=device),
        }


class _FusionBiasPeakGuidedModel(nn.Module):
    def __init__(self, n_channels: int, peak_candidates: int = 16):
        super().__init__()
        self.config = SimpleNamespace(n_channels=int(n_channels), in_channels=1, peak_candidates=int(peak_candidates))
        self._param = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        x: torch.Tensor,
        peak_time: torch.Tensor | None = None,
        peak_amp: torch.Tensor | None = None,
        peak_valid: torch.Tensor | None = None,
        peak_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch = int(x.shape[0])
        device = x.device
        n_channels = int(self.config.n_channels)
        k_count = int(peak_time.shape[-1] if peak_time is not None else int(self.config.peak_candidates))
        complete_time = torch.linspace(30.0, 30.0 + (n_channels - 1) * 20.0, n_channels, device=device, dtype=torch.float32)
        complete_time = (complete_time / 599.0).view(1, 1, -1).repeat(batch, 1, 1)
        anchor = torch.full((batch, 1, n_channels, k_count + 1), -8.0, dtype=torch.float32, device=device)
        if k_count >= 2:
            anchor[..., 0] = 1.0
            anchor[..., 1] = 2.0
        else:
            anchor[..., 0] = 2.0
        if peak_valid is not None:
            anchor[..., :k_count] = anchor[..., :k_count].masked_fill(~peak_valid[:, None, :, :].to(device=device), -8.0)
            anchor[..., -1] = -2.0
        return {
            "num_regular_queries": 1,
            "objectness_logits": torch.full((batch, 1), 8.0, dtype=torch.float32, device=device),
            "direction_logits": torch.tensor([[[8.0, -8.0]]], dtype=torch.float32, device=device).repeat(batch, 1, 1),
            "speed": torch.full((batch, 1), 0.52, dtype=torch.float32, device=device),
            "complete_time": complete_time,
            "complete_valid_logits": torch.full((batch, 1, n_channels), 7.0, dtype=torch.float32, device=device),
            "anchor_peak_logits": anchor,
            "peak_time": peak_time if peak_time is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.float32, device=device),
            "peak_amp": peak_amp if peak_amp is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.float32, device=device),
            "peak_valid": peak_valid if peak_valid is not None else torch.zeros((batch, n_channels, k_count), dtype=torch.bool, device=device),
            "peak_index": peak_index if peak_index is not None else torch.full((batch, n_channels, k_count), -1, dtype=torch.long, device=device),
        }


class VehiclePeakSetDatasetTest(unittest.TestCase):
    def test_dataset_shapes_and_missing_patterns(self) -> None:
        ds = SimpleLinearVehiclePeakDataset(
            config=SimplePeakSetDatasetConfig(
                length=32,
                seed=7,
                return_raw_window=True,
                vehicles_min=4,
                vehicles_max=8,
                speed_min_kmh=60.0,
                speed_max_kmh=90.0,
                missing_random_ratio_min=0.05,
                missing_random_ratio_max=0.20,
                missing_segment_count_max=2,
                missing_segment_min_len=2,
                missing_segment_max_len=6,
            )
        )
        saw_gap = False
        for idx in range(len(ds)):
            _, target = ds[idx]
            self.assertEqual(target["full_time"].shape[1], 50)
            self.assertEqual(tuple(target["full_valid"].shape), tuple(target["observed_visibility"].shape))
            self.assertTrue(torch.all(target["observed_visibility"] <= target["full_valid"] + 1e-6))
            speeds = target["speed"] * 150.0
            self.assertTrue(torch.all(speeds >= 60.0 - 1e-4))
            self.assertTrue(torch.all(speeds <= 90.0 + 1e-4))
            for row in target["observed_visibility"]:
                vis = torch.where(row > 0.5)[0]
                if int(vis.numel()) >= 3:
                    diffs = torch.diff(vis)
                    if bool((diffs > 1).any().item()):
                        saw_gap = True
                        break
        self.assertTrue(saw_gap)


class VehiclePeakSetModelTest(unittest.TestCase):
    def test_forward_and_loss_are_finite(self) -> None:
        model = PeakGuidedVehicleSetTransformer(
            PeakSetModelConfig(
                n_channels=50,
                in_channels=1,
                max_queries=8,
                hidden_dim=32,
                num_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                pooled_time=24,
                peak_candidates=16,
            )
        )
        x = torch.randn(2, 1, 50, 600, dtype=torch.float32)
        ds = SimpleLinearVehiclePeakDataset(
            config=SimplePeakSetDatasetConfig(length=2, seed=11, return_raw_window=True, vehicles_min=4, vehicles_max=6)
        )
        batch = [ds[0], ds[1]]
        xs, targets = peakset_collate(batch)
        outputs = model(
            xs,
            peak_time=targets["peak_time"],
            peak_amp=targets["peak_amp"],
            peak_valid=targets["peak_valid"],
            peak_index=targets["peak_index"],
        )
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2, 8))
        self.assertEqual(tuple(outputs["complete_time"].shape), (2, 8, 50))
        self.assertEqual(tuple(outputs["anchor_peak_logits"].shape[:3]), (2, 8, 50))
        loss, metrics = peak_guided_set_loss(outputs, targets, matcher="greedy")
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss", metrics)
        loss.backward()

    def test_peak_guided_loss_penalizes_anchor_time_inertia(self) -> None:
        model = PeakGuidedVehicleSetTransformer(
            PeakSetModelConfig(
                n_channels=50,
                in_channels=1,
                max_queries=8,
                hidden_dim=32,
                num_heads=4,
                encoder_layers=1,
                decoder_layers=1,
                pooled_time=24,
                peak_candidates=16,
            )
        )
        ds = SimpleLinearVehiclePeakDataset(
            config=SimplePeakSetDatasetConfig(length=1, seed=21, return_raw_window=True, vehicles_min=4, vehicles_max=6)
        )
        xs, targets = peakset_collate([ds[0]])
        peak_time = targets["peak_time"].clone()
        peak_valid = targets["peak_valid"].clone()
        peak_index = targets["peak_index"].clone()
        gt_peak_index = targets["gt_peak_index"].clone()
        n_channels = int(peak_time.shape[-2])
        base = torch.linspace(0.10, 0.90, n_channels)
        if int(peak_time.shape[-1]) < 2:
            self.fail("expected at least 2 peak candidates")
        peak_time[0, :, 0] = base
        peak_time[0, :, 1] = torch.clamp(base + 0.25, 0.0, 1.0)
        peak_valid[0, :, :] = False
        peak_valid[0, :, 0] = True
        peak_valid[0, :, 1] = True
        peak_index[0, :, 0] = torch.arange(n_channels, dtype=torch.long)
        peak_index[0, :, 1] = torch.arange(n_channels, dtype=torch.long)
        gt_peak_index[0, 0, :] = 0
        targets["peak_time"] = peak_time
        targets["peak_valid"] = peak_valid
        targets["peak_index"] = peak_index
        targets["gt_peak_index"] = gt_peak_index

        outputs = model(
            xs,
            peak_time=targets["peak_time"],
            peak_amp=targets["peak_amp"],
            peak_valid=targets["peak_valid"],
            peak_index=targets["peak_index"],
        )
        smooth_outputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in outputs.items()}
        jump_outputs = {key: value.clone() if torch.is_tensor(value) else value for key, value in outputs.items()}
        smooth_logits = torch.full_like(outputs["anchor_peak_logits"], -6.0)
        smooth_logits[..., 0] = 6.0
        jump_logits = smooth_logits.clone()
        for ch in range(n_channels):
            if ch % 2 == 1:
                jump_logits[:, :, ch, 0] = -6.0
                jump_logits[:, :, ch, 1] = 6.0
        smooth_outputs["anchor_peak_logits"] = smooth_logits
        jump_outputs["anchor_peak_logits"] = jump_logits

        smooth_loss, smooth_metrics = peak_guided_set_loss(smooth_outputs, targets, matcher="greedy", inertia_weight=1.0)
        jump_loss, jump_metrics = peak_guided_set_loss(jump_outputs, targets, matcher="greedy", inertia_weight=1.0)
        self.assertTrue(torch.isfinite(smooth_loss))
        self.assertTrue(torch.isfinite(jump_loss))
        self.assertIn("loss_inertia", smooth_metrics)
        self.assertLess(smooth_metrics["loss_inertia"], jump_metrics["loss_inertia"])

    def test_decode_peak_guided_vehicle_tracks_returns_complete_track(self) -> None:
        raw = np.zeros((12, 600), dtype=np.float32)
        for ch in range(12):
            if ch in {3, 6}:
                continue
            raw[ch, 40 + ch * 40] = 1.0
        model = _DummyPeakGuidedModel(n_channels=12)
        tracks = decode_peak_guided_vehicle_tracks(
            model,
            raw,
            fs=100.0,
            x_axis_m=np.arange(12, dtype=np.float32) * 20.0,
            config=PeakSetInferenceConfig(
                time_downsample=1,
                objectness_threshold=0.2,
                complete_valid_threshold=0.3,
                min_visible_channels=5,
                min_anchor_support_channels=3,
                min_anchor_support_ratio=0.2,
                refine_radius_samples=30,
                max_tracks=2,
                dedup_tolerance_samples=8,
            ),
            device="cpu",
        )
        self.assertGreaterEqual(len(tracks), 1)
        self.assertGreaterEqual(len(tracks[0].track.points), 5)
        self.assertTrue(bool((~tracks[0].observed_valid.astype(bool)).any()))

    def test_decode_peak_guided_vehicle_tracks_fuses_model_and_peak_evidence(self) -> None:
        raw = np.zeros((12, 600), dtype=np.float32)
        expected_times = []
        for ch in range(12):
            early = 30 + ch * 20
            late = 220 + ch * 20
            expected_times.append(early)
            raw[ch, early] = 5.0
            raw[ch, late] = 3.0
        model = _FusionBiasPeakGuidedModel(n_channels=12)
        tracks = decode_peak_guided_vehicle_tracks(
            model,
            raw,
            fs=100.0,
            x_axis_m=np.arange(12, dtype=np.float32) * 20.0,
            config=PeakSetInferenceConfig(
                time_downsample=1,
                objectness_threshold=0.2,
                complete_valid_threshold=0.3,
                min_visible_channels=5,
                min_anchor_support_channels=3,
                min_anchor_support_ratio=0.2,
                fused_min_anchor_score=0.35,
                refine_radius_samples=0,
                max_tracks=2,
                dedup_tolerance_samples=8,
                graph_refine=False,
            ),
            device="cpu",
        )
        self.assertGreaterEqual(len(tracks), 1)
        got = [int(point.t_idx) for point in tracks[0].track.points]
        self.assertEqual(len(got), 12)
        for ch, t_idx in enumerate(got):
            self.assertLessEqual(abs(int(t_idx) - int(expected_times[ch])), 2)
        self.assertGreaterEqual(int(np.sum(tracks[0].observed_valid > 0.5)), 8)


if __name__ == "__main__":
    unittest.main()
