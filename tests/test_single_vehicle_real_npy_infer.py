import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from autotrack.dl.predict_single_vehicle_real_npy import _apply_preset, _load_window, _window_activity_score, _window_model_score, _window_starts


class SingleVehicleRealNpyInferTest(unittest.TestCase):
    def test_window_helpers(self) -> None:
        arr = np.arange(2000 * 51, dtype=np.float32).reshape(2000, 51)
        starts = _window_starts(2000, 500, 250, 3)
        self.assertEqual(starts[:2], [0, 250])
        window = _load_window(
            arr,
            layout="time_channel",
            channel_start=1,
            channel_count=50,
            start_t=0,
            window_samples=500,
            background_scale=1.0,
        )
        self.assertEqual(tuple(window.shape), (50, 500))
        self.assertAlmostEqual(float(window[0, 0]), float(arr[0, 1]))

    def test_activity_score_separates_blank_and_signal_windows(self) -> None:
        blank = np.zeros((50, 500), dtype=np.float32)
        signal = np.zeros((50, 500), dtype=np.float32)
        signal[10, 200:220] = 5.0
        self.assertLess(_window_activity_score(blank), 0.1)
        self.assertGreater(_window_activity_score(signal), 0.5)

    def test_model_score_prefers_window_with_objectness_and_peak(self) -> None:
        class _StubModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                batch = x.shape[0]
                score = x.sum(dim=(1, 2, 3))
                heatmap_logits = torch.zeros((batch, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device)
                if batch > 0:
                    heatmap_logits[:, 10, 20] = score
                objectness_logits = score
                return {
                    "heatmap_logits": heatmap_logits,
                    "objectness_logits": objectness_logits,
                    "direction_logits": torch.zeros((batch, 2), dtype=torch.float32, device=x.device),
                    "speed": torch.ones((batch,), dtype=torch.float32, device=x.device),
                }

        blank = np.zeros((50, 500), dtype=np.float32)
        signal = np.zeros((50, 500), dtype=np.float32)
        signal[10, 200:220] = 5.0
        model = _StubModel()
        blank_score = _window_model_score(model, blank, device="cpu", time_downsample=1)
        signal_score = _window_model_score(model, signal, device="cpu", time_downsample=1)
        self.assertGreater(signal_score, blank_score)

    def test_real_vehicle_preset_sets_long_window_defaults(self) -> None:
        args = Namespace(
            preset="real_vehicle",
            window_seconds=10.0,
            window_stride_seconds=10.0,
            window_ranking="activity",
            direction_mode="predicted",
            max_windows=8,
            window_activity_threshold=0.5,
        )
        out = _apply_preset(args)
        self.assertEqual(out.window_seconds, 60.0)
        self.assertEqual(out.window_stride_seconds, 10.0)
        self.assertEqual(out.window_ranking, "model")
        self.assertEqual(out.direction_mode, "both")
        self.assertGreaterEqual(out.max_windows, 24)
        self.assertEqual(out.window_activity_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
