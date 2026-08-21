import tempfile
import unittest
from pathlib import Path

import torch

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.plot_single_vehicle_benchmark_compare import main as plot_compare_main
from autotrack.dl.single_vehicle_net import ModelConfig, SingleVehiclePeakNet, save_checkpoint


class SingleVehiclePlotCompareTest(unittest.TestCase):
    def test_writes_compare_plot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ds = OnlineSyntheticTrajectoryDataset(
                length=1,
                n_channels=50,
                fs=100.0,
                window_seconds=5.0,
                time_downsample=2,
                dx_m=10.0,
                vehicles_min=1,
                vehicles_max=1,
                speed_min_kmh=60.0,
                speed_max_kmh=80.0,
                speed_outlier_ratio=0.0,
                slow_speed_min_kmh=60.0,
                slow_speed_max_kmh=80.0,
                fast_speed_min_kmh=60.0,
                fast_speed_max_kmh=80.0,
                noise_std=0.0,
                amp_min=1.0,
                amp_max=1.0,
                sigma_min_s=0.05,
                sigma_max_s=0.05,
                primary_ratio=1.0,
                min_visible_channels=3,
                speed_norm_kmh=150.0,
                clip_ratio=1.35,
                input_mode="raw",
                seed=10,
                cache_dataset=False,
                return_raw_window=True,
                artifact_competing_ratio=1.0,
                artifact_competing_speed_ratio_min=0.95,
                artifact_competing_speed_ratio_max=1.05,
                artifact_competing_channel_offset_max=1,
                artifact_competing_opposite_direction_ratio=0.0,
            )
            x, target = ds[0]
            benchmark = {
                "format": "single_vehicle_benchmark_v1",
                "meta": {"fs": 100.0, "dx_m": 10.0, "time_downsample": 2, "speed_norm_kmh": 150.0},
                "length": 1,
                "samples": [{"x": x.cpu(), "target": {key: value.cpu() for key, value in target.items()}}],
            }
            benchmark_path = tmp / "bench.pt"
            torch.save(benchmark, benchmark_path)

            model = SingleVehiclePeakNet(ModelConfig(n_channels=50, in_channels=1, hidden_dim=16))
            ckpt_path = tmp / "model.pt"
            save_checkpoint(ckpt_path, model, None, ModelConfig(n_channels=50, in_channels=1, hidden_dim=16), {})

            out_file = tmp / "compare.png"
            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "plot_single_vehicle_benchmark_compare",
                    "--benchmark-file",
                    str(benchmark_path),
                    "--model",
                    str(ckpt_path),
                    "--out-file",
                    str(out_file),
                    "--sample-index",
                    "0",
                    "--device",
                    "cpu",
                ]
                self.assertEqual(plot_compare_main(), 0)
            finally:
                sys.argv = old_argv

            self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()
