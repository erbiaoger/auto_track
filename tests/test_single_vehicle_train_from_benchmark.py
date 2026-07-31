import tempfile
import unittest
from pathlib import Path

import torch

from autotrack.dl.train_single_vehicle import main as train_single_vehicle_main


class SingleVehicleTrainFromBenchmarkTest(unittest.TestCase):
    def test_trains_from_fixed_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bench = tmp / "bench.pt"
            payload = {
                "format": "single_vehicle_benchmark_v1",
                "meta": {"fs": 100.0, "dx_m": 10.0},
                "length": 1,
                "samples": [
                    {
                        "x": torch.zeros((1, 50, 100), dtype=torch.float32),
                        "target": {
                            "time": torch.zeros((1, 50), dtype=torch.float32),
                            "visibility": torch.ones((1, 50), dtype=torch.float32),
                            "direction": torch.zeros((1,), dtype=torch.long),
                            "speed": torch.zeros((1,), dtype=torch.float32),
                            "raw_window": torch.zeros((50, 1000), dtype=torch.float32),
                        },
                    }
                ],
            }
            torch.save(payload, str(bench))
            out_dir = tmp / "out"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_single_vehicle",
                    "--out-dir",
                    str(out_dir),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--benchmark-file",
                    str(bench),
                    "--log-every",
                    "0",
                ]
                self.assertEqual(train_single_vehicle_main(), 0)
            finally:
                sys.argv = old_argv

            self.assertTrue((out_dir / "checkpoint_best.pt").is_file())

    def test_can_resume_from_existing_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bench = tmp / "bench.pt"
            payload = {
                "format": "single_vehicle_benchmark_v1",
                "meta": {"fs": 100.0, "dx_m": 10.0},
                "length": 1,
                "samples": [
                    {
                        "x": torch.zeros((1, 50, 100), dtype=torch.float32),
                        "target": {
                            "time": torch.zeros((1, 50), dtype=torch.float32),
                            "visibility": torch.ones((1, 50), dtype=torch.float32),
                            "direction": torch.zeros((1,), dtype=torch.long),
                            "speed": torch.zeros((1,), dtype=torch.float32),
                            "raw_window": torch.zeros((50, 1000), dtype=torch.float32),
                        },
                    }
                ],
            }
            torch.save(payload, str(bench))
            out_dir = tmp / "out"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_single_vehicle",
                    "--out-dir",
                    str(out_dir),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--benchmark-file",
                    str(bench),
                    "--log-every",
                    "0",
                ]
                self.assertEqual(train_single_vehicle_main(), 0)
            finally:
                sys.argv = old_argv

            resumed = tmp / "out_resumed"
            old_argv = sys.argv
            try:
                sys.argv = [
                    "train_single_vehicle",
                    "--out-dir",
                    str(resumed),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--batch-size",
                    "1",
                    "--benchmark-file",
                    str(bench),
                    "--resume",
                    str(out_dir / "checkpoint_best.pt"),
                    "--log-every",
                    "0",
                ]
                self.assertEqual(train_single_vehicle_main(), 0)
            finally:
                sys.argv = old_argv

            self.assertTrue((resumed / "checkpoint_best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
