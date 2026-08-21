import unittest

import torch

from autotrack.dl.evaluate_single_vehicle import _BenchmarkDataset


class SingleVehicleBenchmarkTest(unittest.TestCase):
    def test_benchmark_dataset_roundtrip(self) -> None:
        payload = {
            "samples": [
                {
                    "x": torch.zeros((1, 50, 100), dtype=torch.float32),
                    "target": {
                        "time": torch.zeros((1, 50), dtype=torch.float32),
                        "visibility": torch.zeros((1, 50), dtype=torch.float32),
                        "direction": torch.zeros((1,), dtype=torch.long),
                        "speed": torch.zeros((1,), dtype=torch.float32),
                        "raw_window": torch.zeros((50, 1000), dtype=torch.float32),
                    },
                }
            ]
        }
        ds = _BenchmarkDataset(payload)
        self.assertEqual(len(ds), 1)
        x, target = ds[0]
        self.assertEqual(tuple(x.shape), (1, 50, 100))
        self.assertIn("raw_window", target)
        self.assertEqual(tuple(target["raw_window"].shape), (50, 1000))


if __name__ == "__main__":
    unittest.main()
