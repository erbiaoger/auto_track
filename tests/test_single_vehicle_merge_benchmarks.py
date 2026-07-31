import tempfile
import unittest
from pathlib import Path

import torch

from autotrack.dl.merge_single_vehicle_benchmarks import main as merge_single_vehicle_benchmarks_main


class SingleVehicleMergeBenchmarksTest(unittest.TestCase):
    def test_merges_multiple_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            a = tmp / "a.pt"
            b = tmp / "b.pt"
            payload = {
                "format": "single_vehicle_benchmark_v1",
                "meta": {"fs": 100.0},
                "length": 1,
                "samples": [{"x": torch.zeros((1, 50, 100), dtype=torch.float32), "target": {}}],
            }
            torch.save(payload, str(a))
            torch.save(payload, str(b))
            out_file = tmp / "merged.pt"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "merge_single_vehicle_benchmarks",
                    "--out-file",
                    str(out_file),
                    "--in-file",
                    str(a),
                    str(b),
                ]
                self.assertEqual(merge_single_vehicle_benchmarks_main(), 0)
            finally:
                sys.argv = old_argv

            merged = torch.load(out_file, map_location="cpu", weights_only=False)
            self.assertEqual(int(merged["length"]), 2)
            self.assertEqual(str(merged["format"]), "single_vehicle_benchmark_v1")


if __name__ == "__main__":
    unittest.main()
