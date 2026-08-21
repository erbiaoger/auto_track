import tempfile
import unittest
from pathlib import Path

import torch

from autotrack.dl.build_single_vehicle_benchmark_from_track_slot import main as build_benchmark_main


class SingleVehicleBenchmarkFromTrackSlotTest(unittest.TestCase):
    def test_builds_single_vehicle_benchmark_from_track_slot_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_dir = tmp / "track_slot"
            data_dir.mkdir(parents=True, exist_ok=True)
            shard = {
                "x": torch.zeros((2, 1, 3, 8), dtype=torch.float16),
                "time": torch.tensor(
                    [
                        [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
                        [[0.4, 0.5, 0.6], [0.5, 0.6, 0.7], [0.6, 0.7, 0.8]],
                    ],
                    dtype=torch.float32,
                ),
                "visibility": torch.tensor(
                    [
                        [[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
                        [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ],
                    dtype=torch.float32,
                ),
                "direction": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
                "speed": torch.tensor([[0.5, 0.7], [0.6, 0.8]], dtype=torch.float32),
                "gt_valid": torch.tensor([[True, True, False], [False, True, True]], dtype=torch.bool),
            }
            torch.save(shard, str(data_dir / "shard_000000.pt"))
            (data_dir / "meta.json").write_text(
                '{"window_seconds": 8.0, "dx_m": 100.0, "speed_norm_kmh": 150.0}',
                encoding="utf-8",
            )
            out_file = tmp / "bench.pt"

            import sys

            old_argv = sys.argv
            try:
                sys.argv = [
                    "build_single_vehicle_benchmark_from_track_slot",
                    "--out-file",
                    str(out_file),
                    "--data-dir",
                    str(data_dir),
                    "--slot-policy",
                    "most_visible",
                ]
                self.assertEqual(build_benchmark_main(), 0)
            finally:
                sys.argv = old_argv

            payload = torch.load(str(out_file), map_location="cpu", weights_only=False)
            self.assertEqual(str(payload.get("format")), "single_vehicle_benchmark_v1")
            self.assertEqual(int(payload.get("length", 0)), 2)
            self.assertEqual(tuple(payload["samples"][0]["x"].shape), (1, 3, 8))
            self.assertEqual(tuple(payload["samples"][0]["target"]["raw_window"].shape), (3, 8))
            self.assertEqual(tuple(payload["samples"][0]["target"]["time"].shape), (1, 3))
            self.assertEqual(tuple(payload["samples"][0]["target"]["visibility"].shape), (1, 3))


if __name__ == "__main__":
    unittest.main()
