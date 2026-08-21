import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autotrack.dl.train_single_vehicle_real_vehicle import main as train_real_vehicle_main


class SingleVehicleRealVehicleTrainWrapperTest(unittest.TestCase):
    def test_wrapper_calls_build_merge_and_train_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            with mock.patch("autotrack.dl.train_single_vehicle_real_vehicle.build_realistic_benchmark_main", return_value=0) as build_clean, \
                mock.patch("autotrack.dl.train_single_vehicle_real_vehicle.build_trackslot_benchmark_main", return_value=0) as build_trackslot, \
                mock.patch("autotrack.dl.train_single_vehicle_real_vehicle.merge_benchmarks_main", return_value=0) as merge_bench, \
                mock.patch("autotrack.dl.train_single_vehicle_real_vehicle.train_single_vehicle_main", return_value=0) as train_main:
                rc = train_real_vehicle_main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--clean-samples",
                        "1",
                        "--realbg-samples",
                        "1",
                        "--trackslot-samples",
                        "1",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "1",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(build_clean.call_count, 2)
            build_trackslot.assert_called_once()
            merge_bench.assert_called_once()
            train_main.assert_called_once()
            forwarded = train_main.call_args.args[0]
            self.assertIn("--benchmark-file", forwarded)
            self.assertIn("--out-dir", forwarded)


if __name__ == "__main__":
    unittest.main()
