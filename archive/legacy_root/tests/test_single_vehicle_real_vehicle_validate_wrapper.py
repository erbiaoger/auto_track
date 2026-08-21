import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autotrack.dl.validate_single_vehicle_real_vehicle import main as validate_real_vehicle_main


class SingleVehicleRealVehicleValidateWrapperTest(unittest.TestCase):
    def test_wrapper_calls_evaluation_and_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            with mock.patch("autotrack.dl.validate_single_vehicle_real_vehicle.build_realistic_benchmark_main", return_value=0), \
                mock.patch("autotrack.dl.validate_single_vehicle_real_vehicle.build_trackslot_benchmark_main", return_value=0), \
                mock.patch("autotrack.dl.validate_single_vehicle_real_vehicle.merge_benchmarks_main", return_value=0), \
                mock.patch("autotrack.dl.validate_single_vehicle_real_vehicle.evaluate_single_vehicle_main", return_value=0) as eval_main, \
                mock.patch("autotrack.dl.validate_single_vehicle_real_vehicle.predict_real_vehicle_main", return_value=0) as smoke_main:
                rc = validate_real_vehicle_main(
                    [
                        "--model",
                        "model.pt",
                        "--out-dir",
                        str(out_dir),
                        "--device",
                        "cpu",
                        "--eval-samples",
                        "4",
                        "--smoke-windows",
                        "3",
                    ]
                )
            self.assertEqual(rc, 0)
            eval_main.assert_called_once()
            smoke_main.assert_called_once()
            forwarded = smoke_main.call_args.args[0]
            self.assertIn("--model", forwarded)
            self.assertIn("model.pt", forwarded)


if __name__ == "__main__":
    unittest.main()
