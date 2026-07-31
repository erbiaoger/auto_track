import tempfile
import unittest
from pathlib import Path
from unittest import mock

from autotrack.dl.run_single_vehicle_real_vehicle import main as run_real_vehicle_main


class SingleVehicleRealVehicleRunWrapperTest(unittest.TestCase):
    def test_wrapper_runs_train_then_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            with mock.patch("autotrack.dl.run_single_vehicle_real_vehicle.train_real_vehicle_main", return_value=0) as train_main, \
                mock.patch("autotrack.dl.run_single_vehicle_real_vehicle.predict_real_vehicle_main", return_value=0) as predict_main:
                rc = run_real_vehicle_main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--device",
                        "cpu",
                        "--epochs",
                        "1",
                        "--batch-size",
                        "1",
                        "--smoke-windows",
                        "3",
                    ]
                )
            self.assertEqual(rc, 0)
            train_main.assert_called_once()
            predict_main.assert_called_once()
            forwarded = predict_main.call_args.args[0]
            self.assertIn("--preset", forwarded)
            self.assertIn("real_vehicle", forwarded)

    def test_quick_mode_reduces_workload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "run"
            with mock.patch("autotrack.dl.run_single_vehicle_real_vehicle.train_real_vehicle_main", return_value=0) as train_main, \
                mock.patch("autotrack.dl.run_single_vehicle_real_vehicle.predict_real_vehicle_main", return_value=0) as predict_main:
                rc = run_real_vehicle_main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--quick",
                        "--device",
                        "cpu",
                    ]
                )
            self.assertEqual(rc, 0)
            train_args = train_main.call_args.args[0]
            self.assertIn("--epochs", train_args)
            self.assertIn("1", train_args)
            self.assertIn("--batch-size", train_args)
            self.assertIn("1", train_args)
            self.assertIn("--clean-samples", train_args)
            self.assertIn("8", train_args)
            self.assertIn("--realbg-samples", train_args)
            self.assertIn("4", train_args)
            self.assertIn("--trackslot-samples", train_args)
            self.assertIn("8", train_args)
            smoke_args = predict_main.call_args.args[0]
            self.assertIn("--max-windows", smoke_args)


if __name__ == "__main__":
    unittest.main()
