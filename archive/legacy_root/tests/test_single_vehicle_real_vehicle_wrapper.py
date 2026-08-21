import unittest
from unittest import mock

from autotrack.dl.predict_single_vehicle_real_vehicle import main as wrapper_main


class SingleVehicleRealVehicleWrapperTest(unittest.TestCase):
    def test_wrapper_injects_real_vehicle_preset(self) -> None:
        with mock.patch("autotrack.dl.predict_single_vehicle_real_vehicle.predict_main", return_value=0) as mocked:
            rc = wrapper_main(["--model", "m.pt", "--input", "x.npy", "--out-dir", "/tmp/out"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()
        forwarded = mocked.call_args.args[0]
        self.assertIn("--preset", forwarded)
        preset_idx = forwarded.index("--preset")
        self.assertEqual(forwarded[preset_idx + 1], "real_vehicle")


if __name__ == "__main__":
    unittest.main()
