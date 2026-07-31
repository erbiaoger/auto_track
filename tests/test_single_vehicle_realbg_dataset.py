import tempfile
import unittest
from pathlib import Path

import numpy as np

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset


class SingleVehicleRealBackgroundDatasetTest(unittest.TestCase):
    def test_real_background_window_and_raw_window_are_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "background.npy"
            background = np.zeros((2000, 51), dtype=np.float32)
            background[100:120, 3] = 1.0
            np.save(path, background)

            ds = OnlineSyntheticTrajectoryDataset(
                length=2,
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
                seed=1,
                cache_dataset=False,
                return_raw_window=True,
                background_npy=path,
                background_layout="time_channel",
                background_channel_start=0,
                background_scale=1.0,
            )

            x, target = ds[0]
            self.assertEqual(tuple(x.shape), (1, 50, 250))
            self.assertIn("raw_window", target)
            self.assertIn("background_meta", target)
            self.assertEqual(tuple(target["raw_window"].shape), (50, 500))
            self.assertEqual(int(target["time"].shape[0]), 1)
            self.assertEqual(int(target["visibility"].shape[0]), 1)

    def test_artifact_modes_inject_dropout_and_decoys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "background.npy"
            background = np.zeros((2000, 51), dtype=np.float32)
            np.save(path, background)

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
                seed=2,
                cache_dataset=False,
                return_raw_window=True,
                background_npy=path,
                background_layout="time_channel",
                background_channel_start=0,
                background_scale=1.0,
                artifact_dropout_ratio=1.0,
                artifact_dropout_min_channels=2,
                artifact_dropout_max_channels=2,
                artifact_decoy_ratio=1.0,
                artifact_decoy_min_points=2,
                artifact_decoy_max_points=2,
                artifact_decoy_amp_scale_min=1.5,
                artifact_decoy_amp_scale_max=1.5,
                artifact_decoy_time_jitter_s=0.0,
            )

            _, target = ds[0]
            self.assertIn("artifact_meta", target)
            self.assertGreaterEqual(float(target["artifact_meta"][0]), 2.0)
            self.assertGreaterEqual(float(target["artifact_meta"][1]), 2.0)

    def test_competing_vehicle_track_is_injected_without_changing_target_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "background.npy"
            background = np.zeros((2000, 51), dtype=np.float32)
            np.save(path, background)

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
                seed=3,
                cache_dataset=False,
                return_raw_window=True,
                background_npy=path,
                background_layout="time_channel",
                background_channel_start=0,
                background_scale=1.0,
                artifact_competing_ratio=1.0,
                artifact_competing_time_jitter_s=0.2,
                artifact_competing_amp_scale_min=1.2,
                artifact_competing_amp_scale_max=1.2,
                artifact_competing_opposite_direction_ratio=1.0,
            )

            _, target = ds[0]
            self.assertEqual(int(target["time"].shape[0]), 1)
            self.assertEqual(int(target["visibility"].shape[0]), 1)
            self.assertIn("artifact_meta", target)
            self.assertEqual(int(target["artifact_meta"].shape[0]), 3)
            self.assertGreaterEqual(float(target["artifact_meta"][2]), 1.0)
            self.assertIn("artifact_competing_time", target)
            self.assertIn("artifact_competing_visibility", target)
            self.assertEqual(tuple(target["artifact_competing_time"].shape), (50,))
            self.assertEqual(tuple(target["artifact_competing_visibility"].shape), (50,))


if __name__ == "__main__":
    unittest.main()
