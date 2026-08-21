from __future__ import annotations

import numpy as np
import torch

from hybrid_vehicle_tracker.data.synthetic import SyntheticSettings, SyntheticVehicleDataset
from hybrid_vehicle_tracker.evaluation.null_control import independently_shift_feature_tensor
from hybrid_vehicle_tracker.types import Station, StationGeometry


def _geometry() -> StationGeometry:
    positions = [0.0, 100.0, 200.0, 300.0, 400.0, 600.0, 700.0, 800.0]
    return StationGeometry(
        tuple(
            Station(index, index + 1, f"S{index}", position)
            for index, position in enumerate(positions)
        )
    )


def test_fully_synthetic_waveform_scene_is_finite_and_speed_bounded():
    dataset = SyntheticVehicleDataset(
        _geometry(),
        SyntheticSettings(
            window_s=48.0,
            feature_rate_hz=10.0,
            waveform_rate_hz=100.0,
            samples=1,
            seed=11,
        ),
    )
    scene = dataset[0]
    assert scene["input"].shape == (5, 8, 480)
    assert bool(scene["input"].isfinite().all())
    valid = scene["track_params"][:, 2] > 0.5
    if bool(valid.any()):
        speeds = 3.6 / scene["track_params"][valid, 0]
        assert bool(((speeds >= 60.0) & (speeds <= 90.0)).all())


def test_synthetic_vehicle_uses_pre_valley_and_broad_gauss_peak():
    rate = 20.0
    dataset = SyntheticVehicleDataset(
        _geometry(),
        SyntheticSettings(
            window_s=48.0,
            feature_rate_hz=rate,
            waveform_rate_hz=200.0,
            samples=8,
            seed=29,
            vehicle_rate=8.0,
        ),
    )
    checked = False
    for scene_index in range(len(dataset)):
        scene = dataset[scene_index]
        features = scene["input"].numpy()
        times = scene["track_times"].numpy()
        observed = scene["track_observed"].numpy() > 0.5
        params = scene["track_params"].numpy()
        for track_index in np.flatnonzero(params[:, 2] > 0.5):
            for channel in np.flatnonzero(observed[track_index]):
                center = int(round(float(times[track_index, channel]) * rate))
                if not 30 <= center < features.shape[-1] - 30:
                    continue
                pre = features[1, channel]
                gauss = features[2, channel]
                local_peak = center - 2 + int(np.argmax(gauss[center - 2 : center + 3]))
                half_height = 0.5 * gauss[local_peak]
                left = local_peak
                while left > 0 and gauss[left - 1] >= half_height:
                    left -= 1
                right = local_peak
                while right + 1 < gauss.size and gauss[right + 1] >= half_height:
                    right += 1
                width_s = (right - left + 1) / rate
                # Skip overlapping station-local decoys; verify one isolated
                # injected observation has the intended morphology.
                if not 0.9 <= width_s <= 1.5:
                    continue
                assert pre[local_peak] < 0.12
                assert gauss[local_peak] > 0.60
                checked = True
                break
            if checked:
                break
        if checked:
            break
    assert checked


def test_vehicle_peakset_profile_is_continuous_and_not_random_peak_cloud():
    dataset = SyntheticVehicleDataset(
        _geometry(),
        SyntheticSettings(
            simulator_version="vehicle_peakset_realshape_v1",
            window_s=120.0,
            feature_rate_hz=20.0,
            waveform_rate_hz=200.0,
            samples=1,
            seed=21260707,
            motion_direction=-1,
            max_vehicles=6,
            peakset_vehicle_min=6,
            peakset_vehicle_max=6,
            peakset_min_visible_channels=5,
            peakset_dead_channels="",
            peakset_false_event_min=0,
            peakset_false_event_max=0,
            return_modalities=True,
        ),
    )
    scene = dataset[0]
    valid = scene["track_params"][:, 2] > 0.5
    assert int(valid.sum()) == 6
    assert bool((scene["track_observed"][valid].sum(dim=1) >= 5).all())
    speeds = 3.6 / scene["track_params"][valid, 0].abs()
    assert bool(((speeds >= 60.0) & (speeds <= 90.0)).all())
    # With no decoys, high Gauss pixels are concentrated in the injected line
    # neighbourhoods instead of filling most station/time bins.
    assert int((scene["gauss_feature"] > 0.5).sum()) < 800
    assert scene["raw_waveform"].shape[-1] == 120 * 200


def test_complex_peakset_has_day11_calibrated_decoys_outages_and_corner_tracks():
    dataset = SyntheticVehicleDataset(
        _geometry(),
        SyntheticSettings(
            simulator_version="vehicle_peakset_complex_v2",
            window_s=120.0,
            feature_rate_hz=20.0,
            waveform_rate_hz=200.0,
            samples=1,
            seed=21260707,
            motion_direction=-1,
            max_vehicles=14,
            peakset_vehicle_min=10,
            peakset_vehicle_max=12,
            peakset_min_visible_channels=5,
            peakset_dead_channels="",
            peakset_false_event_min=50,
            peakset_false_event_max=82,
            peakset_missing_ratio_min=0.08,
            peakset_missing_ratio_max=0.18,
            peakset_gap_probability=1.0,
            peakset_gap_min_channels=2,
            peakset_gap_max_channels=6,
            peakset_boundary_vehicle_ratio=0.40,
            peakset_outage_probability=1.0,
            return_modalities=True,
        ),
    )
    scene = dataset[0]
    valid_tracks = scene["track_params"][:, 2] > 0.5
    assert 10 <= int(valid_tracks.sum()) <= 12
    assert int(scene["boundary_track_mask"].sum()) >= 4
    assert int(scene["false_event_count"].item()) >= 50
    assert 0.20 <= float(scene["isolated_false_event_count"].item()) / float(
        scene["false_event_count"].item()
    ) <= 0.50
    assert int(scene["outage_count"].item()) >= 1
    times = scene["track_times"]
    observed = scene["track_observed"] > 0.5
    visible = torch.isfinite(times) & (times >= 0.0) & (times < 120.0)
    assert bool((visible[valid_tracks] & ~observed[valid_tracks]).any())
    # The extra decoys are meaningful Gauss events, but are still much sparser
    # than a full 50x2400 random cloud.
    high_gauss = int((scene["gauss_feature"] > 0.5).sum())
    assert 1000 < high_gauss < 9000


def test_null_shift_keeps_modalities_synchronised_and_geometry_fixed():
    features = np.zeros((5, 3, 12), dtype=np.float32)
    for plane in range(5):
        for channel in range(3):
            features[plane, channel] = 100 * plane + 10 * channel + np.arange(12)
    shifts = np.asarray([0.0, 0.2, 0.4])
    shifted = independently_shift_feature_tensor(
        features, shifts_s=shifts, feature_rate_hz=5.0
    )
    for channel, bins in enumerate((0, 1, 2)):
        for plane in range(3):
            assert np.array_equal(
                shifted[plane, channel], np.roll(features[plane, channel], bins)
            )
    assert np.array_equal(shifted[3:], features[3:])
