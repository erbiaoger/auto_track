from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset

from hybrid_vehicle_tracker.data.peakset_reference import (
    DAY11_GAUSS_FWHM_QUANTILES_S,
    DAY11_GAUSS_HEIGHT_QUANTILES,
    DAY11_ISOLATED_EVENT_FRACTION,
    station_sampling_weights,
)
from hybrid_vehicle_tracker.types import StationGeometry


@dataclass
class SyntheticSettings:
    """Parameters for the fully synthetic, multimodal waveform simulator."""

    simulator_version: str = "vehicle_peakset_realshape_v1"
    window_s: float = 48.0
    feature_rate_hz: float = 20.0
    waveform_rate_hz: float = 200.0
    max_vehicles: int = 20
    samples: int = 128
    seed: int = 20260724
    min_speed_kmh: float = 60.0
    max_speed_kmh: float = 90.0
    vehicle_rate: float = 5.0
    interaction_probability: float = 0.55
    # Sign of dt/dx in the station file.  DAY11 traffic is high-position to
    # low-position, so time increases with motion_direction=-1.
    motion_direction: int = 1
    # Export/debug mode only; keep disabled during training to avoid copying
    # large waveform arrays through every DataLoader batch.
    return_modalities: bool = False
    waveform_noise_min: float = 0.025
    waveform_noise_max: float = 0.15
    vehicle_snr_min: float = 0.70
    vehicle_snr_max: float = 1.00
    raw_sigma_min_s: float = 0.18
    raw_sigma_max_s: float = 0.55
    pre_valley_sigma_min_s: float = 0.42
    pre_valley_sigma_max_s: float = 0.60
    gauss_sigma_min_s: float = 0.47
    gauss_sigma_max_s: float = 0.54
    gauss_height_min: float = 0.65
    gauss_height_max: float = 0.69
    false_event_min: int = 80
    false_event_max: int = 140
    # The reference project has a deliberately clean ``vehicle peak-set``
    # curriculum.  It is useful for the first training stage because every
    # injected vehicle is a visible, continuous Gaussian event line rather
    # than an arbitrary cloud of independent peaks.  The profile is selected
    # by setting simulator_version to ``vehicle_peakset_realshape_v1``.  These
    # values intentionally mirror the read-only reference configuration, but
    # are implemented locally in this project.
    peakset_vehicle_min: int = 5
    peakset_vehicle_max: int = 8
    peakset_min_visible_channels: int = 8
    peakset_dead_channels: str = "5,6,15,16,22,36,38,45"
    peakset_false_event_min: int = 0
    peakset_false_event_max: int = 4
    peakset_missing_ratio_min: float = 0.0
    peakset_missing_ratio_max: float = 0.03
    peakset_gap_probability: float = 0.45
    peakset_gap_min_channels: int = 1
    peakset_gap_max_channels: int = 3
    peakset_background_noise: float = 0.012
    peakset_vehicle_amp_min: float = 0.45
    peakset_vehicle_amp_max: float = 0.65
    # Complex peak-set profile controls.  The default values retain the clean
    # v1 behaviour; ``vehicle_peakset_complex_v2`` opts into the harder scene.
    peakset_boundary_vehicle_ratio: float = 0.35
    peakset_boundary_time_margin_s: float = 3.5
    # Explicit head/tail occlusion curriculum.  The centreline remains in the
    # target while the first/last few observations are removed, teaching the
    # network to complete a physically valid line rather than truncating it.
    peakset_edge_dropout_probability: float = 0.45
    peakset_edge_dropout_min_channels: int = 1
    peakset_edge_dropout_max_channels: int = 4
    # Pair geometry curriculum for identity preservation at intersections and
    # near-parallel traffic.  The local event kernel remains shared by every
    # vehicle and nuisance peak; only cross-station geometry identifies it.
    peakset_near_parallel_probability: float = 0.30
    peakset_isolated_fraction_min: float = 0.28
    peakset_isolated_fraction_max: float = 0.40
    peakset_outage_probability: float = 0.70
    peakset_outage_min_channels: int = 1
    peakset_outage_max_channels: int = 3
    peakset_outage_min_duration_s: float = 0.8
    peakset_outage_max_duration_s: float = 5.5


@dataclass
class _ModalScene:
    raw: np.ndarray
    pre: np.ndarray
    gauss: np.ndarray
    quality: np.ndarray
    raw_noise_scale: np.ndarray
    gauss_peak_height: np.ndarray
    # The peak-set profile keeps an explicit low-rate positive envelope.  The
    # older reference model trained on this envelope, while ``raw`` remains a
    # full-rate waveform for inspection/export.
    raw_feature: np.ndarray | None = None
    false_event_count: int = 0
    isolated_false_event_count: int = 0
    outage_count: int = 0
    # Complex peak-set scenes deliberately use one shared event kernel for
    # vehicles and nuisance peaks.  Only the cross-station trajectory makes a
    # vehicle identifiable.
    event_sigma_s: float | None = None
    event_height: float | None = None
    event_raw_sigma_s: float | None = None
    event_pre_sigma_s: float | None = None
    event_amplitude: float | None = None
    event_pre_depth: float | None = None


class SyntheticVehicleDataset(Dataset[dict[str, torch.Tensor]]):
    """Generate every training sample without reading measured waveform data.

    The simulator independently implements the useful ideas from the read-only
    reference project: Gaussian vehicle wave packets, isolated nuisance pulses,
    missing channels, dead intervals and interacting vehicles.  It additionally
    creates aligned Raw/Pre/Gauss modalities and uses the mapping's physical
    positions when integrating travel time.
    """

    def __init__(self, geometry: StationGeometry, settings: SyntheticSettings) -> None:
        self.geometry = geometry
        self.settings = settings
        # The dataset is generated on demand.  Keep the epoch in the seed so
        # that an index is not the same scene forever when a DataLoader is
        # iterated repeatedly during multi-stage training.
        self._epoch = 0
        if int(settings.motion_direction) not in (-1, 1):
            raise ValueError("motion_direction must be +1 or -1")
        self.settings.motion_direction = int(settings.motion_direction)
        ratio = settings.waveform_rate_hz / settings.feature_rate_hz
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("waveform_rate_hz must be an integer multiple of feature_rate_hz")
        if settings.waveform_rate_hz < 4.0 / settings.raw_sigma_min_s:
            raise ValueError("waveform_rate_hz is too low for the narrowest simulated pulse")

    def __len__(self) -> int:
        return self.settings.samples

    def set_epoch(self, epoch: int) -> None:
        """Select a new deterministic scene population for an epoch."""
        self._epoch = int(epoch)

    def _scene_seed(self, index: int) -> int:
        # Large odd multipliers keep neighbouring indices/epochs decorrelated
        # while retaining reproducibility for an exact training run.
        return int(self.settings.seed + self._epoch * 1_000_003 + int(index) * 104_729)

    @property
    def _time_bins(self) -> int:
        return int(round(self.settings.window_s * self.settings.feature_rate_hz))

    @property
    def _waveform_samples(self) -> int:
        return int(round(self.settings.window_s * self.settings.waveform_rate_hz))

    @property
    def _is_peakset_profile(self) -> bool:
        version = str(self.settings.simulator_version).strip().lower()
        return version.startswith("vehicle_peakset") or version.startswith("realshape")

    @property
    def _is_complex_peakset_profile(self) -> bool:
        version = str(self.settings.simulator_version).strip().lower()
        return version in {
            "vehicle_peakset_complex_v2",
            "vehicle_peakset_complex",
            "realshape_complex_v2",
        }

    def _peakset_background(self, rng: np.random.Generator) -> _ModalScene:
        """Create the clean waveform used by the reference peak-set curriculum.

        This is intentionally conservative: a small positive baseline, a few
        station-local decoys, and no dense random event field.  Vehicle events
        are added later by :meth:`_inject_peakset_track`, so their exact centre
        times and visibility masks remain available as ground truth.
        """
        station_count = len(self.geometry)
        sample_count = self._waveform_samples
        time_bins = self._time_bins
        noise_level = float(max(self.settings.peakset_background_noise, 1e-6))
        station_noise = rng.uniform(0.65, 1.35, size=station_count).astype(np.float32)
        # A low, positive waveform baseline matches the old overlay's quiet
        # station wiggles.  Vehicle envelopes are positive Gaussian pulses.
        raw = np.abs(
            rng.normal(
                0.0,
                noise_level * station_noise[:, None],
                size=(station_count, sample_count),
            )
        ).astype(np.float32)
        raw_feature = np.abs(
            rng.normal(
                0.0,
                noise_level * 0.45 * station_noise[:, None],
                size=(station_count, time_bins),
            )
        ).astype(np.float32)
        pre = np.clip(
            rng.normal(0.68, 0.035, size=(station_count, time_bins)), 0.40, 0.90
        ).astype(np.float32)
        gauss = np.zeros((station_count, time_bins), dtype=np.float32)
        quality = np.ones(station_count, dtype=np.float32)
        if self._is_complex_peakset_profile:
            # DAY11's median FWHM is 1.177 s, i.e. sigma ~= 0.500 s.  Use one
            # kernel and one height for every positive and nuisance event in a
            # scene.  A nuisance peak is therefore indistinguishable from a
            # vehicle at one station; only a physically continuous line can
            # identify it.
            event_sigma_s = float(np.median(DAY11_GAUSS_FWHM_QUANTILES_S) / 2.354820045)
            event_height = float(np.median(DAY11_GAUSS_HEIGHT_QUANTILES))
            event_raw_sigma_s = float(0.32)
            event_pre_sigma_s = float(0.50)
            event_amplitude = float(0.56)
            event_pre_depth = float(0.96)
            gauss_peak_height = np.full(
                station_count, event_height, dtype=np.float32
            )
        else:
            event_sigma_s = None
            event_height = None
            event_raw_sigma_s = None
            event_pre_sigma_s = None
            event_amplitude = None
            event_pre_depth = None
            gauss_peak_height = rng.uniform(0.65, 0.70, size=station_count).astype(np.float32)

        # v1 keeps only a handful of isolated hard negatives.  The complex
        # profile uses the DAY11-derived station occupancy distribution and
        # approximately one third isolated events.  Remaining events are
        # small station-local bursts (two or three nearby stations) rather
        # than a dense random cloud, so they remain realistic but cannot be
        # mistaken for a full vehicle line without cross-station support.
        false_count = int(
            rng.integers(
                max(0, self.settings.peakset_false_event_min),
                max(
                    max(0, self.settings.peakset_false_event_min),
                    self.settings.peakset_false_event_max,
                )
                + 1,
            )
        )
        isolated_count = 0
        if self._is_complex_peakset_profile and false_count:
            isolated_fraction = float(
                rng.uniform(
                    max(0.0, self.settings.peakset_isolated_fraction_min),
                    max(
                        max(0.0, self.settings.peakset_isolated_fraction_min),
                        self.settings.peakset_isolated_fraction_max,
                    ),
                )
            )
            # Keep the empirical DAY11 fraction central even if a caller uses
            # broad custom bounds.  This also makes the calibration explicit
            # in the generated scene rather than an accidental random ratio.
            isolated_fraction = float(
                np.clip(
                    0.5 * isolated_fraction + 0.5 * DAY11_ISOLATED_EVENT_FRACTION,
                    0.05,
                    0.95,
                )
            )
            isolated_count = int(np.clip(round(false_count * isolated_fraction), 0, false_count))

        weights = station_sampling_weights(station_count)
        false_locations: list[tuple[int, float, bool]] = []
        for _ in range(isolated_count):
            false_locations.append(
                (
                    int(rng.choice(station_count, p=weights)),
                    float(rng.uniform(0.0, self.settings.window_s)),
                    True,
                )
            )
        remaining = false_count - isolated_count
        while remaining > 0:
            base_channel = int(rng.choice(station_count, p=weights))
            center_s = float(rng.uniform(0.0, self.settings.window_s))
            cluster_size = min(remaining, int(rng.integers(2, 4)))
            for member in range(cluster_size):
                # Adjacent-channel local bursts have no physical travel slope:
                # their time jitter is deliberately much larger than the
                # 1 ms event precision but smaller than a vehicle's travel
                # time over a station gap.
                channel = int(np.clip(base_channel + member - cluster_size // 2, 0, station_count - 1))
                false_locations.append(
                    (
                        channel,
                        float(center_s + rng.normal(0.0, 0.30 + 0.15 * member)),
                        False,
                    )
                )
            remaining -= cluster_size

        for channel, center_s, is_isolated in false_locations:
            sigma_s = (
                float(event_sigma_s)
                if event_sigma_s is not None
                else float(
                    rng.uniform(
                        max(0.40, float(DAY11_GAUSS_FWHM_QUANTILES_S[0] / 2.355)),
                        min(0.62, float(DAY11_GAUSS_FWHM_QUANTILES_S[-1] / 2.355) + 0.04),
                    )
                )
            )
            self._add_feature_valley(
                pre[channel],
                center_s,
                float(event_pre_sigma_s or sigma_s),
                float(event_pre_depth if event_pre_depth is not None else rng.uniform(0.55, 0.82)),
            )
            height_low = float(DAY11_GAUSS_HEIGHT_QUANTILES[0])
            height_high = float(DAY11_GAUSS_HEIGHT_QUANTILES[-1])
            # False peaks use the same broad height range as the measured
            # Gauss plane.  Their rejection must come from continuity and
            # physics, not from an artificial amplitude shortcut.
            self._add_feature_pulse(
                gauss[channel],
                center_s,
                sigma_s,
                float(event_height if event_height is not None else np.clip(rng.uniform(height_low, height_high), 0.0, 1.0)),
            )
            self._add_feature_pulse(
                raw_feature[channel],
                center_s,
                float(event_raw_sigma_s or 0.32),
                float(event_amplitude or rng.uniform(0.10, 0.24)),
            )
            self._add_feature_pulse(
                raw[channel],
                center_s,
                float(event_raw_sigma_s or 0.32),
                float(event_amplitude or rng.uniform(0.10, 0.24)),
                rate=self.settings.waveform_rate_hz,
            )

        return _ModalScene(
            raw=raw,
            pre=pre,
            gauss=gauss,
            quality=quality,
            raw_noise_scale=station_noise,
            gauss_peak_height=gauss_peak_height,
            raw_feature=raw_feature,
            false_event_count=false_count,
            isolated_false_event_count=isolated_count,
            event_sigma_s=event_sigma_s,
            event_height=event_height,
            event_raw_sigma_s=event_raw_sigma_s,
            event_pre_sigma_s=event_pre_sigma_s,
            event_amplitude=event_amplitude,
            event_pre_depth=event_pre_depth,
        )

    def _sample_peakset_track_params(
        self, rng: np.random.Generator, count: int
    ) -> list[tuple[float, float]]:
        """Sample visible, mostly continuous 60--90 km/h vehicle lines.

        ``intercept`` is the time at the first physical station.  Sampling an
        anchor position/time rather than an unconstrained intercept guarantees
        that each accepted vehicle contributes at least a few real stations in
        the 120 s window.  A subset of pairs is made to cross at a controlled
        position, matching the reference overlay without filling the scene
        with unrelated clutter.
        """
        station_positions = self.geometry.relative_positions_m.astype(np.float64)
        span_m = float(station_positions[-1])
        direction = int(self.settings.motion_direction)
        minimum = int(max(1, min(self.settings.peakset_vehicle_min, self.settings.max_vehicles)))
        maximum = int(max(minimum, min(self.settings.peakset_vehicle_max, self.settings.max_vehicles)))
        target_count = int(np.clip(count, minimum, maximum))
        tracks: list[tuple[float, float]] = []
        attempts = 0
        while len(tracks) < target_count and attempts < max(200, 200 * target_count):
            attempts += 1
            speed = float(rng.uniform(self.settings.min_speed_kmh, self.settings.max_speed_kmh))
            slope = 3.6 / speed
            if tracks and rng.random() < float(self.settings.interaction_probability):
                # Pick a prior line and force a same-direction crossing or a
                # near-parallel pair.  The physical constraint is still the
                # only source of the line; no hand-drawn labels are involved.
                other_speed, other_intercept = tracks[int(rng.integers(0, len(tracks)))]
                cross_position = float(rng.uniform(0.18 * span_m, 0.82 * span_m))
                cross_time = float(
                    other_intercept
                    + direction * (3.6 / other_speed) * cross_position
                )
                if not 0.0 <= cross_time < self.settings.window_s:
                    continue
                intercept = cross_time - direction * slope * cross_position
                if abs(speed - other_speed) < 4.0:
                    speed = float(
                        np.clip(
                            other_speed + (7.0 if other_speed < 76.0 else -7.0),
                            self.settings.min_speed_kmh,
                            self.settings.max_speed_kmh,
                        )
                    )
                    slope = 3.6 / speed
                    intercept = cross_time - direction * slope * cross_position
            else:
                # Spread clean scenes across the physical span.  The old
                # peak-set overlay deliberately shows vehicles entering at
                # several offsets; sampling every line from one random
                # anchor would collapse them into a visually misleading
                # bundle near the centre of the plot.
                display_start_fraction = (len(tracks) + 1.0) / (target_count + 1.0)
                display_start_fraction += float(rng.normal(0.0, 0.035))
                display_start_fraction = float(np.clip(display_start_fraction, 0.04, 0.96))
                anchor_position = (
                    span_m * (1.0 - display_start_fraction)
                    if direction < 0
                    else span_m * display_start_fraction
                )
                anchor_time = float(rng.uniform(-3.0, 7.0))
                intercept = anchor_time - direction * slope * anchor_position

            times = intercept + direction * slope * station_positions
            visible_count = int(np.count_nonzero((times >= 0.0) & (times < self.settings.window_s)))
            if visible_count < int(max(5, self.settings.peakset_min_visible_channels)):
                continue
            tracks.append((speed, float(intercept)))
        return tracks

    def _sample_complex_peakset_track_params(
        self, rng: np.random.Generator, count: int
    ) -> list[tuple[float, float, bool]]:
        """Sample a harder peak-set scene with four boundary/corner vehicles.

        The returned flag marks vehicles deliberately anchored near one of the
        four time--space corners.  A corner line is still a normal physical
        line: only its anchor is moved close to ``t=0``/``t=window`` and to a
        travel-oriented end of the station span.  The interior vehicles are
        allowed to cross or run nearly parallel, which gives the association
        network both identity and ambiguity examples.
        """

        station_positions = self.geometry.relative_positions_m.astype(np.float64)
        span_m = float(station_positions[-1])
        direction = int(self.settings.motion_direction)
        minimum = int(max(1, min(self.settings.peakset_vehicle_min, self.settings.max_vehicles)))
        maximum = int(max(minimum, min(self.settings.peakset_vehicle_max, self.settings.max_vehicles)))
        target_count = int(np.clip(count, minimum, maximum))
        boundary_count = int(
            np.clip(
                round(target_count * float(self.settings.peakset_boundary_vehicle_ratio)),
                2,
                min(4, target_count),
            )
        )
        margin = float(max(0.2, self.settings.peakset_boundary_time_margin_s))
        # Fractions are plotted in travel-oriented offset coordinates.  The
        # physical position is reversed automatically for DAY11's -1 motion.
        corner_specs = (
            (0.18, 0.0),
            (0.82, 0.0),
            (0.18, self.settings.window_s),
            (0.82, self.settings.window_s),
        )
        tracks: list[tuple[float, float, bool]] = []
        corner_index = 0
        attempts = 0
        while len(tracks) < target_count and attempts < max(400, 400 * target_count):
            attempts += 1
            speed = float(rng.uniform(self.settings.min_speed_kmh, self.settings.max_speed_kmh))
            slope = 3.6 / speed
            is_boundary = len(tracks) < boundary_count
            if is_boundary:
                display_fraction, edge_time = corner_specs[corner_index % len(corner_specs)]
                display_fraction += float(rng.normal(0.0, 0.035))
                display_fraction = float(np.clip(display_fraction, 0.12, 0.88))
                # Keep enough of a corner line inside the 50-station array;
                # this produces 8+ observed stations even at the edge.
                physical_fraction = (
                    display_fraction if direction > 0 else 1.0 - display_fraction
                )
                anchor_position = span_m * physical_fraction
                anchor_time = float(edge_time + rng.uniform(-margin, margin))
                anchor_time = float(np.clip(anchor_time, -margin, self.settings.window_s + margin))
                intercept = anchor_time - direction * slope * anchor_position
            elif tracks and rng.random() < float(self.settings.interaction_probability):
                other_speed, other_intercept, _ = tracks[int(rng.integers(0, len(tracks)))]
                cross_position = float(rng.uniform(0.12 * span_m, 0.88 * span_m))
                cross_time = float(
                    other_intercept + direction * (3.6 / other_speed) * cross_position
                )
                if not 2.0 <= cross_time <= self.settings.window_s - 2.0:
                    continue
                if abs(speed - other_speed) < 3.5:
                    if rng.random() < float(self.settings.peakset_near_parallel_probability):
                        # Keep a genuinely close pair (0.5--3 km/h apart).
                        delta = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 3.0))
                        speed = float(
                            np.clip(
                                other_speed + delta,
                                self.settings.min_speed_kmh,
                                self.settings.max_speed_kmh,
                            )
                        )
                    else:
                        # Also retain a clearly different-speed crossing pair.
                        speed = float(
                            np.clip(
                                other_speed + (6.0 if other_speed < 76.0 else -6.0),
                                self.settings.min_speed_kmh,
                                self.settings.max_speed_kmh,
                            )
                        )
                    slope = 3.6 / speed
                intercept = cross_time - direction * slope * cross_position
            else:
                display_fraction = float(rng.uniform(0.12, 0.88))
                physical_fraction = (
                    display_fraction if direction > 0 else 1.0 - display_fraction
                )
                anchor_position = span_m * physical_fraction
                anchor_time = float(rng.uniform(5.0, self.settings.window_s - 5.0))
                intercept = anchor_time - direction * slope * anchor_position

            times = intercept + direction * slope * station_positions
            visible_count = int(
                np.count_nonzero((times >= 0.0) & (times < self.settings.window_s))
            )
            if visible_count < int(max(5, self.settings.peakset_min_visible_channels)):
                if is_boundary:
                    # Small unit-test geometries may not have enough stations
                    # to show a mathematically exact corner.  Try the next
                    # corner rather than repeatedly retrying the same one.
                    corner_index += 1
                continue
            if is_boundary:
                corner_index += 1
            tracks.append((speed, float(intercept), is_boundary))

        # Extremely short custom geometries can make an edge anchor impossible
        # to satisfy.  Fall back to the clean sampler rather than returning a
        # scene with an incorrect vehicle count.
        if len(tracks) < target_count:
            fallback = self._sample_peakset_track_params(rng, target_count - len(tracks))
            tracks.extend((speed, intercept, False) for speed, intercept in fallback)
        return tracks[:target_count]

    def _inject_peakset_track(
        self,
        scene: _ModalScene,
        centerline: np.ndarray,
        slowness_target: np.ndarray,
        crossing: np.ndarray,
        *,
        times: np.ndarray,
        speed_kmh: float,
        rng: np.random.Generator,
        positive: bool = True,
    ) -> np.ndarray:
        """Render one clean vehicle line and return its observed mask."""
        valid = (times >= 0.0) & (times < self.settings.window_s)
        observed = valid.copy()
        if positive and np.any(valid):
            drop_ratio = float(
                rng.uniform(
                    self.settings.peakset_missing_ratio_min,
                    self.settings.peakset_missing_ratio_max,
                )
            )
            observed &= rng.random(observed.size) >= drop_ratio
            if (
                rng.random() < float(self.settings.peakset_gap_probability)
                and int(np.count_nonzero(observed)) > int(self.settings.peakset_min_visible_channels) + 2
            ):
                gap_len = int(
                    rng.integers(
                        max(1, self.settings.peakset_gap_min_channels),
                        max(
                            max(1, self.settings.peakset_gap_min_channels),
                            self.settings.peakset_gap_max_channels,
                        )
                        + 1,
                    )
                )
                valid_indices = np.flatnonzero(valid)
                if valid_indices.size > gap_len:
                    start = int(rng.integers(0, valid_indices.size - gap_len + 1))
                    observed[valid_indices[start : start + gap_len]] = False
            # Head/tail dropout is separate from an internal gap.  It creates
            # examples where the measured line starts late or ends early,
            # while the dense centreline target still contains the full
            # in-window physical trajectory.
            if (
                rng.random() < float(self.settings.peakset_edge_dropout_probability)
                and int(np.count_nonzero(observed)) > int(self.settings.peakset_min_visible_channels) + 2
            ):
                valid_indices = np.flatnonzero(valid)
                edge_len = int(
                    rng.integers(
                        max(1, self.settings.peakset_edge_dropout_min_channels),
                        max(
                            max(1, self.settings.peakset_edge_dropout_min_channels),
                            self.settings.peakset_edge_dropout_max_channels,
                        )
                        + 1,
                    )
                )
                edge_len = min(edge_len, max(1, valid_indices.size - int(self.settings.peakset_min_visible_channels)))
                if edge_len > 0 and valid_indices.size:
                    if rng.random() < 0.5:
                        observed[valid_indices[:edge_len]] = False
                    else:
                        observed[valid_indices[-edge_len:]] = False
            # Do not let a difficult window disappear from the supervision.
            valid_indices = np.flatnonzero(valid)
            if int(np.count_nonzero(observed)) < min(
                int(self.settings.peakset_min_visible_channels), valid_indices.size
            ):
                keep = rng.choice(
                    valid_indices,
                    size=min(int(self.settings.peakset_min_visible_channels), valid_indices.size),
                    replace=False,
                )
                observed[keep] = True

        slope = 3.6 / max(abs(float(speed_kmh)), 1e-6)
        for channel in np.flatnonzero(valid):
            time_s = float(times[channel])
            # The dense centreline is narrow; the input modalities carry the
            # wider, approximately 1.2 s FWHM Gaussian response.
            target_pulse = np.zeros(self._time_bins, dtype=np.float32)
            self._add_feature_pulse(target_pulse, time_s, 0.10, 1.0)
            existing = centerline[0, channel].copy()
            centerline[0, channel] = np.maximum(existing, target_pulse)
            if positive:
                crossing[0, channel] = np.maximum(
                    crossing[0, channel],
                    ((existing > 0.25) & (target_pulse > 0.25)).astype(np.float32),
                )
                slowness_target[0, channel, target_pulse > 0.20] = float(
                    np.clip((slope - 0.04) / 0.02, 0.0, 1.0)
                )
            if not observed[channel]:
                continue
            amplitude = float(
                scene.event_amplitude
                if self._is_complex_peakset_profile and scene.event_amplitude is not None
                else rng.uniform(self.settings.peakset_vehicle_amp_min, self.settings.peakset_vehicle_amp_max)
            )
            raw_sigma = float(
                scene.event_raw_sigma_s
                if self._is_complex_peakset_profile and scene.event_raw_sigma_s is not None
                else rng.uniform(self.settings.raw_sigma_min_s, self.settings.raw_sigma_max_s)
            )
            pre_sigma = float(
                scene.event_pre_sigma_s
                if self._is_complex_peakset_profile and scene.event_pre_sigma_s is not None
                else rng.uniform(self.settings.pre_valley_sigma_min_s, self.settings.pre_valley_sigma_max_s)
            )
            gauss_sigma = float(
                scene.event_sigma_s
                if self._is_complex_peakset_profile and scene.event_sigma_s is not None
                else rng.uniform(self.settings.gauss_sigma_min_s, self.settings.gauss_sigma_max_s)
            )
            if not positive and not self._is_complex_peakset_profile:
                amplitude *= 0.30
            self._add_feature_pulse(scene.raw_feature[channel], time_s, raw_sigma, amplitude)
            self._add_feature_valley(
                scene.pre[channel],
                time_s,
                pre_sigma,
                float(
                    (
                        scene.event_pre_depth
                        if self._is_complex_peakset_profile and scene.event_pre_depth is not None
                        else (
                            rng.uniform(0.45, 0.65)
                            if not positive
                            else rng.uniform(0.92, 1.0)
                        )
                    )
                ),
            )
            self._add_feature_pulse(
                scene.gauss[channel],
                time_s
                + (
                    0.0
                    if self._is_complex_peakset_profile
                    else float(rng.normal(0.0, 0.008))
                ),
                gauss_sigma,
                float(
                    (
                        scene.gauss_peak_height[channel]
                        if self._is_complex_peakset_profile
                        else scene.gauss_peak_height[channel]
                        * (rng.uniform(0.25, 0.40) if not positive else rng.uniform(0.985, 1.0))
                    )
                ),
            )
            # Keep a full-rate positive envelope for waveform-style overlays.
            self._add_feature_pulse(
                scene.raw[channel],
                time_s,
                raw_sigma,
                amplitude,
                rate=self.settings.waveform_rate_hz,
            )
        return observed.astype(np.float32)

    def _apply_complex_outages(
        self,
        scene: _ModalScene,
        track_times: np.ndarray,
        track_observed: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Apply short multi-channel outages and mark affected GT points.

        These are distinct from the fixed dead channels: an outage is a
        temporary loss of several neighbouring channels, so the underlying
        vehicle centreline remains a valid prediction target while the input
        observation is absent.  This is the missing-channel pattern that the
        Edge-GNN and second-order search are expected to bridge.
        """

        if not self._is_complex_peakset_profile or rng.random() >= float(
            self.settings.peakset_outage_probability
        ):
            return
        station_count = len(self.geometry)
        time_bins = self._time_bins
        outage_count = int(rng.integers(1, 4))
        for _ in range(outage_count):
            width = int(
                rng.integers(
                    max(1, self.settings.peakset_outage_min_channels),
                    max(
                        max(1, self.settings.peakset_outage_min_channels),
                        self.settings.peakset_outage_max_channels,
                    )
                    + 1,
                )
            )
            width = min(width, station_count)
            channel_start = int(rng.integers(0, max(1, station_count - width + 1)))
            duration_s = float(
                rng.uniform(
                    max(0.1, self.settings.peakset_outage_min_duration_s),
                    max(
                        max(0.1, self.settings.peakset_outage_min_duration_s),
                        self.settings.peakset_outage_max_duration_s,
                    ),
                )
            )
            start_s = float(
                rng.uniform(0.0, max(0.0, self.settings.window_s - duration_s))
            )
            start_bin = int(np.floor(start_s * self.settings.feature_rate_hz))
            end_bin = min(time_bins, int(np.ceil((start_s + duration_s) * self.settings.feature_rate_hz)))
            if end_bin <= start_bin:
                continue
            channel_slice = slice(channel_start, channel_start + width)
            scene.raw_feature[channel_slice, start_bin:end_bin] = 0.0
            scene.pre[channel_slice, start_bin:end_bin] = 0.0
            scene.gauss[channel_slice, start_bin:end_bin] = 0.0
            raw_start = int(np.floor(start_s * self.settings.waveform_rate_hz))
            raw_end = min(
                scene.raw.shape[1],
                int(np.ceil((start_s + duration_s) * self.settings.waveform_rate_hz)),
            )
            scene.raw[channel_slice, raw_start:raw_end] = 0.0
            for track_index in range(track_times.shape[0]):
                times = track_times[track_index]
                affected = (
                    np.isfinite(times)
                    & (np.arange(station_count) >= channel_start)
                    & (np.arange(station_count) < channel_start + width)
                    & (times >= start_s)
                    & (times < start_s + duration_s)
                )
                track_observed[track_index, affected] = 0.0
            scene.outage_count += 1

    def _getitem_peakset(self, index: int) -> dict[str, torch.Tensor]:
        """Generate one reference-style clean peak-set scene."""
        rng = np.random.default_rng(self._scene_seed(index))
        station_count = len(self.geometry)
        scene = self._peakset_background(rng)
        centerline = np.zeros((1, station_count, self._time_bins), dtype=np.float32)
        slowness = np.zeros_like(centerline)
        crossing = np.zeros_like(centerline)
        track_times = np.full(
            (int(self.settings.max_vehicles), station_count), np.nan, dtype=np.float32
        )
        track_observed = np.zeros_like(track_times)
        params = np.zeros((int(self.settings.max_vehicles), 3), dtype=np.float32)
        boundary_track_mask = np.zeros(int(self.settings.max_vehicles), dtype=np.float32)

        requested = int(
            rng.integers(
                max(1, self.settings.peakset_vehicle_min),
                max(
                    max(1, self.settings.peakset_vehicle_min),
                    self.settings.peakset_vehicle_max,
                )
                + 1,
            )
        )
        if self._is_complex_peakset_profile:
            tracks = self._sample_complex_peakset_track_params(rng, requested)
        else:
            tracks = self._sample_peakset_track_params(rng, requested)
        for track_index, track in enumerate(tracks[: int(self.settings.max_vehicles)]):
            if len(track) == 3:
                speed, intercept, boundary = track
                boundary_track_mask[track_index] = float(bool(boundary))
            else:
                speed, intercept = track
            positions = self.geometry.relative_positions_m.astype(np.float64)
            slope = 3.6 / max(float(speed), 1e-6)
            times = (
                float(intercept)
                + int(self.settings.motion_direction) * slope * positions
            ).astype(np.float32)
            observed = self._inject_peakset_track(
                scene,
                centerline,
                slowness,
                crossing,
                times=times,
                speed_kmh=float(speed),
                rng=rng,
                positive=True,
            )
            params[track_index] = (
                int(self.settings.motion_direction) * slope,
                float(intercept),
                1.0,
            )
            track_times[track_index] = times
            track_observed[track_index] = observed

        # A very small number of off-range or reverse-direction hard negatives
        # are retained for the speed/direction gate, but they never form the
        # dense vehicle labels.  Their amplitudes are intentionally weaker than
        # the positive lines, unlike the old v2 random event field.
        if rng.random() < 0.35:
            speed = float(rng.uniform(45.0, 59.0))
            direction = int(self.settings.motion_direction)
            intercept = float(rng.uniform(-40.0, self.settings.window_s + 40.0))
            times = (
                intercept
                + direction * (3.6 / speed) * self.geometry.relative_positions_m
            ).astype(np.float32)
            self._inject_peakset_track(
                scene,
                centerline=np.zeros_like(centerline),
                slowness_target=np.zeros_like(slowness),
                crossing=np.zeros_like(crossing),
                times=times,
                speed_kmh=speed,
                rng=rng,
                positive=False,
            )

        # Temporary outage blocks are applied after rendering all vehicles so
        # every affected observation is explicitly labelled as missing while
        # the centreline target remains available for recovery supervision.
        self._apply_complex_outages(scene, track_times, track_observed, rng)

        dead = {
            int(item.strip())
            for item in str(self.settings.peakset_dead_channels).split(",")
            if item.strip().lstrip("-").isdigit()
            and 0 <= int(item.strip()) < station_count
        }
        dead_channels = np.asarray(sorted(dead), dtype=np.int64)
        if dead_channels.size:
            scene.raw[dead_channels] = 0.0
            scene.raw_feature[dead_channels] = 0.0
            scene.pre[dead_channels] = 0.0
            scene.gauss[dead_channels] = 0.0
            scene.quality[dead_channels] = 0.0
            track_observed[:, dead_channels] = 0.0

        features = self._finalize_features(scene, dead_channels)
        result: dict[str, torch.Tensor] = {
            "input": torch.from_numpy(features),
            "centerline": torch.from_numpy(centerline),
            "slowness": torch.from_numpy(slowness),
            "crossing": torch.from_numpy(crossing),
            "track_params": torch.from_numpy(params),
            "track_times": torch.from_numpy(track_times),
            "track_observed": torch.from_numpy(track_observed),
            "boundary_track_mask": torch.from_numpy(boundary_track_mask),
            "false_event_count": torch.tensor(float(scene.false_event_count), dtype=torch.float32),
            "isolated_false_event_count": torch.tensor(
                float(scene.isolated_false_event_count), dtype=torch.float32
            ),
            "outage_count": torch.tensor(float(scene.outage_count), dtype=torch.float32),
            "event_kernel_sigma_s": torch.tensor(
                float(scene.event_sigma_s) if scene.event_sigma_s is not None else float("nan"),
                dtype=torch.float32,
            ),
            "event_kernel_height": torch.tensor(
                float(scene.event_height) if scene.event_height is not None else float("nan"),
                dtype=torch.float32,
            ),
            "event_kernel_raw_sigma_s": torch.tensor(
                float(scene.event_raw_sigma_s)
                if scene.event_raw_sigma_s is not None
                else float("nan"),
                dtype=torch.float32,
            ),
            "event_kernel_pre_sigma_s": torch.tensor(
                float(scene.event_pre_sigma_s)
                if scene.event_pre_sigma_s is not None
                else float("nan"),
                dtype=torch.float32,
            ),
        }
        if self.settings.return_modalities:
            result.update(
                {
                    "raw_waveform": torch.from_numpy(scene.raw.copy()),
                    "pre_feature": torch.from_numpy(scene.pre.copy()),
                    "gauss_feature": torch.from_numpy(scene.gauss.copy()),
                    "quality_vector": torch.from_numpy(scene.quality.copy()),
                    "bad_channel_mask": torch.from_numpy(
                        np.isin(np.arange(station_count), dead_channels).astype(np.float32)
                    ),
                }
            )
        return result

    def _synthetic_background(self, rng: np.random.Generator) -> _ModalScene:
        """Create colored station noise, common mode, bursts and sparse false peaks."""
        station_count = len(self.geometry)
        sample_count = self._waveform_samples
        time_bins = self._time_bins

        station_noise = rng.uniform(
            self.settings.waveform_noise_min,
            self.settings.waveform_noise_max,
            size=(station_count, 1),
        ).astype(np.float32)
        white = rng.standard_t(5.0, size=(station_count, sample_count)).astype(np.float32)
        colored = gaussian_filter1d(white, sigma=1.2, axis=1, mode="reflect")
        raw = station_noise * colored

        # Weak coherent machinery/environmental components are not vehicle lines:
        # phase and station gain vary enough that they cannot form a fixed-speed track.
        waveform_t = np.arange(sample_count, dtype=np.float32) / self.settings.waveform_rate_hz
        common = np.zeros(sample_count, dtype=np.float32)
        for _ in range(int(rng.integers(1, 5))):
            frequency = float(rng.uniform(0.15, 7.0))
            phase = float(rng.uniform(0.0, 2.0 * np.pi))
            common += float(rng.uniform(0.005, 0.04)) * np.sin(
                2.0 * np.pi * frequency * waveform_t + phase
            )
        raw += rng.normal(0.0, 0.6, size=(station_count, 1)).astype(np.float32) * common

        # DAY11's percentile-calibrated Pre plane has a broad background around
        # 0.5.  Candidate centres are *negative valleys*, not positive peaks.
        # Generate this normalized plane parametrically; no measured sample is
        # copied into training.
        pre = (0.05 + 0.95 * rng.beta(2.0, 2.0, size=(station_count, time_bins))).astype(
            np.float32
        )
        pre = gaussian_filter1d(pre, sigma=0.35, axis=1, mode="reflect")
        gauss = np.zeros((station_count, time_bins), dtype=np.float32)
        quality = (
            0.65 + 0.35 * rng.beta(5.0, 1.0, size=station_count)
        ).astype(np.float32)
        gauss_peak_height = rng.uniform(
            self.settings.gauss_height_min,
            self.settings.gauss_height_max,
            size=station_count,
        ).astype(np.float32)

        # Non-stationary noise patches and isolated raw bursts.
        for _ in range(int(rng.integers(4, 22))):
            channel = int(rng.integers(0, station_count))
            center_s = float(rng.uniform(0.0, self.settings.window_s))
            sigma_s = float(rng.uniform(0.05, 0.45))
            amplitude = float(station_noise[channel, 0] * rng.uniform(0.08, 1.5))
            self._add_raw_wave_packet(raw[channel], center_s, sigma_s, amplitude, rng)
            if rng.random() < 0.7:
                self._add_feature_pulse(
                    pre[channel], center_s, sigma_s, float(rng.uniform(0.05, 0.35))
                )

        # Station-local candidate events match the morphology of the supplied
        # Gauss array: a ~1.25 s FWHM Gaussian over a Pre valley, with nearly
        # station-constant peak height.  Their times remain independent across
        # stations, so they are decoys rather than injected vehicle tracks.
        for _ in range(
            int(rng.integers(self.settings.false_event_min, self.settings.false_event_max + 1))
        ):
            channel = int(rng.integers(0, station_count))
            center_s = float(rng.uniform(0.0, self.settings.window_s))
            pre_sigma_s = float(
                rng.uniform(
                    self.settings.pre_valley_sigma_min_s,
                    self.settings.pre_valley_sigma_max_s,
                )
            )
            gauss_sigma_s = float(
                rng.uniform(self.settings.gauss_sigma_min_s, self.settings.gauss_sigma_max_s)
            )
            self._add_feature_valley(
                pre[channel], center_s, pre_sigma_s, float(rng.uniform(0.90, 1.0))
            )
            self._add_feature_pulse(
                gauss[channel],
                center_s,
                gauss_sigma_s,
                float(gauss_peak_height[channel] * rng.uniform(0.985, 1.0)),
            )
            if rng.random() < 0.80:
                self._add_raw_wave_packet(
                    raw[channel],
                    center_s + float(rng.normal(0.0, 0.035)),
                    float(rng.uniform(0.16, 0.50)),
                    float(station_noise[channel, 0] * rng.uniform(0.08, 1.2)),
                    rng,
                )

        # Intermittent interference blocks affect one or a few adjacent stations.
        for _ in range(int(rng.integers(0, 7))):
            width_channels = int(rng.integers(1, 5))
            channel_start = int(rng.integers(0, max(1, station_count - width_channels + 1)))
            duration_s = float(rng.uniform(0.3, 4.0))
            start_s = float(rng.uniform(0.0, max(self.settings.window_s - duration_s, 0.0)))
            raw_start = int(start_s * self.settings.waveform_rate_hz)
            raw_end = min(sample_count, int((start_s + duration_s) * self.settings.waveform_rate_hz))
            feature_start = int(start_s * self.settings.feature_rate_hz)
            feature_end = min(time_bins, int((start_s + duration_s) * self.settings.feature_rate_hz))
            gain = float(rng.uniform(0.0, 0.25))
            raw[channel_start : channel_start + width_channels, raw_start:raw_end] *= gain
            pre[channel_start : channel_start + width_channels, feature_start:feature_end] *= gain
            gauss[channel_start : channel_start + width_channels, feature_start:feature_end] = 0.0

        return _ModalScene(
            raw=raw,
            pre=pre,
            gauss=gauss,
            quality=quality,
            raw_noise_scale=station_noise[:, 0].copy(),
            gauss_peak_height=gauss_peak_height,
        )

    def _add_raw_wave_packet(
        self,
        row: np.ndarray,
        center_s: float,
        sigma_s: float,
        amplitude: float,
        rng: np.random.Generator,
    ) -> None:
        rate = self.settings.waveform_rate_hz
        half_width = int(max(2, np.ceil(4.0 * sigma_s * rate)))
        center = int(round(center_s * rate))
        left = max(0, center - half_width)
        right = min(row.size, center + half_width + 1)
        if right <= left:
            return
        local_t = np.arange(left, right, dtype=np.float32) / rate - center_s
        envelope = np.exp(-0.5 * (local_t / max(sigma_s, 1e-5)) ** 2)
        carrier_hz = float(rng.uniform(3.0, min(42.0, 0.40 * rate)))
        second_hz = float(rng.uniform(2.0, min(30.0, 0.32 * rate)))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        second_phase = float(rng.uniform(0.0, 2.0 * np.pi))
        # A signed, zero-mean vibration burst.  The first implementation used a
        # positive hump and saturated the Raw input plane, unlike DAY11.
        carrier = np.cos(2.0 * np.pi * carrier_hz * local_t + phase)
        carrier += 0.35 * np.cos(2.0 * np.pi * second_hz * local_t + second_phase)
        carrier /= np.sqrt(0.5 * (1.0 + 0.35**2))
        row[left:right] += (amplitude * envelope * carrier).astype(np.float32)

    def _add_feature_pulse(
        self,
        row: np.ndarray,
        center_s: float,
        sigma_s: float,
        amplitude: float,
        *,
        rate: float | None = None,
    ) -> np.ndarray:
        rate = float(rate or self.settings.feature_rate_hz)
        half_width = int(max(1, np.ceil(4.0 * sigma_s * rate)))
        center = int(round(center_s * rate))
        left = max(0, center - half_width)
        right = min(row.size, center + half_width + 1)
        if right <= left:
            return np.empty(0, dtype=np.float32)
        local_t = np.arange(left, right, dtype=np.float32) / rate - center_s
        pulse = (amplitude * np.exp(-0.5 * (local_t / max(sigma_s, 1e-5)) ** 2)).astype(
            np.float32
        )
        row[left:right] = np.maximum(row[left:right], pulse)
        return pulse

    def _add_feature_valley(
        self, row: np.ndarray, center_s: float, sigma_s: float, depth: float
    ) -> np.ndarray:
        rate = self.settings.feature_rate_hz
        half_width = int(max(1, np.ceil(4.0 * sigma_s * rate)))
        center = int(round(center_s * rate))
        left = max(0, center - half_width)
        right = min(row.size, center + half_width + 1)
        if right <= left:
            return np.empty(0, dtype=np.float32)
        local_t = np.arange(left, right, dtype=np.float32) / rate - center_s
        valley = np.exp(-0.5 * (local_t / max(sigma_s, 1e-5)) ** 2).astype(np.float32)
        row[left:right] *= np.clip(1.0 - float(depth) * valley, 0.0, 1.0)
        return valley

    def _sample_times(
        self,
        rng: np.random.Generator,
        *,
        speed_kmh: float,
        intercept_s: float,
        direction: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate local speed over the actual (possibly nonuniform) station gaps."""
        positions = self.geometry.relative_positions_m.astype(np.float64)
        if len(positions) <= 1:
            return np.asarray([intercept_s]), np.empty(0)
        base = abs(float(speed_kmh))
        smooth = gaussian_filter1d(rng.normal(size=len(positions) - 1), sigma=2.0, mode="reflect")
        scale = max(float(np.max(np.abs(smooth))), 1e-6)
        local_speed = np.clip(base * (1.0 + 0.05 * smooth / scale), 60.0, 90.0)
        if not 60.0 <= base <= 90.0:
            # Hard negatives retain their off-range speed instead of being clipped into positives.
            local_speed = np.clip(base * (1.0 + 0.03 * smooth / scale), 30.0, 130.0)
        dt = np.diff(positions) / (local_speed / 3.6)
        cumulative = np.concatenate([[0.0], np.cumsum(dt)])
        times = intercept_s + float(direction) * cumulative
        return times.astype(np.float32), local_speed.astype(np.float32)

    def _dropout_mask(
        self, rng: np.random.Generator, valid: np.ndarray, *, positive: bool
    ) -> np.ndarray:
        observed = valid.copy()
        if not positive:
            return observed
        observed &= rng.random(observed.size) >= float(rng.uniform(0.05, 0.35))
        if rng.random() < 0.75 and np.any(valid):
            positions = self.geometry.relative_positions_m
            gap_m = float(rng.uniform(100.0, 600.0))
            start_m = float(rng.uniform(positions[0], max(positions[-1] - gap_m, positions[0])))
            observed[(positions >= start_m) & (positions <= start_m + gap_m)] = False
        # Keep enough observations for useful supervision whenever possible.
        valid_indices = np.flatnonzero(valid)
        if np.count_nonzero(observed) < min(5, valid_indices.size):
            restore = rng.choice(
                valid_indices,
                size=min(5, valid_indices.size),
                replace=False,
            )
            observed[restore] = True
        return observed

    def _inject_track(
        self,
        scene: _ModalScene,
        centerline: np.ndarray,
        slowness_target: np.ndarray,
        crossing: np.ndarray,
        *,
        times: np.ndarray,
        speed_kmh: float,
        rng: np.random.Generator,
        positive: bool,
    ) -> np.ndarray:
        valid = (times >= 0.0) & (times < self.settings.window_s)
        observed = self._dropout_mask(rng, valid, positive=positive)
        raw_sigma_s = float(rng.uniform(self.settings.raw_sigma_min_s, self.settings.raw_sigma_max_s))
        vehicle_snr = float(
            np.exp(
                rng.uniform(
                    np.log(self.settings.vehicle_snr_min),
                    np.log(self.settings.vehicle_snr_max),
                )
            )
        )
        pre_sigma_s = float(
            rng.uniform(
                self.settings.pre_valley_sigma_min_s,
                self.settings.pre_valley_sigma_max_s,
            )
        )
        gauss_sigma_s = float(
            rng.uniform(self.settings.gauss_sigma_min_s, self.settings.gauss_sigma_max_s)
        )
        slope = 3.6 / max(abs(speed_kmh), 1e-6)
        slope_normalized = float(np.clip((slope - 0.04) / 0.02, 0.0, 1.0))

        # The dense target includes recoverable missing stations; inputs only contain observed ones.
        if positive:
            target_width = float(rng.uniform(0.08, 0.16))
            for channel in np.flatnonzero(valid):
                existing = centerline[0, channel].copy()
                target_pulse = np.zeros(self._time_bins, dtype=np.float32)
                self._add_feature_pulse(target_pulse, float(times[channel]), target_width, 1.0)
                centerline[0, channel] = np.maximum(existing, target_pulse)
                crossing[0, channel] = np.maximum(
                    crossing[0, channel],
                    ((existing > 0.25) & (target_pulse > 0.25)).astype(np.float32),
                )
                slowness_target[0, channel, target_pulse > 0.20] = slope_normalized

        for channel in np.flatnonzero(observed):
            time_s = float(times[channel])
            channel_amp = (
                float(scene.raw_noise_scale[channel])
                * vehicle_snr
                * float(rng.lognormal(mean=0.0, sigma=0.22))
            )
            self._add_raw_wave_packet(
                scene.raw[channel], time_s, raw_sigma_s, channel_amp, rng
            )
            self._add_feature_valley(
                scene.pre[channel],
                time_s,
                pre_sigma_s * float(rng.uniform(0.92, 1.08)),
                float(rng.uniform(0.92, 1.0)),
            )
            self._add_feature_pulse(
                scene.gauss[channel],
                time_s + float(rng.normal(0.0, 0.012)),
                gauss_sigma_s * float(rng.uniform(0.97, 1.03)),
                float(scene.gauss_peak_height[channel] * rng.uniform(0.985, 1.0)),
            )
        return observed.astype(np.float32)

    def _sample_positive_tracks(
        self, rng: np.random.Generator, count: int
    ) -> list[tuple[float, float]]:
        """Return (speed, intercept) pairs, including overtaking/near-parallel scenes."""
        span = float(self.geometry.relative_positions_m[-1])
        tracks: list[tuple[float, float]] = []
        attempts = 0
        while len(tracks) < count and attempts < max(100, 100 * count):
            attempts += 1
            speed = float(rng.uniform(self.settings.min_speed_kmh, self.settings.max_speed_kmh))
            slope = 3.6 / speed
            intercept = float(rng.uniform(-slope * span, self.settings.window_s))
            if tracks and rng.random() < self.settings.interaction_probability:
                other_speed, other_intercept = tracks[int(rng.integers(0, len(tracks)))]
                mode = int(rng.integers(0, 2))
                if mode == 0:
                    # Same-direction intersection (overtake) at a random physical position.
                    cross_x = float(rng.uniform(0.1 * span, 0.9 * span))
                    speed = float(rng.uniform(self.settings.min_speed_kmh, self.settings.max_speed_kmh))
                    if abs(speed - other_speed) < 5.0:
                        speed = float(
                            np.clip(
                                other_speed + (7.0 if other_speed < 75.0 else -7.0),
                                self.settings.min_speed_kmh,
                                self.settings.max_speed_kmh,
                            )
                        )
                    intercept = other_intercept + self.settings.motion_direction * (
                        3.6 / other_speed - 3.6 / speed
                    ) * cross_x
                else:
                    speed = float(
                        np.clip(
                            other_speed + rng.uniform(-3.0, 3.0),
                            self.settings.min_speed_kmh,
                            self.settings.max_speed_kmh,
                        )
                    )
                    intercept = float(other_intercept + rng.choice([-1.0, 1.0]) * rng.uniform(0.15, 1.2))
            nominal = (
                intercept
                + self.settings.motion_direction
                * 3.6
                / speed
                * self.geometry.relative_positions_m
            )
            if np.count_nonzero((nominal >= 0.0) & (nominal < self.settings.window_s)) >= 5:
                tracks.append((speed, intercept))
        return tracks

    def _finalize_features(self, scene: _ModalScene, bad_channels: np.ndarray) -> np.ndarray:
        bins_per_feature = int(round(self.settings.waveform_rate_hz / self.settings.feature_rate_hz))
        if scene.raw_feature is not None:
            # Peak-set scenes already carry the low-rate envelope explicitly;
            # preserve its positive Gaussian shape instead of applying a
            # per-station z-score that would turn a clean line into speckle.
            raw_plane = np.clip(scene.raw_feature, 0.0, 1.0).astype(np.float32, copy=True)
        else:
            raw = scene.raw[:, : self._time_bins * bins_per_feature]
            rms = np.sqrt(np.mean(raw.reshape(len(self.geometry), self._time_bins, -1) ** 2, axis=2))
            raw_log = np.log1p(rms)
            median = np.median(raw_log, axis=1, keepdims=True)
            mad = np.median(np.abs(raw_log - median), axis=1, keepdims=True)
            raw_z = (raw_log - median) / np.maximum(1.4826 * mad, 1e-5)
            raw_plane = np.clip(raw_z / 6.0, -1.0, 1.0).astype(np.float32)

        scene.pre[:] = np.clip(scene.pre, 0.0, 1.0)
        scene.gauss[:] = np.clip(scene.gauss, 0.0, 1.0)
        if bad_channels.size:
            raw_plane[bad_channels] = 0.0
            scene.pre[bad_channels] = 0.0
            scene.gauss[bad_channels] = 0.0
            scene.quality[bad_channels] = 0.0
        coordinates = self.geometry.relative_positions_m.astype(np.float32)
        coordinates /= max(float(coordinates[-1]), 1.0)
        features = np.stack(
            [
                raw_plane,
                scene.pre,
                scene.gauss,
                np.broadcast_to(scene.quality[:, None], raw_plane.shape),
                np.broadcast_to(coordinates[:, None], raw_plane.shape),
            ],
            axis=0,
        )
        return np.asarray(features, dtype=np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._is_peakset_profile:
            return self._getitem_peakset(index)
        rng = np.random.default_rng(self._scene_seed(index))
        scene = self._synthetic_background(rng)
        station_count = len(self.geometry)
        centerline = np.zeros((1, station_count, self._time_bins), dtype=np.float32)
        slowness = np.zeros_like(centerline)
        crossing = np.zeros_like(centerline)
        params = np.zeros((self.settings.max_vehicles, 3), dtype=np.float32)
        track_times = np.full((self.settings.max_vehicles, station_count), np.nan, dtype=np.float32)
        track_observed = np.zeros((self.settings.max_vehicles, station_count), dtype=np.float32)

        if rng.random() < 0.12:
            vehicle_count = 0
        elif rng.random() < 0.16:
            vehicle_count = int(rng.integers(0, self.settings.max_vehicles + 1))
        else:
            vehicle_count = int(min(rng.poisson(self.settings.vehicle_rate), self.settings.max_vehicles))

        for vehicle_index, (speed, intercept) in enumerate(
            self._sample_positive_tracks(rng, vehicle_count)
        ):
            times, _ = self._sample_times(
                rng,
                speed_kmh=speed,
                intercept_s=intercept,
                direction=self.settings.motion_direction,
            )
            observed = self._inject_track(
                scene,
                centerline,
                slowness,
                crossing,
                times=times,
                speed_kmh=speed,
                rng=rng,
                positive=True,
            )
            params[vehicle_index] = (
                self.settings.motion_direction * 3.6 / speed,
                intercept,
                1.0,
            )
            track_times[vehicle_index] = times
            track_observed[vehicle_index] = observed

        # Off-range and reverse-time tracks are visible in the inputs but have no positive label.
        span = float(self.geometry.relative_positions_m[-1])
        for _ in range(int(rng.integers(1, 6))):
            mode = int(rng.integers(0, 3))
            if mode == 0:
                speed = float(rng.uniform(45.0, 59.0))
                direction = self.settings.motion_direction
            elif mode == 1:
                speed = float(rng.uniform(91.0, 110.0))
                direction = self.settings.motion_direction
            else:
                speed = float(rng.uniform(60.0, 90.0))
                direction = -self.settings.motion_direction
            travel_s = 3.6 / speed * span
            intercept = float(
                rng.uniform(-travel_s, self.settings.window_s + travel_s)
                if direction > 0
                else rng.uniform(0.0, self.settings.window_s + travel_s)
            )
            times, _ = self._sample_times(
                rng, speed_kmh=speed, intercept_s=intercept, direction=direction
            )
            self._inject_track(
                scene,
                centerline,
                slowness,
                crossing,
                times=times,
                speed_kmh=speed,
                rng=rng,
                positive=False,
            )

        bad_count = int(rng.integers(1, 5)) if rng.random() < 0.35 else 0
        bad_channels = (
            np.sort(rng.choice(station_count, size=bad_count, replace=False)).astype(np.int64)
            if bad_count
            else np.empty(0, dtype=np.int64)
        )
        if bad_channels.size:
            track_observed[:, bad_channels] = 0.0
        features = self._finalize_features(scene, bad_channels)
        result: dict[str, torch.Tensor] = {
            "input": torch.from_numpy(features),
            "centerline": torch.from_numpy(centerline),
            "slowness": torch.from_numpy(slowness),
            "crossing": torch.from_numpy(crossing),
            "track_params": torch.from_numpy(params),
            "track_times": torch.from_numpy(track_times),
            "track_observed": torch.from_numpy(track_observed),
        }
        if self.settings.return_modalities:
            result.update(
                {
                    "raw_waveform": torch.from_numpy(scene.raw.copy()),
                    "pre_feature": torch.from_numpy(scene.pre.copy()),
                    "gauss_feature": torch.from_numpy(scene.gauss.copy()),
                    "quality_vector": torch.from_numpy(scene.quality.copy()),
                    "bad_channel_mask": torch.from_numpy(
                        np.isin(np.arange(station_count), bad_channels).astype(np.float32)
                    ),
                }
            )
        return result
