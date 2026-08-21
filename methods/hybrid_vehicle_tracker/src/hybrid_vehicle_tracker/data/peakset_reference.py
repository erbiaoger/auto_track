"""Statistics-only calibration for the DAY11 Gauss peak-set simulator.

The simulator deliberately does not load the measured DAY11 array.  The
numbers below were computed once from ``gauss_station_fallback_DAY11.npy``
``[:120000]`` with a 0.5 peak-height and 1.25 s minimum-distance gate.  They
are kept as small, reviewable constants so that synthetic scenes can match the
observed peak morphology without copying any real waveform or event time.
"""

from __future__ import annotations

import numpy as np


DAY11_GAUSS_REFERENCE_VERSION = "DAY11[:120000]_gauss_peak_stats_v1"
DAY11_GAUSS_REFERENCE_PATH = (
    "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/"
    "converted_50ch_station_fallback_q65_s8/gauss_station_fallback_DAY11.npy"
)

# Per-channel strong-event counts in the 120 s audit window.  These are used
# only as a sampling weight; the simulator never reuses the measured times or
# amplitudes.
DAY11_STATION_EVENT_COUNTS = np.asarray(
    [
        7,
        5,
        9,
        2,
        7,
        5,
        5,
        7,
        4,
        1,
        7,
        3,
        4,
        4,
        1,
        9,
        5,
        1,
        7,
        3,
        7,
        7,
        4,
        12,
        16,
        10,
        11,
        8,
        9,
        8,
        4,
        10,
        7,
        7,
        5,
        8,
        9,
        8,
        6,
        14,
        7,
        8,
        11,
        13,
        7,
        13,
        7,
        10,
        13,
        8,
    ],
    dtype=np.float64,
)

DAY11_GAUSS_HEIGHT_QUANTILES = np.asarray(
    [0.62119, 0.64611, 0.65540, 0.67166, 0.67711, 0.68568, 0.68944, 0.69283],
    dtype=np.float64,
)
DAY11_GAUSS_FWHM_QUANTILES_S = np.asarray(
    [1.17700, 1.17700, 1.17700, 1.17700, 1.17838], dtype=np.float64
)
DAY11_ISOLATED_EVENT_FRACTION = 0.3361


def station_sampling_weights(station_count: int) -> np.ndarray:
    """Return empirical station occupancy weights for any station count.

    The real mapping has 50 stations.  Interpolation keeps this utility useful
    in unit tests with a small synthetic geometry while preserving the broad
    high-occupancy middle/end stations visible in DAY11.
    """

    count = int(station_count)
    if count <= 0:
        raise ValueError("station_count must be positive")
    if count == DAY11_STATION_EVENT_COUNTS.size:
        weights = DAY11_STATION_EVENT_COUNTS.copy()
    else:
        source_x = np.linspace(0.0, 1.0, DAY11_STATION_EVENT_COUNTS.size)
        target_x = np.linspace(0.0, 1.0, count)
        weights = np.interp(target_x, source_x, DAY11_STATION_EVENT_COUNTS)
    weights = np.clip(weights, 0.05, None)
    return weights / weights.sum()


def reference_summary() -> dict[str, object]:
    """Return JSON-safe metadata for manifests and reports."""

    return {
        "version": DAY11_GAUSS_REFERENCE_VERSION,
        "source_path": DAY11_GAUSS_REFERENCE_PATH,
        "source_window": "[:120000]",
        "source_samples": 120000,
        "source_sample_rate_hz": 1000.0,
        "strong_peak_height_gate": 0.5,
        "minimum_peak_distance_s": 1.25,
        "strong_event_count_total": int(DAY11_STATION_EVENT_COUNTS.sum()),
        "strong_event_count_per_station_mean": float(DAY11_STATION_EVENT_COUNTS.mean()),
        "strong_event_count_per_station_median": float(np.median(DAY11_STATION_EVENT_COUNTS)),
        "station_event_counts": DAY11_STATION_EVENT_COUNTS.astype(int).tolist(),
        "gauss_height_quantiles": DAY11_GAUSS_HEIGHT_QUANTILES.tolist(),
        "gauss_fwhm_quantiles_s": DAY11_GAUSS_FWHM_QUANTILES_S.tolist(),
        "isolated_event_fraction": DAY11_ISOLATED_EVENT_FRACTION,
        "real_samples_used_for_training": False,
    }
