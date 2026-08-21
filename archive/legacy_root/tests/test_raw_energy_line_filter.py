from __future__ import annotations

import numpy as np

from autotrack.dl.raw_energy_line_filter import (
    LineCandidate,
    compute_energy_envelope,
    gaussian_curve_from_picks,
    match_peaks_to_lines,
    pick_probability_peaks,
)


def test_pick_probability_peaks_merges_close_segments() -> None:
    probability = np.zeros(100, dtype=np.float32)
    probability[10:14] = [0.5, 0.8, 0.7, 0.4]
    probability[16:20] = [0.6, 0.9, 0.7, 0.5]
    probability[60:64] = [0.3, 0.65, 0.55, 0.2]

    times, scores = pick_probability_peaks(
        probability,
        fs_hz=10.0,
        threshold=0.4,
        min_gap_s=1.0,
        normalize=False,
    )

    assert times.tolist() == [1.7, 6.1]
    assert np.allclose(scores, [0.9, 0.65])


def test_match_peaks_to_lines_keeps_only_diagonal_candidates() -> None:
    line = LineCandidate(score=10.0, slope_s_per_station=4.0, intercept_s=10.0, valid_station_count=5)
    candidate_times = [
        np.array([10.0, 30.0]),
        np.array([14.0, 32.0]),
        np.array([18.0]),
        np.array([22.0, 50.0]),
        np.array([26.0]),
    ]
    candidate_scores = [
        np.array([0.8, 0.9]),
        np.array([0.8, 0.4]),
        np.array([0.7]),
        np.array([0.6, 0.95]),
        np.array([0.9]),
    ]

    selected, scores, line_ids = match_peaks_to_lines(
        candidate_times,
        candidate_scores,
        [line],
        tolerance_s=0.1,
    )

    assert [value.tolist() for value in selected] == [[10.0], [14.0], [18.0], [22.0], [26.0]]
    assert np.allclose(scores[3], [0.6])
    assert line_ids[0].tolist() == [0]


def test_energy_envelope_and_gaussian_curve_have_expected_shapes() -> None:
    signal = np.zeros(100, dtype=np.float32)
    signal[40:45] = 3.0
    envelope = compute_energy_envelope(signal, fs_hz=10.0, window_s=1.0)
    curve = gaussian_curve_from_picks(signal, np.array([4.2]), fs_hz=10.0, width_s=0.5)

    assert envelope.shape == (10,)
    assert curve.shape == signal.shape
    assert float(curve.max()) > 0.0
