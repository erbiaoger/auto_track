import unittest
import sys
import types

import torch

if "obspy" not in sys.modules:
    obspy_stub = types.ModuleType("obspy")
    obspy_stub.read = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    sys.modules["obspy"] = obspy_stub

from autotrack.dl.convert_track_slot_to_peak_slot import _convert_one_sample
from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_input


class PeakSlotCandidateTest(unittest.TestCase):
    def test_raw_and_prior_sources_keep_sorted_shape(self) -> None:
        x = torch.zeros((2, 2, 80), dtype=torch.float32)
        x[0, 0, 10] = 0.7
        x[0, 0, 30] = 0.9
        x[1, 1, 20] = 0.8
        cfg = PeakDetectionConfig(candidates_per_channel=4, min_height=0.05, prominence=0.01)

        raw_time, raw_amp, raw_valid, raw_index = detect_peak_candidates_from_input(
            x,
            fs=10.0,
            time_downsample=1,
            window_samples=80,
            config=cfg,
        )
        prior_time, prior_amp, prior_valid, prior_index = detect_peak_candidates_from_input(
            x,
            fs=10.0,
            time_downsample=1,
            window_samples=80,
            config=PeakDetectionConfig(
                candidates_per_channel=4,
                min_height=0.05,
                prominence=0.01,
                candidate_source="prior",
                prior_min_height=0.05,
                prior_prominence=0.01,
            ),
        )

        self.assertEqual(tuple(raw_time.shape), (2, 4))
        self.assertEqual(raw_index[0, raw_valid[0]].tolist(), [10, 30])
        self.assertEqual(prior_index[1, prior_valid[1]].tolist(), [20])
        self.assertTrue(torch.all(raw_amp[0, raw_valid[0]] > 0))
        self.assertTrue(torch.all(prior_amp[1, prior_valid[1]] > 0))

    def test_raw_prior_union_merges_close_peaks_and_keeps_separate_vehicle(self) -> None:
        x = torch.zeros((2, 1, 80), dtype=torch.float32)
        x[0, 0, 10] = 0.50
        x[1, 0, 12] = 1.00
        x[1, 0, 30] = 0.90
        cfg = PeakDetectionConfig(
            candidates_per_channel=4,
            min_height=0.05,
            prominence=0.01,
            candidate_source="raw_prior_union",
            prior_min_height=0.05,
            prior_prominence=0.01,
            merge_tolerance_s=0.20,
            prior_score_scale=0.85,
        )

        _, amp, valid, index, stats = detect_peak_candidates_from_input(
            x,
            fs=10.0,
            time_downsample=1,
            window_samples=80,
            config=cfg,
            return_stats=True,
        )

        self.assertEqual(index[0, valid[0]].tolist(), [12, 30])
        self.assertAlmostEqual(float(amp[0, 0].item()), 0.85, places=5)
        self.assertEqual(stats["raw_candidate_count"], 1)
        self.assertEqual(stats["prior_candidate_count"], 2)
        self.assertEqual(stats["merged_candidate_count"], 2)

    def test_union_truncates_by_score_then_returns_time_sorted(self) -> None:
        x = torch.zeros((2, 1, 80), dtype=torch.float32)
        x[0, 0, 10] = 0.30
        x[0, 0, 50] = 0.80
        x[1, 0, 30] = 1.00
        cfg = PeakDetectionConfig(
            candidates_per_channel=2,
            min_height=0.05,
            prominence=0.01,
            candidate_source="raw_prior_union",
            prior_min_height=0.05,
            prior_prominence=0.01,
            merge_tolerance_s=0.10,
        )

        _, _, valid, index = detect_peak_candidates_from_input(
            x,
            fs=10.0,
            time_downsample=1,
            window_samples=80,
            config=cfg,
        )

        self.assertEqual(index[0, valid[0]].tolist(), [30, 50])

    def test_gt_injection_preserves_valid_peak_index(self) -> None:
        x = torch.zeros((2, 2, 80), dtype=torch.float32)
        x[0, 0, 10] = 0.9
        time_label = torch.full((2, 2), 0.0, dtype=torch.float32)
        time_label[0, 0] = 10.0 / 79.0
        time_label[0, 1] = 35.0 / 79.0
        visibility = torch.zeros((2, 2), dtype=torch.float32)
        visibility[0] = 1.0
        gt_valid = torch.tensor([True, False])
        cfg = PeakDetectionConfig(
            candidates_per_channel=4,
            min_height=0.05,
            prominence=0.01,
            candidate_source="raw_prior_union",
            prior_min_height=0.05,
            prior_prominence=0.01,
            match_tolerance_s=0.25,
        )

        peak_time, _, peak_valid, _, gt_peak_index, _, injected = _convert_one_sample(
            x,
            time_label,
            visibility,
            gt_valid,
            fs=10.0,
            time_downsample=1,
            window_samples=80,
            peak_cfg=cfg,
        )

        self.assertEqual(injected, 1)
        self.assertLess(int(gt_peak_index[0, 0].item()), 4)
        self.assertLess(int(gt_peak_index[0, 1].item()), 4)
        self.assertTrue(bool(peak_valid[1, int(gt_peak_index[0, 1].item())].item()))
        self.assertAlmostEqual(float(peak_time[1, int(gt_peak_index[0, 1].item())].item()), 35.0 / 79.0, places=5)


if __name__ == "__main__":
    unittest.main()
