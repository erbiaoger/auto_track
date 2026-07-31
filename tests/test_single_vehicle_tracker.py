import unittest

import numpy as np
import torch

from autotrack.core.single_vehicle_tracker import SingleVehicleTrackerConfig, extract_single_vehicle_track


def _add_peak(data: np.ndarray, ch: int, t_idx: int, amp: float) -> None:
    data[int(ch), int(t_idx)] = float(amp)


class SingleVehicleTrackerTest(unittest.TestCase):
    def test_extracts_single_track_on_50_channel_crossing_with_gap(self) -> None:
        fs = 100.0
        dx_m = 10.0
        n_channels = 50
        n_samples = 6000
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        prior = np.zeros_like(data)

        for ch in range(n_channels):
            if 18 <= ch <= 20:
                continue
            t = int(700 + 42 * ch + 0.6 * max(0, ch - 24) * (ch - 24))
            _add_peak(data, ch, t, 5.0)
            _add_peak(data, ch, max(0, t - 1), 3.5)
            _add_peak(data, ch, min(n_samples - 1, t + 1), 3.5)
            prior[ch, t] = 1.0
            prior[ch, max(0, t - 1)] = 0.5
            prior[ch, min(n_samples - 1, t + 1)] = 0.5

        for ch in range(n_channels):
            t = int(1500 + 30 * ch)
            _add_peak(data, ch, t, 6.5)
            _add_peak(data, ch, max(0, t - 1), 4.0)
            _add_peak(data, ch, min(n_samples - 1, t + 1), 4.0)

        for ch, t in [(8, 2600), (12, 2605), (30, 1200), (41, 5000)]:
            _add_peak(data, ch, t, 8.0)

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            50.0,
            140.0,
            SingleVehicleTrackerConfig(
                candidate_prominence=0.08,
                candidate_min_distance=12,
                candidate_max_peaks_per_channel=48,
                max_skip_channels=8,
                min_track_channels=30,
                min_track_score=10.0,
                kalman_bridge_gap_channels=12,
                kalman_gate_seconds=0.35,
                candidate_hypotheses=4,
                hypothesis_prior_weight=4.0,
            ),
            prior_heatmap=prior,
            prior_weight=2.0,
        )

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        self.assertGreaterEqual(len(track.points), 35)
        self.assertLess(abs(track.mean_speed_kmh - 81.0), 12.0)
        self.assertLess(len({p.ch_idx for p in track.points}), 50)

    def test_relaxed_fallback_recovers_low_contrast_track(self) -> None:
        fs = 100.0
        dx_m = 10.0
        n_channels = 50
        n_samples = 4000
        data = np.zeros((n_channels, n_samples), dtype=np.float32)
        prior = np.zeros_like(data)
        for ch in range(n_channels):
            t = int(400 + 26 * ch)
            _add_peak(data, ch, t, 0.22)
            prior[ch, t] = 1.0

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            40.0,
            160.0,
            SingleVehicleTrackerConfig(
                candidate_prominence=0.25,
                candidate_min_distance=40,
                candidate_max_peaks_per_channel=16,
                max_skip_channels=4,
                min_track_channels=24,
                min_track_score=12.0,
                candidate_hypotheses=2,
                hypothesis_prior_weight=3.0,
            ),
            prior_heatmap=prior,
            prior_weight=1.5,
        )

        self.assertEqual(len(tracks), 1)
        self.assertGreaterEqual(len(tracks[0].points), 24)

    def test_extracts_one_continuous_track_across_a_small_gap(self) -> None:
        fs = 100.0
        dx_m = 10.0
        step = 45
        data = np.zeros((9, 700), dtype=np.float32)
        for ch in range(9):
            if ch in {4, 5}:
                continue
            _add_peak(data, ch, 100 + ch * step, 5.0)
        # Decoy peaks that should not win the continuity/physics score.
        _add_peak(data, 1, 40, 8.0)
        _add_peak(data, 7, 580, 8.0)
        diagnostics: dict[str, object] = {}

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            60.0,
            100.0,
            SingleVehicleTrackerConfig(
                candidate_prominence=0.1,
                candidate_min_distance=10,
                kalman_bridge_gap_channels=4,
                kalman_gate_seconds=0.4,
            ),
            diagnostics,
        )

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        channels = [point.ch_idx for point in track.points]
        self.assertEqual(channels, list(range(9)))
        self.assertGreaterEqual(int(diagnostics["final_point_count"]), 9)
        self.assertGreaterEqual(int(diagnostics["seed_point_count"]), 4)
        self.assertTrue(all(a < b for a, b in zip([p.time_s for p in track.points[:-1]], [p.time_s for p in track.points[1:]])))
        self.assertTrue(60.0 <= track.mean_speed_kmh <= 100.0)

    def test_rejects_speed_inconsistent_competing_branch(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 700), dtype=np.float32)
        target_step = 45
        bad_step = 30
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * target_step, 5.0)
            _add_peak(data, ch, 150 + ch * bad_step, 8.0)

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            60.0,
            100.0,
            {"candidate_prominence": 0.1, "candidate_min_distance": 10},
        )

        self.assertEqual(len(tracks), 1)
        channels = [point.ch_idx for point in tracks[0].points]
        self.assertEqual(channels, list(range(9)))
        # The recovered line should stay near the slower physical branch.
        t0 = tracks[0].points[0].t_idx
        t8 = tracks[0].points[-1].t_idx
        self.assertGreater(t8 - t0, 300)
        self.assertLess(t8 - t0, 500)

    def test_prior_heatmap_can_override_stronger_decoy_branch(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 700), dtype=np.float32)
        prior = np.zeros_like(data)
        true_step = 45
        decoy_step = 30
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * true_step, 4.0)
            _add_peak(data, ch, 150 + ch * decoy_step, 8.0)
            prior[ch, 100 + ch * true_step] = 1.0
            prior[ch, min(data.shape[1] - 1, 100 + ch * true_step + 1)] = 0.4
            prior[ch, max(0, 100 + ch * true_step - 1)] = 0.4

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            60.0,
            100.0,
            {
                "candidate_prominence": 0.1,
                "candidate_min_distance": 10,
                "min_track_channels": 6,
                "min_track_score": 5.0,
            },
            prior_heatmap=prior,
            prior_weight=2.0,
        )

        self.assertEqual(len(tracks), 1)
        channels = [point.ch_idx for point in tracks[0].points]
        self.assertEqual(channels, list(range(9)))
        t0 = tracks[0].points[0].t_idx
        t8 = tracks[0].points[-1].t_idx
        self.assertGreater(t8 - t0, 300)
        self.assertLess(t8 - t0, 500)

    def test_physics_score_prefers_smoother_branch_over_stronger_kinked_branch(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((12, 900), dtype=np.float32)
        smooth_step = 42
        kink_step_a = 28
        kink_step_b = 58
        for ch in range(12):
            smooth_t = 120 + ch * smooth_step
            _add_peak(data, ch, smooth_t, 4.0)
            if ch < 6:
                kink_t = 180 + ch * kink_step_a
            else:
                kink_t = 180 + 6 * kink_step_a + (ch - 6) * kink_step_b
            _add_peak(data, ch, kink_t, 8.0)

        tracks = extract_single_vehicle_track(
            data,
            fs,
            dx_m,
            "forward",
            55.0,
            100.0,
            SingleVehicleTrackerConfig(
                candidate_prominence=0.1,
                candidate_min_distance=8,
                candidate_max_peaks_per_channel=16,
                max_skip_channels=4,
                min_track_channels=8,
                min_track_score=6.0,
                candidate_hypotheses=4,
                hypothesis_prior_weight=1.0,
                trajectory_fit_weight=2.0,
                trajectory_curvature_weight=0.5,
            ),
        )

        self.assertEqual(len(tracks), 1)
        track = tracks[0]
        times = [p.t_idx for p in track.points]
        self.assertEqual([p.ch_idx for p in track.points], list(range(12)))
        self.assertLess(max(abs(a - b) for a, b in zip(np.diff(times), [smooth_step] * 11)), 20)
        self.assertLess(abs(track.mean_speed_kmh - 85.7), 15.0)

    def test_wide_fallback_speed_window_recovers_track(self) -> None:
        fs = 100.0
        dx_m = 10.0
        data = np.zeros((9, 700), dtype=np.float32)
        for ch in range(9):
            _add_peak(data, ch, 100 + ch * 45, 5.0)

        from autotrack.dl.single_vehicle_net import predict_single_vehicle_track

        class _StubModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                heatmap_logits = x[:, 0] * 8.0
                objectness_logits = torch.full((x.shape[0],), 10.0, dtype=torch.float32, device=x.device)
                direction_logits = torch.tensor([[10.0, 0.0]], dtype=torch.float32, device=x.device).repeat(x.shape[0], 1)
                speed = torch.full((x.shape[0],), 1.4, dtype=torch.float32, device=x.device)
                return {
                    "heatmap_logits": heatmap_logits,
                    "objectness_logits": objectness_logits,
                    "direction_logits": direction_logits,
                    "speed": speed,
                }

        model = _StubModel()
        tracks = predict_single_vehicle_track(
            model,
            data,
            fs,
            dx_m,
            "forward",
            5.0,
            10.0,
            {"time_downsample": 1, "min_visible_channels": 3, "prior_weight": 1.0},
            device="cpu",
        )

        self.assertEqual(len(tracks), 1)
        channels = [p.ch_idx for p in tracks[0].points]
        self.assertGreaterEqual(len(channels), 7)
        self.assertTrue(all(b - a == 1 for a, b in zip(channels[:-1], channels[1:])))


if __name__ == "__main__":
    unittest.main()
