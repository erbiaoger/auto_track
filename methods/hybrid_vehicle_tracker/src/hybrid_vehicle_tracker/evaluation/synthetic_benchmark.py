from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

from hybrid_vehicle_tracker.association.pipeline import associate_observations
from hybrid_vehicle_tracker.config import TrackerConfig
from hybrid_vehicle_tracker.data.synthetic import SyntheticSettings, SyntheticVehicleDataset
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel
from hybrid_vehicle_tracker.models.physical_hough import PhysicalHoughHead
from hybrid_vehicle_tracker.types import StationGeometry, VehicleObservation, VehicleTrack


@dataclass
class BenchmarkMetrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    speed_absolute_errors_kmh: list[float] | None = None

    def __post_init__(self) -> None:
        if self.speed_absolute_errors_kmh is None:
            self.speed_absolute_errors_kmh = []

    @property
    def precision(self) -> float:
        return self.true_positives / max(self.true_positives + self.false_positives, 1)

    @property
    def recall(self) -> float:
        return self.true_positives / max(self.true_positives + self.false_negatives, 1)

    @property
    def f1(self) -> float:
        return 2.0 * self.precision * self.recall / max(self.precision + self.recall, 1e-12)

    def to_dict(self) -> dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "track_f1": self.f1,
            "speed_mae_kmh": (
                float(np.mean(self.speed_absolute_errors_kmh))
                if self.speed_absolute_errors_kmh
                else None
            ),
            "matched_tracks": len(self.speed_absolute_errors_kmh or []),
        }


def _observations_from_features(
    features: np.ndarray,
    geometry: StationGeometry,
    config: TrackerConfig,
    *,
    network: np.ndarray,
    crossing: np.ndarray,
    embedding: np.ndarray | None,
    use_network_candidates: bool,
) -> list[VehicleObservation]:
    rate = config.data.feature_rate_hz
    raw_score = 1.0 / (1.0 + np.exp(-6.0 * features[0]))
    rows: list[VehicleObservation] = []
    distance = max(1, int(round(config.association.candidate_min_distance_s * rate)))
    for channel, station in enumerate(geometry.stations):
        pre_event_score = 1.0 - features[1, channel]
        candidate_signal = np.maximum(pre_event_score, features[2, channel])
        if use_network_candidates:
            candidate_signal = np.maximum(candidate_signal, network[channel])
        peaks, _ = find_peaks(candidate_signal, height=0.52, distance=distance)
        for time_bin in peaks:
            emb = (
                tuple(float(value) for value in embedding[:, channel, time_bin])
                if embedding is not None
                else ()
            )
            gauss_score = float(features[2, channel, time_bin])
            rows.append(
                VehicleObservation(
                    observation_id=-1,
                    channel_index=channel,
                    station_id=station.station_id,
                    position_m=station.position_m,
                    time_s=float(time_bin / rate),
                    gauss_score=gauss_score,
                    pre_score=float(pre_event_score[time_bin]),
                    raw_energy=float(raw_score[channel, time_bin]),
                    network_score=float(network[channel, time_bin]),
                    crossing_score=float(crossing[channel, time_bin]),
                    strong=gauss_score >= config.association.strong_gauss_threshold,
                    embedding=emb,
                )
            )
    rows.sort(key=lambda item: item.evidence_score, reverse=True)
    rows = rows[: config.association.max_candidates]
    rows.sort(key=lambda item: (item.position_m, item.time_s))
    for observation_id, observation in enumerate(rows):
        observation.observation_id = observation_id
    return rows


def _ground_truth_tracks(
    scene: dict[str, torch.Tensor],
    geometry: StationGeometry,
    config: TrackerConfig,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    times = scene["track_times"].numpy()
    observed = scene["track_observed"].numpy() > 0.5
    params = scene["track_params"].numpy()
    result = []
    for track_index in np.flatnonzero(params[:, 2] > 0.5):
        visible = (
            observed[track_index]
            & np.isfinite(times[track_index])
            & (times[track_index] >= 0.0)
            & (times[track_index] < config.data.duration_s)
        )
        channels = np.flatnonzero(visible)
        if channels.size < config.association.min_observations:
            continue
        positions = geometry.positions_m[channels]
        if positions[-1] - positions[0] < config.association.min_span_m:
            continue
        slope, _ = np.polyfit(
            positions - positions[0], times[track_index, channels], deg=1
        )
        speed = float(3.6 / abs(slope))
        result.append((channels, times[track_index, channels], speed))
    return result


def _match_scene(
    predicted: list[VehicleTrack],
    truth: list[tuple[np.ndarray, np.ndarray, float]],
    metrics: BenchmarkMetrics,
    *,
    time_tolerance_s: float = 0.5,
) -> None:
    if not predicted or not truth:
        metrics.false_positives += len(predicted)
        metrics.false_negatives += len(truth)
        return
    costs = np.full((len(predicted), len(truth)), 1e3, dtype=np.float64)
    speed_errors = np.full_like(costs, np.nan)
    for pred_index, track in enumerate(predicted):
        pred_by_channel = {point.channel_index: point.time_s for point in track.points}
        for truth_index, (channels, times, speed) in enumerate(truth):
            residuals = [
                abs(pred_by_channel[int(channel)] - float(time_s))
                for channel, time_s in zip(channels, times)
                if int(channel) in pred_by_channel
            ]
            if len(residuals) >= 3:
                costs[pred_index, truth_index] = float(np.median(residuals))
                speed_errors[pred_index, truth_index] = abs(track.median_speed_kmh - speed)
    row_indices, column_indices = linear_sum_assignment(costs)
    matched = [
        (row, column)
        for row, column in zip(row_indices, column_indices)
        if costs[row, column] <= time_tolerance_s
    ]
    metrics.true_positives += len(matched)
    metrics.false_positives += len(predicted) - len(matched)
    metrics.false_negatives += len(truth) - len(matched)
    metrics.speed_absolute_errors_kmh.extend(
        float(speed_errors[row, column]) for row, column in matched
    )


def run_synthetic_benchmark(
    model: HybridPerceptionModel,
    geometry: StationGeometry,
    config: TrackerConfig,
    *,
    samples: int = 24,
    seed: int = 20260724 + 999983,
    device: torch.device | None = None,
) -> dict:
    device = device or torch.device("cpu")
    window_s = float(config.data.duration_s)
    benchmark_config = copy.deepcopy(config)
    benchmark_config.data.duration_s = window_s
    dataset = SyntheticVehicleDataset(
        geometry,
        SyntheticSettings(
            simulator_version="vehicle_peakset_complex_v2",
            window_s=window_s,
            feature_rate_hz=config.data.feature_rate_hz,
            waveform_rate_hz=200.0,
            max_vehicles=14,
            samples=samples,
            seed=seed,
            vehicle_rate=11.0,
            interaction_probability=0.82,
            motion_direction=config.association.motion_direction,
            peakset_vehicle_min=10,
            peakset_vehicle_max=12,
            peakset_min_visible_channels=8,
            peakset_false_event_min=50,
            peakset_false_event_max=82,
            peakset_dead_channels="5,6,15,16,22,36,38,45",
            peakset_missing_ratio_min=0.08,
            peakset_missing_ratio_max=0.18,
            peakset_gap_probability=0.72,
            peakset_gap_min_channels=2,
            peakset_gap_max_channels=6,
            peakset_boundary_vehicle_ratio=0.40,
            peakset_outage_probability=0.78,
        ),
    )
    methods = {
        "traditional_hough": BenchmarkMetrics(),
        "greedy_graph": BenchmarkMetrics(),
        "resunet_hough": BenchmarkMetrics(),
        "full_gnn_milp": BenchmarkMetrics(),
    }
    positions = torch.from_numpy(geometry.positions_m.astype(np.float32)).to(device)
    model.eval()
    for scene_index in range(samples):
        scene = dataset[scene_index]
        features = scene["input"].numpy()
        inputs = scene["input"][None].to(device)
        with torch.inference_mode():
            outputs = model(inputs)
            learned = torch.sigmoid(outputs["centerline_logits"])[0, 0].float()
            crossing = torch.sigmoid(outputs["crossing_logits"])[0, 0].float()
            raw_score = torch.sigmoid(6.0 * inputs[0, 0].float())
            pre_event_score = 1.0 - inputs[0, 1].float()
            fused = torch.clamp(
                0.55 * learned
                + 0.20 * pre_event_score
                + 0.15 * inputs[0, 2].float()
                + 0.10 * raw_score,
                0.0,
                1.0,
            )
            modal = torch.clamp(
                0.45 * pre_event_score
                + 0.40 * inputs[0, 2].float()
                + 0.15 * raw_score,
                0.0,
                1.0,
            )
            deep_hough = model.hough(
                outputs["feature_map"],
                positions,
                duration_s=window_s,
                feature_rate_hz=config.data.feature_rate_hz,
                evidence_map=fused[None, None],
            )
            traditional_hough = model.hough(
                outputs["feature_map"],
                positions,
                duration_s=window_s,
                feature_rate_hz=config.data.feature_rate_hz,
                evidence_map=modal[None, None],
            )
        fused_np = fused.cpu().numpy()
        modal_np = modal.cpu().numpy()
        crossing_np = crossing.cpu().numpy()
        embedding_np = outputs["embedding"][0].float().cpu().numpy()
        deep_seeds = PhysicalHoughHead.topk_seeds(
            deep_hough,
            top_k=config.model.hough_top_k,
            learned=True,
            min_support=config.association.min_observations,
        )
        traditional_seeds = PhysicalHoughHead.topk_seeds(
            traditional_hough,
            top_k=config.model.hough_top_k,
            learned=False,
            min_support=config.association.min_observations,
        )
        modal_observations = _observations_from_features(
            features,
            geometry,
            benchmark_config,
            network=np.zeros_like(modal_np),
            crossing=np.zeros_like(modal_np),
            embedding=None,
            use_network_candidates=False,
        )
        deep_observations = _observations_from_features(
            features,
            geometry,
            benchmark_config,
            network=fused_np,
            crossing=crossing_np,
            embedding=embedding_np,
            use_network_candidates=True,
        )
        truth = _ground_truth_tracks(scene, geometry, benchmark_config)
        specifications = {
            "traditional_hough": (
                modal_observations,
                traditional_seeds,
                modal_np,
                None,
                "greedy",
                False,
            ),
            "greedy_graph": (
                modal_observations,
                [],
                modal_np,
                None,
                "greedy",
                True,
            ),
            "resunet_hough": (
                deep_observations,
                deep_seeds,
                fused_np,
                None,
                "greedy",
                False,
            ),
            "full_gnn_milp": (
                deep_observations,
                deep_seeds + traditional_seeds,
                fused_np,
                model.edge_gnn,
                "milp",
                True,
            ),
        }
        for name, (observations, seeds, evidence, edge_gnn, mode, pair_seeds) in specifications.items():
            result = associate_observations(
                copy.deepcopy(observations),
                seeds,
                geometry,
                benchmark_config.association,
                duration_s=window_s,
                embedding_dim=config.model.embedding_dim,
                hough_intercept_step_s=config.model.hough_intercept_step_s,
                hough_top_k=config.model.hough_top_k,
                edge_gnn=edge_gnn,
                device=device,
                dense_evidence=evidence,
                feature_rate_hz=config.data.feature_rate_hz,
                selection_mode=mode,
                include_pair_seeds=pair_seeds,
            )
            _match_scene(result.tracks, truth, methods[name])

    payload = {name: metric.to_dict() for name, metric in methods.items()}
    full_f1 = payload["full_gnn_milp"]["track_f1"]
    baseline_best = max(
        payload[name]["track_f1"]
        for name in ("traditional_hough", "greedy_graph", "resunet_hough")
    )
    payload["summary"] = {
        "samples": samples,
        "seed": seed,
        "training_overlap": False,
        "best_baseline_f1": baseline_best,
        "full_f1_gain_percentage_points": 100.0 * (full_f1 - baseline_best),
        "full_speed_mae_le_2_kmh": (
            payload["full_gnn_milp"]["speed_mae_kmh"] is not None
            and payload["full_gnn_milp"]["speed_mae_kmh"] <= 2.0
        ),
        "full_f1_gain_ge_10_points": full_f1 - baseline_best >= 0.10,
        "id_switch_note": "Single-window line matching has no temporal ID-switch event.",
    }
    return payload
