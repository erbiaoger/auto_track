from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch

from hybrid_vehicle_tracker.association.pipeline import associate_observations
from hybrid_vehicle_tracker.config import AssociationConfig, ModelConfig
from hybrid_vehicle_tracker.models.edge_gnn import EdgeAssociationGNN
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel
from hybrid_vehicle_tracker.models.physical_hough import PhysicalHoughHead
from hybrid_vehicle_tracker.types import HoughSeed, StationGeometry, VehicleObservation


@dataclass
class NullControlResult:
    samples: int
    calibration_samples: int
    evaluation_samples: int
    mean_tracks_per_window: float
    maximum_tracks: int
    uncalibrated_mean_tracks_per_window: float
    uncalibrated_maximum_tracks: int
    score_quantile: float
    calibrated_min_score: float
    counts: list[int]
    uncalibrated_counts: list[int]
    maximum_scores: list[float | None]

    def to_dict(self) -> dict:
        return asdict(self)


def _station_shifts(
    station_count: int, duration_s: float, rng: np.random.Generator
) -> np.ndarray:
    return rng.uniform(0.0, duration_s, size=station_count).astype(np.float64)


def independently_shift_observations(
    observations: list[VehicleObservation],
    *,
    duration_s: float,
    shifts_s: np.ndarray,
) -> list[VehicleObservation]:
    """Apply a fixed circular time shift to every item from the same station."""
    shifted = []
    for item in observations:
        clone = copy.copy(item)
        clone.time_s = float((item.time_s + shifts_s[item.channel_index]) % duration_s)
        clone.ambiguous = False
        shifted.append(clone)
    shifted.sort(key=lambda item: (item.position_m, item.time_s))
    for index, item in enumerate(shifted):
        item.observation_id = index
    return shifted


def independently_shift_dense_evidence(
    evidence: np.ndarray,
    *,
    shifts_s: np.ndarray,
    feature_rate_hz: float,
) -> np.ndarray:
    if evidence.ndim != 2 or evidence.shape[0] != shifts_s.size:
        raise ValueError("dense evidence and station shifts have incompatible shapes")
    shifted = np.empty_like(evidence)
    for channel, shift_s in enumerate(shifts_s):
        shift_bins = int(round(float(shift_s) * feature_rate_hz))
        shifted[channel] = np.roll(evidence[channel], shift_bins)
    return shifted


def independently_shift_feature_tensor(
    features: np.ndarray,
    *,
    shifts_s: np.ndarray,
    feature_rate_hz: float,
) -> np.ndarray:
    """Shift Raw/Pre/Gauss together while retaining quality and coordinate planes."""
    if features.ndim != 3 or features.shape[0] != 5 or features.shape[1] != shifts_s.size:
        raise ValueError("feature tensor must have shape [5, station, time]")
    shifted = np.array(features, copy=True)
    for channel, shift_s in enumerate(shifts_s):
        shift_bins = int(round(float(shift_s) * feature_rate_hz))
        shifted[:3, channel] = np.roll(features[:3, channel], shift_bins, axis=-1)
    return shifted


def _run_shifted_deep_model(
    model: HybridPerceptionModel,
    features: np.ndarray,
    geometry: StationGeometry,
    model_config: ModelConfig,
    *,
    duration_s: float,
    feature_rate_hz: float,
    min_observations: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[HoughSeed]]:
    inputs = torch.from_numpy(features[None]).to(device)
    with torch.inference_mode():
        outputs = model(inputs)
        learned = torch.sigmoid(outputs["centerline_logits"])[0, 0].float()
        crossing = torch.sigmoid(outputs["crossing_logits"])[0, 0].float()
        raw_score = torch.sigmoid(6.0 * inputs[0, 0].float())
        # Pre is a negative valley at an event; use the same calibrated event
        # evidence as the real predictor instead of rewarding the valley's
        # background value.
        pre_event_score = 1.0 - inputs[0, 1].float()
        fused = torch.clamp(
            0.55 * learned
            + 0.20 * pre_event_score
            + 0.15 * inputs[0, 2].float()
            + 0.10 * raw_score,
            0.0,
            1.0,
        )
        hough = model.hough(
            outputs["feature_map"],
            torch.from_numpy(geometry.positions_m.astype(np.float32)).to(device),
            duration_s=duration_s,
            feature_rate_hz=feature_rate_hz,
            evidence_map=fused[None, None],
        )
    learned_seeds = PhysicalHoughHead.topk_seeds(
        hough,
        top_k=model_config.hough_top_k,
        learned=True,
        min_support=min_observations,
    )
    physical_seeds = PhysicalHoughHead.topk_seeds(
        hough,
        top_k=model_config.hough_top_k,
        learned=False,
        min_support=min_observations,
    )
    return (
        fused.cpu().numpy(),
        crossing.cpu().numpy(),
        outputs["embedding"][0].float().cpu().numpy(),
        learned_seeds + physical_seeds,
    )


def _update_shifted_deep_fields(
    observations: list[VehicleObservation],
    network: np.ndarray,
    crossing: np.ndarray,
    embedding: np.ndarray,
    *,
    feature_rate_hz: float,
) -> None:
    for observation in observations:
        time_bin = int(
            np.clip(
                round(observation.time_s * feature_rate_hz),
                0,
                network.shape[-1] - 1,
            )
        )
        channel = observation.channel_index
        observation.network_score = float(network[channel, time_bin])
        observation.crossing_score = float(crossing[channel, time_bin])
        observation.embedding = tuple(
            float(value) for value in embedding[:, channel, time_bin]
        )


def _quantile_with_empty(maximum_scores: list[float], quantile: float) -> float:
    finite = np.asarray([value for value in maximum_scores if np.isfinite(value)])
    if finite.size == 0:
        return float("inf")
    filled = np.asarray(
        [value if np.isfinite(value) else float(np.min(finite) - 1.0) for value in maximum_scores],
        dtype=np.float64,
    )
    return float(np.quantile(filled, quantile))


def run_null_controls(
    observations: list[VehicleObservation],
    geometry: StationGeometry,
    association_config: AssociationConfig,
    model_config: ModelConfig,
    *,
    duration_s: float,
    dense_evidence: np.ndarray,
    feature_rate_hz: float,
    feature_tensor: np.ndarray | None = None,
    samples: int = 200,
    seed: int = 20260724,
    edge_gnn: EdgeAssociationGNN | None = None,
    model: HybridPerceptionModel | None = None,
    device: torch.device | None = None,
) -> NullControlResult:
    if samples < 2:
        raise ValueError("at least two null samples are required for held-out calibration")
    rng = np.random.default_rng(seed)
    raw_scores: list[list[float]] = []
    maximum_scores: list[float] = []
    null_config = copy.deepcopy(association_config)
    null_config.min_track_score = -1e9
    for _ in range(samples):
        shifts = _station_shifts(len(geometry), duration_s, rng)
        shifted_observations = independently_shift_observations(
            observations, duration_s=duration_s, shifts_s=shifts
        )
        hough_seeds: list[HoughSeed] = []
        if model is not None and feature_tensor is not None:
            shifted_features = independently_shift_feature_tensor(
                feature_tensor,
                shifts_s=shifts,
                feature_rate_hz=feature_rate_hz,
            )
            shifted_dense, shifted_crossing, shifted_embedding, hough_seeds = (
                _run_shifted_deep_model(
                    model,
                    shifted_features,
                    geometry,
                    model_config,
                    duration_s=duration_s,
                    feature_rate_hz=feature_rate_hz,
                    min_observations=association_config.min_observations,
                    device=device or torch.device("cpu"),
                )
            )
            _update_shifted_deep_fields(
                shifted_observations,
                shifted_dense,
                shifted_crossing,
                shifted_embedding,
                feature_rate_hz=feature_rate_hz,
            )
        else:
            shifted_dense = independently_shift_dense_evidence(
                dense_evidence,
                shifts_s=shifts,
                feature_rate_hz=feature_rate_hz,
            )
        result = associate_observations(
            shifted_observations,
            hough_seeds,
            geometry,
            null_config,
            duration_s=duration_s,
            embedding_dim=model_config.embedding_dim,
            hough_intercept_step_s=model_config.hough_intercept_step_s,
            hough_top_k=model_config.hough_top_k,
            edge_gnn=edge_gnn,
            device=device,
            dense_evidence=shifted_dense,
            feature_rate_hz=feature_rate_hz,
        )
        scores = [float(track.score) for track in result.tracks]
        raw_scores.append(scores)
        maximum_scores.append(max(scores, default=float("-inf")))

    calibration_samples = samples // 2
    evaluation_scores = raw_scores[calibration_samples:]
    threshold = _quantile_with_empty(
        maximum_scores[:calibration_samples], association_config.null_quantile
    )
    # Strict comparison makes the threshold an upper-tail rejection boundary.
    calibrated_counts = [
        sum(score > threshold for score in scores) for scores in evaluation_scores
    ]
    uncalibrated_counts = [len(scores) for scores in evaluation_scores]
    serialized_maximum_scores = [
        float(value) if np.isfinite(value) else None for value in maximum_scores
    ]
    return NullControlResult(
        samples=samples,
        calibration_samples=calibration_samples,
        evaluation_samples=len(evaluation_scores),
        mean_tracks_per_window=float(np.mean(calibrated_counts)),
        maximum_tracks=max(calibrated_counts, default=0),
        uncalibrated_mean_tracks_per_window=float(np.mean(uncalibrated_counts)),
        uncalibrated_maximum_tracks=max(uncalibrated_counts, default=0),
        score_quantile=association_config.null_quantile,
        calibrated_min_score=threshold,
        counts=calibrated_counts,
        uncalibrated_counts=uncalibrated_counts,
        maximum_scores=serialized_maximum_scores,
    )
