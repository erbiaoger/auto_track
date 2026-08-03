from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hybrid_vehicle_tracker.association.pipeline import AssociationResult, associate_observations
from hybrid_vehicle_tracker.config import TrackerConfig
from hybrid_vehicle_tracker.data.features import FeatureBatch, build_feature_batch
from hybrid_vehicle_tracker.data.io import load_modal_window
from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.data.peaks import extract_observations
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel
from hybrid_vehicle_tracker.models.physical_hough import HoughOutput, PhysicalHoughHead
from hybrid_vehicle_tracker.types import StationGeometry, TrackBatch


@dataclass
class InferenceArtifacts:
    features: FeatureBatch
    network_probability: np.ndarray
    crossing_probability: np.ndarray
    hough_scores: np.ndarray
    hough_slopes: np.ndarray
    hough_intercepts: np.ndarray
    association: AssociationResult
    # Keep aligned full-rate modal windows for the reference-style overlay.
    # They are optional so lightweight unit tests and external callers can
    # still build artifacts without retaining the full-rate inputs.
    raw_waveform: np.ndarray | None = None
    gauss_waveform: np.ndarray | None = None
    station_positions_m: np.ndarray | None = None


class HybridVehicleTracker:
    """Public inference API for the full deep/physics association pipeline."""

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config
        self.device = self._resolve_device(config.runtime.device)
        self.model = HybridPerceptionModel(config.model).to(self.device)
        self.has_checkpoint = bool(config.model.checkpoint)
        self.checkpoint_metadata: dict[str, Any] = {}
        if config.model.checkpoint:
            self.checkpoint_metadata = self.model.load_checkpoint(
                config.model.checkpoint, map_location=str(self.device)
            )
        self.model.eval()
        self.last_artifacts: InferenceArtifacts | None = None

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is false; "
                "this project never silently falls back to CPU"
            )
        return device

    @staticmethod
    def _window_array(
        values: np.ndarray,
        *,
        start_s: float,
        duration_s: float,
        sample_rate_hz: float,
    ) -> np.ndarray:
        required = int(round(duration_s * sample_rate_hz))
        if values.shape[0] == required:
            return np.asarray(values, dtype=np.float32)
        begin = int(round(start_s * sample_rate_hz))
        end = begin + required
        if values.shape[0] < end:
            raise ValueError(f"input has {values.shape[0]} samples, need at least {end}")
        return np.asarray(values[begin:end], dtype=np.float32)

    def predict(
        self,
        raw: np.ndarray,
        pre: np.ndarray,
        gauss: np.ndarray,
        station_mapping: str | Path | StationGeometry,
        start_s: float = 0.0,
        duration_s: float = 120.0,
    ) -> TrackBatch:
        geometry = (
            station_mapping
            if isinstance(station_mapping, StationGeometry)
            else load_station_geometry(station_mapping)
        )
        if self.config.model.motion_direction != self.config.association.motion_direction:
            raise ValueError(
                "model.motion_direction and association.motion_direction must match"
            )
        raw_window = self._window_array(
            raw,
            start_s=start_s,
            duration_s=duration_s,
            sample_rate_hz=self.config.data.sample_rate_hz,
        )
        pre_window = self._window_array(
            pre,
            start_s=start_s,
            duration_s=duration_s,
            sample_rate_hz=self.config.data.sample_rate_hz,
        )
        gauss_window = self._window_array(
            gauss,
            start_s=start_s,
            duration_s=duration_s,
            sample_rate_hz=self.config.data.sample_rate_hz,
        )
        features = build_feature_batch(
            raw_window,
            pre_window,
            gauss_window,
            geometry,
            sample_rate_hz=self.config.data.sample_rate_hz,
            feature_rate_hz=self.config.data.feature_rate_hz,
        )
        inputs = torch.from_numpy(features.tensor[None]).to(self.device)
        # The supplied Pre modality has a negative valley at each Gauss centre.
        # Its calibrated event evidence is therefore 1 - Pre, while the network
        # still receives the original calibrated plane so it can learn shape.
        pre_event_score = 1.0 - features.pre_score
        with torch.inference_mode():
            if self.has_checkpoint:
                outputs = self.model(inputs)
                learned_probability = torch.sigmoid(outputs["centerline_logits"])[0, 0]
                crossing_tensor = torch.sigmoid(outputs["crossing_logits"])[0, 0]
                embedding = outputs["embedding"][0].detach().cpu().numpy()
                feature_map = outputs["feature_map"]
                network_probability = (
                    0.55 * learned_probability.detach().cpu().numpy()
                    + 0.20 * pre_event_score
                    + 0.15 * features.gauss_score
                    + 0.10 * features.raw_score
                )
                crossing_probability = crossing_tensor.detach().cpu().numpy()
            else:
                network_probability = (
                    0.45 * pre_event_score
                    + 0.35 * features.gauss_score
                    + 0.20 * features.raw_score
                )
                crossing_probability = np.zeros_like(network_probability)
                embedding = None
                feature_map = torch.zeros(
                    (
                        1,
                        self.config.model.base_channels,
                        features.station_count,
                        features.time_bins,
                    ),
                    device=self.device,
                )
            network_probability = np.clip(network_probability, 0.0, 1.0).astype(np.float32)
            # Keep a physics-only evidence plane alongside the learned plane.
            # On real DAY11, a fixed-kernel synthetic checkpoint can assign low
            # learned probability to a genuine peak whose height is slightly
            # different; Gauss+Pre still provides valid line evidence.
            modal_probability = np.clip(
                0.45 * pre_event_score
                + 0.40 * features.gauss_score
                + 0.15 * features.raw_score,
                0.0,
                1.0,
            ).astype(np.float32)
            evidence = torch.from_numpy(network_probability[None, None]).to(self.device)
            hough_output = self.model.hough(
                feature_map,
                torch.from_numpy(geometry.positions_m.astype(np.float32)),
                duration_s=duration_s,
                feature_rate_hz=features.feature_rate_hz,
                evidence_map=evidence,
            )
            physical_hough_output = self.model.hough(
                feature_map,
                torch.from_numpy(geometry.positions_m.astype(np.float32)),
                duration_s=duration_s,
                feature_rate_hz=features.feature_rate_hz,
                evidence_map=torch.from_numpy(modal_probability[None, None]).to(self.device),
            )
        learned_hough_seeds = PhysicalHoughHead.topk_seeds(
            hough_output,
            top_k=self.config.model.hough_top_k,
            learned=self.has_checkpoint,
            min_support=self.config.association.min_observations,
        )
        physical_hough_seeds = PhysicalHoughHead.topk_seeds(
            physical_hough_output,
            top_k=self.config.model.hough_top_k,
            learned=False,
            min_support=self.config.association.min_observations,
        )
        hough_seeds = learned_hough_seeds + physical_hough_seeds
        observations = extract_observations(
            gauss_window,
            pre_window,
            features,
            geometry,
            self.config.association,
            network_probability=network_probability,
            crossing_probability=crossing_probability,
            embedding=embedding,
        )
        association = associate_observations(
            observations,
            hough_seeds,
            geometry,
            self.config.association,
            duration_s=duration_s,
            embedding_dim=self.config.model.embedding_dim,
            hough_intercept_step_s=self.config.model.hough_intercept_step_s,
            hough_top_k=self.config.model.hough_top_k,
            edge_gnn=self.model.edge_gnn if self.has_checkpoint else None,
            device=self.device,
            dense_evidence=np.maximum(network_probability, modal_probability),
            feature_rate_hz=features.feature_rate_hz,
        )
        diagnostics = {
            "device": str(self.device),
            "checkpoint": self.config.model.checkpoint,
            "checkpoint_metadata": self.checkpoint_metadata,
            "station_count": len(geometry),
            "feature_bins": features.time_bins,
            "observation_count": len(observations),
            "strong_observation_count": sum(item.strong for item in observations),
            "edge_count": int(association.graph.edge_index.shape[1]),
            "hough_seed_count": len(association.seeds),
            "candidate_path_count": len(association.candidate_paths),
            "selected_path_count": len(association.selected_paths),
            "track_count": len(association.tracks),
            "track_status": "physics_candidate_unvalidated",
            "confidence_semantics": "internal multimodal/physics consistency; not calibrated precision",
            "speed_range_kmh": [
                self.config.association.speed_min_kmh,
                self.config.association.speed_max_kmh,
            ],
            "motion_direction": self.config.association.motion_direction,
            "direction": (
                "increasing_time_with_position"
                if self.config.association.motion_direction > 0
                else "increasing_time_with_decreasing_position"
            ),
        }
        batch = TrackBatch(
            tracks=association.tracks,
            observations=observations,
            start_s=start_s,
            duration_s=duration_s,
            diagnostics=diagnostics,
        )
        scores = (
            0.55 * torch.sigmoid(hough_output.logits[0])
            + 0.45 * hough_output.raw_scores[0]
            if self.has_checkpoint
            else hough_output.raw_scores[0]
        )
        self.last_artifacts = InferenceArtifacts(
            features=features,
            network_probability=network_probability,
            crossing_probability=crossing_probability,
            hough_scores=scores.detach().cpu().numpy(),
            hough_slopes=hough_output.slopes.detach().cpu().numpy(),
            hough_intercepts=hough_output.intercepts.detach().cpu().numpy(),
            association=association,
            raw_waveform=np.asarray(raw_window, dtype=np.float32),
            gauss_waveform=np.asarray(gauss_window, dtype=np.float32),
            station_positions_m=np.asarray(geometry.positions_m, dtype=np.float32),
        )
        return batch

    def predict_from_paths(self) -> TrackBatch:
        data = self.config.data
        raw, pre, gauss = load_modal_window(
            data.raw_path,
            data.pre_path,
            data.gauss_path,
            start_s=data.start_s,
            duration_s=data.duration_s,
            sample_rate_hz=data.sample_rate_hz,
        )
        return self.predict(
            raw,
            pre,
            gauss,
            data.mapping_path,
            start_s=data.start_s,
            duration_s=data.duration_s,
        )
