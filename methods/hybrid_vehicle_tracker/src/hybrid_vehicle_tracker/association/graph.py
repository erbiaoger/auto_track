from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.models.edge_gnn import EdgeAssociationGNN
from hybrid_vehicle_tracker.types import StationGeometry, VehicleObservation


@dataclass
class CandidateGraph:
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    edge_scores: np.ndarray
    edge_speeds_kmh: np.ndarray
    outgoing: dict[int, list[int]]


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-8 else 0.0


def build_candidate_graph(
    observations: list[VehicleObservation],
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    duration_s: float,
    embedding_dim: int,
) -> CandidateGraph:
    positions = geometry.positions_m
    direction = int(config.motion_direction)
    motion_positions = direction * positions
    motion_origin = float(np.min(motion_positions))
    span = max(float(np.max(motion_positions) - motion_origin), 1.0)
    node_rows = []
    for observation in observations:
        embedding = list(observation.embedding[:embedding_dim])
        embedding.extend([0.0] * (embedding_dim - len(embedding)))
        node_rows.append(
            [
                observation.gauss_score,
                observation.pre_score,
                observation.raw_energy,
                observation.network_score,
                observation.crossing_score,
                float(observation.strong),
                observation.evidence_score,
                (direction * observation.position_m - motion_origin) / span,
                observation.time_s / max(duration_s, 1e-6),
                *embedding,
            ]
        )

    edge_pairs: list[tuple[int, int]] = []
    edge_rows: list[list[float]] = []
    heuristic_scores: list[float] = []
    speeds: list[float] = []
    mid_speed = 0.5 * (config.speed_min_kmh + config.speed_max_kmh)
    half_band = 0.5 * (config.speed_max_kmh - config.speed_min_kmh)
    for source_index, source in enumerate(observations):
        if direction > 0:
            target_indices = range(source_index + 1, len(observations))
        else:
            target_indices = range(source_index - 1, -1, -1)
        for target_index in target_indices:
            target = observations[target_index]
            # Work in the travel coordinate.  This permits a physical array
            # stored from high kilometre marker to low marker while retaining
            # the same positive-time graph semantics.
            dx = direction * (target.position_m - source.position_m)
            if dx <= 0.0:
                continue
            if dx > config.max_gap_m:
                # Observations are sorted by physical position, so all later
                # targets in this direction are even farther away.
                break
            dt = target.time_s - source.time_s
            lower = config.min_slowness_s_per_m * dx - config.edge_time_tolerance_s
            upper = config.max_slowness_s_per_m * dx + config.edge_time_tolerance_s
            if dt <= 0.0 or dt < lower or dt > upper:
                continue
            speed = 3.6 * dx / dt
            speed_score = np.exp(-0.5 * ((speed - mid_speed) / max(half_band, 1.0)) ** 2)
            gap_score = np.exp(-0.4 * dx / config.max_gap_m)
            similarity = _cosine(source.embedding, target.embedding)
            embedding_score = 0.5 * (similarity + 1.0) if source.embedding and target.embedding else 0.5
            evidence = np.sqrt(source.evidence_score * target.evidence_score)
            strong_bonus = 0.10 if source.strong and target.strong else 0.0
            heuristic = np.clip(
                0.40 * evidence + 0.30 * speed_score + 0.15 * gap_score + 0.15 * embedding_score + strong_bonus,
                1e-4,
                1.0 - 1e-4,
            )
            skipped = max(abs(target.channel_index - source.channel_index) - 1, 0)
            edge_pairs.append((source_index, target_index))
            edge_rows.append(
                [
                    dx / config.max_gap_m,
                    dt / max(config.max_slowness_s_per_m * config.max_gap_m, 1e-6),
                    (speed - config.speed_min_kmh)
                    / max(config.speed_max_kmh - config.speed_min_kmh, 1e-6),
                    abs(speed - mid_speed) / max(half_band, 1.0),
                    source.evidence_score,
                    target.evidence_score,
                    embedding_score,
                    skipped / max(len(geometry) - 1, 1),
                    speed_score,
                ]
            )
            heuristic_scores.append(float(heuristic))
            speeds.append(float(speed))

    if edge_pairs:
        edge_index = np.asarray(edge_pairs, dtype=np.int64).T
        edge_features = np.asarray(edge_rows, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_features = np.empty((0, 9), dtype=np.float32)
    outgoing: dict[int, list[int]] = {index: [] for index in range(len(observations))}
    for edge_id, (source_index, _) in enumerate(edge_pairs):
        outgoing[source_index].append(edge_id)
    return CandidateGraph(
        node_features=np.asarray(node_rows, dtype=np.float32),
        edge_index=edge_index,
        edge_features=edge_features,
        edge_scores=np.asarray(heuristic_scores, dtype=np.float32),
        edge_speeds_kmh=np.asarray(speeds, dtype=np.float32),
        outgoing=outgoing,
    )


def apply_learned_edge_scores(
    graph: CandidateGraph,
    observations: list[VehicleObservation],
    model: EdgeAssociationGNN,
    *,
    device: torch.device,
) -> CandidateGraph:
    if not len(observations):
        return graph
    with torch.inference_mode():
        node_tensor = torch.from_numpy(graph.node_features).to(device)
        edge_index = torch.from_numpy(graph.edge_index).to(device)
        edge_tensor = torch.from_numpy(graph.edge_features).to(device)
        edge_logits, merge_logits = model(node_tensor, edge_index, edge_tensor)
        learned = torch.sigmoid(edge_logits).cpu().numpy()
        merge = torch.sigmoid(merge_logits).cpu().numpy()
    graph.edge_scores = np.clip(0.65 * learned + 0.35 * graph.edge_scores, 1e-4, 1 - 1e-4)
    for observation, merge_score in zip(observations, merge):
        if merge_score >= 0.65 and observation.crossing_score >= 0.45:
            observation.ambiguous = True
    return graph
