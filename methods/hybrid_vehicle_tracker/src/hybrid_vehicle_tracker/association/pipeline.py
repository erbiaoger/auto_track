from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from hybrid_vehicle_tracker.association.beam import deduplicate_paths, paths_for_seed
from hybrid_vehicle_tracker.association.graph import (
    CandidateGraph,
    apply_learned_edge_scores,
    build_candidate_graph,
)
from hybrid_vehicle_tracker.association.milp import (
    select_paths_globally,
    select_paths_greedily,
)
from hybrid_vehicle_tracker.association.refine import refine_path
from hybrid_vehicle_tracker.association.seeds import (
    merge_seeds,
    observation_pair_seeds,
    rescore_seeds_on_dense_evidence,
)
from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.models.edge_gnn import EdgeAssociationGNN
from hybrid_vehicle_tracker.types import (
    CandidatePath,
    HoughSeed,
    StationGeometry,
    VehicleObservation,
    VehicleTrack,
)


@dataclass
class AssociationResult:
    tracks: list[VehicleTrack]
    graph: CandidateGraph
    seeds: list[HoughSeed]
    candidate_paths: list[CandidatePath]
    selected_paths: list[CandidatePath]


def associate_observations(
    observations: list[VehicleObservation],
    hough_seeds: list[HoughSeed],
    geometry: StationGeometry,
    config: AssociationConfig,
    *,
    duration_s: float,
    embedding_dim: int,
    hough_intercept_step_s: float,
    hough_top_k: int,
    edge_gnn: EdgeAssociationGNN | None = None,
    device: torch.device | None = None,
    dense_evidence: np.ndarray | None = None,
    feature_rate_hz: float | None = None,
    selection_mode: str = "milp",
    include_pair_seeds: bool = True,
) -> AssociationResult:
    graph = build_candidate_graph(
        observations,
        geometry,
        config,
        duration_s=duration_s,
        embedding_dim=embedding_dim,
    )
    if edge_gnn is not None:
        graph = apply_learned_edge_scores(
            graph, observations, edge_gnn, device=device or torch.device("cpu")
        )
    pair_seeds = (
        observation_pair_seeds(
            observations,
            geometry,
            config,
            top_k=hough_top_k,
            intercept_step_s=hough_intercept_step_s,
        )
        if include_pair_seeds
        else []
    )
    seed_candidates = hough_seeds + pair_seeds
    if dense_evidence is not None:
        seed_candidates = rescore_seeds_on_dense_evidence(
            seed_candidates,
            np.asarray(dense_evidence),
            geometry,
            config,
            feature_rate_hz=float(feature_rate_hz or 20.0),
            duration_s=duration_s,
        )
    seeds = merge_seeds(seed_candidates, top_k=hough_top_k)
    paths = []
    for seed in seeds:
        paths.extend(
            paths_for_seed(
                observations,
                graph,
                seed,
                config,
                reference_position_m=float(geometry.positions_m[0]),
            )
        )
    paths = deduplicate_paths(paths)
    if selection_mode == "milp":
        selected = select_paths_globally(paths, observations, min_score=config.min_track_score)
    elif selection_mode == "greedy":
        selected = select_paths_greedily(paths, observations, min_score=config.min_track_score)
    else:
        raise ValueError(f"unknown selection mode: {selection_mode}")
    tracks = []
    for path in sorted(selected, key=lambda item: item.seed.intercept_s):
        track = refine_path(
            path,
            observations,
            geometry,
            config,
            duration_s=duration_s,
            track_index=len(tracks),
        )
        if track is not None:
            tracks.append(track)
    return AssociationResult(
        tracks=tracks,
        graph=graph,
        seeds=seeds,
        candidate_paths=paths,
        selected_paths=selected,
    )
