from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_vehicle_tracker.association.graph import CandidateGraph
from hybrid_vehicle_tracker.config import AssociationConfig
from hybrid_vehicle_tracker.types import CandidatePath, HoughSeed, VehicleObservation


def _logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return float(np.log(clipped / (1.0 - clipped)))


@dataclass(frozen=True)
class _State:
    ids: tuple[int, ...]
    score: float
    last_speed_kmh: float | None
    strong_count: int


def paths_for_seed(
    observations: list[VehicleObservation],
    graph: CandidateGraph,
    seed: HoughSeed,
    config: AssociationConfig,
    *,
    reference_position_m: float,
) -> list[CandidatePath]:
    direction = int(config.motion_direction)
    residuals = np.asarray(
        [
            abs(
                observation.time_s
                - (
                    seed.intercept_s
                    + seed.slope_s_per_m * (observation.position_m - reference_position_m)
                )
            )
            for observation in observations
        ],
        dtype=np.float64,
    )
    allowed = set(np.flatnonzero(residuals <= config.seed_time_tolerance_s).tolist())
    if len(allowed) < config.min_observations:
        return []

    states: dict[int, list[_State]] = {}
    final_states: list[_State] = []
    # The observation list is stored in physical channel order, which is the
    # reverse of the travel order for DAY11.  Dynamic programming must process
    # nodes in motion order; otherwise reverse-direction states are appended
    # after their target bucket has already been visited and can never grow.
    node_order = sorted(allowed, key=lambda index: direction * observations[index].position_m)
    for node_index in node_order:
        observation = observations[node_index]
        start_score = _logit(observation.evidence_score) - residuals[node_index]
        states.setdefault(node_index, []).append(
            _State(
                ids=(node_index,),
                score=float(start_score),
                last_speed_kmh=None,
                strong_count=int(observation.strong),
            )
        )
        current = sorted(
            states[node_index], key=lambda item: item.score, reverse=True
        )[: config.beam_width]
        states[node_index] = current
        for state in current:
            for edge_id in graph.outgoing.get(node_index, []):
                target = int(graph.edge_index[1, edge_id])
                if target not in allowed:
                    continue
                edge_speed = float(graph.edge_speeds_kmh[edge_id])
                smooth_penalty = (
                    0.08 * abs(edge_speed - state.last_speed_kmh)
                    if state.last_speed_kmh is not None
                    else 0.0
                )
                target_observation = observations[target]
                transition_score = (
                    _logit(float(graph.edge_scores[edge_id]))
                    + 0.65 * _logit(target_observation.evidence_score)
                    - smooth_penalty
                    - 0.5 * residuals[target]
                )
                new_state = _State(
                    ids=state.ids + (target,),
                    score=state.score + transition_score,
                    last_speed_kmh=edge_speed,
                    strong_count=state.strong_count + int(target_observation.strong),
                )
                bucket = states.setdefault(target, [])
                bucket.append(new_state)
                bucket.sort(key=lambda item: item.score, reverse=True)
                del bucket[config.beam_width :]

    for bucket in states.values():
        for state in bucket:
            if len(state.ids) < config.min_observations or state.strong_count < 2:
                continue
            span = direction * (
                observations[state.ids[-1]].position_m
                - observations[state.ids[0]].position_m
            )
            if span >= config.min_span_m:
                final_states.append(state)
    final_states.sort(key=lambda item: (item.score, len(item.ids)), reverse=True)
    paths = []
    seen: set[tuple[int, ...]] = set()
    for state in final_states:
        if state.ids in seen:
            continue
        seen.add(state.ids)
        path_score = state.score / np.sqrt(len(state.ids)) + 3.0 * _logit(seed.score)
        paths.append(
            CandidatePath(
                observation_ids=state.ids,
                score=float(path_score),
                seed=seed,
            )
        )
        if len(paths) >= config.paths_per_seed:
            break
    return paths


def deduplicate_paths(paths: list[CandidatePath]) -> list[CandidatePath]:
    selected: list[CandidatePath] = []
    for path in sorted(paths, key=lambda item: item.score, reverse=True):
        current = set(path.observation_ids)
        duplicate = False
        for old in selected:
            previous = set(old.observation_ids)
            overlap = len(current & previous) / max(min(len(current), len(previous)), 1)
            if overlap >= 0.80:
                duplicate = True
                break
        if not duplicate:
            selected.append(path)
    return selected
