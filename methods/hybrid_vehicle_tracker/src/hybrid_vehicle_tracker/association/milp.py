from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from hybrid_vehicle_tracker.types import CandidatePath, VehicleObservation


def select_paths_greedily(
    paths: list[CandidatePath],
    observations: list[VehicleObservation],
    *,
    min_score: float,
) -> list[CandidatePath]:
    capacities = np.ones(len(observations), dtype=np.int8)
    for observation in observations:
        if observation.ambiguous:
            capacities[observation.observation_id] = 2
    chosen: list[CandidatePath] = []
    usage = np.zeros(len(observations), dtype=np.int8)
    for path in sorted(paths, key=lambda item: item.score, reverse=True):
        if path.score < min_score:
            continue
        if all(usage[index] < capacities[index] for index in path.observation_ids):
            chosen.append(path)
            usage[list(path.observation_ids)] += 1
    return chosen


def select_paths_globally(
    paths: list[CandidatePath],
    observations: list[VehicleObservation],
    *,
    min_score: float,
) -> list[CandidatePath]:
    eligible = [path for path in paths if path.score >= min_score]
    if not eligible:
        return []
    rows = []
    columns = []
    values = []
    capacities = np.ones(len(observations), dtype=np.float64)
    for observation in observations:
        if observation.ambiguous:
            capacities[observation.observation_id] = 2.0
    for path_index, path in enumerate(eligible):
        for observation_id in path.observation_ids:
            rows.append(observation_id)
            columns.append(path_index)
            values.append(1.0)
    constraint_matrix = csr_matrix(
        (values, (rows, columns)), shape=(len(observations), len(eligible)), dtype=np.float64
    )
    constraints = LinearConstraint(constraint_matrix, lb=0.0, ub=capacities)
    result = milp(
        c=-np.asarray([path.score for path in eligible], dtype=np.float64),
        integrality=np.ones(len(eligible), dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        constraints=constraints,
        options={"time_limit": 30.0},
    )
    if result.success and result.x is not None:
        return [path for path, selected in zip(eligible, result.x) if selected >= 0.5]

    return select_paths_greedily(eligible, observations, min_score=min_score)
