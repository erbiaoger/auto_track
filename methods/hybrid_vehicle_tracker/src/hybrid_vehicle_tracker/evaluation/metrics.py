from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from hybrid_vehicle_tracker.types import VehicleTrack


@dataclass
class TrackMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    speed_mae_kmh: float

    def to_dict(self) -> dict:
        return asdict(self)


def match_tracks(
    predictions: list[VehicleTrack],
    truth: list[tuple[float, float]],
    *,
    max_speed_error_kmh: float = 5.0,
    max_intercept_error_s: float = 2.0,
) -> TrackMetrics:
    """Match predictions to (speed_kmh, intercept_s) synthetic truth."""
    if not predictions or not truth:
        tp = 0
        fp = len(predictions)
        fn = len(truth)
        return TrackMetrics(tp, fp, fn, 0.0, 0.0, 0.0, float("nan"))
    costs = np.full((len(predictions), len(truth)), 1e6, dtype=np.float64)
    for row, prediction in enumerate(predictions):
        first_time = prediction.points[0].time_s
        for column, (speed, intercept) in enumerate(truth):
            speed_error = abs(prediction.median_speed_kmh - speed)
            intercept_error = abs(first_time - intercept)
            if speed_error <= max_speed_error_kmh and intercept_error <= max_intercept_error_s:
                costs[row, column] = speed_error / max_speed_error_kmh + intercept_error / max_intercept_error_s
    rows, columns = linear_sum_assignment(costs)
    valid = costs[rows, columns] < 1e5
    tp = int(np.count_nonzero(valid))
    fp = len(predictions) - tp
    fn = len(truth) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    speed_errors = [
        abs(predictions[row].median_speed_kmh - truth[column][0])
        for row, column, keep in zip(rows, columns, valid)
        if keep
    ]
    return TrackMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        speed_mae_kmh=float(np.mean(speed_errors)) if speed_errors else float("nan"),
    )
