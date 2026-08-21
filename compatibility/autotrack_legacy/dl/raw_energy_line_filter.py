"""Raw-energy diagonal consistency filter for DAS vehicle peak candidates.

This module implements the filtering method documented in
``docs/raw_energy_line_filter.md``:

1. calculate a robust short-time energy envelope from every raw station;
2. scan physically plausible diagonal lines using station spacing and a speed
   range;
3. keep high-scoring, non-overlapping raw-energy lines;
4. retain only probability peaks close to those lines.

The red guide lines in the inspection figures are the output of step 3. They
are raw-data candidates, not neural-network predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


STATION_ID_RE = re.compile(r"_([^_]+)_EHZ_", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class LineCandidate:
    """One raw-energy diagonal candidate.

    ``intercept_s`` is the time at station index 0 and
    ``slope_s_per_station`` is the time change between adjacent stations.
    A negative slope means the event arrives earlier at larger station
    indices. ``score`` is the normalized raw-energy score.
    """

    score: float
    slope_s_per_station: float
    intercept_s: float
    valid_station_count: int

    def time_at_station(self, station_index: int) -> float:
        """Return the candidate time for a zero-based station index."""
        return self.intercept_s + self.slope_s_per_station * float(station_index)


@dataclass(slots=True)
class LineFilterResult:
    """Results returned by :func:`run_line_consistent_filter`."""

    lines: list[LineCandidate]
    candidate_times_s: list[np.ndarray]
    candidate_scores: list[np.ndarray]
    selected_times_s: list[np.ndarray]
    selected_scores: list[np.ndarray]
    selected_line_ids: list[np.ndarray]
    energy_envelope: np.ndarray
    energy_dt_s: float


def _as_station_list(values: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
    result = [np.asarray(value).reshape(-1) for value in values]
    if not result:
        raise ValueError(f"{name} must contain at least one station")
    return result


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float32)
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def pick_probability_peaks(
    probability: np.ndarray,
    *,
    fs_hz: float,
    threshold: float,
    min_gap_s: float = 2.0,
    flip: bool = False,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick one maximum from each above-threshold probability segment.

    Parameters
    ----------
    probability:
        One station's prediction array. Logits are accepted; values outside
        ``[0, 1]`` are passed through a sigmoid.
    fs_hz:
        Sampling frequency of the prediction array.
    threshold:
        Confidence threshold for this station. It may be different for every
        station when this function is called in a loop.
    min_gap_s:
        Minimum separation between retained peaks.
    flip:
        Set to ``True`` when the prediction uses background-high semantics.
    normalize:
        Normalize by the station maximum before applying ``flip``.
    """
    if fs_hz <= 0 or min_gap_s < 0:
        raise ValueError("fs_hz must be positive and min_gap_s must be non-negative")
    values = np.asarray(probability, dtype=np.float32).reshape(-1).copy()
    if values.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
    if float(values.min()) < 0.0 or float(values.max()) > 1.0:
        values = _stable_sigmoid(values)
    if normalize:
        maximum = float(values.max())
        if maximum > 1e-8:
            values /= maximum
    if flip:
        values = 1.0 - values

    above = np.flatnonzero(values > float(threshold))
    if above.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float32)
    breaks = np.flatnonzero(np.diff(above) > 1)
    starts = np.concatenate(([above[0]], above[breaks + 1]))
    stops = np.concatenate((above[breaks] + 1, [above[-1] + 1]))
    peak_indices = np.asarray(
        [start + int(np.argmax(values[start:stop])) for start, stop in zip(starts, stops)],
        dtype=np.int64,
    )
    times = peak_indices.astype(np.float64) / float(fs_hz)
    scores = values[peak_indices].astype(np.float32)
    min_gap_samples = max(1, int(round(float(min_gap_s) * float(fs_hz))))
    keep_times = [times[0]]
    keep_scores = [scores[0]]
    keep_indices = [peak_indices[0]]
    for index in range(1, len(peak_indices)):
        if int(peak_indices[index]) - int(keep_indices[-1]) < min_gap_samples:
            if scores[index] > keep_scores[-1]:
                keep_times[-1] = times[index]
                keep_scores[-1] = scores[index]
                keep_indices[-1] = peak_indices[index]
        else:
            keep_times.append(times[index])
            keep_scores.append(scores[index])
            keep_indices.append(peak_indices[index])
    return np.asarray(keep_times), np.asarray(keep_scores, dtype=np.float32)


def compute_energy_envelope(
    signal: np.ndarray,
    *,
    fs_hz: float,
    window_s: float = 1.0,
    baseline_quantile: float = 0.50,
    scale_quantile: float = 0.95,
    clip_max: float = 8.0,
) -> np.ndarray:
    """Convert one raw station into a robust short-time RMS energy envelope."""
    if fs_hz <= 0 or window_s <= 0:
        raise ValueError("fs_hz and window_s must be positive")
    if not 0.0 <= baseline_quantile < 1.0 or not 0.0 < scale_quantile <= 1.0:
        raise ValueError("invalid energy quantiles")
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    block = max(1, int(round(float(fs_hz) * float(window_s))))
    count = values.size // block
    if count == 0:
        raise ValueError("signal is shorter than one energy window")
    trimmed = values[: count * block].reshape(count, block)
    with np.errstate(over="ignore", invalid="ignore"):
        rms = np.sqrt(np.nanmean(np.square(trimmed, dtype=np.float64), axis=1)).astype(np.float32)
    rms = np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)
    baseline = float(np.quantile(rms, baseline_quantile))
    scale = float(np.quantile(rms, scale_quantile)) - baseline
    if scale <= 1e-12:
        scale = max(float(np.std(rms)), 1e-12)
    normalized = np.clip((rms - baseline) / scale, 0.0, float(clip_max))
    return normalized.astype(np.float32)


def compute_energy_matrix(
    raw_signals: Sequence[np.ndarray],
    *,
    fs_hz: float,
    window_s: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Return ``[time_bin, station]`` energy and its bin interval."""
    stations = _as_station_list(raw_signals, "raw_signals")
    lengths = {int(value.size) for value in stations}
    if len(lengths) != 1:
        raise ValueError(f"all raw_signals arrays must have the same length; got {sorted(lengths)}")
    envelopes = [compute_energy_envelope(signal, fs_hz=fs_hz, window_s=window_s) for signal in stations]
    return np.stack(envelopes, axis=1), float(window_s)


def _slope_bounds_from_speed(
    *, station_spacing_m: float, speed_min_kmh: float, speed_max_kmh: float
) -> tuple[float, float]:
    if station_spacing_m <= 0 or speed_min_kmh <= 0 or speed_max_kmh < speed_min_kmh:
        raise ValueError("invalid station spacing or speed range")
    # seconds per station: distance / speed, with both directions handled later
    slowest = station_spacing_m / (speed_min_kmh / 3.6)
    fastest = station_spacing_m / (speed_max_kmh / 3.6)
    return float(fastest), float(slowest)


def scan_raw_energy_lines(
    energy: np.ndarray,
    *,
    energy_dt_s: float,
    station_spacing_m: float = 100.0,
    speed_min_kmh: float = 60.0,
    speed_max_kmh: float = 100.0,
    slope_step_s_per_station: float = 0.2,
    top_k: int = 70,
    nonmax_intercept_s: float = 20.0,
    nonmax_slope_s_per_station: float = 0.4,
) -> list[LineCandidate]:
    """Scan physically plausible raw-energy diagonals.

    The line equation is ``time = intercept + slope * station_index``. The
    slope range is derived from the speed range and station spacing. Both
    signs are scanned. A line score is the mean normalized energy sampled along
    the line, scaled to the full station count. Non-maximum suppression avoids
    returning many nearly identical lines.
    """
    values = np.asarray(energy, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < 1:
        raise ValueError("energy must have shape [time_bin, station]")
    if energy_dt_s <= 0 or slope_step_s_per_station <= 0 or top_k <= 0:
        raise ValueError("energy_dt_s, slope_step_s_per_station, and top_k must be positive")
    fastest, slowest = _slope_bounds_from_speed(
        station_spacing_m=station_spacing_m,
        speed_min_kmh=speed_min_kmh,
        speed_max_kmh=speed_max_kmh,
    )
    positive = np.arange(fastest, slowest + slope_step_s_per_station * 0.5, slope_step_s_per_station)
    slopes = np.unique(np.r_[positive, -positive])
    time_axis = np.arange(values.shape[0], dtype=np.float64) * float(energy_dt_s)
    station_indices = np.arange(values.shape[1], dtype=np.float64)
    score_surface = np.zeros((len(slopes), values.shape[0]), dtype=np.float32)
    count_surface = np.zeros_like(score_surface)
    for slope_index, slope in enumerate(slopes):
        score = np.zeros(values.shape[0], dtype=np.float32)
        valid_count = np.zeros(values.shape[0], dtype=np.float32)
        for station_index in range(values.shape[1]):
            requested = time_axis + float(slope) * float(station_index)
            valid = (requested >= time_axis[0]) & (requested <= time_axis[-1])
            if not np.any(valid):
                continue
            score[valid] += np.interp(requested[valid], time_axis, values[:, station_index]).astype(np.float32)
            valid_count[valid] += 1.0
        score_surface[slope_index] = np.divide(
            score * float(values.shape[1]),
            np.maximum(valid_count, 1.0),
            out=np.zeros_like(score),
            where=valid_count > 0,
        )
        count_surface[slope_index] = valid_count

    work = score_surface.copy()
    lines: list[LineCandidate] = []
    suppress_bins = max(1, int(round(float(nonmax_intercept_s) / float(energy_dt_s))))
    for _ in range(int(top_k)):
        slope_index, intercept_bin = np.unravel_index(int(np.argmax(work)), work.shape)
        score = float(work[slope_index, intercept_bin])
        if not np.isfinite(score) or score <= 0:
            break
        slope = float(slopes[slope_index])
        lines.append(
            LineCandidate(
                score=score,
                slope_s_per_station=slope,
                intercept_s=float(intercept_bin) * float(energy_dt_s),
                valid_station_count=int(count_surface[slope_index, intercept_bin]),
            )
        )
        slope_mask = np.abs(slopes - slope) <= float(nonmax_slope_s_per_station)
        start = max(0, int(intercept_bin) - suppress_bins)
        stop = min(work.shape[1], int(intercept_bin) + suppress_bins + 1)
        work[slope_mask, start:stop] = 0.0
    return lines


def _nearest_candidate(
    times_s: np.ndarray,
    scores: np.ndarray,
    target_s: float,
    tolerance_s: float,
) -> tuple[float, float, int] | None:
    if times_s.size == 0:
        return None
    index = int(np.searchsorted(times_s, target_s))
    indices = [min(max(index, 0), times_s.size - 1)]
    if index > 0:
        indices.append(index - 1)
    best = min(indices, key=lambda item: abs(float(times_s[item]) - float(target_s)))
    if abs(float(times_s[best]) - float(target_s)) > float(tolerance_s):
        return None
    return float(times_s[best]), float(scores[best]), best


def match_peaks_to_lines(
    candidate_times_s: Sequence[np.ndarray],
    candidate_scores: Sequence[np.ndarray] | None,
    lines: Sequence[LineCandidate],
    *,
    tolerance_s: float = 0.8,
    dedup_gap_s: float = 1.0,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Keep candidate peaks that lie close to at least one raw-energy line."""
    times_list = _as_station_list(candidate_times_s, "candidate_times_s")
    if candidate_scores is None:
        scores_list = [np.ones(times.size, dtype=np.float32) for times in times_list]
    else:
        scores_list = _as_station_list(candidate_scores, "candidate_scores")
        if len(scores_list) != len(times_list):
            raise ValueError("candidate times and scores must have the same station count")
        if any(a.size != b.size for a, b in zip(times_list, scores_list)):
            raise ValueError("candidate times and scores must have matching lengths")
    selected_times: list[np.ndarray] = []
    selected_scores: list[np.ndarray] = []
    selected_line_ids: list[np.ndarray] = []
    for station_index, (times, scores) in enumerate(zip(times_list, scores_list)):
        proposals: list[tuple[float, float, int]] = []
        for line_index, line in enumerate(lines):
            match = _nearest_candidate(
                times,
                scores,
                line.time_at_station(station_index),
                tolerance_s,
            )
            if match is not None:
                match_time, match_score, _ = match
                proposals.append((match_time, match_score, line_index))
        proposals.sort(key=lambda value: value[0])
        kept: list[tuple[float, float, int]] = []
        for proposal in proposals:
            if kept and proposal[0] - kept[-1][0] < float(dedup_gap_s):
                if proposal[1] > kept[-1][1]:
                    kept[-1] = proposal
            else:
                kept.append(proposal)
        selected_times.append(np.asarray([value[0] for value in kept], dtype=np.float64))
        selected_scores.append(np.asarray([value[1] for value in kept], dtype=np.float32))
        selected_line_ids.append(np.asarray([value[2] for value in kept], dtype=np.int64))
    return selected_times, selected_scores, selected_line_ids


def derive_station_thresholds(
    candidate_times_s: Sequence[np.ndarray],
    candidate_scores: Sequence[np.ndarray],
    lines: Sequence[LineCandidate],
    *,
    quantile: float = 0.5,
    lower: float = 0.3,
    upper: float = 0.9,
    tolerance_s: float = 0.8,
    fallback: float = 0.9,
) -> np.ndarray:
    """Derive one confidence threshold per station from line-supported peaks."""
    if not 0.0 <= quantile <= 1.0 or lower > upper:
        raise ValueError("invalid threshold quantile or bounds")
    times_list = _as_station_list(candidate_times_s, "candidate_times_s")
    scores_list = _as_station_list(candidate_scores, "candidate_scores")
    thresholds = []
    for station_index, (times, scores) in enumerate(zip(times_list, scores_list)):
        matched_scores = []
        for line in lines:
            match = _nearest_candidate(times, scores, line.time_at_station(station_index), tolerance_s)
            if match is not None:
                matched_scores.append(match[1])
        value = float(np.quantile(matched_scores, quantile)) if matched_scores else float(fallback)
        thresholds.append(float(np.clip(value, lower, upper)))
    return np.asarray(thresholds, dtype=np.float32)


def gaussian_curve_from_picks(
    signal: np.ndarray,
    pick_times_s: np.ndarray,
    *,
    fs_hz: float,
    width_s: float = 0.5,
    amplitude_scale: float = 0.8,
) -> np.ndarray:
    """Build a sparse Gaussian curve from selected peak times."""
    if fs_hz <= 0 or width_s <= 0:
        raise ValueError("fs_hz and width_s must be positive")
    values = np.asarray(signal, dtype=np.float32).reshape(-1)
    output = np.zeros(values.size, dtype=np.float32)
    amplitude = float(np.percentile(np.abs(values), 99.0)) * float(amplitude_scale)
    if amplitude < 1e-10:
        amplitude = float(amplitude_scale)
    radius = max(1, int(np.ceil(4.0 * float(width_s) * float(fs_hz))))
    for pick_time in np.asarray(pick_times_s, dtype=np.float64):
        center = int(round(float(pick_time) * float(fs_hz)))
        center = min(max(center, 0), values.size - 1)
        lo = max(0, center - radius)
        hi = min(values.size, center + radius + 1)
        local_time = (np.arange(lo, hi, dtype=np.float64) - center) / float(fs_hz)
        output[lo:hi] += amplitude * np.exp(-0.5 * (local_time / float(width_s)) ** 2).astype(np.float32)
    return output


def run_line_consistent_filter(
    raw_signals: Sequence[np.ndarray],
    candidate_times_s: Sequence[np.ndarray],
    candidate_scores: Sequence[np.ndarray] | None = None,
    *,
    fs_hz: float,
    station_spacing_m: float = 100.0,
    speed_min_kmh: float = 60.0,
    speed_max_kmh: float = 100.0,
    energy_window_s: float = 1.0,
    line_slope_step_s_per_station: float = 0.2,
    line_count: int = 70,
    line_match_tolerance_s: float = 0.8,
    peak_dedup_gap_s: float = 1.0,
) -> LineFilterResult:
    """Run raw-energy line detection and peak filtering."""
    raw_list = _as_station_list(raw_signals, "raw_signals")
    time_list = _as_station_list(candidate_times_s, "candidate_times_s")
    if len(raw_list) != len(time_list):
        raise ValueError("raw_signals and candidate_times_s must have the same station count")
    if candidate_scores is None:
        score_list = [np.ones(times.size, dtype=np.float32) for times in time_list]
    else:
        score_list = _as_station_list(candidate_scores, "candidate_scores")
        if len(score_list) != len(time_list):
            raise ValueError("candidate times and scores must have the same station count")
    energy, energy_dt_s = compute_energy_matrix(raw_list, fs_hz=fs_hz, window_s=energy_window_s)
    lines = scan_raw_energy_lines(
        energy,
        energy_dt_s=energy_dt_s,
        station_spacing_m=station_spacing_m,
        speed_min_kmh=speed_min_kmh,
        speed_max_kmh=speed_max_kmh,
        slope_step_s_per_station=line_slope_step_s_per_station,
        top_k=line_count,
    )
    selected = match_peaks_to_lines(
        time_list,
        score_list,
        lines,
        tolerance_s=line_match_tolerance_s,
        dedup_gap_s=peak_dedup_gap_s,
    )
    return LineFilterResult(
        lines=lines,
        candidate_times_s=time_list,
        candidate_scores=score_list,
        selected_times_s=selected[0],
        selected_scores=selected[1],
        selected_line_ids=selected[2],
        energy_envelope=energy,
        energy_dt_s=energy_dt_s,
    )


def run_from_probability_arrays(
    raw_signals: Sequence[np.ndarray],
    probability_signals: Sequence[np.ndarray],
    *,
    fs_hz: float,
    candidate_threshold: float | Sequence[float] = 0.3,
    min_gap_s: float = 2.0,
    flip_probability: bool = True,
    normalize_probability: bool = True,
    **filter_kwargs: object,
) -> LineFilterResult:
    """Pick probability peaks and then apply the raw-energy line filter."""
    probability_list = _as_station_list(probability_signals, "probability_signals")
    if isinstance(candidate_threshold, (float, int)):
        thresholds = [float(candidate_threshold)] * len(probability_list)
    else:
        thresholds = [float(value) for value in candidate_threshold]
        if len(thresholds) != len(probability_list):
            raise ValueError("candidate_threshold sequence must match station count")
    times = []
    scores = []
    for probability, threshold in zip(probability_list, thresholds):
        station_times, station_scores = pick_probability_peaks(
            probability,
            fs_hz=fs_hz,
            threshold=threshold,
            min_gap_s=min_gap_s,
            flip=flip_probability,
            normalize=normalize_probability,
        )
        times.append(station_times)
        scores.append(station_scores)
    return run_line_consistent_filter(
        raw_signals,
        times,
        scores,
        fs_hz=fs_hz,
        **filter_kwargs,
    )


def _station_id(path: Path) -> str | None:
    match = STATION_ID_RE.search(path.name)
    return match.group(1).upper() if match else None


def _load_station_order(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        rows = data.get("workbook_station_rows") or data.get("selected_channels") or data.get("rows")
    else:
        rows = data
    if not isinstance(rows, list):
        raise ValueError(f"unsupported station order JSON: {path}")
    result = []
    for row in rows:
        station = row if isinstance(row, str) else row.get("station_id")
        if station is not None:
            result.append(str(station).strip().upper())
    return result or None


def _scan_directory_pairs(raw_dir: Path, probability_dir: Path, order_json: Path | None) -> list[tuple[str, Path, Path]]:
    raw = {_station_id(path): path for path in raw_dir.glob("*.npy") if _station_id(path)}
    probabilities = {_station_id(path): path for path in probability_dir.glob("*.npy") if _station_id(path)}
    station_order = _load_station_order(order_json) or sorted(raw)
    pairs = []
    for station in station_order:
        if station in raw and station in probabilities:
            pairs.append((station, raw[station], probabilities[station]))
    if not pairs:
        raise FileNotFoundError("no matching raw/probability station files were found")
    return pairs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--probability-dir", required=True, type=Path)
    parser.add_argument("--order-json", type=Path, help="Workbook mapping JSON used to order stations")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fs-hz", type=float, default=1000.0)
    parser.add_argument("--station-spacing-m", type=float, default=100.0)
    parser.add_argument("--speed-min-kmh", type=float, default=60.0)
    parser.add_argument("--speed-max-kmh", type=float, default=100.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.3)
    parser.add_argument("--min-gap-s", type=float, default=2.0)
    parser.add_argument("--line-count", type=int, default=70)
    parser.add_argument("--line-match-tolerance-s", type=float, default=0.8)
    parser.add_argument("--no-flip-probability", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pairs = _scan_directory_pairs(args.raw_dir, args.probability_dir, args.order_json)
    raw_signals = [np.load(raw_path, mmap_mode="r") for _, raw_path, _ in pairs]
    probability_signals = [np.load(probability_path, mmap_mode="r") for _, _, probability_path in pairs]
    result = run_from_probability_arrays(
        raw_signals,
        probability_signals,
        fs_hz=args.fs_hz,
        candidate_threshold=args.candidate_threshold,
        min_gap_s=args.min_gap_s,
        flip_probability=not args.no_flip_probability,
        station_spacing_m=args.station_spacing_m,
        speed_min_kmh=args.speed_min_kmh,
        speed_max_kmh=args.speed_max_kmh,
        line_count=args.line_count,
        line_match_tolerance_s=args.line_match_tolerance_s,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lines.json").write_text(
        json.dumps([asdict(line) for line in result.lines], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.out_dir / "selected_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["station_index", "station_id", "peak_time_s", "peak_score", "line_id"])
        writer.writeheader()
        for station_index, ((station_id, _, _), times, scores, line_ids) in enumerate(
            zip(pairs, result.selected_times_s, result.selected_scores, result.selected_line_ids)
        ):
            for time_s, score, line_id in zip(times, scores, line_ids):
                writer.writerow({"station_index": station_index, "station_id": station_id, "peak_time_s": float(time_s), "peak_score": float(score), "line_id": int(line_id)})
    summary = {
        "station_count": len(pairs),
        "candidate_threshold": args.candidate_threshold,
        "candidate_peak_count": int(sum(len(value) for value in result.candidate_times_s)),
        "line_count": len(result.lines),
        "selected_peak_count": int(sum(len(value) for value in result.selected_times_s)),
        "fs_hz": args.fs_hz,
        "station_spacing_m": args.station_spacing_m,
        "speed_min_kmh": args.speed_min_kmh,
        "speed_max_kmh": args.speed_max_kmh,
        "line_match_tolerance_s": args.line_match_tolerance_s,
        "station_ids": [station_id for station_id, _, _ in pairs],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
