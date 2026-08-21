"""Neutral signal primitives shared by the classic physical trackers.

The module intentionally contains no association or filtering policy.  It only
normalizes peak candidates and geometry so the Hungarian and Kalman trackers
can be compared on exactly the same input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks

from .types import StationGeometry, Track, TrackPoint


@dataclass(frozen=True)
class PeakCandidate:
    channel_index: int
    sample_index: int
    time_s: float
    score: float
    amplitude: float


def detect_peak_candidates(
    data_channel_time: np.ndarray,
    *,
    sample_rate_hz: float,
    prominence: float = 0.4,
    min_distance_s: float = 0.5,
    wlen_s: float = 2.0,
) -> list[list[PeakCandidate]]:
    """Detect comparable candidates on a ``[channel, time]`` array."""

    data = np.asarray(data_channel_time, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("data_channel_time must have shape [channel, time]")
    fs = float(sample_rate_hz)
    if fs <= 0:
        raise ValueError("sample_rate_hz must be positive")
    distance = max(1, int(round(float(min_distance_s) * fs)))
    wlen = max(3, int(round(float(wlen_s) * fs)))
    result: list[list[PeakCandidate]] = []
    for channel, row in enumerate(data):
        signal = np.abs(np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0))
        peaks, props = find_peaks(signal, prominence=float(prominence), distance=distance, wlen=wlen)
        if peaks.size == 0:
            result.append([])
            continue
        prominences = np.asarray(props.get("prominences", np.zeros(peaks.size)), dtype=np.float64)
        amplitudes = signal[peaks].astype(np.float64, copy=False)
        scale = max(float(np.median(amplitudes)), 1e-6)
        scores = prominences + 0.2 * amplitudes / scale
        result.append([
            PeakCandidate(
                channel_index=int(channel),
                sample_index=int(sample),
                time_s=float(sample) / fs,
                score=float(score),
                amplitude=float(amplitude),
            )
            for sample, score, amplitude in zip(peaks, scores, amplitudes)
        ])
    return result


def travel_positions(geometry: StationGeometry, *, direction: str = "reverse") -> np.ndarray:
    """Return distance along the configured travel direction from the origin."""

    positions = np.asarray(geometry.positions_m, dtype=np.float64)
    if direction == "reverse":
        return positions[-1] - positions
    if direction == "forward":
        return positions - positions[0]
    raise ValueError("direction must be `forward` or `reverse`")


def speed_kmh(points: Iterable[TrackPoint], geometry: StationGeometry) -> float:
    ordered = sorted(points, key=lambda point: float(point.position_m))
    if len(ordered) < 2:
        return float("nan")
    speeds: list[float] = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        distance = abs(float(right.position_m) - float(left.position_m))
        dt = abs(float(right.time_s) - float(left.time_s))
        if distance > 0 and dt > 1e-9:
            speeds.append(3.6 * distance / dt)
    return float(np.median(speeds)) if speeds else float("nan")


def track_score(points: Iterable[TrackPoint]) -> float:
    observed = [point for point in points if point.observed]
    return float(sum(float(point.score or 0.0) for point in observed))


def as_track(
    track_id: str,
    points: list[TrackPoint],
    geometry: StationGeometry,
    *,
    direction: str = "reverse",
    confidence: float | None = None,
) -> Track:
    ordered = sorted(points, key=lambda point: (float(point.time_s), int(point.channel_index)))
    return Track(
        track_id=str(track_id),
        points=ordered,
        direction=str(direction),
        median_speed_kmh=speed_kmh(ordered, geometry),
        confidence=confidence,
        score=track_score(ordered),
    )


def deduplicate_tracks(
    tracks: list[Track],
    *,
    overlap_ratio: float = 0.5,
    residual_s: float = 0.3,
) -> list[Track]:
    """Keep the stronger track when two decoders return the same vehicle."""

    def observed_map(track: Track) -> dict[int, float]:
        return {
            int(point.channel_index): float(point.time_s)
            for point in track.points
            if point.observed
        }

    ordered = sorted(
        tracks,
        key=lambda track: (sum(point.observed for point in track.points), float(track.score or 0.0)),
        reverse=True,
    )
    kept: list[Track] = []
    kept_maps: list[dict[int, float]] = []
    for track in ordered:
        current = observed_map(track)
        duplicate = False
        for previous in kept_maps:
            common = set(current).intersection(previous)
            if not common:
                continue
            ratio = len(common) / max(1, min(len(current), len(previous)))
            residual = float(np.median([abs(current[ch] - previous[ch]) for ch in common]))
            if ratio >= float(overlap_ratio) and residual <= float(residual_s):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
            kept_maps.append(current)
    return kept
