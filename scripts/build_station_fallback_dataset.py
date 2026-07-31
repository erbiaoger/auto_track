#!/usr/bin/env python3
"""Build a station-level fallback Gaussian dataset for vehicle picking.

The base candidates are picked with a strict, station-specific threshold.
Fallback candidates are picked with a lower station-specific threshold, but
are added only when they lie close to a strong raw-energy diagonal supported
by several stations.  This is deliberately different from lowering the
threshold globally: isolated low-confidence peaks are never added.

The output is a normal ``[time, station]`` ``.npy`` array in workbook order,
plus a time-reversed copy suitable for the current repro pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from autotrack.dl.raw_energy_line_filter import (
    LineCandidate,
    gaussian_curve_from_picks,
    pick_probability_peaks,
)


def _station_id(path: Path) -> str:
    match = re.search(r"_([^_]+)_EHZ_", path.name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"cannot parse station id from {path.name}")
    return match.group(1).upper()


def _load_thresholds(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = csv.DictReader(handle)
            return {
                str(row["station_id"]).upper(): float(row["adaptive_threshold"])
                for row in rows
            }
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    return {
        str(row["station_id"]).upper(): float(
            row.get("adaptive_threshold", row.get("threshold"))
        )
        for row in rows
    }


def _nearest_candidate(
    times: np.ndarray, scores: np.ndarray, target_s: float, tolerance_s: float
) -> tuple[float, float] | None:
    if times.size == 0:
        return None
    index = int(np.searchsorted(times, target_s))
    candidates = [min(max(index, 0), times.size - 1)]
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda item: abs(float(times[item]) - target_s))
    if abs(float(times[best]) - target_s) > tolerance_s:
        return None
    return float(times[best]), float(scores[best])


def _load_lines(path: Path) -> list[LineCandidate]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [LineCandidate(**row) for row in rows]


def _apply_agc(signal: np.ndarray, fs_hz: float, window_s: float) -> np.ndarray:
    """Match the AGC used by ``0002pre2guass.py``."""
    values = np.asarray(signal, dtype=np.float64)
    half = max(1, int(window_s * fs_hz / 2.0))
    cumulative = np.concatenate(([0.0], np.cumsum(values * values)))
    indices0 = np.maximum(0, np.arange(values.size) - half)
    indices1 = np.minimum(values.size, np.arange(values.size) + half + 1)
    rms = np.sqrt((cumulative[indices1] - cumulative[indices0]) / (indices1 - indices0))
    output = values / np.maximum(rms, 1e-10)
    clip_value = np.percentile(np.abs(output), 99.5)
    return (np.clip(output, -clip_value, clip_value) / (clip_value + 1e-10)).astype(
        np.float32
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--base-thresholds", type=Path, required=True)
    parser.add_argument("--fallback-thresholds", type=Path, required=True)
    parser.add_argument("--lines", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fs-hz", type=float, default=1000.0)
    parser.add_argument("--line-score-quantile", type=float, default=0.75)
    parser.add_argument("--min-fallback-support", type=int, default=8)
    parser.add_argument("--line-tolerance-s", type=float, default=0.8)
    parser.add_argument("--min-gap-s", type=float, default=2.0)
    parser.add_argument("--agc-window-s", type=float, default=10.0)
    parser.add_argument("--gaussian-width-s", type=float, default=0.5)
    parser.add_argument("--gaussian-amplitude-scale", type=float, default=0.8)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    selected = mapping["selected_channels"]
    station_ids = [str(row["station_id"]).upper() for row in selected]
    base_thresholds = _load_thresholds(args.base_thresholds)
    fallback_thresholds = _load_thresholds(args.fallback_thresholds)
    probability_files = {_station_id(path): path for path in args.probability_dir.glob("*.npy")}
    raw_files = {_station_id(path): path for path in args.raw_dir.glob("*.npy")}
    missing = [sid for sid in station_ids if sid not in probability_files or sid not in raw_files]
    if missing:
        raise FileNotFoundError(f"missing selected stations: {missing}")

    base_times: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    fallback_times: list[np.ndarray] = []
    fallback_scores: list[np.ndarray] = []
    raw_max = 0.0
    signal_length = None
    for sid in station_ids:
        probability = np.load(probability_files[sid], mmap_mode="r")
        base = pick_probability_peaks(
            probability,
            fs_hz=args.fs_hz,
            threshold=base_thresholds[sid],
            min_gap_s=args.min_gap_s,
            flip=True,
            normalize=True,
        )
        fallback = pick_probability_peaks(
            probability,
            fs_hz=args.fs_hz,
            threshold=fallback_thresholds[sid],
            min_gap_s=args.min_gap_s,
            flip=True,
            normalize=True,
        )
        base_times.append(base[0])
        base_scores.append(base[1])
        fallback_times.append(fallback[0])
        fallback_scores.append(fallback[1])
        signal_length = int(probability.size) if signal_length is None else signal_length
        raw_max = max(raw_max, float(np.max(np.abs(np.load(raw_files[sid], mmap_mode="r")))))

    if signal_length is None or raw_max <= 0:
        raise ValueError("empty input or zero raw amplitude")

    lines = _load_lines(args.lines)
    score_cutoff = float(np.quantile([line.score for line in lines], args.line_score_quantile))
    strong_lines = [line for line in lines if line.score >= score_cutoff]
    selected_lines: list[LineCandidate] = []
    line_support_rows = []
    for line in strong_lines:
        base_support = 0
        fallback_support = 0
        for station_index in range(len(station_ids)):
            target = line.time_at_station(station_index)
            if _nearest_candidate(
                base_times[station_index], base_scores[station_index], target, args.line_tolerance_s
            ) is not None:
                base_support += 1
            if _nearest_candidate(
                fallback_times[station_index],
                fallback_scores[station_index],
                target,
                args.line_tolerance_s,
            ) is not None:
                fallback_support += 1
        keep = fallback_support >= args.min_fallback_support
        line_support_rows.append(
            {
                "score": line.score,
                "slope_s_per_station": line.slope_s_per_station,
                "intercept_s": line.intercept_s,
                "valid_station_count": line.valid_station_count,
                "base_support": base_support,
                "fallback_support": fallback_support,
                "selected": keep,
            }
        )
        if keep:
            selected_lines.append(line)

    # Add only fallback candidates that are close to a selected raw line and
    # are not already represented by a strict/base candidate.
    final_times: list[np.ndarray] = []
    final_scores: list[np.ndarray] = []
    fallback_added_counts = []
    fallback_added_times: list[np.ndarray] = []
    for station_index, sid in enumerate(station_ids):
        additions: list[tuple[float, float]] = []
        for line in selected_lines:
            target = line.time_at_station(station_index)
            base_match = _nearest_candidate(
                base_times[station_index], base_scores[station_index], target, args.line_tolerance_s
            )
            if base_match is not None:
                continue
            fallback_match = _nearest_candidate(
                fallback_times[station_index],
                fallback_scores[station_index],
                target,
                args.line_tolerance_s,
            )
            if fallback_match is not None:
                additions.append(fallback_match)
        additions.sort(key=lambda item: item[0])
        deduped: list[tuple[float, float]] = []
        for item in additions:
            if deduped and item[0] - deduped[-1][0] < args.min_gap_s:
                if item[1] > deduped[-1][1]:
                    deduped[-1] = item
            else:
                deduped.append(item)
        add_times = np.asarray([item[0] for item in deduped], dtype=np.float64)
        add_scores = np.asarray([item[1] for item in deduped], dtype=np.float32)
        final_t = np.sort(np.concatenate((base_times[station_index], add_times)))
        # Scores are only needed for the audit; all final picks are rendered as
        # Gaussians using the same amplitude rule as the original conversion.
        final_s = np.ones(final_t.size, dtype=np.float32)
        final_times.append(final_t)
        final_scores.append(final_s)
        fallback_added_times.append(add_times)
        fallback_added_counts.append(int(add_times.size))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_signals = [np.load(raw_files[sid], mmap_mode="r") for sid in station_ids]
    global_scale = raw_max
    forward_path = args.output_dir / "gauss_station_fallback_DAY11.npy"
    forward = np.lib.format.open_memmap(
        forward_path, mode="w+", dtype=np.float32, shape=(signal_length, len(station_ids))
    )
    for station_index, raw in enumerate(raw_signals):
        normalized = np.asarray(raw, dtype=np.float32) / global_scale
        processed = _apply_agc(normalized, args.fs_hz, args.agc_window_s)
        forward[:, station_index] = gaussian_curve_from_picks(
            processed,
            final_times[station_index],
            fs_hz=args.fs_hz,
            width_s=args.gaussian_width_s,
            amplitude_scale=args.gaussian_amplitude_scale,
        )
    forward.flush()
    reversed_path = args.output_dir / "gauss_station_fallback_DAY11_time_reversed.npy"
    reversed_array = np.lib.format.open_memmap(
        reversed_path, mode="w+", dtype=np.float32, shape=forward.shape
    )
    reversed_array[:] = forward[::-1]
    reversed_array.flush()

    rows = []
    for index, sid in enumerate(station_ids):
        rows.append(
            {
                "sequence": index + 1,
                "station_id": sid,
                "base_threshold": base_thresholds[sid],
                "fallback_threshold": fallback_thresholds[sid],
                "base_pick_count": int(base_times[index].size),
                "fallback_added_count": fallback_added_counts[index],
                "final_pick_count": int(final_times[index].size),
            }
        )
    with (args.output_dir / "station_fallback_thresholds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "selected_lines.json").write_text(
        json.dumps(line_support_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "mapping.json").write_text(
        json.dumps(
            {
                "source": "strict station threshold + raw-energy line-gated station fallback",
                "shape_time_channel": [signal_length, len(station_ids)],
                "fs_hz": args.fs_hz,
                "line_score_quantile": args.line_score_quantile,
                "line_score_cutoff": score_cutoff,
                "min_fallback_support": args.min_fallback_support,
                "line_tolerance_s": args.line_tolerance_s,
                "selected_line_count": len(selected_lines),
                "station_order": station_ids,
                "forward_array": forward_path.name,
                "time_reversed_array": reversed_path.name,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_lines": len(selected_lines),
        "line_score_cutoff": score_cutoff,
        "base_picks": int(sum(row["base_pick_count"] for row in rows)),
        "fallback_added": int(sum(row["fallback_added_count"] for row in rows)),
        "final_picks": int(sum(row["final_pick_count"] for row in rows)),
        "forward_array": str(forward_path),
        "time_reversed_array": str(reversed_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
