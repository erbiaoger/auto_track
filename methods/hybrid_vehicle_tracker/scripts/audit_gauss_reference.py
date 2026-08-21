#!/usr/bin/env python3
"""Audit DAY11 Gauss peak statistics without writing into the source dataset.

This is a reproducibility utility for the statistics-only calibration used by
``vehicle_peakset_complex_v2``.  It reads ``gauss[:120000]`` and writes a small
JSON report; it never exports, caches, or uses the measured array as training
input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, peak_widths

from hybrid_vehicle_tracker.data.peakset_reference import reference_summary


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gauss",
        type=Path,
        default=Path(
            "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/"
            "converted_50ch_station_fallback_q65_s8/gauss_station_fallback_DAY11.npy"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    return parser.parse_args()


def main() -> None:
    args = _args()
    source = np.load(args.gauss, mmap_mode="r")
    sample_count = min(source.shape[0], int(round(args.duration_s * args.sample_rate_hz)))
    window = np.asarray(source[:sample_count], dtype=np.float32)
    rows: list[dict[str, float | int]] = []
    for channel in range(window.shape[1]):
        peaks, properties = find_peaks(
            window[:, channel],
            height=0.5,
            distance=max(1, int(round(1.25 * args.sample_rate_hz))),
            prominence=0.02,
        )
        if peaks.size == 0:
            continue
        widths = peak_widths(window[:, channel], peaks, rel_height=0.5)[0] / args.sample_rate_hz
        heights = np.asarray(properties["peak_heights"], dtype=np.float64)
        for peak, height, width in zip(peaks, heights, widths):
            rows.append(
                {
                    "channel_index": int(channel),
                    "time_s": float(peak / args.sample_rate_hz),
                    "height": float(height),
                    "fwhm_s": float(width),
                }
            )
    rows.sort(key=lambda row: (row["channel_index"], row["time_s"]))
    payload = {
        "reference_constants": reference_summary(),
        "read_only_source": str(args.gauss),
        "window_samples": sample_count,
        "event_count": len(rows),
        "event_count_per_station": np.bincount(
            [int(row["channel_index"]) for row in rows], minlength=window.shape[1]
        ).astype(int).tolist(),
        "height_quantiles": np.quantile([row["height"] for row in rows], np.linspace(0.0, 1.0, 8)).tolist(),
        "fwhm_quantiles_s": np.quantile([row["fwhm_s"] for row in rows], [0.0, 0.25, 0.5, 0.75, 1.0]).tolist(),
        # A local event is called isolated when no event is within 2 s on the
        # same or an adjacent station.  This is only a calibration statistic.
        "isolated_event_count": sum(
            not any(
                other is not row
                and abs(float(other["time_s"]) - float(row["time_s"])) <= 2.0
                and abs(int(other["channel_index"]) - int(row["channel_index"])) <= 2
                for other in rows
            )
            for row in rows
        ),
        "training_samples_exported": False,
    }
    payload["isolated_event_fraction"] = payload["isolated_event_count"] / max(len(rows), 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("event_count", "isolated_event_count", "isolated_event_fraction")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
