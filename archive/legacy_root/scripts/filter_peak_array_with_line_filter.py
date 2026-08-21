#!/usr/bin/env python3
"""Apply the raw-energy line filter directly to a peak/Gaussian array.

Unlike the normal raw-data workflow, ``--input`` here is already a
``[time, station]`` peak array.  The same array is used as the energy image
and as the source of candidate peaks; no original waveform and no neural
network inference are involved.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from autotrack.dl.raw_energy_line_filter import (
    compute_energy_matrix,
    gaussian_curve_from_picks,
    match_peaks_to_lines,
    pick_probability_peaks,
    scan_raw_energy_lines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fs-hz", type=float, default=1000.0)
    parser.add_argument("--station-spacing-m", type=float, default=100.0)
    parser.add_argument("--speed-min-kmh", type=float, default=60.0)
    parser.add_argument("--speed-max-kmh", type=float, default=100.0)
    parser.add_argument("--candidate-threshold", type=float, default=0.05)
    parser.add_argument("--min-gap-s", type=float, default=2.0)
    parser.add_argument("--line-count", type=int, default=100)
    parser.add_argument("--line-score-quantile", type=float, default=0.65)
    parser.add_argument("--min-valid-stations", type=int, default=40)
    parser.add_argument("--line-tolerance-s", type=float, default=0.8)
    parser.add_argument("--plot-lines", type=int, default=30)
    args = parser.parse_args()

    array = np.load(args.input, mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"expected [time, station], got {array.shape}")
    signals = [array[:, station] for station in range(array.shape[1])]
    candidate_times = []
    candidate_scores = []
    for signal in signals:
        times, scores = pick_probability_peaks(
            signal,
            fs_hz=args.fs_hz,
            threshold=args.candidate_threshold,
            min_gap_s=args.min_gap_s,
            flip=False,
            normalize=True,
        )
        candidate_times.append(times)
        candidate_scores.append(scores)

    energy, energy_dt_s = compute_energy_matrix(signals, fs_hz=args.fs_hz, window_s=1.0)
    lines = scan_raw_energy_lines(
        energy,
        energy_dt_s=energy_dt_s,
        station_spacing_m=args.station_spacing_m,
        speed_min_kmh=args.speed_min_kmh,
        speed_max_kmh=args.speed_max_kmh,
        slope_step_s_per_station=0.2,
        top_k=args.line_count,
    )
    score_cutoff = float(np.quantile([line.score for line in lines], args.line_score_quantile))
    selected_lines = [
        line
        for line in lines
        if line.score >= score_cutoff and line.valid_station_count >= args.min_valid_stations
    ]
    selected_times, selected_scores, selected_line_ids = match_peaks_to_lines(
        candidate_times,
        candidate_scores,
        selected_lines,
        tolerance_s=args.line_tolerance_s,
        dedup_gap_s=args.min_gap_s,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "lines.json").write_text(
        json.dumps(
            [
                {
                    "score": line.score,
                    "slope_s_per_station": line.slope_s_per_station,
                    "intercept_s": line.intercept_s,
                    "valid_station_count": line.valid_station_count,
                    "selected": line in selected_lines,
                }
                for line in lines
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with (args.out_dir / "selected_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_index", "time_s", "score", "line_id"])
        for station, (times, scores, line_ids) in enumerate(
            zip(selected_times, selected_scores, selected_line_ids)
        ):
            writer.writerows(
                [station, float(time), float(score), int(line_id)]
                for time, score, line_id in zip(times, scores, line_ids)
            )

    filtered = np.lib.format.open_memmap(
        args.out_dir / "peaks_line_filtered.npy",
        mode="w+",
        dtype=np.float32,
        shape=array.shape,
    )
    for station, signal in enumerate(signals):
        filtered[:, station] = gaussian_curve_from_picks(
            signal,
            selected_times[station],
            fs_hz=args.fs_hz,
            width_s=0.5,
            amplitude_scale=0.8,
        )
    filtered.flush()

    # Inspection plot: the input peaks are gray wiggles, selected peaks are
    # red dots, and raw-energy candidates are red guide lines.
    n_time, n_station = array.shape
    t_axis = np.arange(n_time, dtype=np.float64) / float(args.fs_hz)
    plot_stride = max(1, int(round(float(args.fs_hz) * 0.01)))
    t_plot = t_axis[::plot_stride]
    x_axis = np.arange(n_station, dtype=np.float64) * float(args.station_spacing_m) * 1e-3
    vmax = max(float(np.quantile(np.abs(np.asarray(array[::plot_stride], dtype=np.float32)), 0.995)), 1e-8)
    spacing = float(np.median(np.diff(x_axis))) if n_station > 1 else 1.0
    wiggle = 0.27 * spacing
    fig, ax = plt.subplots(figsize=(15, 8))
    for station in range(n_station):
        ratio = np.clip(np.asarray(array[::plot_stride, station], dtype=np.float64) / vmax, -1.35, 1.35)
        ax.plot(x_axis[station] + ratio * wiggle, t_plot, color="0.45", lw=0.55, alpha=0.8)
    colors = plt.get_cmap("tab20", max(1, min(args.plot_lines, len(selected_lines))))
    for line_index, line in enumerate(selected_lines[: args.plot_lines]):
        xs = []
        ys = []
        for station in range(n_station):
            time_s = line.time_at_station(station)
            if 0.0 <= time_s <= t_axis[-1]:
                xs.append(x_axis[station])
                ys.append(time_s)
        if len(xs) >= 2:
            ax.plot(xs, ys, color=colors(line_index % max(1, colors.N)), lw=1.7, alpha=0.85)
    for station, times in enumerate(selected_times):
        if times.size:
            ax.scatter(
                np.full(times.size, x_axis[station]),
                times,
                s=10,
                c="#d62728",
                alpha=0.85,
                zorder=5,
            )
    ax.set_xlim(x_axis[0] - 0.05, x_axis[-1] + 0.05)
    ax.set_ylim(0.0, t_axis[-1])
    ax.invert_yaxis()
    ax.set_xlabel("Channel offset [km]")
    ax.set_ylabel("Time (s)")
    ax.set_title(
        f"Line filter directly on peaks: {len(selected_lines)} lines, "
        f"{sum(len(x) for x in selected_times)} selected peaks"
    )
    fig.tight_layout()
    fig.savefig(args.out_dir / "peaks_line_filter_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (args.out_dir / "summary.json").write_text(
        json.dumps(
            {
                "input": str(args.input),
                "input_shape_time_station": list(array.shape),
                "candidate_threshold": args.candidate_threshold,
                "candidate_peak_count": int(sum(len(x) for x in candidate_times)),
                "line_score_quantile": args.line_score_quantile,
                "line_score_cutoff": score_cutoff,
                "min_valid_stations": args.min_valid_stations,
                "selected_line_count": len(selected_lines),
                "selected_peak_count": int(sum(len(x) for x in selected_times)),
                "output_array": "peaks_line_filtered.npy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((args.out_dir / "summary.json").read_text()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
