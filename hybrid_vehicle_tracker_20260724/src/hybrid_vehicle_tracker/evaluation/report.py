from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from hybrid_vehicle_tracker.config import TrackerConfig
from hybrid_vehicle_tracker.tracker import InferenceArtifacts
from hybrid_vehicle_tracker.types import TrackBatch


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _plot_overlay(
    batch: TrackBatch,
    artifacts: InferenceArtifacts,
    path: Path,
    *,
    time_window: tuple[float, float] | None = None,
    time_offset_s: float = 0.0,
) -> None:
    """Render the reference-style waveform/track overlay.

    Each station's threshold-generated Gauss-window curve is drawn as a small horizontal
    displacement around its physical position.  A selected real observation
    is a square; a point inserted by the continuous-track fit (missing station
    or an in-window prediction) is an open circle.
    """
    figure, axis = plt.subplots(figsize=(15, 9), constrained_layout=True)
    positions_km = artifacts.features.coordinates
    # Coordinates in FeatureBatch are normalized.  Track points carry the
    # authoritative physical coordinates, so recover the station positions
    # from the selected observations/track points for the waveform backdrop.
    station_positions: dict[int, float] = {}
    if artifacts.station_positions_m is not None:
        station_positions.update(
            {index: float(position) / 1000.0 for index, position in enumerate(artifacts.station_positions_m)}
        )
    for item in batch.observations:
        station_positions[item.channel_index] = item.position_m / 1000.0
    for track in batch.tracks:
        for point in track.points:
            station_positions[point.channel_index] = point.position_m / 1000.0

    gauss = artifacts.gauss_waveform
    bins = int(artifacts.features.time_bins)
    pooled = None
    if gauss is not None and gauss.ndim == 2 and gauss.shape[1] == len(artifacts.features.coordinates):
        usable = gauss.shape[0] - gauss.shape[0] % max(1, bins)
        if usable > 0:
            pooled = np.max(
                gauss[:usable].reshape(bins, usable // bins, gauss.shape[1]), axis=1
            )
    if pooled is None:
        # Artifacts made by external callers may only retain the binned Gauss
        # plane; still never fall back to Raw for this reference-style plot.
        pooled = np.asarray(artifacts.features.gauss_score.T, dtype=np.float32)
    if pooled.shape[0] > 0:
            low = np.quantile(pooled, 0.50, axis=0, keepdims=True)
            high = np.quantile(pooled, 0.995, axis=0, keepdims=True)
            trace = np.clip((pooled - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
            time_axis = np.linspace(0.0, batch.duration_s, bins, endpoint=False) + time_offset_s
            if station_positions:
                spacing = np.median(np.diff(sorted(station_positions.values())))
            else:
                spacing = 0.1
            trace_width = float(np.clip(0.32 * spacing, 0.018, 0.045))
            for channel, position in station_positions.items():
                if channel >= trace.shape[1]:
                    continue
                axis.plot(
                    position + trace[:, channel] * trace_width,
                    time_axis,
                    color="0.40",
                    linewidth=0.55,
                    alpha=0.85,
                    zorder=1,
                )
                axis.plot(
                    [position, position],
                    [0.0, batch.duration_s],
                    color="0.25",
                    linewidth=0.35,
                    alpha=0.55,
                    zorder=0,
                )

    # Candidate observations remain available in observations.csv, but are
    # intentionally omitted from this visual so the Gauss windows and selected
    # vehicle tracks are not obscured by weak/noise candidates.
    palette = plt.get_cmap("tab20")
    for index, track in enumerate(batch.tracks):
        color = palette(index % 20)
        axis.plot(
            [point.position_m / 1000.0 for point in track.points],
            [point.time_s + time_offset_s for point in track.points],
            color=color,
            linewidth=1.7,
            label=f"{track.track_id} {track.median_speed_kmh:.1f} km/h",
            zorder=4,
        )
        observed = [point for point in track.points if point.observed]
        estimated = [point for point in track.points if not point.observed]
        if observed:
            axis.scatter(
                [point.position_m / 1000.0 for point in observed],
                [point.time_s + time_offset_s for point in observed],
                marker="s",
                s=30,
                facecolors=color,
                edgecolors="white",
                linewidths=0.55,
                label="observed (square)" if index == 0 else None,
                zorder=6,
            )
        if estimated:
            axis.scatter(
                [point.position_m / 1000.0 for point in estimated],
                [point.time_s + time_offset_s for point in estimated],
                marker="o",
                s=38,
                facecolors="none",
                edgecolors=color,
                linewidths=1.25,
                label="estimated / missing (circle)" if index == 0 else None,
                zorder=7,
            )
        if track.points:
            anchor = track.points[len(track.points) // 2]
            axis.text(
                anchor.position_m / 1000.0 + 0.025,
                anchor.time_s + time_offset_s - 1.0,
                f"{track.median_speed_kmh:.1f} km/h",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
                alpha=0.95,
                zorder=8,
                clip_on=True,
            )
    axis.set_xlabel("physical station position (km)")
    axis.set_ylabel("time within window (s)")
    if time_window is None:
        axis.set_ylim(batch.duration_s + time_offset_s, time_offset_s)
        window_title = f"{time_offset_s:.0f}–{time_offset_s + batch.duration_s:.0f} s"
    else:
        start_s, end_s = time_window[0] + time_offset_s, time_window[1] + time_offset_s
        axis.set_ylim(end_s, start_s)
        window_title = f"{start_s:.0f}–{end_s:.0f} s"
    axis.set_title(
        "Gauss-window peaks + vehicle association | "
        f"{window_title} | square=observed, circle=estimated/missing"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_probability(artifacts: InferenceArtifacts, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, constrained_layout=True)
    extent = [0.0, artifacts.features.duration_s, artifacts.features.station_count - 0.5, -0.5]
    axes[0].imshow(
        artifacts.network_probability,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    axes[0].set_title("multimodal trajectory probability")
    axes[1].imshow(
        artifacts.crossing_probability,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
    )
    axes[1].set_title("crossing / merged-observation probability")
    axes[1].set_xlabel("time (s)")
    for axis in axes:
        axis.set_ylabel("channel")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_hough(artifacts: InferenceArtifacts, path: Path) -> None:
    speeds = 3.6 / np.abs(artifacts.hough_slopes)
    extent = [
        float(artifacts.hough_intercepts[0]),
        float(artifacts.hough_intercepts[-1]),
        float(speeds[-1]),
        float(speeds[0]),
    ]
    figure, axis = plt.subplots(figsize=(13, 5), constrained_layout=True)
    axis.imshow(artifacts.hough_scores, aspect="auto", cmap="magma", extent=extent)
    axis.set_xlabel("time at first physical station (s)")
    axis.set_ylabel("speed (km/h)")
    direction = "increasing time with position" if artifacts.hough_slopes[0] > 0 else "increasing time with decreasing position"
    axis.set_title(f"physical Hough score (60–90 km/h, {direction})")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_graph(batch: TrackBatch, artifacts: InferenceArtifacts, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 7), constrained_layout=True)
    observations = batch.observations
    graph = artifacts.association.graph
    if graph.edge_index.shape[1]:
        ranked = np.argsort(graph.edge_scores)[::-1][:5000]
        for edge_id in ranked:
            source = observations[int(graph.edge_index[0, edge_id])]
            target = observations[int(graph.edge_index[1, edge_id])]
            alpha = 0.03 + 0.20 * float(graph.edge_scores[edge_id])
            axis.plot(
                [source.position_m / 1000.0, target.position_m / 1000.0],
                [source.time_s, target.time_s],
                color="tab:blue",
                alpha=alpha,
                linewidth=0.4,
            )
    if observations:
        axis.scatter(
            [item.position_m / 1000.0 for item in observations],
            [item.time_s for item in observations],
            s=7,
            color="black",
        )
    axis.set_ylim(batch.duration_s, 0.0)
    axis.set_xlabel("physical station position (km)")
    axis.set_ylabel("time (s)")
    axis.set_title("physics-gated candidate association graph")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_prediction_outputs(
    batch: TrackBatch,
    artifacts: InferenceArtifacts,
    output_dir: str | Path,
    *,
    config: TrackerConfig | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "tracks.jsonl").open("w", encoding="utf-8") as handle:
        for track in batch.tracks:
            handle.write(json.dumps(track.to_dict(), ensure_ascii=False) + "\n")
    with (destination / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "observation_id",
                "channel_index",
                "station_id",
                "position_m",
                "time_s",
                "gauss_score",
                "pre_score",
                "raw_energy",
                "network_score",
                "crossing_score",
                "strong",
                "ambiguous",
                "evidence_score",
            ],
        )
        writer.writeheader()
        for item in batch.observations:
            row = vars(item).copy()
            row.pop("embedding", None)
            row["evidence_score"] = item.evidence_score
            writer.writerow(row)
    with (destination / "track_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "track_id",
            "channel_index",
            "station_id",
            "position_m",
            "time_s",
            "observed",
            "observation_id",
            "residual_s",
            "ambiguous",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in batch.tracks:
            for point in track.points:
                writer.writerow({"track_id": track.track_id, **vars(point)})
    _write_json(destination / "diagnostics.json", batch.diagnostics)
    if config is not None:
        with (destination / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
    _plot_overlay(batch, artifacts, destination / "tracks_overlay.png")
    _plot_probability(artifacts, destination / "network_probability.png")
    _plot_hough(artifacts, destination / "physical_hough.png")
    _plot_graph(batch, artifacts, destination / "association_graph.png")
    report = build_markdown_report(batch)
    with (destination / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(report)
    return destination


def build_markdown_report(batch: TrackBatch) -> str:
    lines = [
        "# Hybrid Vehicle Tracker report",
        "",
        f"- Window: {batch.start_s:.1f}–{batch.start_s + batch.duration_s:.1f} s",
        f"- Candidate observations: {len(batch.observations)}",
        f"- Physics-consistent candidate tracks: {len(batch.tracks)}",
        f"- Direction: {batch.diagnostics.get('direction', 'configured motion direction')}",
        "- Required speed band: 60–90 km/h",
        "",
        "## Candidate tracks",
        "",
        "| track | speed (km/h) | observed | span (m) | residual (s) | confidence | boundary |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for track in batch.tracks:
        boundary = "/".join(
            part
            for part, enabled in (("enter", track.enters_window), ("exit", track.exits_window))
            if enabled
        ) or "inside"
        lines.append(
            f"| {track.track_id} | {track.median_speed_kmh:.2f} | {track.observed_count} "
            f"| {track.span_m:.0f} | {track.median_residual_s:.3f} "
            f"| {track.confidence:.3f} | {boundary} |"
        )
    lines.extend(
        [
            "",
            "> DAY11 currently has no manual ground truth. These are candidates, and the confidence column is an internal consistency score rather than calibrated precision. See the null-control report for statistical acceptance.",
            "",
        ]
    )
    return "\n".join(lines)
