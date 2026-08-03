#!/usr/bin/env python3
"""Persist a one-scene, vehicle-by-vehicle simulation/prediction comparison.

The aggregate benchmark is useful for regression testing, but it cannot answer
whether a particular injected vehicle was recovered.  This script writes the
actual simulator arrays, every injected vehicle (including missing stations),
each recognizer's tracks, matching rows, and a single overlay figure.  It never
loads a measured waveform; only the station geometry and the synthetic scene
are used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from hybrid_vehicle_tracker.association.pipeline import AssociationResult, associate_observations
from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.data.peakset_reference import reference_summary
from hybrid_vehicle_tracker.data.synthetic import SyntheticSettings, SyntheticVehicleDataset
from hybrid_vehicle_tracker.evaluation.synthetic_benchmark import (
    _ground_truth_tracks,
    _observations_from_features,
)
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel
from hybrid_vehicle_tracker.models.physical_hough import PhysicalHoughHead
from hybrid_vehicle_tracker.types import StationGeometry, VehicleTrack


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/day11_120s.yaml"))
    parser.add_argument("--output", type=Path, default=Path("reports/synthetic_scene_compare"))
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=21260707)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def _run_model(
    scene: dict[str, torch.Tensor],
    model: HybridPerceptionModel,
    geometry: StationGeometry,
    config,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, AssociationResult]]:
    features = scene["input"].numpy()
    window_s = float(features.shape[-1] / config.data.feature_rate_hz)
    inputs = scene["input"][None].to(device)
    positions = torch.from_numpy(geometry.positions_m.astype(np.float32)).to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)
        learned = torch.sigmoid(outputs["centerline_logits"])[0, 0].float()
        crossing = torch.sigmoid(outputs["crossing_logits"])[0, 0].float()
        raw_score = torch.sigmoid(6.0 * inputs[0, 0].float())
        pre_event_score = 1.0 - inputs[0, 1].float()
        fused = torch.clamp(
            0.55 * learned
            + 0.20 * pre_event_score
            + 0.15 * inputs[0, 2].float()
            + 0.10 * raw_score,
            0.0,
            1.0,
        )
        modal = torch.clamp(
            0.45 * pre_event_score + 0.40 * inputs[0, 2].float() + 0.15 * raw_score,
            0.0,
            1.0,
        )
        deep_hough = model.hough(
            outputs["feature_map"],
            positions,
            duration_s=window_s,
            feature_rate_hz=config.data.feature_rate_hz,
            evidence_map=fused[None, None],
        )
        traditional_hough = model.hough(
            outputs["feature_map"],
            positions,
            duration_s=window_s,
            feature_rate_hz=config.data.feature_rate_hz,
            evidence_map=modal[None, None],
        )
    fused_np = fused.cpu().numpy()
    modal_np = modal.cpu().numpy()
    crossing_np = crossing.cpu().numpy()
    embedding_np = outputs["embedding"][0].float().cpu().numpy()
    deep_seeds = PhysicalHoughHead.topk_seeds(
        deep_hough,
        top_k=config.model.hough_top_k,
        learned=True,
        min_support=config.association.min_observations,
    )
    traditional_seeds = PhysicalHoughHead.topk_seeds(
        traditional_hough,
        top_k=config.model.hough_top_k,
        learned=False,
        min_support=config.association.min_observations,
    )
    modal_observations = _observations_from_features(
        features,
        geometry,
        config,
        network=np.zeros_like(modal_np),
        crossing=np.zeros_like(modal_np),
        embedding=None,
        use_network_candidates=False,
    )
    deep_observations = _observations_from_features(
        features,
        geometry,
        config,
        network=fused_np,
        crossing=crossing_np,
        embedding=embedding_np,
        use_network_candidates=True,
    )
    method_specs = {
        "traditional_hough": (
            modal_observations,
            traditional_seeds,
            modal_np,
            None,
            "greedy",
            False,
        ),
        "resunet_hough": (
            deep_observations,
            deep_seeds,
            fused_np,
            None,
            "greedy",
            False,
        ),
        "full_gnn_milp": (
            deep_observations,
            deep_seeds + traditional_seeds,
            fused_np,
            model.edge_gnn,
            "milp",
            False,
        ),
    }
    results: dict[str, AssociationResult] = {}
    for name, (observations, seeds, evidence, edge_gnn, mode, pair_seeds) in method_specs.items():
        results[name] = associate_observations(
            copy.deepcopy(observations),
            seeds,
            geometry,
            config.association,
            duration_s=window_s,
            embedding_dim=config.model.embedding_dim,
            hough_intercept_step_s=config.model.hough_intercept_step_s,
            hough_top_k=config.model.hough_top_k,
            edge_gnn=edge_gnn,
            device=device,
            dense_evidence=evidence,
            feature_rate_hz=config.data.feature_rate_hz,
            selection_mode=mode,
            include_pair_seeds=pair_seeds,
        )
    return {
        "network_probability": fused_np,
        "modal_probability": modal_np,
        "crossing_probability": crossing_np,
    }, results


def _truth_rows(scene: dict[str, torch.Tensor], geometry: StationGeometry, config) -> list[dict[str, object]]:
    params = scene["track_params"].numpy()
    times = scene["track_times"].numpy()
    observed = scene["track_observed"].numpy() > 0.5
    rows: list[dict[str, object]] = []
    for index in np.flatnonzero(params[:, 2] > 0.5):
        valid = np.isfinite(times[index]) & (times[index] >= 0.0) & (
            times[index] < config.data.duration_s
        )
        valid_channels = np.flatnonzero(valid)
        observed_channels = np.flatnonzero(valid & observed[index])
        slope = float(params[index, 0])
        rows.append(
            {
                "truth_track_id": int(index),
                "speed_kmh": float(3.6 / max(abs(slope), 1e-8)),
                "slope_s_per_m": slope,
                "intercept_s": float(params[index, 1]),
                "valid_count": int(valid_channels.size),
                "observed_count": int(observed_channels.size),
                "valid_channels": valid_channels.astype(int).tolist(),
                "observed_channels": observed_channels.astype(int).tolist(),
                "missing_channels": [
                    int(channel)
                    for channel in valid_channels
                    if not observed[index, channel]
                ],
                "times_s": {
                    str(int(channel)): float(times[index, channel]) for channel in valid_channels
                },
                "positions_m": {
                    str(int(channel)): float(geometry.positions_m[channel])
                    for channel in valid_channels
                },
            }
        )
    return rows


def _match_rows(
    tracks: list[VehicleTrack],
    truth: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if not tracks or not truth:
        rows = [
            {
                "truth_track_id": int(index),
                "pred_track_id": None,
                "matched": False,
                "residual_s": None,
                "speed_error_kmh": None,
            }
            for index in range(len(truth))
        ]
        rows.extend(
            {
                "truth_track_id": None,
                "pred_track_id": track.track_id,
                "matched": False,
                "residual_s": None,
                "speed_error_kmh": None,
            }
            for track in tracks
        )
        return {
            "true_positives": 0,
            "false_positives": len(tracks),
            "false_negatives": len(truth),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "speed_mae_kmh": None,
        }, rows
    costs = np.full((len(tracks), len(truth)), 1e6, dtype=np.float64)
    speed_errors = np.full_like(costs, np.nan)
    residual_cache: dict[tuple[int, int], float] = {}
    for pi, track in enumerate(tracks):
        by_channel = {point.channel_index: point.time_s for point in track.points}
        for ti, (channels, times, speed) in enumerate(truth):
            residuals = [
                abs(by_channel[int(channel)] - float(time))
                for channel, time in zip(channels, times)
                if int(channel) in by_channel
            ]
            if len(residuals) >= 3:
                residual = float(np.median(residuals))
                costs[pi, ti] = residual
                residual_cache[(pi, ti)] = residual
                speed_errors[pi, ti] = abs(track.median_speed_kmh - speed)
    rows_i, cols_i = linear_sum_assignment(costs)
    matched = {
        (int(row), int(col)) for row, col in zip(rows_i, cols_i) if costs[row, col] <= 0.5
    }
    rows: list[dict[str, object]] = []
    for ti in range(len(truth)):
        pair = next(((pi, ti) for pi in range(len(tracks)) if (pi, ti) in matched), None)
        if pair is None:
            rows.append(
                {
                    "truth_track_id": ti,
                    "pred_track_id": None,
                    "matched": False,
                    "residual_s": None,
                    "speed_error_kmh": None,
                }
            )
        else:
            pi, _ = pair
            rows.append(
                {
                    "truth_track_id": ti,
                    "pred_track_id": tracks[pi].track_id,
                    "matched": True,
                    "residual_s": residual_cache[(pi, ti)],
                    "speed_error_kmh": float(speed_errors[pi, ti]),
                }
            )
    rows.extend(
        {
            "truth_track_id": None,
            "pred_track_id": track.track_id,
            "matched": any(pi == index for pi, _ in matched),
            "residual_s": None,
            "speed_error_kmh": None,
        }
        for index, track in enumerate(tracks)
        if not any(pi == index for pi, _ in matched)
    )
    tp = len(matched)
    errors = [float(speed_errors[pi, ti]) for pi, ti in matched]
    precision = tp / max(len(tracks), 1)
    recall = tp / max(len(truth), 1)
    return {
        "true_positives": tp,
        "false_positives": len(tracks) - tp,
        "false_negatives": len(truth) - tp,
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "speed_mae_kmh": float(np.mean(errors)) if errors else None,
    }, rows


def _track_line(track: VehicleTrack) -> tuple[list[float], list[int]]:
    points = sorted(track.points, key=lambda point: point.channel_index)
    return [point.time_s for point in points], [point.channel_index for point in points]


def _plot(
    path: Path,
    scene: dict[str, torch.Tensor],
    probabilities: dict[str, np.ndarray],
    results: dict[str, AssociationResult],
    truth_rows: list[dict[str, object]],
    metrics: dict[str, dict[str, object]],
    feature_rate_hz: float,
) -> None:
    features = scene["input"].numpy()
    gauss = features[2]
    truth_times = scene["track_times"].numpy()
    truth_obs = scene["track_observed"].numpy() > 0.5
    valid_truth = scene["track_params"].numpy()[:, 2] > 0.5
    window_s = features.shape[-1] / feature_rate_hz
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes[0, 0].imshow(
        gauss,
        origin="lower",
        aspect="auto",
        extent=(0.0, window_s, 0, gauss.shape[0] - 1),
        cmap="gray_r",
        vmin=0.0,
        vmax=0.72,
    )
    for index in np.flatnonzero(valid_truth):
        valid = np.isfinite(truth_times[index]) & (truth_times[index] >= 0.0) & (
            truth_times[index] < window_s
        )
        channels = np.flatnonzero(valid)
        axes[0, 0].plot(truth_times[index, channels], channels, lw=1.3, label=f"GT {index}")
        observed_channels = channels[truth_obs[index, channels]]
        axes[0, 0].scatter(
            truth_times[index, observed_channels], observed_channels, s=12, facecolors="none", edgecolors="tab:blue"
        )
        missing_channels = channels[~truth_obs[index, channels]]
        axes[0, 0].scatter(
            truth_times[index, missing_channels], missing_channels, s=15, marker="x", color="tab:orange"
        )
    axes[0, 0].set_title("Simulator Gauss + known vehicles\nblue circle=observed, orange x=missing")
    axes[0, 0].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("station channel")

    axes[0, 1].imshow(
        probabilities["network_probability"],
        origin="lower",
        aspect="auto",
        extent=(0.0, window_s, 0, gauss.shape[0] - 1),
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0, 1].set_title("ResUNet + modal fused probability")
    axes[0, 1].set_xlabel("time (s)")
    axes[0, 1].set_ylabel("station channel")

    axes[0, 2].imshow(
        gauss,
        origin="lower",
        aspect="auto",
        extent=(0.0, window_s, 0, gauss.shape[0] - 1),
        cmap="gray_r",
        vmin=0.0,
        vmax=0.72,
    )
    for index in np.flatnonzero(valid_truth):
        valid = np.isfinite(truth_times[index]) & (truth_times[index] >= 0.0) & (
            truth_times[index] < window_s
        )
        channels = np.flatnonzero(valid)
        axes[0, 2].plot(truth_times[index, channels], channels, "--", color="deepskyblue", lw=1.0)
    for track in results["full_gnn_milp"].tracks:
        times, channels = _track_line(track)
        axes[0, 2].plot(times, channels, color="crimson", lw=1.8)
        axes[0, 2].scatter(times, channels, color="crimson", s=10)
    axes[0, 2].set_title(
        "Full GNN + MILP: blue dashed GT / red prediction\n"
        f"F1={metrics['full_gnn_milp']['f1']:.3f}, tracks={len(results['full_gnn_milp'].tracks)}"
    )
    axes[0, 2].set_xlabel("time (s)")
    axes[0, 2].set_ylabel("station channel")

    raw_plane = features[0]
    axes[1, 0].imshow(
        raw_plane,
        origin="lower",
        aspect="auto",
        extent=(0.0, window_s, 0, raw_plane.shape[0] - 1),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    axes[1, 0].set_title("Simulator 20-Hz robust Raw plane")
    axes[1, 0].set_xlabel("time (s)")
    axes[1, 0].set_ylabel("station channel")

    axes[1, 1].imshow(
        gauss,
        origin="lower",
        aspect="auto",
        extent=(0.0, window_s, 0, gauss.shape[0] - 1),
        cmap="gray_r",
        vmin=0.0,
        vmax=0.72,
    )
    colors = {"traditional_hough": "#2ca02c", "resunet_hough": "#9467bd", "full_gnn_milp": "#d62728"}
    for name, result in results.items():
        for track in result.tracks:
            times, channels = _track_line(track)
            axes[1, 1].plot(times, channels, color=colors[name], lw=1.2, alpha=0.75)
    axes[1, 1].set_title("All recognizers on the same simulated scene")
    axes[1, 1].set_xlabel("time (s)")
    axes[1, 1].set_ylabel("station channel")
    for name, color in colors.items():
        axes[1, 1].plot([], [], color=color, label=name)
    axes[1, 1].legend(fontsize=8, loc="upper right")

    names = list(results)
    f1 = [float(metrics[name]["f1"]) for name in names]
    speed_mae = [float(metrics[name]["speed_mae_kmh"] or 0.0) for name in names]
    x = np.arange(len(names))
    axis = axes[1, 2]
    bars = axis.bar(x - 0.18, f1, width=0.36, color=[colors[name] for name in names], alpha=0.85)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("track F1")
    axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    axis2 = axis.twinx()
    axis2.plot(x + 0.18, speed_mae, "ko--", label="speed MAE")
    axis2.set_ylabel("speed MAE (km/h)")
    for bar, value in zip(bars, f1):
        axis.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.2f}", ha="center", fontsize=8)
    axis.set_title("Same injected GT, method-by-method metrics")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_vehicle_peakset_overlay(
    path: Path,
    scene: dict[str, torch.Tensor],
    geometry: StationGeometry,
    results: dict[str, AssociationResult],
    *,
    direction: int,
    feature_rate_hz: float,
) -> None:
    """Render the same waveform/GT/prediction view as the reference overlay.

    The x axis is a travel-oriented physical offset.  Reversing the offset for
    ``motion_direction=-1`` makes the vehicle lines slope down/right like the
    supplied reference image while preserving the actual station mapping and
    200 m gap in the manifest.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feature = scene["input"][0].numpy()
    times = scene["track_times"].numpy()
    params = scene["track_params"].numpy()
    observed = scene["track_observed"].numpy() > 0.5
    boundary_tracks = scene.get("boundary_track_mask", torch.zeros(params.shape[0])).numpy() > 0.5
    window_s = float(feature.shape[-1] / feature_rate_hz)
    relative = geometry.relative_positions_m.astype(np.float64)
    if int(direction) < 0:
        x_axis = relative[-1] - relative
    else:
        x_axis = relative.copy()
    x_axis_km = x_axis * 1e-3
    spacing = float(np.median(np.diff(np.sort(x_axis_km)))) if len(x_axis_km) > 1 else 0.1
    spacing = max(spacing, 1e-3)
    finite = feature[np.isfinite(feature)]
    vmax = max(float(np.quantile(np.abs(finite), 0.995)) if finite.size else 1.0, 1e-6)
    figure, axis = plt.subplots(figsize=(13.0, 7.6), constrained_layout=True)
    t_axis = np.linspace(0.0, window_s, feature.shape[-1], dtype=np.float64)
    wiggle_amp = 0.27 * spacing
    for channel in range(feature.shape[0]):
        ratio = np.clip(feature[channel].astype(np.float64) / vmax, -1.35, 1.35)
        axis.plot(
            x_axis_km[channel] + ratio * wiggle_amp,
            t_axis,
            color="0.46",
            linewidth=0.80,
            alpha=0.88,
            zorder=1,
        )

    valid_track_count = 0
    boundary_label_added = False
    for track_index in np.flatnonzero(params[:, 2] > 0.5):
        valid = np.isfinite(times[track_index]) & (times[track_index] >= 0.0) & (
            times[track_index] < window_s
        )
        channels = np.flatnonzero(valid)
        if channels.size < 2:
            continue
        is_boundary = bool(boundary_tracks[track_index])
        axis.plot(
            x_axis_km[channels],
            times[track_index, channels],
            color="#5d6670" if is_boundary else "#a7a7a7",
            linestyle="--",
            linewidth=1.15 if is_boundary else 0.85,
            alpha=0.92,
            zorder=6 if is_boundary else 5,
            label="GT boundary vehicle" if is_boundary and not boundary_label_added else None,
        )
        axis.scatter(
            x_axis_km[channels],
            times[track_index, channels],
            s=13,
            color="#5d6670" if is_boundary else "#a7a7a7",
            marker="o",
            linewidths=0.0,
            label="GT" if valid_track_count == 0 else None,
            zorder=7 if is_boundary else 6,
        )
        if is_boundary:
            axis.scatter(
                x_axis_km[channels[[0, -1]]],
                times[track_index, channels[[0, -1]]],
                s=32,
                marker="D",
                facecolors="none",
                edgecolors="#263746",
                linewidths=0.9,
                zorder=8,
            )
            boundary_label_added = True
        valid_track_count += 1

    full_result = results.get("full_gnn_milp")
    tracks = full_result.tracks if full_result is not None else []
    palette = plt.get_cmap("tab20", max(1, len(tracks)))
    prediction_label_added = False
    for index, track in enumerate(tracks):
        points = sorted(track.points, key=lambda point: point.channel_index)
        if not points:
            continue
        xs = [x_axis_km[int(point.channel_index)] for point in points]
        ys = [float(point.time_s) for point in points]
        color = palette(index % max(1, palette.N))
        axis.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8,
            alpha=0.96,
            label="Prediction" if not prediction_label_added else None,
            zorder=7,
        )
        observed_points = [point for point in points if point.observed]
        missing_points = [point for point in points if not point.observed]
        if observed_points:
            axis.scatter(
                [x_axis_km[int(point.channel_index)] for point in observed_points],
                [float(point.time_s) for point in observed_points],
                s=22,
                marker="s",
                color=color,
                linewidths=0.65,
                zorder=8,
            )
        if missing_points:
            axis.scatter(
                [x_axis_km[int(point.channel_index)] for point in missing_points],
                [float(point.time_s) for point in missing_points],
                s=30,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                zorder=8,
            )
        middle = len(points) // 2
        text_x = min(
            float(x_axis_km.max()) - 0.02 * max(float(np.ptp(x_axis_km)), 1e-3),
            xs[middle] + 0.01,
        )
        axis.text(
            text_x,
            ys[middle],
            f"{track.median_speed_kmh:.1f} km/h",
            color=color,
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.58, "edgecolor": "none", "pad": 0.8},
            zorder=9,
        )
        prediction_label_added = True

    axis.set_xlim(float(x_axis_km.min() - 0.4 * spacing), float(x_axis_km.max() + 0.4 * spacing))
    axis.set_ylim(window_s, 0.0)
    axis.set_xlabel("Offset [km] (travel direction)")
    axis.set_ylabel("Time (s)")
    axis.set_title(
        f"Vehicle peak-set overlay | GT={valid_track_count} | "
        f"Prediction={len(tracks)}"
    )
    if valid_track_count or prediction_label_added:
        axis.legend(loc="upper right", frameon=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("GPU comparison requested but CUDA is unavailable")
    config = load_tracker_config(args.config)
    if args.checkpoint is not None:
        config.model.checkpoint = str(args.checkpoint)
    config.runtime.device = args.device
    if config.model.motion_direction != config.association.motion_direction:
        raise ValueError("model and association motion_direction differ")
    device = torch.device(args.device)
    geometry = load_station_geometry(config.data.mapping_path)
    window_s = float(config.data.duration_s)
    dataset = SyntheticVehicleDataset(
        geometry,
        SyntheticSettings(
            simulator_version="vehicle_peakset_complex_v2",
            window_s=window_s,
            feature_rate_hz=config.data.feature_rate_hz,
            waveform_rate_hz=200.0,
            max_vehicles=14,
            samples=max(args.scene_index + 1, 1),
            seed=args.seed,
            vehicle_rate=11.0,
            interaction_probability=0.82,
            motion_direction=config.association.motion_direction,
            peakset_vehicle_min=10,
            peakset_vehicle_max=12,
            peakset_min_visible_channels=8,
            peakset_false_event_min=50,
            peakset_false_event_max=82,
            peakset_dead_channels="5,6,15,16,22,36,38,45",
            peakset_missing_ratio_min=0.08,
            peakset_missing_ratio_max=0.18,
            peakset_gap_probability=0.72,
            peakset_gap_min_channels=2,
            peakset_gap_max_channels=6,
            peakset_boundary_vehicle_ratio=0.40,
            peakset_outage_probability=0.78,
            return_modalities=True,
        ),
    )
    scene = dataset[args.scene_index]
    model = HybridPerceptionModel(config.model).to(device)
    if not config.model.checkpoint:
        raise ValueError("a trained checkpoint is required for the comparison")
    metadata = model.load_checkpoint(config.model.checkpoint, map_location=str(device))
    probabilities, results = _run_model(scene, model, geometry, config, device)
    truth = _ground_truth_tracks(scene, geometry, config)
    truth_rows = _truth_rows(scene, geometry, config)
    metrics: dict[str, dict[str, object]] = {}
    match_rows: list[dict[str, object]] = []
    for name, result in results.items():
        metric, rows = _match_rows(result.tracks, truth)
        metrics[name] = metric
        for row in rows:
            match_rows.append({"method": name, **row})

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "scene_arrays.npz",
        input=scene["input"].numpy(),
        raw_waveform=scene["raw_waveform"].numpy(),
        pre_feature=scene["pre_feature"].numpy(),
        gauss_feature=scene["gauss_feature"].numpy(),
        track_params=scene["track_params"].numpy(),
        track_times=scene["track_times"].numpy(),
        track_observed=scene["track_observed"].numpy(),
        event_kernel_sigma_s=scene["event_kernel_sigma_s"].numpy(),
        event_kernel_height=scene["event_kernel_height"].numpy(),
        event_kernel_raw_sigma_s=scene["event_kernel_raw_sigma_s"].numpy(),
        event_kernel_pre_sigma_s=scene["event_kernel_pre_sigma_s"].numpy(),
        network_probability=probabilities["network_probability"],
    )
    with (args.output / "simulation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "scene_index": args.scene_index,
                "seed": args.seed,
                "simulator_version": dataset.settings.simulator_version,
                "reference_style": "vehicle_peakset_complex_v2: reimplemented clean-crossing morphology plus DAY11-statistics-calibrated decoys/outages",
                "reference_source_read_only": "/csim2/zhangzhiyu/MyProjects/auto_track/repro_vehicle_pipeline_20260629",
                "gauss_reference_calibration": reference_summary(),
                "gauss_reference_usage": "statistics only; the measured DAY11 array is not copied into this scene or training",
                "window_s": dataset.settings.window_s,
                "feature_rate_hz": dataset.settings.feature_rate_hz,
                "waveform_rate_hz": dataset.settings.waveform_rate_hz,
                "motion_direction": dataset.settings.motion_direction,
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "checkpoint": config.model.checkpoint,
                "checkpoint_metadata": metadata,
                "stations": [
                    {
                        "channel_index": station.channel_index,
                        "station_id": station.station_id,
                        "position_m": station.position_m,
                        "location": station.location,
                    }
                    for station in geometry.stations
                ],
                "vehicles": truth_rows,
                "bad_channels": np.flatnonzero(scene["bad_channel_mask"].numpy() > 0.5).astype(int).tolist(),
                "boundary_vehicle_ids": np.flatnonzero(scene["boundary_track_mask"].numpy() > 0.5).astype(int).tolist(),
                "false_event_count": int(scene["false_event_count"].item()),
                "isolated_false_event_count": int(scene["isolated_false_event_count"].item()),
                "temporary_outage_count": int(scene["outage_count"].item()),
                "event_kernel": {
                    "gauss_sigma_s": float(scene["event_kernel_sigma_s"].item()),
                    "gauss_height": float(scene["event_kernel_height"].item()),
                    "raw_sigma_s": float(scene["event_kernel_raw_sigma_s"].item()),
                    "pre_sigma_s": float(scene["event_kernel_pre_sigma_s"].item()),
                    "all_vehicle_and_false_peaks_share_kernel": True,
                },
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    for name, result in results.items():
        with (args.output / f"tracks_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for track in result.tracks:
                handle.write(json.dumps(track.to_dict(), ensure_ascii=False) + "\n")
    with (args.output / "matching.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["method", "truth_track_id", "pred_track_id", "matched", "residual_s", "speed_error_kmh"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(match_rows)
    with (args.output / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "semantics": "Synthetic injected vehicles are ground truth; no measured DAY11 samples are used.",
                "reference_style": "vehicle_peakset_complex_v2: DAY11-calibrated isolated peaks, outages, crossings, and boundary vehicles",
                "gauss_reference_calibration": reference_summary(),
                "scene_index": args.scene_index,
                "seed": args.seed,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "truth_vehicle_count": len(truth_rows),
                "predicted_track_counts": {name: len(result.tracks) for name, result in results.items()},
                "overlay": "vehicle_peakset_overlay.png",
                "metrics": metrics,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    _plot(
        args.output / "comparison.png",
        scene,
        probabilities,
        results,
        truth_rows,
        metrics,
        config.data.feature_rate_hz,
    )
    _plot_vehicle_peakset_overlay(
        args.output / "vehicle_peakset_overlay.png",
        scene,
        geometry,
        results,
        direction=config.association.motion_direction,
        feature_rate_hz=config.data.feature_rate_hz,
    )
    print(json.dumps({"output": str(args.output), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
