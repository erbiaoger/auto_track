#!/usr/bin/env python3
"""Compare the synthetic training scenes with the DAY11 inference input.

This is deliberately an audit, not a model evaluation: DAY11 has no vehicle
identity ground truth.  Real event centres are the strong Gauss peaks and the
subset used by the current physics-consistent candidate tracks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.data.features import build_feature_batch
from hybrid_vehicle_tracker.data.io import load_modal_window
from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.data.synthetic import SyntheticSettings, SyntheticVehicleDataset
from hybrid_vehicle_tracker.models.hybrid import HybridPerceptionModel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/day11_120s.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--skip-model", action="store_true")
    return parser.parse_args()


def _strong_real_events(gauss: np.ndarray, *, sample_rate_hz: float, bins: int) -> list[tuple[int, int]]:
    events: list[tuple[int, int]] = []
    for channel in range(gauss.shape[1]):
        peaks, _ = find_peaks(
            gauss[:, channel],
            height=0.5,
            distance=max(1, int(round(1.25 * sample_rate_hz))),
        )
        for peak in peaks:
            feature_bin = int(round(float(peak) * 20.0 / sample_rate_hz))
            if 12 <= feature_bin < bins - 12:
                events.append((channel, feature_bin))
    return events


def _recognized_events(path: Path, *, feature_rate_hz: float, bins: int) -> list[tuple[int, int]]:
    if not path.exists():
        return []
    events: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("observed", "").lower() != "true":
                continue
            feature_bin = int(round(float(row["time_s"]) * feature_rate_hz))
            if 12 <= feature_bin < bins - 12:
                events.append((int(row["channel_index"]), feature_bin))
    return events


def _synthetic_scenes(dataset: SyntheticVehicleDataset) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    inputs: list[np.ndarray] = []
    events: list[tuple[int, int, int]] = []
    rate = dataset.settings.feature_rate_hz
    for scene_index in range(len(dataset)):
        scene = dataset[scene_index]
        features = scene["input"].numpy()
        inputs.append(features)
        times = scene["track_times"].numpy()
        observed = scene["track_observed"].numpy() > 0.5
        params = scene["track_params"].numpy()
        for track_index in np.flatnonzero(params[:, 2] > 0.5):
            for channel in np.flatnonzero(observed[track_index] & np.isfinite(times[track_index])):
                feature_bin = int(round(float(times[track_index, channel]) * rate))
                if 12 <= feature_bin < features.shape[-1] - 12:
                    lo = max(0, feature_bin - 2)
                    hi = min(features.shape[-1], feature_bin + 3)
                    feature_bin = lo + int(np.argmax(features[2, channel, lo:hi]))
                    events.append((scene_index, int(channel), feature_bin))
    return inputs, events


def _synthetic_strong_event_count(inputs: list[np.ndarray], feature_rate_hz: float) -> int:
    count = 0
    distance = max(1, int(round(1.25 * feature_rate_hz)))
    for features in inputs:
        for channel in range(features.shape[1]):
            peaks, _ = find_peaks(features[2, channel], height=0.5, distance=distance)
            count += int(peaks.size)
    return count


def _profiles_single(features: np.ndarray, events: list[tuple[int, int]], half_bins: int = 20) -> np.ndarray:
    rows = []
    for channel, feature_bin in events:
        if half_bins <= feature_bin < features.shape[-1] - half_bins:
            rows.append(features[:3, channel, feature_bin - half_bins : feature_bin + half_bins + 1])
    if not rows:
        return np.empty((0, 3, 2 * half_bins + 1), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _profiles_synthetic(
    inputs: list[np.ndarray], events: list[tuple[int, int, int]], half_bins: int = 20
) -> np.ndarray:
    rows = []
    for scene_index, channel, feature_bin in events:
        features = inputs[scene_index]
        if half_bins <= feature_bin < features.shape[-1] - half_bins:
            rows.append(features[:3, channel, feature_bin - half_bins : feature_bin + half_bins + 1])
    if not rows:
        return np.empty((0, 3, 2 * half_bins + 1), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _event_summary(profiles: np.ndarray, feature_rate_hz: float) -> dict[str, dict[str, float]]:
    if profiles.shape[0] == 0:
        return {}
    result: dict[str, dict[str, float]] = {}
    centre = profiles.shape[-1] // 2
    names = ("raw", "pre", "gauss")
    for plane, name in enumerate(names):
        values = profiles[:, plane, centre]
        offsets = (np.argmax(profiles[:, plane], axis=1) - centre) / feature_rate_hz
        row = {
            "center_q25": float(np.quantile(values, 0.25)),
            "center_median": float(np.median(values)),
            "center_q75": float(np.quantile(values, 0.75)),
            "absolute_max_offset_median_s": float(np.median(np.abs(offsets))),
            "absolute_max_offset_p90_s": float(np.quantile(np.abs(offsets), 0.9)),
        }
        if name == "gauss":
            widths = []
            for profile in profiles[:, plane]:
                level = 0.5 * profile[centre]
                left = centre
                while left > 0 and profile[left - 1] >= level:
                    left -= 1
                right = centre
                while right + 1 < profile.size and profile[right + 1] >= level:
                    right += 1
                widths.append((right - left + 1) / feature_rate_hz)
            row["fwhm_median_s"] = float(np.median(widths))
        result[name] = row
    return result


def _profile_stat(profiles: np.ndarray, statistic: str) -> np.ndarray:
    if profiles.shape[0] == 0:
        return np.full(profiles.shape[1:], np.nan, dtype=np.float32)
    if statistic == "median":
        return np.median(profiles, axis=0)
    quantile = {"q25": 0.25, "q75": 0.75}[statistic]
    return np.quantile(profiles, quantile, axis=0)


def _model_probabilities(
    model: HybridPerceptionModel,
    real_features: np.ndarray,
    real_events: list[tuple[int, int]],
    synthetic_inputs: list[np.ndarray],
    synthetic_events: list[tuple[int, int, int]],
) -> dict[str, object]:
    device = torch.device("cuda")
    model = model.to(device).eval()
    with torch.inference_mode():
        real_probability = torch.sigmoid(
            model(torch.from_numpy(real_features[None]).to(device))["centerline_logits"]
        )[0, 0].cpu().numpy()
        synthetic_probability = []
        for features in synthetic_inputs:
            synthetic_probability.append(
                torch.sigmoid(model(torch.from_numpy(features[None]).to(device))["centerline_logits"])[
                    0, 0
                ].cpu().numpy()
            )
    real_at_events = np.asarray([real_probability[channel, time_bin] for channel, time_bin in real_events])
    synthetic_at_events = np.asarray(
        [
            synthetic_probability[scene][channel, time_bin]
            for scene, channel, time_bin in synthetic_events
        ]
    )
    return {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "real_all_quantiles": np.quantile(real_probability, [0.5, 0.9, 0.99, 0.999]).tolist(),
        "synthetic_all_quantiles": np.quantile(
            np.concatenate([item.ravel() for item in synthetic_probability]),
            [0.5, 0.9, 0.99, 0.999],
        ).tolist(),
        "real_event_quantiles": np.quantile(real_at_events, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
        "synthetic_event_quantiles": np.quantile(
            synthetic_at_events, [0.1, 0.25, 0.5, 0.75, 0.9]
        ).tolist(),
        "real_probability": real_probability,
        "synthetic_probability": synthetic_probability[0],
    }


def _plot_profiles(
    destination: Path,
    time_s: np.ndarray,
    real: np.ndarray,
    recognized: np.ndarray,
    synthetic: np.ndarray,
) -> None:
    names = ("Raw model plane", "Pre model plane", "Gauss model plane")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for plane, axis in enumerate(axes):
        for values, label, color in (
            (real, "DAY11 strong Gauss events", "#59636e"),
            (recognized, "current selected points", "#d55e00"),
            (synthetic, "synthetic GT observations", "#0072b2"),
        ):
            if values.size == 0:
                continue
            median = np.median(values[:, plane], axis=0)
            q25, q75 = np.quantile(values[:, plane], [0.25, 0.75], axis=0)
            axis.plot(time_s, median, label=label, color=color, linewidth=1.8)
            axis.fill_between(time_s, q25, q75, color=color, alpha=0.13)
        axis.axvline(0.0, color="black", alpha=0.3, linewidth=0.8)
        axis.set_title(names[plane])
        axis.set_xlabel("offset from Gauss/GT centre (s)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("actual network input value")
    axes[-1].legend(loc="best", fontsize=8)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_scene_comparison(
    destination: Path,
    real_gauss: np.ndarray,
    synthetic: np.ndarray,
    track_points: Path,
    synthetic_scene: dict[str, torch.Tensor],
    feature_rate_hz: float,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    real_display = real_gauss[:, ::2]
    axes[0].imshow(
        real_display,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, 120.0, 0, real_display.shape[0] - 1),
        cmap="gray_r",
        vmin=0.0,
        vmax=0.68,
    )
    grouped: dict[str, list[tuple[float, int, bool]]] = {}
    if track_points.exists():
        with track_points.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped.setdefault(row["track_id"], []).append(
                    (float(row["time_s"]), int(row["channel_index"]), row["observed"].lower() == "true")
                )
    for rows in grouped.values():
        rows.sort(key=lambda item: item[1])
        axes[0].plot([row[0] for row in rows], [row[1] for row in rows], color="#d55e00", linewidth=1.1)
        observed = [row for row in rows if row[2]]
        axes[0].scatter([row[0] for row in observed], [row[1] for row in observed], s=8, color="#d55e00")
    axes[0].set_title("DAY11 Gauss + current physics candidates (not ground truth)")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("channel")

    axes[1].imshow(
        synthetic[2],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(0.0, synthetic.shape[-1] / feature_rate_hz, 0, synthetic.shape[1] - 1),
        cmap="gray_r",
        vmin=0.0,
        vmax=0.72,
    )
    times = synthetic_scene["track_times"].numpy()
    params = synthetic_scene["track_params"].numpy()
    for track_index in np.flatnonzero(params[:, 2] > 0.5):
        valid = np.isfinite(times[track_index]) & (times[track_index] >= 0.0) & (
            times[track_index] < synthetic.shape[-1] / feature_rate_hz
        )
        channels = np.flatnonzero(valid)
        axes[1].plot(times[track_index, channels], channels, color="#0072b2", linewidth=1.1)
    axes[1].set_title("Synthetic Gauss + known injected vehicle truth")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("channel")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_tracker_config(args.config)
    geometry = load_station_geometry(config.data.mapping_path)
    raw, pre, gauss = load_modal_window(
        config.data.raw_path,
        config.data.pre_path,
        config.data.gauss_path,
        start_s=config.data.start_s,
        duration_s=config.data.duration_s,
        sample_rate_hz=config.data.sample_rate_hz,
    )
    real_batch = build_feature_batch(
        raw,
        pre,
        gauss,
        geometry,
        sample_rate_hz=config.data.sample_rate_hz,
        feature_rate_hz=config.data.feature_rate_hz,
    )
    real_events = _strong_real_events(
        gauss, sample_rate_hz=config.data.sample_rate_hz, bins=real_batch.time_bins
    )
    track_points = Path(config.runtime.output_dir) / "track_points.csv"
    recognized_events = _recognized_events(
        track_points, feature_rate_hz=config.data.feature_rate_hz, bins=real_batch.time_bins
    )
    dataset = SyntheticVehicleDataset(
        geometry,
        SyntheticSettings(
            simulator_version="vehicle_peakset_complex_v2",
            window_s=config.data.duration_s,
            feature_rate_hz=config.data.feature_rate_hz,
            waveform_rate_hz=200.0,
            max_vehicles=14,
            samples=args.synthetic_samples,
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
        ),
    )
    synthetic_inputs, synthetic_events = _synthetic_scenes(dataset)
    synthetic_strong_count = _synthetic_strong_event_count(
        synthetic_inputs, config.data.feature_rate_hz
    )
    real_profiles = _profiles_single(real_batch.tensor, real_events)
    recognized_profiles = _profiles_single(real_batch.tensor, recognized_events)
    synthetic_profiles = _profiles_synthetic(synthetic_inputs, synthetic_events)
    real_flat = real_batch.tensor.reshape(5, -1)
    synthetic_flat = np.concatenate([item.reshape(5, -1) for item in synthetic_inputs], axis=1)
    quantiles = [0.01, 0.1, 0.5, 0.9, 0.99, 0.999]
    input_summary = {}
    for plane, name in enumerate(("raw", "pre", "gauss", "quality", "coordinate")):
        input_summary[name] = {
            "real_quantiles": np.quantile(real_flat[plane], quantiles).tolist(),
            "synthetic_quantiles": np.quantile(synthetic_flat[plane], quantiles).tolist(),
            "wasserstein_distance": float(
                wasserstein_distance(real_flat[plane], synthetic_flat[plane])
            ),
        }

    payload: dict[str, object] = {
        "audit_semantics": (
            "DAY11 events are strong Gauss peaks, not confirmed vehicle ground truth; "
            "recognized events are points in the current unvalidated candidate tracks."
        ),
        "real_window_s": config.data.duration_s,
        "synthetic_window_s": dataset.settings.window_s,
        "synthetic_simulator_version": dataset.settings.simulator_version,
        "synthetic_samples": len(dataset),
        "real_strong_event_count": len(real_events),
        "recognized_event_count": len(recognized_events),
        "synthetic_gt_observation_count": len(synthetic_events),
        "synthetic_strong_event_count": synthetic_strong_count,
        "real_strong_events_per_48s_per_station": len(real_events)
        / config.data.duration_s
        * 48.0
        / len(geometry),
        "synthetic_gt_events_per_48s_per_station": len(synthetic_events)
        / len(dataset)
        / len(geometry),
        "synthetic_strong_events_per_48s_per_station": synthetic_strong_count
        / len(dataset)
        / len(geometry),
        "input_distribution": input_summary,
        "real_event_profile": _event_summary(real_profiles, config.data.feature_rate_hz),
        "recognized_event_profile": _event_summary(
            recognized_profiles, config.data.feature_rate_hz
        ),
        "synthetic_gt_event_profile": _event_summary(
            synthetic_profiles, config.data.feature_rate_hz
        ),
    }
    model_probability = None
    if not args.skip_model:
        if not torch.cuda.is_available():
            raise RuntimeError("audit requested model comparison but CUDA is unavailable")
        model = HybridPerceptionModel(config.model)
        model.load_checkpoint(config.model.checkpoint, map_location="cpu")
        model_probability = _model_probabilities(
            model,
            real_batch.tensor,
            real_events,
            synthetic_inputs,
            synthetic_events,
        )
        payload["model_probability"] = {
            key: value
            for key, value in model_probability.items()
            if key not in {"real_probability", "synthetic_probability"}
        }

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    profile_time = (
        np.arange(real_profiles.shape[-1], dtype=np.float32)
        - real_profiles.shape[-1] // 2
    ) / config.data.feature_rate_hz
    np.savez_compressed(
        args.output / "profiles.npz",
        time_s=profile_time,
        real_q25=_profile_stat(real_profiles, "q25"),
        real_median=_profile_stat(real_profiles, "median"),
        real_q75=_profile_stat(real_profiles, "q75"),
        recognized_q25=_profile_stat(recognized_profiles, "q25"),
        recognized_median=_profile_stat(recognized_profiles, "median"),
        recognized_q75=_profile_stat(recognized_profiles, "q75"),
        synthetic_q25=_profile_stat(synthetic_profiles, "q25"),
        synthetic_median=_profile_stat(synthetic_profiles, "median"),
        synthetic_q75=_profile_stat(synthetic_profiles, "q75"),
    )
    _plot_profiles(
        args.output / "event_profile_comparison.png",
        profile_time,
        real_profiles,
        recognized_profiles,
        synthetic_profiles,
    )
    synthetic_scene = dataset[0]
    _plot_scene_comparison(
        args.output / "scene_comparison.png",
        real_batch.gauss_score,
        synthetic_inputs[0],
        track_points,
        synthetic_scene,
        config.data.feature_rate_hz,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
