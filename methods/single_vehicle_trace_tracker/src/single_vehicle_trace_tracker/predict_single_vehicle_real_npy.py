"""Run the single-vehicle model on a full real DAS `.npy` file with sliding windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.single_vehicle_net import (
    InferenceConfig,
    SingleVehicleTrackerConfig,
    load_checkpoint_model,
    predict_single_vehicle_track,
    prepare_window_input,
)
from autotrack.core.single_vehicle_tracker import merge_single_vehicle_track_fragments


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-vehicle model over a real DAS .npy file.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--input", required=True, type=Path, help="Real DAS .npy file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for CSV, summary, and optional plots.")
    parser.add_argument(
        "--preset",
        default="real_vehicle",
        choices=["real_vehicle", "fast_scan", "manual"],
        help="Inference preset: real_vehicle uses the validated long-window default; fast_scan keeps shorter windows.",
    )
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--array-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Input array layout.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Sliding window length in seconds.")
    parser.add_argument("--window-stride-seconds", type=float, default=10.0, help="Stride between windows in seconds.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to each extracted window.")
    parser.add_argument("--max-windows", type=int, default=64, help="Maximum number of windows to process; 0 processes all.")
    parser.add_argument("--window-activity-threshold", type=float, default=0.5, help="Skip windows whose robust activity score is below this threshold.")
    parser.add_argument("--activity-sorted", action=argparse.BooleanOptionalAction, default=True, help="Process the most active windows first.")
    parser.add_argument("--window-ranking", default="model", choices=["activity", "model"], help="Ranking strategy used before decoding.")
    parser.add_argument(
        "--direction-mode",
        default="auto",
        choices=["auto", "predicted", "forward", "reverse", "both"],
        help="Direction handling for decoding.",
    )
    parser.add_argument("--plot-samples", type=int, default=8, help="Number of overlay plots to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay figure DPI.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by the checkpoint.")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Weight applied to the model prior heatmap.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for tracker.")
    parser.add_argument("--candidate-prominence", type=float, default=0.22, help="Tracker peak prominence.")
    parser.add_argument("--candidate-min-distance", type=int, default=180, help="Tracker minimum peak distance.")
    parser.add_argument("--candidate-max-peaks-per-channel", type=int, default=32, help="Tracker peak cap per channel.")
    parser.add_argument("--max-skip-channels", type=int, default=8, help="Tracker graph skip limit.")
    parser.add_argument("--min-track-channels", type=int, default=8, help="Minimum track length.")
    parser.add_argument("--min-track-score", type=float, default=8.0, help="Minimum track score.")
    parser.add_argument("--kalman-bridge-gap-channels", type=int, default=12, help="Maximum Kalman bridge gap.")
    parser.add_argument("--kalman-fill-missing", action=argparse.BooleanOptionalAction, default=True, help="Fill missing channels after smoothing.")
    parser.add_argument("--kalman-gate-seconds", type=float, default=0.35, help="Gate for Hungarian gap reassignment.")
    parser.add_argument("--kalman-speed-gate-kmh", type=float, default=30.0, help="Kept for config parity.")
    return parser.parse_args(argv)


def _apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = str(getattr(args, "preset", "manual")).strip().lower()
    if preset == "real_vehicle":
        args.window_seconds = 60.0
        args.window_stride_seconds = 10.0
        args.window_ranking = "model"
        args.direction_mode = "both"
        args.max_windows = max(int(args.max_windows), 24)
        args.window_activity_threshold = 0.0
    elif preset == "fast_scan":
        args.window_seconds = min(float(args.window_seconds), 20.0)
        args.window_stride_seconds = min(float(args.window_stride_seconds), 10.0)
        args.window_ranking = "activity"
        args.direction_mode = "both"
        args.max_windows = max(int(args.max_windows), 32)
        args.window_activity_threshold = max(0.0, float(args.window_activity_threshold))
    return args


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def _load_window(
    arr: np.ndarray,
    *,
    layout: str,
    channel_start: int,
    channel_count: int,
    start_t: int,
    window_samples: int,
    background_scale: float,
) -> np.ndarray:
    if layout not in {"time_channel", "channel_time"}:
        raise ValueError("layout must be time_channel or channel_time")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D npy array, got {arr.shape}")
    if layout == "time_channel":
        n_time, n_channels_all = arr.shape
        end_ch = int(channel_start + channel_count)
        if channel_start < 0 or end_ch > int(n_channels_all):
            raise ValueError(f"Channel slice [{channel_start}, {end_ch}) outside {arr.shape}")
        if start_t < 0 or start_t + window_samples > int(n_time):
            raise ValueError("Window slice outside array")
        window = np.array(arr[start_t : start_t + window_samples, channel_start:end_ch], dtype=np.float32, copy=True).T
    else:
        n_channels_all, n_time = arr.shape
        end_ch = int(channel_start + channel_count)
        if channel_start < 0 or end_ch > int(n_channels_all):
            raise ValueError(f"Channel slice [{channel_start}, {end_ch}) outside {arr.shape}")
        if start_t < 0 or start_t + window_samples > int(n_time):
            raise ValueError("Window slice outside array")
        window = np.array(arr[channel_start:end_ch, start_t : start_t + window_samples], dtype=np.float32, copy=True)
    window = np.nan_to_num(window, copy=False)
    if float(background_scale) != 1.0:
        window *= float(background_scale)
    return window


def _window_starts(n_time: int, window_samples: int, stride_samples: int, max_windows: int) -> list[int]:
    if window_samples <= 0 or window_samples > n_time:
        return []
    starts = list(range(0, max(1, n_time - window_samples + 1), max(1, stride_samples)))
    last = int(n_time - window_samples)
    if not starts or starts[-1] != last:
        starts.append(last)
    starts = sorted(set(int(item) for item in starts))
    if int(max_windows) > 0:
        starts = starts[: int(max_windows)]
    return starts


def _window_activity_score(window: np.ndarray) -> float:
    arr = np.asarray(window, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    centered = finite - float(np.median(finite))
    abs_vals = np.abs(centered)
    q995 = float(np.quantile(abs_vals, 0.995))
    rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
    return max(0.0, q995 + 4.0 * rms)


def _window_model_score(
    model: torch.nn.Module,
    window: np.ndarray,
    *,
    device: str,
    time_downsample: int,
) -> float:
    x = prepare_window_input(window, int(time_downsample)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        obj_prob = float(torch.sigmoid(outputs["objectness_logits"][0]).item())
        heatmap_prob = torch.sigmoid(outputs["heatmap_logits"][0]).detach()
        heatmap_peak = float(heatmap_prob.max().item())
        heatmap_mean = float(heatmap_prob.mean().item())
        heatmap_contrast = max(0.0, heatmap_peak - heatmap_mean)
    return 0.65 * obj_prob + 0.35 * min(1.0, heatmap_contrast / max(1e-6, heatmap_peak + 1e-3))


def _decode_window_tracks(
    model: torch.nn.Module,
    window: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    pred_dir: int,
    pred_speed: float,
    device: str,
    time_downsample: int,
    min_visible_channels: int,
    prior_weight: float,
    candidate_prominence: float,
    candidate_min_distance: int,
    candidate_max_peaks_per_channel: int,
    max_skip_channels: int,
    min_track_channels: int,
    min_track_score: float,
    kalman_bridge_gap_channels: int,
    kalman_fill_missing: bool,
    kalman_gate_seconds: float,
    kalman_speed_gate_kmh: float,
    direction_mode: str,
) -> tuple[list[Any], str]:
    speed_margin = max(15.0, 0.2 * pred_speed)
    directions: list[str]
    mode = str(direction_mode)
    if mode == "forward":
        directions = ["forward"]
    elif mode == "reverse":
        directions = ["reverse"]
    elif mode == "both":
        directions = ["forward", "reverse"]
    elif mode == "auto":
        directions = ["auto"]
    else:
        directions = ["forward" if pred_dir == 0 else "reverse"]
        if pred_speed < 1.0:
            directions = ["forward", "reverse"]

    candidates: list[tuple[float, str, list[Any]]] = []
    for direction in directions:
        tracks = predict_single_vehicle_track(
            model,
            window,
            float(fs),
            float(dx_m),
            direction,
            max(1.0, pred_speed - speed_margin),
            pred_speed + speed_margin,
            InferenceConfig(
                time_downsample=int(time_downsample),
                min_visible_channels=int(min_visible_channels),
                prior_weight=float(prior_weight),
                single_vehicle_tracker=SingleVehicleTrackerConfig(
                    candidate_prominence=float(candidate_prominence),
                    candidate_min_distance=int(candidate_min_distance),
                    candidate_max_peaks_per_channel=int(candidate_max_peaks_per_channel),
                    max_skip_channels=int(max_skip_channels),
                    min_track_channels=int(min_track_channels),
                    min_track_score=float(min_track_score),
                    kalman_bridge_gap_channels=int(kalman_bridge_gap_channels),
                    kalman_fill_missing=bool(kalman_fill_missing),
                    kalman_gate_seconds=float(kalman_gate_seconds),
                    kalman_speed_gate_kmh=float(kalman_speed_gate_kmh),
                ),
            ),
            device=device,
        )
        top_score = float(tracks[0].total_score) if tracks else float("-inf")
        candidates.append((top_score, direction, tracks))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0]
    resolved_direction = best[2][0].direction if best[2] else best[1]
    return best[2], resolved_direction


def _plot_window(
    out_path: Path,
    *,
    window: np.ndarray,
    fs: float,
    dx_m: float,
    start_s: float,
    tracks: list[Any],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=int(dpi))
    im = ax.imshow(
        window,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[start_s, start_s + window.shape[1] / float(fs), 0.0, float(window.shape[0] - 1) * float(dx_m)],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="scaled amplitude")
    for track in tracks:
        xs = [start_s + float(point.time_s) for point in track.points]
        ys = [float(point.ch_idx) * float(dx_m) for point in track.points]
        ax.plot(xs, ys, linewidth=2.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("channel offset (m)")
    ax.set_title(f"Single-vehicle inference overlay, t0={start_s:.2f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = _apply_preset(parse_args(argv))
    device = _resolve_device(args.device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _ = load_checkpoint_model(args.model, device=device)

    arr = np.load(str(args.input), mmap_mode="r")
    window_samples = int(round(float(args.window_seconds) * float(args.fs)))
    stride_samples = int(round(float(args.window_stride_seconds) * float(args.fs)))
    n_time = int(arr.shape[0] if str(args.array_layout) == "time_channel" else arr.shape[1])
    starts = _window_starts(n_time, window_samples, stride_samples, 0 if int(args.max_windows) > 0 else 0)
    if not starts:
        raise ValueError("No windows available for the requested window size")
    if str(args.window_ranking) == "model":
        coarse_limit = max(64, int(args.max_windows) * 4 if int(args.max_windows) > 0 else 64)
        activity_ranked: list[tuple[float, int, np.ndarray]] = []
        for start in starts:
            window = _load_window(
                arr,
                layout=str(args.array_layout),
                channel_start=int(args.channel_start),
                channel_count=int(args.channel_count),
                start_t=int(start),
                window_samples=int(window_samples),
                background_scale=float(args.background_scale),
            )
            activity_score = _window_activity_score(window)
            if activity_score < float(args.window_activity_threshold):
                continue
            activity_ranked.append((float(activity_score), int(start), window))
        activity_ranked.sort(key=lambda item: item[0], reverse=True)
        ranked: list[tuple[float, int]] = []
        for activity_score, start, window in activity_ranked[:coarse_limit]:
            model_score = _window_model_score(model, window, device=device, time_downsample=int(args.time_downsample))
            ranked.append((0.65 * float(model_score) + 0.35 * min(1.0, float(activity_score) / 2.0), int(start)))
        starts = [start for _, start in sorted(ranked, key=lambda item: item[0], reverse=True)]
    elif bool(args.activity_sorted):
        starts = sorted(starts, key=lambda start: _window_activity_score(_load_window(
            arr,
            layout=str(args.array_layout),
            channel_start=int(args.channel_start),
            channel_count=int(args.channel_count),
            start_t=int(start),
            window_samples=int(window_samples),
            background_scale=float(args.background_scale),
        )), reverse=True)
    if int(args.max_windows) > 0:
        starts = starts[: int(args.max_windows)]

    rows: list[dict[str, Any]] = []
    all_tracks: list[dict[str, Any]] = []
    fragments: list[Any] = []
    fragment_directions: list[int] = []
    plot_limit = max(0, int(args.plot_samples))
    for i, start_t in enumerate(starts):
        window = _load_window(
            arr,
            layout=str(args.array_layout),
            channel_start=int(args.channel_start),
            channel_count=int(args.channel_count),
            start_t=int(start_t),
            window_samples=int(window_samples),
            background_scale=float(args.background_scale),
        )
        activity_score = _window_activity_score(window)
        if activity_score < float(args.window_activity_threshold):
            rows.append(
                {
                    "window_index": int(i),
                    "window_start_s": float(start_t) / float(args.fs),
                    "track_count": 0,
                    "mean_speed_kmh": None,
                    "total_score": None,
                    "direction": "skipped",
                    "activity_score": float(activity_score),
                }
            )
            continue
        x = prepare_window_input(window, int(args.time_downsample)).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(x)
        pred_dir = int(outputs["direction_logits"].argmax(dim=-1).item())
        pred_speed = float(outputs["speed"].item()) * float(args.speed_norm_kmh)
        if not math.isfinite(pred_speed) or pred_speed <= 0:
            pred_speed = 80.0
        tracks, track_direction = _decode_window_tracks(
            model,
            window,
            fs=float(args.fs),
            dx_m=float(args.dx_m),
            pred_dir=int(pred_dir),
            pred_speed=float(pred_speed),
            device=device,
            time_downsample=int(args.time_downsample),
            min_visible_channels=int(args.min_visible_channels),
            prior_weight=float(args.prior_weight),
            candidate_prominence=float(args.candidate_prominence),
            candidate_min_distance=int(args.candidate_min_distance),
            candidate_max_peaks_per_channel=int(args.candidate_max_peaks_per_channel),
            max_skip_channels=int(args.max_skip_channels),
            min_track_channels=int(args.min_track_channels),
            min_track_score=float(args.min_track_score),
            kalman_bridge_gap_channels=int(args.kalman_bridge_gap_channels),
            kalman_fill_missing=bool(args.kalman_fill_missing),
            kalman_gate_seconds=float(args.kalman_gate_seconds),
            kalman_speed_gate_kmh=float(args.kalman_speed_gate_kmh),
            direction_mode=str(args.direction_mode),
        )
        rows.append(
            {
                "window_index": int(i),
                "window_start_s": float(start_t) / float(args.fs),
                "track_count": int(len(tracks)),
                "mean_speed_kmh": float(tracks[0].mean_speed_kmh) if tracks else None,
                "total_score": float(tracks[0].total_score) if tracks else None,
                "direction": str(track_direction),
                "activity_score": float(activity_score),
            }
        )
        if tracks:
            fragments.append(tracks[0])
            fragment_directions.append(0 if str(track_direction) == "forward" else 1)
            for point in tracks[0].points:
                all_tracks.append(
                    {
                        "window_index": int(i),
                        "window_start_s": float(start_t) / float(args.fs),
                        "ch_idx": int(point.ch_idx),
                        "abs_time_s": float(start_t) / float(args.fs) + float(point.time_s),
                        "time_s": float(point.time_s),
                        "offset_m": float(point.offset_m),
                        "amp": float(point.amp),
                        "score": float(point.score),
                    }
                )
        if i < plot_limit:
            _plot_window(
                out_dir / "plots" / f"window_{i:06d}.png",
                window=window,
                fs=float(args.fs),
                dx_m=float(args.dx_m),
                start_s=float(start_t) / float(args.fs),
                tracks=tracks,
                dpi=int(args.plot_dpi),
            )

    merged_tracks = merge_single_vehicle_track_fragments(
        fragments,
        fs=float(args.fs),
        dx_m=float(args.dx_m),
        direction="forward" if (sum(1 for d in fragment_directions if d == 0) >= sum(1 for d in fragment_directions if d == 1)) else "reverse",
        merge_tolerance_samples=int(round(float(args.window_stride_seconds) * float(args.fs) * 0.5)),
        max_gap_channels=max(12, int(round(float(args.window_seconds) / max(1e-6, float(args.window_stride_seconds))))),
        kalman_process_var=0.8,
        kalman_meas_var=0.18,
        kalman_fill_missing=True,
    )
    merged_track_rows: list[dict[str, Any]] = []
    if merged_tracks:
        merged_track = merged_tracks[0]
        for point in merged_track.points:
            merged_track_rows.append(
                {
                    "ch_idx": int(point.ch_idx),
                    "abs_time_s": float(point.time_s),
                    "time_s": float(point.time_s),
                    "offset_m": float(point.offset_m),
                    "amp": float(point.amp),
                    "score": float(point.score),
                }
            )
        with (out_dir / "merged_track_points.csv").open("w", encoding="utf-8", newline="") as fp:
            fieldnames = ["ch_idx", "abs_time_s", "time_s", "offset_m", "amp", "score"]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_track_rows:
                writer.writerow(row)

    (out_dir / "sample_summary.csv").write_text("", encoding="utf-8")
    with (out_dir / "sample_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["window_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with (out_dir / "predicted_track_points.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["window_index", "window_start_s", "ch_idx", "abs_time_s", "time_s", "offset_m", "amp", "score"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_tracks:
            writer.writerow(row)

    summary = {
        "model": str(args.model),
        "input": str(args.input),
        "device": device,
        "sample_count": int(len(starts)),
        "processed_window_count": int(sum(1 for row in rows if str(row.get("direction")) != "skipped")),
        "track_found_rate": float(sum(1 for row in rows if int(row["track_count"]) > 0) / max(1, len(rows))),
        "avg_track_count": float(sum(int(row["track_count"]) for row in rows) / max(1, len(rows))),
        "avg_mean_speed_kmh": float(np.mean([row["mean_speed_kmh"] for row in rows if row["mean_speed_kmh"] is not None])) if any(row["mean_speed_kmh"] is not None for row in rows) else None,
        "merged_track_count": int(len(merged_tracks)),
        "merged_track_points": int(len(merged_track_rows)),
        "outputs": {
            "sample_summary_csv": str(out_dir / "sample_summary.csv"),
            "predicted_track_points_csv": str(out_dir / "predicted_track_points.csv"),
            "merged_track_points_csv": str(out_dir / "merged_track_points.csv") if merged_track_rows else None,
            "plots_dir": str(out_dir / "plots"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
