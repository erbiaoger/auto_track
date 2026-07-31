"""Run the compact slot model on a full real DAS `.npy` file."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.compact_slot_model import InferenceConfig, load_checkpoint_model, predict_tracks_from_window
from autotrack.dl.predict_single_vehicle_real_npy import _load_window, _window_activity_score, _window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the compact slot model over a real DAS .npy file.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--input", required=True, type=Path, help="Real DAS .npy file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for CSV, summary, and optional plots.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--array-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Input array layout.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Sliding window length in seconds.")
    parser.add_argument("--window-stride-seconds", type=float, default=10.0, help="Stride between windows in seconds.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to each extracted window.")
    parser.add_argument("--max-windows", type=int, default=64, help="Maximum number of windows to process; 0 processes all.")
    parser.add_argument("--window-activity-threshold", type=float, default=0.5, help="Skip windows whose robust activity score is below this threshold.")
    parser.add_argument("--activity-sorted", action=argparse.BooleanOptionalAction, default=True, help="Process the most active windows first.")
    parser.add_argument("--window-ranking", default="activity", choices=["activity", "model"], help="Ranking strategy used before decoding.")
    parser.add_argument("--plot-samples", type=int, default=8, help="Number of overlay plots to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay figure DPI.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by the checkpoint.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--objectness-threshold", type=float, default=0.15, help="Slot objectness threshold.")
    parser.add_argument("--visibility-threshold", type=float, default=0.35, help="Per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for a track.")
    parser.add_argument("--max-tracks-per-window", type=int, default=24, help="Maximum predicted tracks kept per window.")
    parser.add_argument("--candidate-objectness-floor", type=float, default=0.05, help="Minimum slot objectness considered as a candidate.")
    parser.add_argument("--objectness-count-scale", type=float, default=1.05, help="Soft count scale for how many slots to keep.")
    parser.add_argument("--dedup-tolerance-samples", type=int, default=180, help="Track deduplication tolerance in samples.")
    parser.add_argument("--dedup-min-overlap-channels", type=int, default=3, help="Minimum shared channels for deduplication.")
    parser.add_argument("--kalman-smooth", action=argparse.BooleanOptionalAction, default=True, help="Apply a simple Kalman smoother to each track.")
    parser.add_argument("--refine-radius-samples", type=int, default=120, help="Local peak refinement radius in samples.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


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


def _track_time_map(track: Track) -> dict[int, int]:
    return {int(p.ch_idx): int(p.t_idx) for p in track.points}


def _deduplicate_tracks(tracks: list[Track], tol_samples: int, min_overlap: int) -> list[Track]:
    kept: list[Track] = []
    for track in sorted(tracks, key=lambda item: item.total_score, reverse=True):
        track_map = _track_time_map(track)
        duplicate = False
        for existing in kept:
            existing_map = _track_time_map(existing)
            common = sorted(set(track_map) & set(existing_map))
            if len(common) < int(min_overlap):
                continue
            diffs = np.array([abs(track_map[ch] - existing_map[ch]) for ch in common], dtype=np.float64)
            if float(np.median(diffs)) <= float(tol_samples):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return kept


def _shift_track(track: Track, start_s: float, fs: float, dx_m: float) -> Track:
    points = [
        TrackPoint(
            ch_idx=int(p.ch_idx),
            t_idx=int(p.t_idx + round(start_s * fs)),
            time_s=float(start_s + p.time_s),
            offset_m=float(p.offset_m if np.isfinite(p.offset_m) else float(p.ch_idx) * float(dx_m)),
            amp=float(p.amp),
            score=float(p.score),
        )
        for p in track.points
    ]
    return Track(
        track_id=int(track.track_id),
        direction=str(track.direction),
        points=points,
        total_score=float(track.total_score),
        mean_speed_kmh=float(track.mean_speed_kmh),
    )


def _plot_window(
    out_path: Path,
    *,
    window: np.ndarray,
    fs: float,
    dx_m: float,
    start_s: float,
    tracks: list[Track],
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
        xs = [float(point.time_s) for point in track.points]
        ys = [float(point.ch_idx) * float(dx_m) for point in track.points]
        ax.plot(xs, ys, linewidth=2.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("channel offset (m)")
    ax.set_title(f"Compact slot inference overlay, t0={start_s:.2f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
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

    if str(args.window_ranking) == "activity":
        starts = sorted(
            starts,
            key=lambda start: _window_activity_score(
                _load_window(
                    arr,
                    layout=str(args.array_layout),
                    channel_start=int(args.channel_start),
                    channel_count=int(args.channel_count),
                    start_t=int(start),
                    window_samples=int(window_samples),
                    background_scale=float(args.background_scale),
                )
            ),
            reverse=True,
        )
    if int(args.max_windows) > 0:
        starts = starts[: int(args.max_windows)]

    rows: list[dict[str, Any]] = []
    all_tracks: list[Track] = []
    plot_limit = max(0, int(args.plot_samples))
    infer_cfg = InferenceConfig(
        time_downsample=int(args.time_downsample),
        objectness_threshold=float(args.objectness_threshold),
        visibility_threshold=float(args.visibility_threshold),
        min_visible_channels=int(args.min_visible_channels),
        max_tracks=int(args.max_tracks_per_window),
        candidate_objectness_floor=float(args.candidate_objectness_floor),
        objectness_count_scale=float(args.objectness_count_scale),
        dedup_tolerance_samples=int(args.dedup_tolerance_samples),
        dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
        speed_norm_kmh=float(args.speed_norm_kmh),
        kalman_smooth=bool(args.kalman_smooth),
        refine_radius_samples=int(args.refine_radius_samples),
    )

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

        tracks = predict_tracks_from_window(
            model,
            window,
            float(args.fs),
            np.arange(int(args.channel_count), dtype=np.float32) * float(args.dx_m),
            config=infer_cfg,
            device=device,
        )
        rows.append(
            {
                "window_index": int(i),
                "window_start_s": float(start_t) / float(args.fs),
                "track_count": int(len(tracks)),
                "mean_speed_kmh": float(np.mean([tr.mean_speed_kmh for tr in tracks])) if tracks else None,
                "total_score": float(np.sum([tr.total_score for tr in tracks])) if tracks else None,
                "direction": "multi",
                "activity_score": float(activity_score),
            }
        )
        shifted = [_shift_track(tr, float(start_t) / float(args.fs), float(args.fs), float(args.dx_m)) for tr in tracks]
        all_tracks.extend(shifted)

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

    merged_tracks = _deduplicate_tracks(
        all_tracks,
        tol_samples=int(args.dedup_tolerance_samples),
        min_overlap=int(args.dedup_min_overlap_channels),
    )
    merged_tracks = sorted(merged_tracks, key=lambda tr: tr.total_score, reverse=True)

    predicted_rows: list[dict[str, Any]] = []
    for tr in all_tracks:
        for point in tr.points:
            predicted_rows.append(
                {
                    "track_id": int(tr.track_id),
                    "direction": str(tr.direction),
                    "ch_idx": int(point.ch_idx),
                    "t_idx": int(point.t_idx),
                    "time_s": float(point.time_s),
                    "offset_m": float(point.offset_m),
                    "amp": float(point.amp),
                    "score": float(point.score),
                    "track_score": float(tr.total_score),
                    "mean_speed_kmh": float(tr.mean_speed_kmh),
                }
            )

    merged_rows: list[dict[str, Any]] = []
    for tr in merged_tracks:
        for point in tr.points:
            merged_rows.append(
                {
                    "track_id": int(tr.track_id),
                    "direction": str(tr.direction),
                    "ch_idx": int(point.ch_idx),
                    "t_idx": int(point.t_idx),
                    "time_s": float(point.time_s),
                    "offset_m": float(point.offset_m),
                    "amp": float(point.amp),
                    "score": float(point.score),
                    "track_score": float(tr.total_score),
                    "mean_speed_kmh": float(tr.mean_speed_kmh),
                }
            )

    with (out_dir / "sample_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["window_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with (out_dir / "predicted_track_points.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["track_id", "direction", "ch_idx", "t_idx", "time_s", "offset_m", "amp", "score", "track_score", "mean_speed_kmh"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in predicted_rows:
            writer.writerow(row)

    with (out_dir / "merged_track_points.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["track_id", "direction", "ch_idx", "t_idx", "time_s", "offset_m", "amp", "score", "track_score", "mean_speed_kmh"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
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
        "merged_track_points": int(len(merged_rows)),
        "outputs": {
            "sample_summary_csv": str(out_dir / "sample_summary.csv"),
            "predicted_track_points_csv": str(out_dir / "predicted_track_points.csv"),
            "merged_track_points_csv": str(out_dir / "merged_track_points.csv"),
            "plots_dir": str(out_dir / "plots"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
