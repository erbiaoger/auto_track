"""Run the multi-vehicle candidate pipeline on a real DAS `.npy` segment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.multi_vehicle_pipeline import MultiVehiclePipelineConfig, extract_multi_vehicle_tracks
from autotrack.dl.predict_single_vehicle_real_npy import _load_window, _window_activity_score, _window_starts
from autotrack.dl.vehicle_trace_net import (
    InferenceConfig,
    SingleVehicleTrackerConfig,
    load_checkpoint_model,
    predict_single_vehicle_track,
    prepare_window_input,
)
from autotrack.dl.vehicle_proposal_net import (
    load_checkpoint_model as load_proposal_checkpoint_model,
    prepare_window_input as prepare_proposal_window_input,
    score_vehicle_proposal_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the multi-vehicle pipeline over a real DAS .npy segment.")
    parser.add_argument("--input", required=True, type=Path, help="Real DAS .npy file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for CSV, summary, and plots.")
    parser.add_argument("--model", type=Path, default=None, help="Optional single-vehicle checkpoint for per-candidate refinement.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--array-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Input array layout.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--proposal-model", type=Path, default=None, help="Optional dense proposal checkpoint for full-segment candidate extraction.")
    parser.add_argument("--proposal-prior-weight", type=float, default=1.25, help="Weight of the proposal prior before graph extraction.")
    parser.add_argument("--proposal-time-downsample", type=int, default=10, help="Time downsample factor used by the proposal model.")
    parser.add_argument(
        "--direction",
        default="auto",
        choices=["forward", "reverse", "both", "auto"],
        help="Vehicle motion direction.",
    )
    parser.add_argument("--vmin-kmh", type=float, default=40.0, help="Minimum plausible speed.")
    parser.add_argument("--vmax-kmh", type=float, default=140.0, help="Maximum plausible speed.")
    parser.add_argument("--candidate-mode", default="hybrid", choices=["windowed", "graph", "hybrid"], help="Candidate extraction strategy.")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window length used in windowed candidate mode.")
    parser.add_argument("--window-stride-seconds", type=float, default=5.0, help="Window stride used in windowed candidate mode.")
    parser.add_argument("--max-windows", type=int, default=0, help="Maximum windows in windowed candidate mode; 0 means all.")
    parser.add_argument("--window-activity-threshold", type=float, default=0.0, help="Skip windows whose robust activity score is below this threshold.")
    parser.add_argument("--activity-sorted", action=argparse.BooleanOptionalAction, default=True, help="Process the most active windows first in windowed candidate mode.")
    parser.add_argument("--candidate-limit", type=int, default=48, help="Maximum candidate tracks to refine.")
    parser.add_argument("--candidate-min-score", type=float, default=8.0, help="Minimum candidate score before refinement.")
    parser.add_argument("--dedup-tolerance-samples", type=int, default=180, help="Track deduplication tolerance in samples.")
    parser.add_argument("--dedup-min-overlap-channels", type=int, default=2, help="Minimum shared channels for deduplication.")
    parser.add_argument("--dedup-min-overlap-ratio", type=float, default=0.45, help="Minimum overlap ratio for deduplication.")
    parser.add_argument("--crop-channel-margin", type=int, default=4, help="Channels to add around each candidate crop.")
    parser.add_argument("--crop-time-margin-seconds", type=float, default=4.0, help="Time margin around each candidate crop.")
    parser.add_argument("--no-refine", action="store_true", help="Disable single-vehicle refinement and only use graph candidates.")
    parser.add_argument("--min-model-confidence", type=float, default=0.18, help="Minimum single-vehicle model confidence before refinement.")
    parser.add_argument("--model-confidence-weight", type=float, default=0.7, help="Weight of model confidence in final track score.")
    parser.add_argument("--graph-confidence-weight", type=float, default=0.3, help="Weight of graph candidate score in final track score.")
    parser.add_argument("--plot", action="store_true", help="Render an overlay plot of the final tracks.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay plot DPI.")
    return parser.parse_args()


def _load_segment(arr: np.ndarray, *, layout: str, channel_start: int, channel_count: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D npy array, got {arr.shape}")
    if layout not in {"time_channel", "channel_time"}:
        raise ValueError("layout must be time_channel or channel_time")
    end_ch = int(channel_start + channel_count)
    if layout == "time_channel":
        if channel_start < 0 or end_ch > int(arr.shape[1]):
            raise ValueError(f"Channel slice [{channel_start}, {end_ch}) outside {arr.shape}")
        segment = np.array(arr[:, channel_start:end_ch], dtype=np.float32, copy=True).T
    else:
        if channel_start < 0 or end_ch > int(arr.shape[0]):
            raise ValueError(f"Channel slice [{channel_start}, {end_ch}) outside {arr.shape}")
        segment = np.array(arr[channel_start:end_ch, :], dtype=np.float32, copy=True)
    return np.nan_to_num(segment, copy=False)


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def _direction_list(direction: str) -> list[str]:
    if direction in {"both", "auto"}:
        return ["forward", "reverse"]
    return [direction]


def _shift_track(track: Track, ch_offset: int, t_offset: int, *, fs: float, dx_m: float) -> Track:
    points = [
        TrackPoint(
            ch_idx=int(p.ch_idx) + int(ch_offset),
            t_idx=int(p.t_idx) + int(t_offset),
            time_s=float(p.time_s) + float(t_offset) / float(fs),
            offset_m=float(int(p.ch_idx) + int(ch_offset)) * float(dx_m),
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


def _track_overlap(a: Track, b: Track, *, tol_samples: int, min_overlap_channels: int) -> tuple[float, int]:
    amap = {int(p.ch_idx): int(p.t_idx) for p in a.points}
    bmap = {int(p.ch_idx): int(p.t_idx) for p in b.points}
    common = sorted(set(amap).intersection(bmap))
    if len(common) < int(min_overlap_channels):
        return 0.0, 0
    diffs = [abs(float(amap[ch] - bmap[ch])) for ch in common]
    if float(np.median(diffs)) > float(tol_samples):
        return 0.0, len(common)
    ratio = len(common) / float(max(1, min(len(a.points), len(b.points))))
    return float(ratio), len(common)


def _track_line_fit(track: Track) -> tuple[float, float] | None:
    if len(track.points) < 2:
        return None
    ch = np.asarray([int(p.ch_idx) for p in track.points], dtype=np.float64)
    t = np.asarray([float(p.time_s) for p in track.points], dtype=np.float64)
    if np.ptp(ch) < 1e-9:
        return None
    slope, intercept = np.polyfit(ch, t, deg=1)
    return float(slope), float(intercept)


def _track_line_distance(a: Track, b: Track) -> float:
    fa = _track_line_fit(a)
    fb = _track_line_fit(b)
    if fa is None or fb is None:
        return float("inf")
    slope_a, intercept_a = fa
    slope_b, intercept_b = fb
    center_a = 0.5 * (float(min(int(p.ch_idx) for p in a.points)) + float(max(int(p.ch_idx) for p in a.points)))
    center_b = 0.5 * (float(min(int(p.ch_idx) for p in b.points)) + float(max(int(p.ch_idx) for p in b.points)))
    center_ch = 0.5 * (center_a + center_b)
    pred_a = slope_a * center_ch + intercept_a
    pred_b = slope_b * center_ch + intercept_b
    return abs(pred_a - pred_b) + 0.2 * abs(slope_a - slope_b)


def _deduplicate_tracks(
    tracks: list[Track],
    *,
    tol_samples: int,
    min_overlap_channels: int,
    min_overlap_ratio: float,
) -> list[Track]:
    ordered = sorted(tracks, key=lambda tr: tr.total_score, reverse=True)
    kept: list[Track] = []
    for tr in ordered:
        keep = True
        for existing in kept:
            ratio, common = _track_overlap(
                tr,
                existing,
                tol_samples=int(tol_samples),
                min_overlap_channels=int(min_overlap_channels),
            )
            line_distance = _track_line_distance(tr, existing)
            if (common >= int(min_overlap_channels) and ratio >= float(min_overlap_ratio)) or (
                line_distance < max(0.45, 1.5 * float(tol_samples) / 1000.0)
            ):
                keep = False
                break
        if keep:
            kept.append(tr)
    return kept


def _track_channel_bounds(track: Track) -> tuple[TrackPoint | None, TrackPoint | None, set[int]]:
    if not track.points:
        return None, None, set()
    pts = sorted(track.points, key=lambda p: int(p.ch_idx))
    return pts[0], pts[-1], {int(p.ch_idx) for p in pts}


def _dt_link_bounds(
    direction: str,
    delta_x_m: float,
    vmin_mps: float,
    vmax_mps: float,
    slack_ratio: float,
) -> tuple[float, float]:
    slack = max(0.0, float(slack_ratio))
    if direction == "forward":
        lo = (delta_x_m / vmax_mps) * max(0.0, 1.0 - slack)
        hi = (delta_x_m / vmin_mps) * (1.0 + slack)
    else:
        base_lo = -(delta_x_m / vmin_mps)
        base_hi = -(delta_x_m / vmax_mps)
        lo = base_lo * (1.0 + slack)
        hi = base_hi * max(0.0, 1.0 - slack)
    return (lo, hi) if lo <= hi else (hi, lo)


def _stitch_link_cost(
    left: Track,
    right: Track,
    *,
    direction: str,
    fs: float,
    dx_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    max_gap_channels: int,
    dt_slack_ratio: float,
    max_speed_diff_kmh: float,
) -> float | None:
    if left.direction != direction or right.direction != direction:
        return None
    l_start, l_end, l_set = _track_channel_bounds(left)
    r_start, _, r_set = _track_channel_bounds(right)
    if l_end is None or r_start is None:
        return None
    if l_set & r_set:
        return None
    dch = int(r_start.ch_idx) - int(l_end.ch_idx)
    if dch < 1 or dch > int(max_gap_channels):
        return None
    dx = abs(float(r_start.offset_m) - float(l_end.offset_m))
    if dx <= 1e-9:
        dx = float(dch) * float(dx_m)
    if dx <= 1e-9:
        return None
    dt = float(r_start.time_s) - float(l_end.time_s)
    vmin_mps = float(speed_min_kmh) / 3.6
    vmax_mps = float(speed_max_kmh) / 3.6
    dt_lo, dt_hi = _dt_link_bounds(direction, dx, vmin_mps, vmax_mps, dt_slack_ratio)
    if not (dt_lo <= dt <= dt_hi):
        return None
    speed_link_kmh = 3.6 * dx / max(1e-9, abs(dt))
    if not (float(speed_min_kmh) * 0.65 <= speed_link_kmh <= float(speed_max_kmh) * 1.35):
        return None
    speed_diff = 0.0
    if np.isfinite(left.mean_speed_kmh) and np.isfinite(right.mean_speed_kmh):
        speed_diff = abs(float(left.mean_speed_kmh) - float(right.mean_speed_kmh))
        if speed_diff > float(max_speed_diff_kmh):
            return None
    dt_mid = 0.5 * (dt_lo + dt_hi)
    dt_scale = max(1e-6, abs(dt_hi - dt_lo))
    dt_cost = abs(dt - dt_mid) / dt_scale
    speed_cost = speed_diff / max(1e-6, float(max_speed_diff_kmh))
    return float(dch) + 0.6 * float(dt_cost) + 0.8 * float(speed_cost)


def _stitch_track_fragments(
    tracks: list[Track],
    *,
    direction: str,
    fs: float,
    dx_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    tol_samples: int,
    max_gap_channels: int = 8,
    dt_slack_ratio: float = 0.35,
    max_speed_diff_kmh: float = 35.0,
) -> list[Track]:
    if len(tracks) < 2:
        return tracks
    merged = [Track(track_id=i, direction=tr.direction, points=list(tr.points), total_score=float(tr.total_score), mean_speed_kmh=float(tr.mean_speed_kmh)) for i, tr in enumerate(tracks)]
    while True:
        best_pair: tuple[int, int] | None = None
        best_cost = float("inf")
        for i in range(len(merged)):
            for j in range(len(merged)):
                if i == j:
                    continue
                cost = _stitch_link_cost(
                    merged[i],
                    merged[j],
                    direction=direction,
                    fs=float(fs),
                    dx_m=float(dx_m),
                    speed_min_kmh=float(speed_min_kmh),
                    speed_max_kmh=float(speed_max_kmh),
                    max_gap_channels=int(max_gap_channels),
                    dt_slack_ratio=float(dt_slack_ratio),
                    max_speed_diff_kmh=float(max_speed_diff_kmh),
                )
                if cost is None:
                    continue
                if cost < best_cost:
                    best_cost = cost
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        left = merged[i]
        right = merged[j]
        points: dict[int, TrackPoint] = {int(p.ch_idx): p for p in left.points}
        for p in right.points:
            old = points.get(int(p.ch_idx))
            if old is None or abs(int(old.t_idx) - int(p.t_idx)) <= int(tol_samples):
                points[int(p.ch_idx)] = p if old is None or p.score >= old.score else old
            elif p.score > old.score:
                points[int(p.ch_idx)] = p
        stitched_points = sorted(points.values(), key=lambda p: int(p.ch_idx))
        stitched = Track(
            track_id=left.track_id,
            direction=str(direction),
            points=stitched_points,
            total_score=float(sum(float(p.score) for p in stitched_points)),
            mean_speed_kmh=float(np.mean([float(left.mean_speed_kmh), float(right.mean_speed_kmh)]) if np.isfinite(left.mean_speed_kmh) and np.isfinite(right.mean_speed_kmh) else np.nan),
        )
        next_tracks = [tr for k, tr in enumerate(merged) if k not in {i, j}]
        next_tracks.append(stitched)
        merged = next_tracks
    return sorted(merged, key=lambda tr: tr.total_score, reverse=True)


def _windowed_multi_vehicle_tracks(
    segment: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    model_path: str,
    proposal_model_path: str | None,
    proposal_prior_weight: float,
    proposal_time_downsample: int,
    device: str,
    window_seconds: float,
    window_stride_seconds: float,
    max_windows: int,
    window_activity_threshold: float,
    activity_sorted: bool,
    candidate_limit: int,
    candidate_min_score: float,
    dedup_tolerance_samples: int,
    dedup_min_overlap_channels: int,
    dedup_min_overlap_ratio: float,
    crop_channel_margin: int,
    crop_time_margin_s: float,
    min_model_confidence: float,
    model_confidence_weight: float,
    graph_confidence_weight: float,
) -> list[Track]:
    model, _ = load_checkpoint_model(model_path, device=device)
    proposal_model = None
    if proposal_model_path is not None:
        proposal_model, _ = load_proposal_checkpoint_model(proposal_model_path, device=device)
    n_time = int(segment.shape[1])
    window_samples = int(round(float(window_seconds) * float(fs)))
    stride_samples = int(round(float(window_stride_seconds) * float(fs)))
    starts = _window_starts(n_time, window_samples, stride_samples, 0 if int(max_windows) > 0 else 0)
    if proposal_model is not None:
        ranked: list[tuple[float, int]] = []
        for start in starts:
            window = _load_window(
                segment.T,
                layout="time_channel",
                channel_start=0,
                channel_count=int(segment.shape[0]),
                start_t=int(start),
                window_samples=int(window_samples),
                background_scale=1.0,
            )
            x_prop = prepare_proposal_window_input(window, int(proposal_time_downsample)).unsqueeze(0).to(device)
            with torch.no_grad():
                prop_out = proposal_model(x_prop)
            score = float(score_vehicle_proposal_window(prop_out)["confidence"])
            ranked.append((score, int(start)))
        starts = [start for score, start in sorted(ranked, key=lambda item: item[0], reverse=True) if score >= float(window_activity_threshold)]
    elif activity_sorted:
        starts = sorted(
            starts,
            key=lambda start: _window_activity_score(
                _load_window(
                    segment.T,
                    layout="time_channel",
                    channel_start=0,
                    channel_count=int(segment.shape[0]),
                    start_t=int(start),
                    window_samples=int(window_samples),
                    background_scale=1.0,
                )
            ),
            reverse=True,
        )
    if int(max_windows) > 0:
        starts = starts[: int(max_windows)]

    fragments: list[Track] = []
    for start_t in starts:
        window = _load_window(
            segment.T,
            layout="time_channel",
            channel_start=0,
            channel_count=int(segment.shape[0]),
            start_t=int(start_t),
            window_samples=int(window_samples),
            background_scale=1.0,
        )
        activity_score = _window_activity_score(window)
        if activity_score < float(window_activity_threshold):
            continue
        with torch.no_grad():
            outputs = model(prepare_window_input(window, 10).unsqueeze(0).to(device))
        pred_dir = int(outputs["direction_logits"].argmax(dim=-1).item())
        pred_speed = float(outputs["speed"].item()) * float(getattr(getattr(model, "config", None), "speed_norm_kmh", 150.0))
        if not np.isfinite(pred_speed) or pred_speed <= 0:
            pred_speed = 80.0
        speed_margin = max(15.0, 0.2 * pred_speed)
        direction_order = _direction_list(str(direction))
        if len(direction_order) == 2:
            direction_order = ["forward" if pred_dir == 0 else "reverse", "reverse" if pred_dir == 0 else "forward"]
        for scan_direction in direction_order:
            tracks = predict_single_vehicle_track(
                model,
                window,
                float(fs),
                float(dx_m),
                str(scan_direction),
                max(1.0, pred_speed - speed_margin),
                pred_speed + speed_margin,
                InferenceConfig(
                    time_downsample=10,
                    min_visible_channels=4,
                    prior_weight=1.0,
                    single_vehicle_tracker=SingleVehicleTrackerConfig(
                        candidate_prominence=0.22,
                        candidate_min_distance=180,
                        candidate_max_peaks_per_channel=32,
                        max_skip_channels=8,
                        min_track_channels=8,
                        min_track_score=8.0,
                        kalman_bridge_gap_channels=12,
                        kalman_fill_missing=True,
                        kalman_gate_seconds=0.35,
                        kalman_speed_gate_kmh=30.0,
                    ),
                ),
                device=device,
            )
            local_graph_tracks = extract_multi_vehicle_tracks(
                window,
                float(fs),
                float(dx_m),
                str(scan_direction),
                float(vmin_kmh),
                float(vmax_kmh),
                config=MultiVehiclePipelineConfig(
                    candidate_limit=max(1, int(round(float(candidate_limit) * 0.25))),
                    candidate_min_score=float(candidate_min_score),
                    dedup_tolerance_samples=int(dedup_tolerance_samples),
                    dedup_min_overlap_channels=int(dedup_min_overlap_channels),
                    dedup_min_overlap_ratio=float(dedup_min_overlap_ratio),
                    crop_channel_margin=int(crop_channel_margin),
                    crop_time_margin_s=float(crop_time_margin_s),
                    refine_with_model=False,
                    proposal_model_path=proposal_model_path,
                    proposal_prior_weight=float(proposal_prior_weight),
                    proposal_time_downsample=int(proposal_time_downsample),
                ),
                model_path=None,
                device=device,
            )
            if tracks:
                fragments.append(_shift_track(tracks[0], 0, int(start_t), fs=float(fs), dx_m=float(dx_m)))
            for tr in local_graph_tracks:
                fragments.append(_shift_track(tr, 0, int(start_t), fs=float(fs), dx_m=float(dx_m)))

    if not fragments:
        return []

    stitched = _stitch_track_fragments(
        fragments,
        direction=str(direction),
        fs=float(fs),
        dx_m=float(dx_m),
        speed_min_kmh=float(vmin_kmh),
        speed_max_kmh=float(vmax_kmh),
        tol_samples=int(dedup_tolerance_samples),
        max_gap_channels=max(8, int(round(float(window_seconds) / max(1e-6, float(window_stride_seconds))))),
    )

    return _deduplicate_tracks(
        stitched,
        tol_samples=int(dedup_tolerance_samples),
        min_overlap_channels=int(dedup_min_overlap_channels),
        min_overlap_ratio=float(dedup_min_overlap_ratio),
    )


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    arr = np.load(str(Path(args.input).expanduser()), mmap_mode="r")
    segment = _load_segment(arr, layout=str(args.array_layout), channel_start=int(args.channel_start), channel_count=int(args.channel_count))
    device = _resolve_device(args.device)
    model_path = str(Path(args.model).expanduser()) if args.model is not None else None
    window_tracks: list[Track] = []
    graph_tracks: list[Track] = []
    if str(args.candidate_mode) in {"windowed", "hybrid"}:
        if model_path is None:
            raise ValueError("--model is required for candidate-mode=windowed")
        window_tracks = _windowed_multi_vehicle_tracks(
            segment,
            fs=float(args.fs),
            dx_m=float(args.dx_m),
            direction=str(args.direction),
            vmin_kmh=float(args.vmin_kmh),
            vmax_kmh=float(args.vmax_kmh),
            model_path=model_path,
            proposal_model_path=str(Path(args.proposal_model).expanduser()) if args.proposal_model is not None else None,
            proposal_prior_weight=float(args.proposal_prior_weight),
            proposal_time_downsample=int(args.proposal_time_downsample),
            device=device,
            window_seconds=float(args.window_seconds),
            window_stride_seconds=float(args.window_stride_seconds),
            max_windows=int(args.max_windows),
            window_activity_threshold=float(args.window_activity_threshold),
            activity_sorted=bool(args.activity_sorted),
            candidate_limit=int(args.candidate_limit),
            candidate_min_score=float(args.candidate_min_score),
            dedup_tolerance_samples=int(args.dedup_tolerance_samples),
            dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
            dedup_min_overlap_ratio=float(args.dedup_min_overlap_ratio),
            crop_channel_margin=int(args.crop_channel_margin),
            crop_time_margin_s=float(args.crop_time_margin_seconds),
            min_model_confidence=float(args.min_model_confidence),
            model_confidence_weight=float(args.model_confidence_weight),
            graph_confidence_weight=float(args.graph_confidence_weight),
        )
    if str(args.candidate_mode) in {"graph", "hybrid"}:
        pipeline_cfg = MultiVehiclePipelineConfig(
            candidate_limit=int(args.candidate_limit),
            candidate_min_score=float(args.candidate_min_score),
            dedup_tolerance_samples=int(args.dedup_tolerance_samples),
            dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
            dedup_min_overlap_ratio=float(args.dedup_min_overlap_ratio),
            crop_channel_margin=int(args.crop_channel_margin),
            crop_time_margin_s=float(args.crop_time_margin_seconds),
            refine_with_model=not bool(args.no_refine),
            min_model_confidence=float(args.min_model_confidence),
            model_confidence_weight=float(args.model_confidence_weight),
            graph_confidence_weight=float(args.graph_confidence_weight),
            proposal_model_path=str(Path(args.proposal_model).expanduser()) if args.proposal_model is not None else None,
            proposal_prior_weight=float(args.proposal_prior_weight),
            proposal_time_downsample=int(args.proposal_time_downsample),
        )
        graph_tracks = []
        for scan_direction in _direction_list(str(args.direction)):
            graph_tracks.extend(
                extract_multi_vehicle_tracks(
                    segment,
                    float(args.fs),
                    float(args.dx_m),
                    str(scan_direction),
                    float(args.vmin_kmh),
                    float(args.vmax_kmh),
                    config=pipeline_cfg,
                    model_path=model_path,
                    device=device,
                )
            )
    if str(args.candidate_mode) == "windowed":
        tracks = window_tracks
    elif str(args.candidate_mode) == "graph":
        tracks = graph_tracks
    else:
        tracks = _deduplicate_tracks(
            list(window_tracks) + list(graph_tracks),
            tol_samples=int(args.dedup_tolerance_samples),
            min_overlap_channels=int(args.dedup_min_overlap_channels),
            min_overlap_ratio=float(args.dedup_min_overlap_ratio),
        )

    rows: list[dict[str, Any]] = []
    for tr in tracks:
        for point in tr.points:
            rows.append(
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

    (out_dir / "tracks.csv").parent.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tracks.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["track_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(11, 5.5), constrained_layout=True, dpi=int(args.plot_dpi))
        ax.imshow(segment, aspect="auto", origin="lower", cmap="magma")
        colors = ["cyan", "lime", "yellow", "orange", "red", "deepskyblue", "white"]
        for idx, tr in enumerate(tracks):
            pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
            ax.plot(
                [float(p.t_idx) for p in pts],
                [int(p.ch_idx) for p in pts],
                linewidth=2.0,
                color=colors[idx % len(colors)],
                label=f"track {idx} score={tr.total_score:.1f}",
            )
        ax.set_title(f"multi-vehicle extraction | tracks={len(tracks)}")
        ax.set_xlabel("time [sample idx]")
        ax.set_ylabel("channel")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.75)
        fig.savefig(str(out_dir / "tracks_overlay.png"))
        plt.close(fig)

    summary = {
        "input": str(Path(args.input).expanduser()),
        "model": model_path,
        "device": device,
        "tracks": int(len(tracks)),
        "points": int(len(rows)),
        "segment_shape": list(map(int, segment.shape)),
        "tracks_csv": str(out_dir / "tracks.csv"),
        "plot": str(out_dir / "tracks_overlay.png") if args.plot else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
