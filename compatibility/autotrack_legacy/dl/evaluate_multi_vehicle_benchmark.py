"""Evaluate the multi-vehicle windowed pipeline on a fixed benchmark file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.multi_vehicle_pipeline import MultiVehiclePipelineConfig, extract_multi_vehicle_tracks
from autotrack.dl.predict_multi_vehicle_real_npy import _deduplicate_tracks, _direction_list, _windowed_multi_vehicle_tracks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the multi-vehicle pipeline on a benchmark .pt file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="multi_vehicle_benchmark_v1 .pt file")
    parser.add_argument("--model", required=True, type=Path, help="Single-vehicle checkpoint used for window refinement")
    parser.add_argument("--proposal-model", type=Path, default=None, help="Optional dense proposal checkpoint for full-segment candidate extraction.")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum benchmark samples to evaluate; 0 means all")
    parser.add_argument("--window-seconds", type=float, default=10.0, help="Window length for candidate scanning")
    parser.add_argument("--window-stride-seconds", type=float, default=5.0, help="Stride for candidate scanning")
    parser.add_argument("--window-activity-threshold", type=float, default=0.0, help="Skip windows whose activity falls below this threshold")
    parser.add_argument("--candidate-mode", default="hybrid", choices=["windowed", "graph", "hybrid"], help="Candidate extraction strategy")
    parser.add_argument("--candidate-limit", type=int, default=48, help="Maximum candidate tracks to refine")
    parser.add_argument("--candidate-min-score", type=float, default=8.0, help="Minimum candidate score before refinement")
    parser.add_argument("--dedup-tolerance-samples", type=int, default=180, help="Track deduplication tolerance in samples")
    parser.add_argument("--dedup-min-overlap-channels", type=int, default=2, help="Minimum shared channels for deduplication")
    parser.add_argument("--dedup-min-overlap-ratio", type=float, default=0.45, help="Minimum overlap ratio for deduplication")
    parser.add_argument("--crop-channel-margin", type=int, default=4, help="Channels to add around each candidate crop")
    parser.add_argument("--crop-time-margin-seconds", type=float, default=4.0, help="Time margin around each candidate crop")
    parser.add_argument("--min-model-confidence", type=float, default=0.18, help="Minimum single-vehicle confidence before refinement")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary JSON")
    return parser.parse_args(argv)


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
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _build_track(
    track_id: int,
    time: torch.Tensor,
    visibility: torch.Tensor,
    direction: int,
    speed: float,
    *,
    time_scale: int,
) -> Track:
    points: list[TrackPoint] = []
    for ch in torch.where(visibility > 0.5)[0].tolist():
        t_idx = int(round(float(time[int(ch)].item()) * float(max(1, int(time_scale) - 1))))
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx),
                offset_m=float(ch),
                amp=1.0,
                score=1.0,
            )
        )
    return Track(
        track_id=int(track_id),
        direction="forward" if int(direction) == 0 else "reverse",
        points=points,
        total_score=float(len(points)),
        mean_speed_kmh=float(speed),
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


def _match_tracks(
    pred_tracks: list[Track],
    gt_tracks: list[Track],
    *,
    tol_samples: int,
    min_overlap_channels: int,
    min_overlap_ratio: float,
) -> tuple[int, int, int]:
    scored: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_tracks):
        for g_idx, gt in enumerate(gt_tracks):
            ratio, common = _track_overlap(pred, gt, tol_samples=int(tol_samples), min_overlap_channels=int(min_overlap_channels))
            if common >= int(min_overlap_channels) and ratio >= float(min_overlap_ratio):
                scored.append((ratio, p_idx, g_idx))
    scored.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    for _, p_idx, g_idx in scored:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        used_pred.add(int(p_idx))
        used_gt.add(int(g_idx))
        tp += 1
    return int(tp), int(len(pred_tracks) - tp), int(len(gt_tracks) - tp)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    samples = list(payload.get("samples", []))
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise ValueError("benchmark contains no samples")

    total_tp = total_fp = total_fn = 0
    total_count_err = 0.0
    per_sample: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        target = sample["target"]
        raw_window = target.get("raw_window")
        if raw_window is None:
            raw_window = sample["x"][0]
        segment = raw_window.to(torch.float32).cpu().numpy()
        if segment.ndim != 2:
            raise ValueError(f"sample {idx} raw_window must be 2-D, got {segment.shape}")
        gt_time = target["time"].to(torch.float32)
        gt_visibility = target["visibility"].to(torch.float32)
        gt_direction = target["direction"].to(torch.long)
        gt_speed = target["speed"].to(torch.float32)
        gt_valid = target.get("gt_valid", torch.ones((gt_time.shape[0],), dtype=torch.bool))
        gt_tracks = [
            _build_track(
                int(g_idx),
                gt_time[g_idx],
                gt_visibility[g_idx],
                int(gt_direction[g_idx].item()),
                float(gt_speed[g_idx].item()),
                time_scale=int(segment.shape[1]),
            )
            for g_idx in torch.where(gt_valid)[0].tolist()
        ]
        pred_tracks: list[Track] = []
        if str(args.candidate_mode) in {"windowed", "hybrid"}:
            for scan_direction in _direction_list("both"):
                pred_tracks.extend(
                    _windowed_multi_vehicle_tracks(
                        segment,
                        fs=float(payload.get("meta", {}).get("fs", 1000.0)),
                        dx_m=float(payload.get("meta", {}).get("dx_m", 100.0)),
                        direction=scan_direction,
                        vmin_kmh=70.0,
                        vmax_kmh=90.0,
                        model_path=str(Path(args.model).expanduser()),
                        proposal_model_path=str(Path(args.proposal_model).expanduser()) if args.proposal_model is not None else None,
                        proposal_prior_weight=1.25,
                        proposal_time_downsample=10,
                        device=device,
                        window_seconds=float(args.window_seconds),
                        window_stride_seconds=float(args.window_stride_seconds),
                        max_windows=0,
                        window_activity_threshold=float(args.window_activity_threshold),
                        activity_sorted=True,
                        candidate_limit=int(args.candidate_limit),
                        candidate_min_score=float(args.candidate_min_score),
                        dedup_tolerance_samples=int(args.dedup_tolerance_samples),
                        dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
                        dedup_min_overlap_ratio=float(args.dedup_min_overlap_ratio),
                        crop_channel_margin=int(args.crop_channel_margin),
                        crop_time_margin_s=float(args.crop_time_margin_seconds),
                        min_model_confidence=float(args.min_model_confidence),
                        model_confidence_weight=0.7,
                        graph_confidence_weight=0.3,
                    )
                )
        if str(args.candidate_mode) in {"graph", "hybrid"}:
            for scan_direction in _direction_list("both"):
                graph_tracks = extract_multi_vehicle_tracks(
                    segment,
                    float(payload.get("meta", {}).get("fs", 1000.0)),
                    float(payload.get("meta", {}).get("dx_m", 100.0)),
                    scan_direction,
                    70.0,
                    90.0,
                    config=MultiVehiclePipelineConfig(
                        candidate_limit=int(args.candidate_limit),
                        candidate_min_score=float(args.candidate_min_score),
                        dedup_tolerance_samples=int(args.dedup_tolerance_samples),
                        dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
                        dedup_min_overlap_ratio=float(args.dedup_min_overlap_ratio),
                        crop_channel_margin=int(args.crop_channel_margin),
                        crop_time_margin_s=float(args.crop_time_margin_seconds),
                        refine_with_model=True,
                        proposal_model_path=str(Path(args.proposal_model).expanduser()) if args.proposal_model is not None else None,
                        proposal_prior_weight=1.25,
                        proposal_time_downsample=10,
                    ),
                    model_path=str(Path(args.model).expanduser()),
                    device=device,
                )
                pred_tracks.extend(graph_tracks)
        pred_tracks = _deduplicate_tracks(
            pred_tracks,
            tol_samples=int(args.dedup_tolerance_samples),
            min_overlap_channels=int(args.dedup_min_overlap_channels),
            min_overlap_ratio=float(args.dedup_min_overlap_ratio),
        )
        tp, fp, fn = _match_tracks(
            pred_tracks,
            gt_tracks,
            tol_samples=int(args.dedup_tolerance_samples),
            min_overlap_channels=int(args.dedup_min_overlap_channels),
            min_overlap_ratio=float(args.dedup_min_overlap_ratio),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_count_err += abs(len(pred_tracks) - len(gt_tracks))
        per_sample.append(
            {
                "sample_index": int(idx),
                "gt_count": int(len(gt_tracks)),
                "pred_count": int(len(pred_tracks)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2.0 * precision * recall / max(1e-9, precision + recall)
    summary = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "device": device,
        "samples": int(len(samples)),
        "track_precision": float(precision),
        "track_recall": float(recall),
        "track_f1": float(f1),
        "count_mae": float(total_count_err / max(1, len(samples))),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "per_sample": per_sample,
    }
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_ready(summary), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
