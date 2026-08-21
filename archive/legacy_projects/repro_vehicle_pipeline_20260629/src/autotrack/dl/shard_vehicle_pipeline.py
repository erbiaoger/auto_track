from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.multi_vehicle_pipeline import (
    MultiVehiclePipelineConfig,
    _deduplicate_tracks,
    _merge_track_fragments,
    _prune_short_fragments,
    _shift_track,
    extract_multi_vehicle_tracks_with_models,
)
from autotrack.dl.vehicle_proposal_net import load_checkpoint_model as load_proposal_checkpoint_model
from autotrack.dl.vehicle_trace_net import load_checkpoint_model as load_trace_checkpoint_model


@dataclass
class ShardBundle:
    shard_path: Path
    meta_path: Path
    payload: dict[str, Any]
    meta: dict[str, Any]

    @property
    def sample_count(self) -> int:
        samples = self.payload.get("samples")
        if isinstance(samples, list):
            return len(samples)
        x = self.payload.get("x")
        if isinstance(x, torch.Tensor):
            return int(x.shape[0])
        if isinstance(x, np.ndarray):
            return int(x.shape[0])
        return int(self.meta.get("num_samples", 0))

    def sample_at(self, index: int) -> dict[str, Any]:
        samples = self.payload.get("samples")
        if isinstance(samples, list):
            return samples[int(index)]
        x = self.payload.get("x")
        if isinstance(x, torch.Tensor):
            return {"x": x[int(index)]}
        if isinstance(x, np.ndarray):
            return {"x": x[int(index)]}
        raise KeyError("shard payload does not contain samples or x")


@dataclass
class ShardRunConfig:
    model_path: Optional[str | Path] = None
    proposal_model_path: Optional[str | Path] = None
    device: str = "auto"
    direction: str = "auto"
    vmin_kmh: float = 40.0
    vmax_kmh: float = 140.0
    max_samples: int = 0
    candidate_limit: int = 48
    candidate_min_score: float = 8.0
    dedup_tolerance_samples: int = 180
    dedup_min_overlap_channels: int = 2
    dedup_min_overlap_ratio: float = 0.45
    crop_channel_margin: int = 4
    crop_time_margin_s: float = 4.0
    refine_with_model: bool = True
    min_model_confidence: float = 0.18
    model_confidence_weight: float = 0.7
    graph_confidence_weight: float = 0.3
    proposal_prior_weight: float = 1.25
    proposal_time_downsample: int = 10
    iterative_extraction: bool = True
    max_iterations: int = 12


@dataclass
class ShardExtractionResult:
    tracks: list[Track]
    window_track_counts: list[int] = field(default_factory=list)
    samples_processed: int = 0
    shifted_track_count: int = 0
    input_scale: int = 1
    input_fs: float = 0.0


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def load_shard_bundle(shard_path: str | Path, meta_path: str | Path | None = None) -> ShardBundle:
    resolved_shard = Path(shard_path).expanduser()
    resolved_meta = Path(meta_path).expanduser() if meta_path is not None else resolved_shard.with_name("meta.json")
    payload = torch.load(str(resolved_shard), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"shard payload must be a dict, got {type(payload).__name__}")
    if resolved_meta.exists():
        meta = json.loads(resolved_meta.read_text(encoding="utf-8"))
    else:
        meta = dict(payload.get("meta", {}))
    if not isinstance(meta, dict):
        raise TypeError("shard meta must be a dict")

    samples = payload.get("samples")
    if samples is not None and not isinstance(samples, list):
        raise TypeError("shard payload samples must be a list when present")
    x = payload.get("x")
    if x is not None and not isinstance(x, (torch.Tensor, np.ndarray)):
        raise TypeError("shard payload x must be a tensor or ndarray when present")
    count = len(samples) if isinstance(samples, list) else int(x.shape[0]) if x is not None else 0
    expected = int(meta.get("num_samples", count))
    if count and expected != count:
        raise ValueError(f"sample count mismatch: meta={expected}, payload={count}")
    return ShardBundle(shard_path=resolved_shard, meta_path=resolved_meta, payload=payload, meta=meta)


def _sample_window_and_scale(sample: dict[str, Any], *, time_downsample: int) -> tuple[np.ndarray, int]:
    target = sample.get("target", {})
    if isinstance(target, dict) and "raw_window" in target:
        raw = target["raw_window"]
        scale = 1
    elif "x" in sample:
        raw = sample["x"]
        scale = int(max(1, time_downsample))
    else:
        raise KeyError("sample does not contain target.raw_window or x")

    if isinstance(raw, torch.Tensor):
        arr = raw.detach().cpu().numpy()
    else:
        arr = np.asarray(raw)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return np.asarray(arr[0], dtype=np.float32), int(scale)
    if arr.ndim == 2:
        return arr, int(scale)
    raise ValueError(f"expected raw window with 2 or 3 dims, got {arr.shape}")


def _window_start_samples(meta: dict[str, Any], count: int) -> list[int]:
    starts = meta.get("window_start_samples")
    if isinstance(starts, list) and len(starts) >= count:
        return [int(v) for v in starts[:count]]
    stride = int(meta.get("stride_samples", meta.get("window_samples", 0)))
    if stride <= 0:
        stride = 1
    return [int(idx) * stride for idx in range(count)]


def build_pipeline_config(run_cfg: ShardRunConfig) -> MultiVehiclePipelineConfig:
    return MultiVehiclePipelineConfig(
        candidate_limit=int(run_cfg.candidate_limit),
        candidate_min_score=float(run_cfg.candidate_min_score),
        dedup_tolerance_samples=int(run_cfg.dedup_tolerance_samples),
        dedup_min_overlap_channels=int(run_cfg.dedup_min_overlap_channels),
        dedup_min_overlap_ratio=float(run_cfg.dedup_min_overlap_ratio),
        crop_channel_margin=int(run_cfg.crop_channel_margin),
        crop_time_margin_s=float(run_cfg.crop_time_margin_s),
        refine_with_model=bool(run_cfg.refine_with_model),
        min_model_confidence=float(run_cfg.min_model_confidence),
        model_confidence_weight=float(run_cfg.model_confidence_weight),
        graph_confidence_weight=float(run_cfg.graph_confidence_weight),
        proposal_model_path=run_cfg.proposal_model_path,
        proposal_prior_weight=float(run_cfg.proposal_prior_weight),
        proposal_time_downsample=int(run_cfg.proposal_time_downsample),
        iterative_extraction=bool(run_cfg.iterative_extraction),
        max_iterations=int(run_cfg.max_iterations),
    )


def shift_tracks_by_samples(tracks: list[Track], sample_offset: int, *, fs: float, dx_m: float) -> list[Track]:
    return [_shift_track(track, 0, int(sample_offset), fs=float(fs), dx_m=float(dx_m)) for track in tracks]


def merge_global_tracks(
    tracks: list[Track],
    *,
    fs: float,
    dx_m: float,
    dedup_tolerance_samples: int,
    dedup_min_overlap_channels: int,
    dedup_min_overlap_ratio: float,
    crop_time_margin_s: float,
    n_samples: int | None = None,
) -> list[Track]:
    merged = _deduplicate_tracks(
        tracks,
        tol_samples=int(dedup_tolerance_samples),
        min_overlap_channels=int(dedup_min_overlap_channels),
        min_overlap_ratio=float(dedup_min_overlap_ratio),
    )
    merged = _merge_track_fragments(
        merged,
        fs=float(fs),
        dx_m=float(dx_m),
        max_gap_channels=max(6, int(dedup_tolerance_samples // 30)),
        max_gap_seconds=max(0.45, float(crop_time_margin_s) * 0.25),
        line_distance_threshold=max(0.30, float(crop_time_margin_s) * 0.08),
        speed_diff_kmh=12.0,
    )
    merged = _prune_short_fragments(
        merged,
        min_keep_points=6,
        edge_margin_points=max(60, int(fs * 0.08)),
        n_samples=int(n_samples) if n_samples is not None else None,
    )
    return merged


def extract_tracks_from_shard_bundle(
    bundle: ShardBundle,
    run_cfg: Optional[ShardRunConfig] = None,
) -> ShardExtractionResult:
    cfg = run_cfg or ShardRunConfig()
    device = _resolve_device(cfg.device)
    meta = dict(bundle.meta)
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    time_downsample = int(meta.get("time_downsample", 1))
    source_shape = meta.get("source_shape_time_channel", [])
    first_scale = 1
    if bundle.sample_count > 0:
        _, first_scale = _sample_window_and_scale(bundle.sample_at(0), time_downsample=time_downsample)
    input_scale = int(max(1, first_scale))
    sample_fs = float(fs) / float(input_scale)
    total_n_samples = int(round(float(source_shape[0]) / float(input_scale))) if isinstance(source_shape, list) and source_shape else None
    window_starts = [int(round(float(v) / float(input_scale))) for v in _window_start_samples(meta, bundle.sample_count)]

    trace_model = None
    if cfg.model_path is not None:
        trace_model, _ = load_trace_checkpoint_model(str(Path(cfg.model_path).expanduser()), device=device)
    proposal_model = None
    if cfg.proposal_model_path is not None:
        proposal_model, _ = load_proposal_checkpoint_model(str(Path(cfg.proposal_model_path).expanduser()), device=device)

    pipeline_cfg = build_pipeline_config(cfg)
    shifted_tracks: list[Track] = []
    window_track_counts: list[int] = []

    max_samples = int(cfg.max_samples)
    sample_limit = bundle.sample_count if max_samples <= 0 else min(bundle.sample_count, max_samples)
    for sample_idx in range(sample_limit):
        sample = bundle.sample_at(sample_idx)
        raw_window, sample_scale = _sample_window_and_scale(sample, time_downsample=time_downsample)
        local_tracks = extract_multi_vehicle_tracks_with_models(
            raw_window,
            fs=float(sample_fs) if sample_scale > 1 else float(fs),
            dx_m=float(dx_m),
            direction=str(cfg.direction),
            vmin_kmh=float(cfg.vmin_kmh),
            vmax_kmh=float(cfg.vmax_kmh),
            config=pipeline_cfg,
            model=trace_model,
            proposal_model=proposal_model,
            device=device,
        )
        window_track_counts.append(len(local_tracks))
        shifted_tracks.extend(
            shift_tracks_by_samples(
                local_tracks,
                int(window_starts[sample_idx]),
                fs=float(fs) / float(max(1, sample_scale)),
                dx_m=float(dx_m),
            )
        )

    merged = merge_global_tracks(
        shifted_tracks,
        fs=float(sample_fs),
        dx_m=float(dx_m),
        dedup_tolerance_samples=int(cfg.dedup_tolerance_samples),
        dedup_min_overlap_channels=int(cfg.dedup_min_overlap_channels),
        dedup_min_overlap_ratio=float(cfg.dedup_min_overlap_ratio),
        crop_time_margin_s=float(cfg.crop_time_margin_s),
        n_samples=total_n_samples,
    )
    return ShardExtractionResult(
        tracks=merged,
        window_track_counts=window_track_counts,
        samples_processed=sample_limit,
        shifted_track_count=len(shifted_tracks),
        input_scale=int(input_scale),
        input_fs=float(sample_fs),
    )


def tracks_to_rows(tracks: list[Track]) -> list[dict[str, Any]]:
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
    return rows


def write_tracks_csv(path: str | Path, tracks: list[Track]) -> Path:
    resolved = Path(path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rows = tracks_to_rows(tracks)
    fieldnames = ["track_id", "direction", "ch_idx", "t_idx", "time_s", "offset_m", "amp", "score", "track_score", "mean_speed_kmh"]
    with resolved.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def build_summary(bundle: ShardBundle, result: ShardExtractionResult, run_cfg: Optional[ShardRunConfig] = None) -> dict[str, Any]:
    cfg = run_cfg or ShardRunConfig()
    meta = dict(bundle.meta)
    lengths = [len(tr.points) for tr in result.tracks]
    total_points = int(sum(lengths))
    return {
        "shard_path": str(bundle.shard_path),
        "meta_path": str(bundle.meta_path),
        "direction": str(cfg.direction),
        "device": _resolve_device(cfg.device),
        "samples_available": int(bundle.sample_count),
        "samples_processed": int(result.samples_processed),
        "window_track_counts": list(result.window_track_counts),
        "shifted_track_count": int(result.shifted_track_count),
        "merged_track_count": int(len(result.tracks)),
        "total_points": total_points,
        "track_length_min": int(min(lengths)) if lengths else 0,
        "track_length_median": float(np.median(lengths)) if lengths else 0.0,
        "track_length_max": int(max(lengths)) if lengths else 0,
        "fs": float(meta.get("fs", 1000.0)),
        "input_fs": float(result.input_fs),
        "input_time_scale": int(result.input_scale),
        "dx_m": float(meta.get("dx_m", 100.0)),
        "window_seconds": float(meta.get("window_seconds", 0.0)),
        "proposal_model_path": str(cfg.proposal_model_path) if cfg.proposal_model_path is not None else None,
        "model_path": str(cfg.model_path) if cfg.model_path is not None else None,
    }
