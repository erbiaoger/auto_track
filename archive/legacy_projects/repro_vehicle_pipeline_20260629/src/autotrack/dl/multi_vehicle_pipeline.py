"""Multi-vehicle candidate extraction and refinement pipeline.

This sits above the single-vehicle decoder:

1. Extract coarse candidate tracks from the full segment.
2. Optionally crop around each candidate and refine it with the single-vehicle
   decoder / network.
3. Deduplicate the resulting tracks with trajectory-level overlap checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from autotrack.core.single_vehicle_tracker import SingleVehicleTrackerConfig, extract_single_vehicle_track
from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, extract_all
from autotrack.dl.vehicle_trace_net import (
    InferenceConfig,
    load_checkpoint_model,
    predict_single_vehicle_track,
    score_single_vehicle_window,
)
from autotrack.dl.vehicle_proposal_net import (
    load_checkpoint_model as load_proposal_checkpoint_model,
    prepare_window_input as prepare_proposal_window_input,
    score_vehicle_proposal_window,
)


@dataclass
class MultiVehiclePipelineConfig:
    candidate_graph: ExtractorConfig = field(default_factory=ExtractorConfig)
    candidate_limit: int = 64
    candidate_min_score: float = 4.0
    dedup_tolerance_samples: int = 180
    dedup_min_overlap_channels: int = 3
    dedup_min_overlap_ratio: float = 0.55
    crop_channel_margin: int = 4
    crop_time_margin_s: float = 4.0
    refine_with_model: bool = True
    min_model_confidence: float = 0.10
    model_confidence_weight: float = 0.7
    graph_confidence_weight: float = 0.3
    proposal_model_path: Optional[str | Path] = None
    proposal_prior_weight: float = 1.40
    proposal_time_downsample: int = 10
    iterative_extraction: bool = True
    max_iterations: int = 12
    residual_suppression_weight: float = 0.80
    residual_suppression_sigma_channels: float = 1.2
    residual_suppression_sigma_time: float = 3.5
    single_vehicle: SingleVehicleTrackerConfig = field(default_factory=SingleVehicleTrackerConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)


def _as_graph_config(config: Optional[ExtractorConfig | dict[str, Any]]) -> ExtractorConfig:
    if config is None:
        return ExtractorConfig()
    if isinstance(config, ExtractorConfig):
        return config
    if isinstance(config, dict):
        return ExtractorConfig(**config)
    raise TypeError("candidate_graph must be ExtractorConfig / dict / None")


def _as_pipeline_config(config: Optional[MultiVehiclePipelineConfig | dict[str, Any]]) -> MultiVehiclePipelineConfig:
    if config is None:
        return MultiVehiclePipelineConfig()
    if isinstance(config, MultiVehiclePipelineConfig):
        return config
    if isinstance(config, dict):
        defaults = MultiVehiclePipelineConfig()
        candidate_graph = config.get("candidate_graph", None)
        if isinstance(candidate_graph, ExtractorConfig):
            graph_cfg = candidate_graph
        elif isinstance(candidate_graph, dict):
            graph_cfg = ExtractorConfig(**candidate_graph)
        else:
            graph_cfg = defaults.candidate_graph
        single_vehicle = config.get("single_vehicle", None)
        if isinstance(single_vehicle, SingleVehicleTrackerConfig):
            single_cfg = single_vehicle
        elif isinstance(single_vehicle, dict):
            single_cfg = SingleVehicleTrackerConfig(**single_vehicle)
        else:
            single_cfg = defaults.single_vehicle
        inference = config.get("inference", None)
        if isinstance(inference, InferenceConfig):
            infer_cfg = inference
        elif isinstance(inference, dict):
            infer_cfg = InferenceConfig(**inference)
        else:
            infer_cfg = defaults.inference
        return MultiVehiclePipelineConfig(
            candidate_graph=graph_cfg,
            candidate_limit=int(config.get("candidate_limit", defaults.candidate_limit)),
            candidate_min_score=float(config.get("candidate_min_score", defaults.candidate_min_score)),
            dedup_tolerance_samples=int(config.get("dedup_tolerance_samples", defaults.dedup_tolerance_samples)),
            dedup_min_overlap_channels=int(config.get("dedup_min_overlap_channels", defaults.dedup_min_overlap_channels)),
            dedup_min_overlap_ratio=float(config.get("dedup_min_overlap_ratio", defaults.dedup_min_overlap_ratio)),
            crop_channel_margin=int(config.get("crop_channel_margin", defaults.crop_channel_margin)),
            crop_time_margin_s=float(config.get("crop_time_margin_s", defaults.crop_time_margin_s)),
            refine_with_model=bool(config.get("refine_with_model", defaults.refine_with_model)),
            min_model_confidence=float(config.get("min_model_confidence", defaults.min_model_confidence)),
            model_confidence_weight=float(config.get("model_confidence_weight", defaults.model_confidence_weight)),
            graph_confidence_weight=float(config.get("graph_confidence_weight", defaults.graph_confidence_weight)),
            proposal_model_path=config.get("proposal_model_path", defaults.proposal_model_path),
            proposal_prior_weight=float(config.get("proposal_prior_weight", defaults.proposal_prior_weight)),
            proposal_time_downsample=int(config.get("proposal_time_downsample", defaults.proposal_time_downsample)),
            iterative_extraction=bool(config.get("iterative_extraction", defaults.iterative_extraction)),
            max_iterations=int(config.get("max_iterations", defaults.max_iterations)),
            residual_suppression_weight=float(
                config.get("residual_suppression_weight", defaults.residual_suppression_weight)
            ),
            residual_suppression_sigma_channels=float(
                config.get("residual_suppression_sigma_channels", defaults.residual_suppression_sigma_channels)
            ),
            residual_suppression_sigma_time=float(config.get("residual_suppression_sigma_time", defaults.residual_suppression_sigma_time)),
            single_vehicle=single_cfg,
            inference=infer_cfg,
        )
    raise TypeError("config must be MultiVehiclePipelineConfig / dict / None")


def _track_channel_time_bounds(track: Track) -> tuple[int, int, int, int]:
    pts = sorted(track.points, key=lambda p: int(p.ch_idx))
    ch_min = int(min(int(p.ch_idx) for p in pts))
    ch_max = int(max(int(p.ch_idx) for p in pts))
    t_min = int(min(int(p.t_idx) for p in pts))
    t_max = int(max(int(p.t_idx) for p in pts))
    return ch_min, ch_max, t_min, t_max


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


def _track_channel_overlap_ratio(a: Track, b: Track) -> tuple[float, int]:
    aset = {int(p.ch_idx) for p in a.points}
    bset = {int(p.ch_idx) for p in b.points}
    common = len(aset & bset)
    denom = float(max(1, min(len(aset), len(bset))))
    return float(common / denom), int(common)


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
    if not tracks:
        return []
    ordered = sorted(tracks, key=lambda tr: tr.total_score, reverse=True)
    kept: list[Track] = []
    for tr in ordered:
        keep = True
        for existing in kept:
            ratio, common = _track_overlap(tr, existing, tol_samples=int(tol_samples), min_overlap_channels=int(min_overlap_channels))
            channel_ratio, channel_common = _track_channel_overlap_ratio(tr, existing)
            line_distance = _track_line_distance(tr, existing)
            speed_diff = abs(float(tr.mean_speed_kmh) - float(existing.mean_speed_kmh))
            if (common >= int(min_overlap_channels) and ratio >= float(min_overlap_ratio)) or (
                channel_common >= int(min_overlap_channels)
                and channel_ratio >= 0.55
                and line_distance < 10.0
                and speed_diff < 18.0
            ):
                keep = False
                break
        if keep:
            kept.append(tr)
    return sorted(kept, key=lambda tr: tr.total_score, reverse=True)


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: int(p.ch_idx))
    speeds: list[float] = []
    for left, right in zip(pts[:-1], pts[1:]):
        dch = abs(int(right.ch_idx) - int(left.ch_idx))
        dt = abs(float(right.time_s) - float(left.time_s))
        if dch > 0 and dt > 1e-9:
            speeds.append(3.6 * float(dch) * float(dx_m) / dt)
    return float(np.mean(speeds)) if speeds else float("nan")


def _merge_track_points(left: Track, right: Track, *, dx_m: float, fs: float) -> Track:
    grouped: dict[int, list[TrackPoint]] = {}
    for track in (left, right):
        for point in track.points:
            grouped.setdefault(int(point.ch_idx), []).append(point)
    merged_points: list[TrackPoint] = []
    for ch in sorted(grouped):
        pts = grouped[ch]
        if len(pts) == 1:
            point = pts[0]
            merged_points.append(
                TrackPoint(
                    ch_idx=int(point.ch_idx),
                    t_idx=int(point.t_idx),
                    time_s=float(point.time_s),
                    offset_m=float(point.offset_m),
                    amp=float(point.amp),
                    score=float(point.score),
                )
            )
            continue
        scores = np.asarray([max(1e-6, float(p.score)) for p in pts], dtype=np.float64)
        times = np.asarray([float(p.time_s) for p in pts], dtype=np.float64)
        merged_points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(round(float(np.average(times, weights=scores)) * float(fs))),
                time_s=float(np.average(times, weights=scores)),
                offset_m=float(ch) * float(dx_m),
                amp=float(max(float(p.amp) for p in pts)),
                score=float(max(float(p.score) for p in pts)),
            )
        )
    merged_points.sort(key=lambda p: int(p.ch_idx))
    return Track(
        track_id=int(left.track_id),
        direction=str(left.direction),
        points=merged_points,
        total_score=float(left.total_score + right.total_score),
        mean_speed_kmh=float(_mean_speed_kmh(merged_points, float(dx_m))),
    )


def _merge_track_fragments(
    tracks: list[Track],
    *,
    fs: float,
    dx_m: float,
    max_gap_channels: int,
    max_gap_seconds: float,
    line_distance_threshold: float,
    speed_diff_kmh: float,
) -> list[Track]:
    if not tracks:
        return []

    def _bounds(track: Track) -> tuple[int, int, float, float]:
        pts = sorted(track.points, key=lambda p: int(p.ch_idx))
        return int(pts[0].ch_idx), int(pts[-1].ch_idx), float(pts[0].time_s), float(pts[-1].time_s)

    merged = [track for track in tracks if track.points]
    for _ in range(4):
        ordered = sorted(merged, key=lambda tr: (min(int(p.ch_idx) for p in tr.points), -float(tr.total_score)))
        next_round: list[Track] = []
        changed = False
        for track in ordered:
            candidate = track
            merged_into_existing = False
            for idx in range(len(next_round) - 1, -1, -1):
                existing = next_round[idx]
                if existing.direction != candidate.direction:
                    continue
                if set(int(p.ch_idx) for p in existing.points) & set(int(p.ch_idx) for p in candidate.points):
                    continue
                ex_ch0, ex_ch1, ex_t0, ex_t1 = _bounds(existing)
                ca_ch0, ca_ch1, ca_t0, ca_t1 = _bounds(candidate)
                gap_channels = min(abs(ca_ch0 - ex_ch1), abs(ex_ch0 - ca_ch1))
                if gap_channels > int(max_gap_channels):
                    continue
                if str(existing.direction) == "forward":
                    signed_gap = ca_t0 - ex_t1 if ca_ch0 >= ex_ch1 else ex_t0 - ca_t1
                else:
                    signed_gap = ex_t1 - ca_t0 if ca_ch0 >= ex_ch1 else ca_t1 - ex_t0
                if signed_gap < 0.0 or signed_gap > float(max_gap_seconds):
                    continue
                if _track_line_distance(existing, candidate) > float(line_distance_threshold):
                    continue
                if abs(float(existing.mean_speed_kmh) - float(candidate.mean_speed_kmh)) > float(speed_diff_kmh):
                    continue
                next_round[idx] = _merge_track_points(existing, candidate, dx_m=float(dx_m), fs=float(fs))
                merged_into_existing = True
                changed = True
                break
            if not merged_into_existing:
                next_round.append(candidate)
        merged = next_round
        if not changed:
            break
    return sorted(merged, key=lambda tr: tr.total_score, reverse=True)


def _prune_short_fragments(
    tracks: list[Track],
    *,
    min_keep_points: int = 6,
    edge_margin_points: int = 80,
    n_samples: int | None = None,
) -> list[Track]:
    if not tracks:
        return []
    kept: list[Track] = []
    for track in tracks:
        if len(track.points) >= int(min_keep_points):
            kept.append(track)
            continue
        if n_samples is not None and track.points:
            t_min = min(int(p.t_idx) for p in track.points)
            t_max = max(int(p.t_idx) for p in track.points)
            if t_min <= int(edge_margin_points) or t_max >= int(n_samples) - int(edge_margin_points):
                kept.append(track)
                continue
        if float(track.total_score) / max(1.0, float(len(track.points))) >= 10.0:
            kept.append(track)
    return sorted(kept, key=lambda tr: tr.total_score, reverse=True)


def _crop_around_track(
    data: np.ndarray,
    track: Track,
    *,
    fs: float,
    channel_margin: int,
    time_margin_s: float,
) -> tuple[np.ndarray, int, int]:
    arr = np.asarray(data, dtype=np.float32)
    ch_min, ch_max, t_min, t_max = _track_channel_time_bounds(track)
    ch0 = max(0, ch_min - int(channel_margin))
    ch1 = min(arr.shape[0] - 1, ch_max + int(channel_margin))
    t_margin = int(max(1, round(float(time_margin_s) * float(fs))))
    t0 = max(0, t_min - t_margin)
    t1 = min(arr.shape[1] - 1, t_max + t_margin)
    crop = arr[ch0 : ch1 + 1, t0 : t1 + 1]
    return crop, ch0, t0


def _track_prior_heatmap(track: Track, n_channels: int, n_samples: int, *, sigma_ch: float = 1.0, sigma_t: float = 3.0) -> np.ndarray:
    prior = np.zeros((int(n_channels), int(n_samples)), dtype=np.float32)
    if not track.points:
        return prior
    ch_axis = np.arange(int(n_channels), dtype=np.float32)[:, None]
    t_axis = np.arange(int(n_samples), dtype=np.float32)[None, :]
    for p in track.points:
        gc = np.exp(-0.5 * ((ch_axis - float(p.ch_idx)) / float(max(1e-6, sigma_ch))) ** 2)
        gt = np.exp(-0.5 * ((t_axis - float(p.t_idx)) / float(max(1e-6, sigma_t))) ** 2)
        prior = np.maximum(prior, gc * gt * float(max(0.1, p.score)))
    return prior


def _suppress_track_region(
    residual: np.ndarray,
    track: Track,
    *,
    weight: float,
    sigma_ch: float,
    sigma_t: float,
) -> np.ndarray:
    arr = np.asarray(residual, dtype=np.float32)
    prior = _track_prior_heatmap(
        track,
        int(arr.shape[0]),
        int(arr.shape[1]),
        sigma_ch=float(max(1e-6, sigma_ch)),
        sigma_t=float(max(1e-6, sigma_t)),
    )
    if not np.isfinite(weight) or float(weight) <= 0.0:
        return arr
    scale = np.clip(1.0 - float(weight) * np.clip(prior, 0.0, 1.0), 0.0, 1.0)
    return (arr * scale).astype(np.float32, copy=False)


def _proposal_bboxes(
    score_map: np.ndarray,
    *,
    min_score: float,
    min_area: int,
    pad_channels: int,
    pad_time: int,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []
    threshold = float(max(min_score, np.quantile(finite, 0.92)))
    mask = arr >= threshold
    if not np.any(mask):
        return []
    labeled, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if count <= 0:
        return []
    boxes: list[tuple[float, tuple[int, int, int, int]]] = []
    slices = ndimage.find_objects(labeled)
    for label_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        ch_sl, t_sl = slc
        ch0 = max(0, int(ch_sl.start) - int(pad_channels))
        ch1 = min(arr.shape[0], int(ch_sl.stop) + int(pad_channels))
        t0 = max(0, int(t_sl.start) - int(pad_time))
        t1 = min(arr.shape[1], int(t_sl.stop) + int(pad_time))
        if ch1 - ch0 <= 0 or t1 - t0 <= 0:
            continue
        patch = arr[ch0:ch1, t0:t1]
        if int(np.count_nonzero(patch >= threshold)) < int(min_area):
            continue
        score = float(np.max(patch) + 0.1 * np.mean(patch))
        boxes.append((score, (ch0, ch1, t0, t1)))
    boxes.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[int, int, int, int]]] = []
    for score, box in boxes:
        if any(_bbox_iou(box, other_box) >= 0.25 for _, other_box in kept):
            continue
        kept.append((score, box))
    peak_boxes = _proposal_peak_boxes(
        arr,
        min_score=threshold,
        pad_channels=pad_channels,
        pad_time=pad_time,
        max_peaks=max(8, len(kept) * 2),
    )
    for score, box in peak_boxes:
        if any(_bbox_iou(box, other_box) >= 0.15 for _, other_box in kept):
            continue
        kept.append((score, box))
    kept.sort(key=lambda item: item[0], reverse=True)
    return kept


def _proposal_peak_boxes(
    score_map: np.ndarray,
    *,
    min_score: float,
    pad_channels: int,
    pad_time: int,
    max_peaks: int,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    arr = np.asarray(score_map, dtype=np.float32)
    if arr.ndim != 2 or arr.size == 0:
        return []
    filt = ndimage.maximum_filter(arr, size=(3, 21), mode="nearest")
    peaks = np.argwhere((arr >= float(min_score)) & (arr >= filt))
    if peaks.size == 0:
        # Recall-first fallback: keep the strongest local maxima even if the
        # absolute score is low. This is intentionally permissive because the
        # proposal model is only a coarse front-end.
        peaks = np.argwhere(arr >= filt)
        if peaks.size == 0:
            flat = arr.ravel()
            if flat.size == 0:
                return []
            topk = min(int(max_peaks), int(flat.size))
            idxs = np.argpartition(flat, -topk)[-topk:]
            idxs = idxs[np.argsort(flat[idxs])[::-1]]
            peaks = np.asarray([np.unravel_index(int(idx), arr.shape) for idx in idxs], dtype=np.int64)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for ch, t in peaks.tolist():
        score = float(arr[int(ch), int(t)])
        ch0 = max(0, int(ch) - int(pad_channels))
        ch1 = min(arr.shape[0], int(ch) + int(pad_channels) + 1)
        t0 = max(0, int(t) - int(pad_time))
        t1 = min(arr.shape[1], int(t) + int(pad_time) + 1)
        scored.append((score, (ch0, ch1, t0, t1)))
    scored.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, tuple[int, int, int, int]]] = []
    for score, box in scored:
        if any(_bbox_iou(box, other_box) >= 0.2 for _, other_box in kept):
            continue
        kept.append((score, box))
        if len(kept) >= int(max_peaks):
            break
    return kept


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ach0, ach1, at0, at1 = a
    bch0, bch1, bt0, bt1 = b
    ch0 = max(ach0, bch0)
    ch1 = min(ach1, bch1)
    t0 = max(at0, bt0)
    t1 = min(at1, bt1)
    inter_ch = max(0, ch1 - ch0)
    inter_t = max(0, t1 - t0)
    inter = float(inter_ch * inter_t)
    if inter <= 0.0:
        return 0.0
    area_a = float(max(1, ach1 - ach0) * max(1, at1 - at0))
    area_b = float(max(1, bch1 - bch0) * max(1, bt1 - bt0))
    return inter / max(1e-6, area_a + area_b - inter)


def _proposal_prior_heatmap(
    model: torch.nn.Module,
    data: np.ndarray,
    *,
    time_downsample: int,
    device: Optional[str],
) -> np.ndarray:
    x = prepare_proposal_window_input(np.asarray(data, dtype=np.float32), int(time_downsample)).unsqueeze(0)
    resolved_device = device or next(model.parameters()).device
    with torch.no_grad():
        outputs = model(x.to(resolved_device))
        prior_ds = torch.sigmoid(outputs["heatmap_logits"][0]).detach().cpu().unsqueeze(0).unsqueeze(0)
        prior = F.interpolate(prior_ds, size=data.shape, mode="bilinear", align_corners=False)[0, 0]
    return prior.numpy().astype(np.float32, copy=False)


def _track_candidate_confidence(
    model: torch.nn.Module,
    crop: np.ndarray,
    *,
    speed_norm_kmh: float,
    device: Optional[str],
) -> float:
    x = np.asarray(crop, dtype=np.float32)
    x_t = torch.from_numpy(x[None, None, :, :]).to(device=device or next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(x_t)
    score = score_single_vehicle_window(outputs, speed_norm_kmh=float(speed_norm_kmh))
    return float(score["confidence"])


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


def extract_multi_vehicle_tracks_with_models(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[MultiVehiclePipelineConfig | dict[str, Any]] = None,
    *,
    model_path: Optional[str | Path] = None,
    model: Optional[torch.nn.Module] = None,
    proposal_model: Optional[torch.nn.Module] = None,
    device: Optional[str] = None,
) -> list[Track]:
    cfg = _as_pipeline_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must have shape [n_channel, n_sample]")
    local_model = model
    local_proposal_model = proposal_model
    if cfg.refine_with_model and model_path is not None and local_model is None:
        local_model, _ = load_checkpoint_model(model_path, device=device)
    if cfg.proposal_model_path is not None and local_proposal_model is None:
        local_proposal_model, _ = load_proposal_checkpoint_model(str(cfg.proposal_model_path), device=device)
    direction_norm = str(direction).strip().lower()
    if direction_norm in {"auto", "both", "dual"}:
        tracks: list[Track] = []
        for sub_direction in ("forward", "reverse"):
            tracks.extend(
                extract_multi_vehicle_tracks_with_models(
                    arr,
                    float(fs),
                    float(dx_m),
                    sub_direction,
                    float(vmin_kmh),
                    float(vmax_kmh),
                    config=cfg,
                    model_path=model_path,
                    model=local_model,
                    proposal_model=local_proposal_model,
                    device=device,
                )
            )
        return _deduplicate_tracks(
            tracks,
            tol_samples=int(cfg.dedup_tolerance_samples),
            min_overlap_channels=int(cfg.dedup_min_overlap_channels),
            min_overlap_ratio=float(cfg.dedup_min_overlap_ratio),
        )
    proposal_prior = None
    if cfg.proposal_model_path is not None:
        proposal_prior = _proposal_prior_heatmap(
            local_proposal_model,
            arr,
            time_downsample=int(cfg.proposal_time_downsample),
            device=device,
        )
        arr = np.abs(arr) + float(cfg.proposal_prior_weight) * np.maximum(proposal_prior, 0.0)

    residual = np.asarray(arr, dtype=np.float32)
    all_candidates: list[Track] = []
    iteration_limit = int(max(1, cfg.max_iterations if bool(cfg.iterative_extraction) else 1))
    for _ in range(iteration_limit):
        candidate_boxes: list[tuple[float, tuple[int, int, int, int]]] = []
        if proposal_prior is not None:
            candidate_boxes = _proposal_bboxes(
                residual,
                min_score=float(cfg.candidate_min_score) / 10.0,
                min_area=max(8, int(cfg.candidate_limit // 2)),
                pad_channels=max(1, int(cfg.crop_channel_margin)),
                pad_time=max(8, int(round(cfg.crop_time_margin_s * 10.0))),
            )
            if len(candidate_boxes) < int(cfg.candidate_limit):
                extra_boxes = _proposal_bboxes(
                    residual,
                    min_score=max(0.015, float(cfg.candidate_min_score) / 20.0),
                    min_area=max(4, int(cfg.candidate_limit // 3)),
                    pad_channels=max(1, int(cfg.crop_channel_margin)),
                    pad_time=max(8, int(round(cfg.crop_time_margin_s * 8.0))),
                )
                for extra in extra_boxes:
                    if all(_bbox_iou(extra[1], box) < 0.15 for _, box in candidate_boxes):
                        candidate_boxes.append(extra)
                candidate_boxes.sort(key=lambda item: item[0], reverse=True)
        iteration_candidates: list[Track] = []
        coarse_candidates: list[Track] = []
        if proposal_prior is not None:
            coarse = residual[:, ::4]
            coarse_candidates = extract_all(
                coarse,
                float(fs) / 4.0,
                float(dx_m),
                str(direction),
                float(vmin_kmh),
                float(vmax_kmh),
                config=cfg.candidate_graph,
            )
            coarse_candidates = [tr for tr in coarse_candidates if float(tr.total_score) >= float(cfg.candidate_min_score)]
            coarse_candidates = sorted(coarse_candidates, key=lambda tr: tr.total_score, reverse=True)[: max(2, int(cfg.candidate_limit // 2))]
        if candidate_boxes:
            max_local = int(max(1, min(12, cfg.candidate_limit)))
            for _, (ch0, ch1, t0, t1) in candidate_boxes[:max_local]:
                candidate_crop = residual[ch0:ch1, t0:t1]
                if np.count_nonzero(candidate_crop) <= 0:
                    continue
                box_mask = np.zeros_like(residual, dtype=np.float32)
                box_mask[ch0:ch1, t0:t1] = 1.0
                decoded = extract_single_vehicle_track(
                    residual * box_mask,
                    float(fs),
                    float(dx_m),
                    str(direction),
                    float(vmin_kmh),
                    float(vmax_kmh),
                    config=cfg.single_vehicle,
                    prior_heatmap=proposal_prior * box_mask if proposal_prior is not None else None,
                    prior_weight=float(cfg.proposal_prior_weight),
                )
                if decoded:
                    iteration_candidates.extend(decoded[:1])
        if coarse_candidates:
            iteration_candidates.extend(coarse_candidates)
        if not iteration_candidates and proposal_prior is None:
            iteration_candidates = extract_all(
                residual,
                float(fs),
                float(dx_m),
                str(direction),
                float(vmin_kmh),
                float(vmax_kmh),
                config=cfg.candidate_graph,
            )
            iteration_candidates = [tr for tr in iteration_candidates if float(tr.total_score) >= float(cfg.candidate_min_score)]
            iteration_candidates = sorted(iteration_candidates, key=lambda tr: tr.total_score, reverse=True)[: int(cfg.candidate_limit)]
        if not iteration_candidates:
            break
        for chosen in iteration_candidates:
            all_candidates.append(chosen)
            residual = _suppress_track_region(
                residual,
                chosen,
                weight=float(cfg.residual_suppression_weight),
                sigma_ch=float(cfg.residual_suppression_sigma_channels),
                sigma_t=float(cfg.residual_suppression_sigma_time),
            )
        if not bool(cfg.iterative_extraction):
            break

    if not all_candidates:
        return []

    refined: list[Track] = []
    for idx, cand in enumerate(all_candidates):
        crop, ch_offset, t_offset = _crop_around_track(
            arr,
            cand,
            fs=float(fs),
            channel_margin=int(cfg.crop_channel_margin),
            time_margin_s=float(cfg.crop_time_margin_s),
        )
        cand_speed = float(cand.mean_speed_kmh if np.isfinite(cand.mean_speed_kmh) else 0.0)
        speed_margin = max(20.0, 0.25 * max(1.0, abs(cand_speed)))
        vmin = max(float(vmin_kmh), max(1.0, cand_speed - speed_margin))
        vmax = min(float(vmax_kmh), max(vmin + 1.0, cand_speed + speed_margin))
        tr = cand
        needs_shift = False
        model_confidence = 1.0
        if local_model is not None:
            model_confidence = _track_candidate_confidence(
                local_model,
                crop,
                speed_norm_kmh=float(cfg.inference.speed_norm_kmh),
                device=device,
            )
            if float(model_confidence) >= float(cfg.min_model_confidence):
                pred_tracks = predict_single_vehicle_track(
                    local_model,
                    crop,
                    float(fs),
                    float(dx_m),
                    str(direction),
                    float(vmin),
                    float(vmax),
                    config=cfg.inference,
                    device=device,
                )
                if pred_tracks:
                    refined_track = pred_tracks[0]
                    if len(refined_track.points) >= max(4, int(round(0.8 * len(cand.points)))):
                        tr = refined_track
                        needs_shift = True
                    else:
                        tr = cand
        else:
            decoded = extract_single_vehicle_track(
                crop,
                float(fs),
                float(dx_m),
                str(direction),
                float(vmin),
                float(vmax),
                config=cfg.single_vehicle,
                prior_heatmap=_track_prior_heatmap(cand, crop.shape[0], crop.shape[1]),
                prior_weight=1.0,
            )
            if decoded:
                refined_track = decoded[0]
                if len(refined_track.points) >= max(4, int(round(0.8 * len(cand.points)))):
                    tr = refined_track
                    needs_shift = True
                else:
                    tr = cand
        shifted = _shift_track(tr, ch_offset, t_offset, fs=float(fs), dx_m=float(dx_m)) if needs_shift else tr
        combined_conf = float(cfg.graph_confidence_weight) * float(cand.total_score) + float(cfg.model_confidence_weight) * float(model_confidence)
        shifted.total_score = float(shifted.total_score + combined_conf)
        refined.append(shifted)

    combined = _deduplicate_tracks(
        [*all_candidates, *refined],
        tol_samples=int(cfg.dedup_tolerance_samples),
        min_overlap_channels=int(cfg.dedup_min_overlap_channels),
        min_overlap_ratio=float(cfg.dedup_min_overlap_ratio),
    )
    combined = _merge_track_fragments(
        combined,
        fs=float(fs),
        dx_m=float(dx_m),
        max_gap_channels=max(6, int(cfg.dedup_tolerance_samples // 30)),
        max_gap_seconds=max(0.45, float(cfg.crop_time_margin_s) * 0.25),
        line_distance_threshold=max(0.30, float(cfg.crop_time_margin_s) * 0.08),
        speed_diff_kmh=12.0,
    )
    combined = _prune_short_fragments(combined, min_keep_points=6, edge_margin_points=max(60, int(fs * 0.08)), n_samples=int(arr.shape[1]))
    return combined


def extract_multi_vehicle_tracks(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[MultiVehiclePipelineConfig | dict[str, Any]] = None,
    *,
    model_path: Optional[str | Path] = None,
    device: Optional[str] = None,
) -> list[Track]:
    return extract_multi_vehicle_tracks_with_models(
        data,
        fs,
        dx_m,
        direction,
        vmin_kmh,
        vmax_kmh,
        config=config,
        model_path=model_path,
        device=device,
    )
