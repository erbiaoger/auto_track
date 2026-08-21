"""Predict and evaluate PeakSlotNet checkpoints on peak-slot tensor shards.

Purpose:
    Inspect a trained `peak_slot` checkpoint directly on converted tensor
    shards. Predictions are exported as selected peak candidates, so plotted
    points always lie on detected heatmap peaks.

Example:
    uv run python -m autotrack.dl.predict_peak_slot_dataset \
        --data-dir datasets/peak_slot/train \
        --model models/peak_slot_cuda/checkpoint_best.pt \
        --out-dir /tmp/peak_slot_prediction_check \
        --device cuda \
        --plot-samples 16

Outputs:
    <out-dir>/summary.json
    <out-dir>/sample_summary.csv
    <out-dir>/predicted_tracks.csv
    <out-dir>/ground_truth_tracks.csv
    <out-dir>/plots/sample_000000.png, ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.core.track_fusion import extend_peakslot_tracks_with_graph
from autotrack.dl.peak_slot_model import (
    InferenceConfig,
    argmax_peak_slot_path,
    decode_peak_slot_path,
    decode_peak_slot_paths,
    load_checkpoint_model,
    peak_slot_detection_metrics,
    peak_slot_physics_metrics,
    peak_slot_set_loss,
)
from autotrack.dl.trajectory_set_model import LABEL_TO_DIRECTION, auto_torch_device, move_targets_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PeakSlotNet outputs from converted .pt tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Peak-slot dataset directory.")
    parser.add_argument("--model", required=True, type=Path, help="PeakSlotNet checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary, CSV, and plots.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum evaluated samples; 0 evaluates all.")
    parser.add_argument("--max-csv-samples", type=int, default=32, help="Detailed CSV sample limit; 0 disables detailed CSV.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of overlay figures to write.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for overlay PNG figures.")
    parser.add_argument("--plot-style", default="waveform", choices=["waveform", "heatmap"], help="Overlay plot style: GUI-like waveform or heatmap.")
    parser.add_argument("--plot-direction-filter", default="all", choices=["all", "forward", "reverse"], help="Direction filter used only for overlay PNG plots.")
    parser.add_argument("--prediction-direction-filter", default="all", choices=["all", "forward", "reverse"], help="Direction filter applied to exported predictions and summary counts.")
    parser.add_argument("--objectness-threshold", type=float, default=0.35, help="Predicted slot objectness threshold. Lower values favor recall.")
    parser.add_argument("--extra-candidate-slots", type=int, default=32, help="Decode this many extra below-threshold slots by objectness rank.")
    parser.add_argument("--candidate-objectness-floor", type=float, default=0.02, help="Minimum objectness for extra decoded candidate slots.")
    parser.add_argument("--peak-threshold", type=float, default=0.4, help="Minimum selected peak probability.")
    parser.add_argument("--min-visible-channels", type=int, default=2, help="Minimum selected peaks for a predicted track.")
    parser.add_argument("--max-predicted-tracks", type=int, default=96, help="Maximum slots kept per sample.")
    parser.add_argument("--decoder-mode", default="beam_global", choices=["argmax", "viterbi", "beam_global"], help="Peak decoding strategy.")
    parser.add_argument("--no-viterbi-decoder", action="store_true", help="Use legacy per-channel argmax decoding instead of Viterbi.")
    parser.add_argument("--viterbi-beam-size", type=int, default=8, help="Number of candidate paths retained per slot in beam_global decoding.")
    parser.add_argument("--time-prior-weight", type=float, default=2.0, help="Penalty weight for peak distance from time_prior.")
    parser.add_argument("--global-conflict-penalty", type=float, default=2.0, help="Enable cross-slot overlap suppression when > 0.")
    parser.add_argument("--conflict-mode", default="soft", choices=["soft", "hard", "off"], help="Cross-slot conflict handling. soft preserves close vehicles with unique support.")
    parser.add_argument("--duplicate-overlap-ratio", type=float, default=0.75, help="Minimum channel overlap ratio before soft conflict handling treats two paths as duplicates.")
    parser.add_argument("--min-unique-support-channels", type=int, default=3, help="Minimum unique peak-supported channels needed to preserve a close path.")
    parser.add_argument("--viterbi-topk", type=int, default=16, help="Top peak candidates per channel considered by Viterbi.")
    parser.add_argument("--viterbi-candidate-threshold", type=float, default=0.01, help="Low probability floor for candidates entering Viterbi.")
    parser.add_argument("--viterbi-speed-min-kmh", type=float, default=60.0, help="Minimum hard transition speed for Viterbi.")
    parser.add_argument("--viterbi-speed-max-kmh", type=float, default=100.0, help="Maximum hard transition speed for Viterbi.")
    parser.add_argument("--viterbi-max-skip-channels", type=int, default=4, help="Maximum channel gap for one Viterbi transition.")
    parser.add_argument("--viterbi-point-bonus", type=float, default=3.0, help="Per-point reward that lets Viterbi prefer long paths.")
    parser.add_argument("--viterbi-skip-penalty", type=float, default=2.0, help="Penalty per skipped channel in Viterbi.")
    parser.add_argument("--viterbi-speed-penalty", type=float, default=1.0, help="Soft penalty for deviation from slot speed.")
    parser.add_argument("--viterbi-smoothness-penalty", type=float, default=0.6, help="Soft penalty for slope changes.")
    parser.add_argument("--viterbi-inertia-penalty", type=float, default=2.5, help="Penalty for deviating from the previous speed prediction.")
    parser.add_argument("--viterbi-slope-memory", type=float, default=0.75, help="Exponential memory for the slot's running slope.")
    parser.add_argument("--fusion-mode", default="off", choices=["off", "graph_extend"], help="Optional graph-search extension after PeakSlotNet decoding.")
    parser.add_argument("--fusion-min-seed-channels", type=int, default=4, help="Minimum PeakSlotNet points before graph extension is attempted.")
    parser.add_argument("--fusion-extend-left", action=argparse.BooleanOptionalAction, default=True, help="Extend graph search toward lower channel indices.")
    parser.add_argument("--fusion-extend-right", action=argparse.BooleanOptionalAction, default=True, help="Extend graph search toward higher channel indices.")
    parser.add_argument("--fusion-graph-prominence", type=float, default=0.18, help="Graph peak prominence used during fusion.")
    parser.add_argument("--fusion-graph-min-peak-distance", type=int, default=120, help="Graph minimum peak distance in samples of the fusion input.")
    parser.add_argument("--fusion-graph-max-skip-channels", type=int, default=8, help="Maximum channel skip used by graph extension.")
    parser.add_argument("--fusion-min-added-channels", type=int, default=1, help="Minimum added channels required to keep a fused track.")
    parser.add_argument("--fusion-nms-tolerance-samples", type=int, default=180, help="Same-channel fusion tolerance in original sample units.")
    parser.add_argument("--fusion-bridge-search-radius-samples", type=int, default=900, help="Internal gap bridge search radius in original sample units.")
    parser.add_argument(
        "--matcher",
        default="hungarian",
        choices=["hungarian", "greedy", "auction", "independent"],
        help="Metric matching strategy. independent is a fast approximate matcher.",
    )
    parser.add_argument("--none-weight", type=float, default=0.35, help="GT-none weight for loss reporting.")
    parser.add_argument("--no-object-weight", type=float, default=0.15, help="Unmatched slot weight for loss reporting.")
    parser.add_argument("--metric-point-threshold", type=float, default=0.05, help="Normalized time-error threshold for TP.")
    parser.add_argument("--close-pair-min-common-channels", type=int, default=8, help="Minimum shared visible channels for close-pair metrics.")
    parser.add_argument("--close-pair-min-gap-s", type=float, default=0.15, help="Minimum mean time gap for close-pair metrics.")
    parser.add_argument("--close-pair-max-gap-s", type=float, default=1.5, help="Maximum mean time gap for close-pair metrics.")
    parser.add_argument("--no-ground-truth-csv", action="store_true", help="Do not write ground_truth_tracks.csv.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fmt = str(meta.get("format", ""))
    if fmt and fmt != "peak_slot_shards_v1":
        raise ValueError(
            f"{data_dir} is not a peak_slot dataset: meta.json format={fmt!r}. "
            "Run convert_track_slot_to_peak_slot.sh first, or set DATA_DIR to a directory "
            "whose shards contain peak_time/peak_valid/peak_index."
        )
    return meta


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _targets_from_payload(payload: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    required = {
        "peak_time",
        "peak_amp",
        "peak_valid",
        "peak_index",
        "gt_peak_index",
        "visibility",
        "direction",
        "speed",
        "gt_valid",
    }
    missing = sorted(required - set(payload))
    if missing:
        keys = ", ".join(sorted(str(key) for key in payload.keys()))
        raise KeyError(
            f"Input shard is missing peak_slot fields: {missing}. "
            f"Available keys: {keys}. This usually means DATA_DIR points to track_slot shards; "
            "convert them with convert_track_slot_to_peak_slot.sh before prediction."
        )
    targets = {
        "peak_time": payload["peak_time"][idx].to(torch.float32),
        "peak_amp": payload["peak_amp"][idx].to(torch.float32),
        "peak_valid": payload["peak_valid"][idx].to(torch.bool),
        "peak_index": payload["peak_index"][idx].to(torch.long),
        "gt_peak_index": payload["gt_peak_index"][idx].to(torch.long),
        "visibility": payload["visibility"][idx].to(torch.float32),
        "direction": payload["direction"][idx].to(torch.long),
        "speed": payload["speed"][idx].to(torch.float32),
        "gt_valid": payload["gt_valid"][idx].to(torch.bool),
    }
    targets["gt_count"] = targets["gt_valid"].sum(dim=1).to(torch.long)
    return targets


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    max_samples: int,
) -> Iterator[tuple[list[int], torch.Tensor, dict[str, torch.Tensor]]]:
    emitted = 0
    global_start = 0
    for shard in shards:
        payload = torch.load(str(data_dir / shard), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        for start in range(0, n, int(batch_size)):
            if int(max_samples) > 0 and emitted >= int(max_samples):
                return
            take = min(int(batch_size), n - start)
            if int(max_samples) > 0:
                take = min(take, int(max_samples) - emitted)
            if take <= 0:
                return
            idx = torch.arange(start, start + take)
            sample_indices = list(range(global_start + start, global_start + start + take))
            emitted += take
            yield sample_indices, payload["x"][idx].to(torch.float32), _targets_from_payload(payload, idx)
        global_start += n


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _weighted_add(sums: defaultdict[str, float], weights: defaultdict[str, float], metrics: dict[str, float], weight: int) -> None:
    for key, value in metrics.items():
        value_f = float(value)
        if math.isfinite(value_f):
            sums[str(key)] += value_f * float(weight)
            weights[str(key)] += float(weight)


def _weighted_mean(sums: defaultdict[str, float], weights: defaultdict[str, float]) -> dict[str, float]:
    return {key: float(sums[key] / max(1e-12, weights[key])) for key in sorted(sums) if weights[key] > 0.0}


def _filter_predictions_by_direction(predictions: list[dict[str, Any]], direction_filter: str) -> list[dict[str, Any]]:
    direction = str(direction_filter).lower()
    if direction == "all":
        return predictions
    return [pred for pred in predictions if str(pred.get("direction", "")).lower() == direction]


def _prediction_to_track(
    pred: dict[str, Any],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    fs: float,
    dx_m: float,
) -> Track:
    points: list[TrackPoint] = []
    for ch, peak_idx, prob in zip(pred["channels"], pred["peak_indices"], pred["peak_probs"]):
        peak_sample = int(targets_cpu["peak_index"][batch_index, int(ch), int(peak_idx)].item())
        t_idx = int(max(0, peak_sample))
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=t_idx,
                time_s=float(t_idx) / float(fs),
                offset_m=float(ch) * float(dx_m),
                amp=float(targets_cpu["peak_amp"][batch_index, int(ch), int(peak_idx)].item()),
                score=float(pred.get("score", 0.0)) * float(prob),
            )
        )
    return Track(
        track_id=int(pred.get("slot", 0)),
        direction=str(pred.get("direction", "forward")),
        points=points,
        total_score=float(pred.get("selection_score", pred.get("score", 0.0))),
        mean_speed_kmh=float(pred.get("speed_kmh", float("nan"))),
    )


def _nearest_peak_index_for_point(
    point: TrackPoint,
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    time_downsample: int,
) -> Optional[int]:
    ch = int(point.ch_idx)
    if ch < 0 or ch >= int(targets_cpu["peak_valid"].shape[1]):
        return None
    valid = torch.where(targets_cpu["peak_valid"][batch_index, ch])[0]
    if valid.numel() <= 0:
        return None
    target_down = int(round(float(point.t_idx) / float(max(1, int(time_downsample)))))
    indices = targets_cpu["peak_index"][batch_index, ch, valid].to(torch.long)
    diffs = torch.abs(indices - int(target_down))
    pos = int(torch.argmin(diffs).item())
    return int(valid[pos].item())


def _tracks_to_predictions(
    tracks: list[Track],
    source_predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    time_downsample: int,
) -> list[dict[str, Any]]:
    by_slot = {int(pred.get("slot", idx)): pred for idx, pred in enumerate(source_predictions)}
    out: list[dict[str, Any]] = []
    for rank, track in enumerate(tracks):
        src = by_slot.get(int(track.track_id), source_predictions[min(rank, len(source_predictions) - 1)] if source_predictions else {})
        channels: list[int] = []
        peak_indices: list[int] = []
        peak_probs: list[float] = []
        original_prob_by_ch = {
            int(ch): float(prob)
            for ch, prob in zip(src.get("channels", []), src.get("peak_probs", []))
        }
        for point in sorted(track.points, key=lambda item: int(item.ch_idx)):
            peak_idx = _nearest_peak_index_for_point(
                point,
                targets_cpu,
                batch_index,
                time_downsample=int(time_downsample),
            )
            if peak_idx is None:
                continue
            channels.append(int(point.ch_idx))
            peak_indices.append(int(peak_idx))
            peak_probs.append(float(original_prob_by_ch.get(int(point.ch_idx), 0.0)))
        if not channels:
            continue
        item = dict(src)
        item.update(
            {
                "rank": int(src.get("rank", rank)),
                "slot": int(src.get("slot", track.track_id)),
                "score": float(src.get("score", max(0.0, track.total_score))),
                "selection_score": float(src.get("selection_score", max(0.0, track.total_score))),
                "path_score": float(src.get("path_score", track.total_score)),
                "direction": str(track.direction),
                "channels": channels,
                "peak_indices": peak_indices,
                "peak_probs": peak_probs,
                "fusion_point_count": int(len(channels)),
            }
        )
        out.append(item)
    return out


def _apply_graph_fusion_to_predictions(
    predictions: list[dict[str, Any]],
    heatmap: torch.Tensor,
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    fs: float,
    dx_m: float,
    time_downsample: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if str(args.fusion_mode).lower() != "graph_extend" or not predictions:
        return predictions, {
            "fusion_enabled": False,
            "fusion_input_track_count": int(len(predictions)),
            "fusion_output_track_count": int(len(predictions)),
            "fusion_added_point_count": 0,
        }
    tracks = [
        _prediction_to_track(
            pred,
            targets_cpu,
            batch_index,
            fs=float(fs) / float(max(1, int(time_downsample))),
            dx_m=float(dx_m),
        )
        for pred in predictions
    ]
    fusion_input = heatmap.detach().cpu().to(torch.float32).numpy()
    fused_tracks: list[Track] = []
    diagnostics: dict[str, Any] = {
        "fusion_enabled": True,
        "fusion_input_track_count": 0,
        "fusion_output_track_count": 0,
        "fusion_added_point_count": 0,
        "fusion_tracks": [],
    }
    for direction in sorted({str(track.direction) for track in tracks}):
        direction_tracks = [track for track in tracks if str(track.direction) == direction]
        direction_diagnostics: dict[str, Any] = {}
        fused_tracks.extend(
            extend_peakslot_tracks_with_graph(
                data=fusion_input,
                fs=float(fs) / float(max(1, int(time_downsample))),
                dx_m=float(dx_m),
                tracks=direction_tracks,
                direction=str(direction),
                vmin_kmh=float(args.viterbi_speed_min_kmh),
                vmax_kmh=float(args.viterbi_speed_max_kmh),
                config={
                    "fusion_mode": str(args.fusion_mode),
                    "fusion_min_seed_channels": int(args.fusion_min_seed_channels),
                    "fusion_extend_left": bool(args.fusion_extend_left),
                    "fusion_extend_right": bool(args.fusion_extend_right),
                    "fusion_graph_prominence": float(args.fusion_graph_prominence),
                    "fusion_graph_min_peak_distance": max(1, int(round(float(args.fusion_graph_min_peak_distance) / float(max(1, int(time_downsample)))))),
                    "fusion_graph_max_skip_channels": int(args.fusion_graph_max_skip_channels),
                    "fusion_min_added_channels": int(args.fusion_min_added_channels),
                    "fusion_nms_tolerance_samples": max(1, int(round(float(args.fusion_nms_tolerance_samples) / float(max(1, int(time_downsample)))))),
                    "fusion_bridge_search_radius_samples": max(1, int(round(float(args.fusion_bridge_search_radius_samples) / float(max(1, int(time_downsample)))))),
                },
                diagnostics=direction_diagnostics,
            )
        )
        diagnostics["fusion_input_track_count"] += int(direction_diagnostics.get("fusion_input_track_count", 0))
        diagnostics["fusion_output_track_count"] += int(direction_diagnostics.get("fusion_output_track_count", 0))
        diagnostics["fusion_added_point_count"] += int(direction_diagnostics.get("fusion_added_point_count", 0))
        diagnostics["fusion_tracks"].extend(list(direction_diagnostics.get("fusion_tracks", [])))
    fused = _tracks_to_predictions(fused_tracks, predictions, targets_cpu, batch_index, time_downsample=1)
    return fused, diagnostics


def _prediction_peak_set(pred: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(ch), int(pk)) for ch, pk in zip(pred["channels"], pred["peak_indices"])}


def _prediction_time_map(
    pred: dict[str, Any],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    time_downsample: int,
) -> dict[int, int]:
    return {
        int(ch): int(targets_cpu["peak_index"][batch_index, int(ch), int(pk)].item()) * int(time_downsample)
        for ch, pk in zip(pred["channels"], pred["peak_indices"])
    }


def _prediction_duplicate_decision(
    pred: dict[str, Any],
    existing: dict[str, Any],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    time_downsample: int,
    dedup_tolerance_samples: int,
    duplicate_overlap_ratio: float,
    min_unique_support_channels: int,
    dedup_min_overlap_channels: int,
) -> tuple[bool, bool, str]:
    pred_times = _prediction_time_map(pred, targets_cpu, batch_index, time_downsample=int(time_downsample))
    existing_times = _prediction_time_map(existing, targets_cpu, batch_index, time_downsample=int(time_downsample))
    common_channels = sorted(set(pred_times) & set(existing_times))
    if len(common_channels) < int(dedup_min_overlap_channels):
        return False, False, "insufficient_overlap"
    min_len = max(1, min(len(pred_times), len(existing_times)))
    overlap_ratio = float(len(common_channels) / min_len)
    peak_set = _prediction_peak_set(pred)
    existing_set = _prediction_peak_set(existing)
    unique_support = len(peak_set - existing_set)
    diffs = np.array([abs(pred_times[ch] - existing_times[ch]) for ch in common_channels], dtype=np.float64)
    median_diff = float(np.median(diffs)) if diffs.size else float("inf")
    same_direction = str(pred.get("direction", "")) == str(existing.get("direction", ""))
    near_duplicate = (
        same_direction
        and overlap_ratio >= float(duplicate_overlap_ratio)
        and median_diff <= float(dedup_tolerance_samples)
    )
    if near_duplicate and unique_support < int(min_unique_support_channels):
        reason = f"duplicate overlap={overlap_ratio:.3f} unique={unique_support} median_dt_samples={median_diff:.1f}"
        return True, False, reason
    if near_duplicate:
        reason = f"close_kept overlap={overlap_ratio:.3f} unique={unique_support} median_dt_samples={median_diff:.1f}"
        return False, True, reason
    return False, False, f"distinct overlap={overlap_ratio:.3f} unique={unique_support} median_dt_samples={median_diff:.1f}"


def _filter_prediction_conflicts(
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    sample_index: int,
    time_downsample: int,
    inference_config: InferenceConfig,
    diagnostics: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    mode = str(inference_config.conflict_mode).lower()
    if diagnostics is not None:
        diagnostics["raw_candidate_count"] = int(len(predictions))
        diagnostics["duplicate_removed_count"] = 0
        diagnostics["close_pair_kept_count"] = 0
        diagnostics.setdefault("rows", [])
    if mode == "off":
        return predictions
    kept: list[dict[str, Any]] = []
    if mode == "soft":
        pending = list(predictions)
        while pending:
            def adjusted_score(item: dict[str, Any]) -> float:
                base = float(item.get("selection_score", item["score"]))
                item_set = _prediction_peak_set(item)
                max_overlap = 0.0
                for existing_item in kept:
                    existing_set = _prediction_peak_set(existing_item)
                    min_len = max(1, min(len(item_set), len(existing_set)))
                    max_overlap = max(max_overlap, float(len(item_set & existing_set) / min_len))
                return base - float(inference_config.global_conflict_penalty) * max_overlap

            pred = max(pending, key=adjusted_score)
            pending.remove(pred)
            duplicate = False
            close_kept = False
            reason = "kept"
            for existing in kept:
                duplicate, close_kept_one, reason = _prediction_duplicate_decision(
                    pred,
                    existing,
                    targets_cpu,
                    batch_index,
                    time_downsample=int(time_downsample),
                    dedup_tolerance_samples=int(inference_config.dedup_tolerance_samples),
                    duplicate_overlap_ratio=float(inference_config.duplicate_overlap_ratio),
                    min_unique_support_channels=int(inference_config.min_unique_support_channels),
                    dedup_min_overlap_channels=int(inference_config.dedup_min_overlap_channels),
                )
                close_kept = close_kept or close_kept_one
                if duplicate:
                    break
            if diagnostics is not None:
                diagnostics["rows"].append(
                    {
                        "sample_index": int(sample_index),
                        "slot": int(pred["slot"]),
                        "rank": int(pred["rank"]),
                        "score": f"{float(pred['score']):.6f}",
                        "direction": str(pred["direction"]),
                        "point_count": int(len(pred["channels"])),
                        "action": "removed_duplicate" if duplicate else "kept",
                        "reason": reason,
                    }
                )
            if duplicate:
                if diagnostics is not None:
                    diagnostics["duplicate_removed_count"] += 1
                continue
            if close_kept and diagnostics is not None:
                diagnostics["close_pair_kept_count"] += 1
            kept.append(pred)
        return kept
    for pred in sorted(predictions, key=lambda item: float(item.get("selection_score", item["score"])), reverse=True):
        duplicate = False
        close_kept = False
        reason = "kept"
        if mode == "hard":
            peak_set = _prediction_peak_set(pred)
            for existing in kept:
                existing_set = _prediction_peak_set(existing)
                common = len(peak_set & existing_set)
                min_len = max(1, min(len(peak_set), len(existing_set)))
                if common >= 3 or common / min_len >= 0.35:
                    duplicate = True
                    reason = f"hard_conflict common={common} ratio={common / min_len:.3f}"
                    break
        else:
            raise ValueError(f"Unsupported conflict_mode: {inference_config.conflict_mode}")
        if diagnostics is not None:
            diagnostics["rows"].append(
                {
                    "sample_index": int(sample_index),
                    "slot": int(pred["slot"]),
                    "rank": int(pred["rank"]),
                    "score": f"{float(pred['score']):.6f}",
                    "direction": str(pred["direction"]),
                    "point_count": int(len(pred["channels"])),
                    "action": "removed_duplicate" if duplicate else "kept",
                    "reason": reason,
                }
            )
        if duplicate:
            if diagnostics is not None:
                diagnostics["duplicate_removed_count"] += 1
            continue
        if close_kept and diagnostics is not None:
            diagnostics["close_pair_kept_count"] += 1
        kept.append(pred)
    return kept


def _active_predictions(
    outputs_cpu: dict[str, torch.Tensor],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    objectness_threshold: float,
    peak_threshold: float,
    min_visible_channels: int,
    max_predicted_tracks: int,
    speed_norm_kmh: float,
    fs: float,
    dx_m: float,
    time_downsample: int,
    inference_config: InferenceConfig,
    sample_index: int = -1,
    diagnostics: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    obj = torch.sigmoid(outputs_cpu["objectness_logits"][batch_index])
    peak_prob = torch.softmax(outputs_cpu["peak_logits"][batch_index], dim=-1)
    direction = torch.argmax(outputs_cpu["direction_logits"][batch_index], dim=-1)
    speed = outputs_cpu["speed"][batch_index]
    time_prior = outputs_cpu.get("time_prior")
    ranked_slots = torch.argsort(obj, descending=True).tolist()
    active_slots = [int(slot) for slot in ranked_slots if float(obj[int(slot)].item()) >= float(objectness_threshold)]
    active_set = set(active_slots)
    extra_slots: list[int] = []
    extra_limit = max(0, int(inference_config.extra_candidate_slots))
    if extra_limit > 0:
        for slot in ranked_slots:
            slot_i = int(slot)
            if slot_i in active_set:
                continue
            if float(obj[slot_i].item()) < float(inference_config.candidate_objectness_floor):
                continue
            extra_slots.append(slot_i)
            if len(extra_slots) >= extra_limit:
                break
    order = (active_slots + extra_slots)[: min(int(max_predicted_tracks), int(obj.shape[0]))]
    slot_rank = {int(slot): int(rank) for rank, slot in enumerate(ranked_slots)}
    predictions: list[dict[str, Any]] = []
    for slot in order:
        score = float(obj[slot].item())
        direction_label = int(direction[slot].item())
        decoder_mode = str(inference_config.decoder_mode).lower()
        path_options: list[tuple[list[dict[str, float | int]], float]] = []
        if bool(inference_config.use_viterbi_decoder) and decoder_mode == "beam_global":
            x_axis_m = np.arange(int(peak_prob.shape[1]), dtype=np.float64) * float(dx_m)
            decoded_options = decode_peak_slot_paths(
                peak_prob[slot].numpy(),
                targets_cpu["peak_time"][batch_index],
                targets_cpu["peak_valid"][batch_index],
                targets_cpu["peak_index"][batch_index],
                direction_label=direction_label,
                predicted_speed_kmh=float(speed[slot].item()) * float(speed_norm_kmh),
                x_axis_m=x_axis_m,
                time_downsample=int(time_downsample),
                fs=float(fs),
                config=inference_config,
                time_prior=None if not torch.is_tensor(time_prior) else time_prior[batch_index, slot].numpy(),
            )
            path_options = [(list(item["path"]), float(item["score"])) for item in decoded_options]
        elif bool(inference_config.use_viterbi_decoder) and decoder_mode != "argmax":
            x_axis_m = np.arange(int(peak_prob.shape[1]), dtype=np.float64) * float(dx_m)
            decoded = decode_peak_slot_path(
                peak_prob[slot].numpy(),
                targets_cpu["peak_time"][batch_index],
                targets_cpu["peak_valid"][batch_index],
                targets_cpu["peak_index"][batch_index],
                direction_label=direction_label,
                predicted_speed_kmh=float(speed[slot].item()) * float(speed_norm_kmh),
                x_axis_m=x_axis_m,
                time_downsample=int(time_downsample),
                fs=float(fs),
                config=inference_config,
                time_prior=None if not torch.is_tensor(time_prior) else time_prior[batch_index, slot].numpy(),
            )
            path_options = [(decoded, 0.0)]
        else:
            decoded = argmax_peak_slot_path(
                peak_prob[slot].numpy(),
                targets_cpu["peak_valid"][batch_index],
                peak_threshold=float(peak_threshold),
            )
            path_options = [(decoded, 0.0)]
        for option_idx, (decoded, path_score) in enumerate(path_options):
            channels = [int(item["ch"]) for item in decoded]
            peak_indices = [int(item["peak_idx"]) for item in decoded]
            probs = [float(item["prob"]) for item in decoded]
            if len(channels) < int(min_visible_channels):
                continue
            norm_path_score = float(path_score) / float(max(1, len(channels)))
            predictions.append(
                {
                    "rank": int(slot_rank.get(slot, len(slot_rank))),
                    "slot": slot,
                    "beam_index": int(option_idx),
                    "score": score,
                    "selection_score": float(score + 0.02 * norm_path_score),
                    "path_score": float(path_score),
                    "direction_label": direction_label,
                    "direction": LABEL_TO_DIRECTION.get(direction_label, str(direction_label)),
                    "speed": float(speed[slot].item()),
                    "speed_kmh": float(speed[slot].item() * float(speed_norm_kmh)),
                    "channels": channels,
                    "peak_indices": peak_indices,
                    "peak_probs": probs,
                }
            )
    if str(inference_config.decoder_mode).lower() == "beam_global":
        best_per_slot: list[dict[str, Any]] = []
        used_slots: set[int] = set()
        for pred in sorted(predictions, key=lambda item: float(item.get("selection_score", item["score"])), reverse=True):
            slot = int(pred["slot"])
            if slot in used_slots:
                continue
            best_per_slot.append(pred)
            used_slots.add(slot)
        predictions = best_per_slot
    if str(inference_config.decoder_mode).lower() == "beam_global" and float(inference_config.global_conflict_penalty) > 0.0:
        predictions = _filter_prediction_conflicts(
            predictions,
            targets_cpu,
            batch_index,
            sample_index=int(sample_index),
            time_downsample=int(time_downsample),
            inference_config=inference_config,
            diagnostics=diagnostics,
        )
    elif diagnostics is not None:
        diagnostics["raw_candidate_count"] = int(len(predictions))
        diagnostics["duplicate_removed_count"] = 0
        diagnostics["close_pair_kept_count"] = 0
        diagnostics.setdefault("rows", [])
    return predictions


def _write_prediction_rows(writer: csv.DictWriter, *, sample_index: int, predictions: list[dict[str, Any]], targets_cpu: dict[str, torch.Tensor], batch_index: int) -> None:
    for pred_id, pred in enumerate(predictions):
        for ch, peak_idx, prob in zip(pred["channels"], pred["peak_indices"], pred["peak_probs"]):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "pred_track_id": int(pred_id),
                    "slot": int(pred["slot"]),
                    "rank": int(pred["rank"]),
                    "score": f"{float(pred['score']):.6f}",
                    "direction": str(pred["direction"]),
                    "speed_kmh": f"{float(pred['speed_kmh']):.6f}",
                    "channel": int(ch),
                    "peak_index": int(peak_idx),
                    "time_norm": f"{float(targets_cpu['peak_time'][batch_index, ch, peak_idx].item()):.8f}",
                    "peak_amp": f"{float(targets_cpu['peak_amp'][batch_index, ch, peak_idx].item()):.6f}",
                    "peak_prob": f"{float(prob):.6f}",
                }
            )


def _write_ground_truth_rows(writer: csv.DictWriter, *, sample_index: int, targets_cpu: dict[str, torch.Tensor], batch_index: int) -> None:
    none_idx = int(targets_cpu["peak_time"].shape[-1])
    valid_indices = torch.where(targets_cpu["gt_valid"][batch_index])[0].tolist()
    for gt_id, gt_idx in enumerate(valid_indices):
        direction_label = int(targets_cpu["direction"][batch_index, gt_idx].item())
        for ch in torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist():
            peak_idx = int(targets_cpu["gt_peak_index"][batch_index, gt_idx, ch].item())
            if peak_idx >= none_idx:
                continue
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "gt_track_id": int(gt_id),
                    "direction": LABEL_TO_DIRECTION.get(direction_label, str(direction_label)),
                    "channel": int(ch),
                    "peak_index": int(peak_idx),
                    "time_norm": f"{float(targets_cpu['peak_time'][batch_index, ch, peak_idx].item()):.8f}",
                }
            )


def _decoded_physics_counts(
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    window_seconds: float,
    dx_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    smooth_tolerance_s: float = 2.0,
) -> dict[str, int]:
    pair_count = 0
    direction_bad = 0
    speed_bad = 0
    triple_count = 0
    smooth_bad = 0
    for pred in predictions:
        chs = [int(ch) for ch in pred["channels"]]
        times = [
            float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds)
            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"])
        ]
        sign = 1.0 if str(pred.get("direction", "forward")) == "forward" else -1.0
        for (ch0, t0), (ch1, t1) in zip(zip(chs[:-1], times[:-1]), zip(chs[1:], times[1:])):
            pair_count += 1
            dt = float(t1 - t0)
            if sign * dt < 0.0:
                direction_bad += 1
            dx = abs(float(ch1 - ch0)) * float(dx_m)
            speed = 3.6 * dx / max(1e-9, abs(dt))
            if speed < float(speed_min_kmh) or speed > float(speed_max_kmh):
                speed_bad += 1
        for t0, t1, t2 in zip(times[:-2], times[1:-1], times[2:]):
            triple_count += 1
            if abs(float(t2 - 2.0 * t1 + t0)) > float(smooth_tolerance_s):
                smooth_bad += 1
    return {
        "pair_count": int(pair_count),
        "direction_bad": int(direction_bad),
        "speed_bad": int(speed_bad),
        "triple_count": int(triple_count),
        "smooth_bad": int(smooth_bad),
    }


def _gt_track_time_map(targets_cpu: dict[str, torch.Tensor], batch_index: int, gt_idx: int) -> dict[int, float]:
    times: dict[int, float] = {}
    for ch in torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist():
        peak_idx = int(targets_cpu["gt_peak_index"][batch_index, gt_idx, ch].item())
        if peak_idx >= int(targets_cpu["peak_time"].shape[-1]):
            continue
        times[int(ch)] = float(targets_cpu["peak_time"][batch_index, int(ch), peak_idx].item())
    return times


def _prediction_time_norm_map(
    pred: dict[str, Any],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
) -> dict[int, float]:
    return {
        int(ch): float(targets_cpu["peak_time"][batch_index, int(ch), int(pk)].item())
        for ch, pk in zip(pred["channels"], pred["peak_indices"])
    }


def _filtered_detection_counts(
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    point_threshold: float,
    min_visible_channels: int,
) -> dict[str, float]:
    gt_indices = [int(idx) for idx in torch.where(targets_cpu["gt_valid"][batch_index])[0].tolist()]
    gt_maps = {gt_idx: _gt_track_time_map(targets_cpu, batch_index, gt_idx) for gt_idx in gt_indices}
    gt_dirs = {
        gt_idx: LABEL_TO_DIRECTION.get(int(targets_cpu["direction"][batch_index, gt_idx].item()), "")
        for gt_idx in gt_indices
    }
    candidates: list[tuple[float, int, int, int, int]] = []
    for pred_idx, pred in enumerate(predictions):
        pred_map = _prediction_time_norm_map(pred, targets_cpu, batch_index)
        if len(pred_map) < int(min_visible_channels):
            continue
        for gt_idx in gt_indices:
            if str(pred.get("direction", "")) != str(gt_dirs[gt_idx]):
                continue
            common = sorted(set(pred_map) & set(gt_maps[gt_idx]))
            if not common:
                continue
            errors = [abs(float(pred_map[ch]) - float(gt_maps[gt_idx][ch])) for ch in common]
            mean_error = float(np.mean(errors))
            if mean_error <= float(point_threshold):
                candidates.append((mean_error, pred_idx, gt_idx, len(common), len(gt_maps[gt_idx])))
    candidates.sort(key=lambda item: (item[0], -item[3]))
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_error_sum = 0.0
    matched_common_sum = 0
    forward_gt = 0
    reverse_gt = 0
    short_gt = 0
    long_gt = 0
    matched_forward = 0
    matched_reverse = 0
    matched_short = 0
    matched_long = 0
    for gt_idx in gt_indices:
        direction = str(gt_dirs[gt_idx])
        visible_count = len(gt_maps[gt_idx])
        forward_gt += int(direction == "forward")
        reverse_gt += int(direction == "reverse")
        short_gt += int(visible_count < int(min_visible_channels))
        long_gt += int(visible_count >= int(min_visible_channels))
    for mean_error, pred_idx, gt_idx, common_count, _gt_visible_count in candidates:
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matched_error_sum += float(mean_error)
        matched_common_sum += int(common_count)
        direction = str(gt_dirs[gt_idx])
        visible_count = len(gt_maps[gt_idx])
        matched_forward += int(direction == "forward")
        matched_reverse += int(direction == "reverse")
        matched_short += int(visible_count < int(min_visible_channels))
        matched_long += int(visible_count >= int(min_visible_channels))
    return {
        "tp": float(len(matched_gt)),
        "pred": float(len(predictions)),
        "gt": float(len(gt_indices)),
        "time_error_sum": float(matched_error_sum),
        "time_error_count": float(len(matched_gt)),
        "matched_common_channels": float(matched_common_sum),
        "gt_forward": float(forward_gt),
        "gt_reverse": float(reverse_gt),
        "gt_short": float(short_gt),
        "gt_long": float(long_gt),
        "matched_forward": float(matched_forward),
        "matched_reverse": float(matched_reverse),
        "matched_short": float(matched_short),
        "matched_long": float(matched_long),
    }


def _plot_sample_overlay(
    out_path: Path,
    *,
    heatmap: torch.Tensor,
    sample_index: int,
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    window_seconds: float,
    dx_m: float,
    plot_style: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    arr = heatmap.detach().cpu().to(torch.float32).numpy()
    if arr.ndim != 2:
        raise ValueError("heatmap must have shape [channel, time]")
    n_ch, n_t = int(arr.shape[0]), int(arr.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finite = arr[np.isfinite(arr)]
    vmax = max(float(np.quantile(np.abs(finite), 0.995)), 1e-6) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    plot_style = str(plot_style).lower()
    if plot_style == "waveform":
        if float(dx_m) > 0.0:
            x_axis = np.arange(n_ch, dtype=np.float64) * float(dx_m) * 1e-3
            x_label = "Offset [km]"
        else:
            x_axis = np.arange(n_ch, dtype=np.float64)
            x_label = "Channel index"
        t_axis = np.linspace(0.0, float(window_seconds), n_t, dtype=np.float64)
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        wiggle_amp = 0.27 * spacing
        clip_ratio = 1.35
        for ch in range(n_ch):
            ratio = np.clip(arr[ch].astype(np.float64) / max(vmax, 1e-12), -clip_ratio, clip_ratio)
            ax.plot(x_axis[ch] + ratio * wiggle_amp, t_axis, color="0.45", linewidth=0.8, alpha=0.9)
        pad = 0.4 * spacing
        x_min_plot = float(x_axis[0] - pad)
        x_max_plot = float(x_axis[-1] + pad)
        x_span_full = max(1e-6, x_max_plot - x_min_plot)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(0.0, float(window_seconds))
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        im = None
    else:
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            vmin=-vmax,
            vmax=vmax,
            extent=(0.0, float(window_seconds), -0.5, float(n_ch) - 0.5),
            interpolation="nearest",
        )
        x_axis = np.arange(n_ch, dtype=np.float64)
        x_max_plot = float(window_seconds)
        x_span_full = max(1e-6, float(window_seconds))
    gt_label_added = False
    for gt_idx in torch.where(targets_cpu["gt_valid"][batch_index])[0].tolist():
        chs = []
        times = []
        for ch in torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist():
            peak_idx = int(targets_cpu["gt_peak_index"][batch_index, gt_idx, ch].item())
            if peak_idx >= int(targets_cpu["peak_time"].shape[-1]):
                continue
            chs.append(int(ch))
            times.append(float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds))
        if len(chs) >= 2:
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) for ch in chs]
                ys = times
            else:
                xs = times
                ys = chs
            ax.plot(xs, ys, color="black", linewidth=0.6, alpha=0.9, label="GT" if not gt_label_added else None)
            gt_label_added = True
    cmap = plt.get_cmap("tab20", max(1, len(predictions)))
    pred_label_added = False
    for pred_id, pred in enumerate(predictions):
        chs = list(pred["channels"])
        times = [
            float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds)
            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"])
        ]
        if len(chs) >= 2:
            color = cmap(pred_id % max(1, cmap.N))
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) for ch in chs]
                ys = times
            else:
                xs = times
                ys = chs
            ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.9, label="Prediction" if not pred_label_added else None)
            ax.scatter(xs, ys, s=9, color=[color], edgecolors="black", linewidths=0.2, alpha=0.95)
            speed_kmh = float(pred.get("speed_kmh", float("nan")))
            if math.isfinite(speed_kmh):
                mid = len(xs) // 2
                x_text = min(x_max_plot - 0.02 * x_span_full, float(xs[mid]) + 0.01 * x_span_full)
                ax.text(
                    x_text,
                    float(ys[mid]),
                    f"{speed_kmh:.1f} km/h",
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    alpha=0.95,
                    bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
                )
            pred_label_added = True
    gt_count = int(targets_cpu["gt_valid"][batch_index].sum().item())
    ax.set_title(f"PeakSlotNet prediction overlay, sample {sample_index}  GT={gt_count}  Pred={len(predictions)}")
    if plot_style != "waveform":
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel index")
        ax.set_xlim(0.0, float(window_seconds))
        ax.set_ylim(-0.5, float(n_ch) - 0.5)
    if gt_label_added or pred_label_added:
        ax.legend(loc="upper right", frameon=True)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized heatmap amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {data_dir / 'meta.json'}")
    device = _resolve_device(str(args.device))
    model, checkpoint = load_checkpoint_model(args.model, device=device)
    if str(checkpoint.get("model_family", "peak_slot")) != "peak_slot":
        raise ValueError(f"Checkpoint is not peak_slot: model_family={checkpoint.get('model_family')}")
    speed_norm_kmh = float(meta.get("speed_norm_kmh", 150.0))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    time_downsample = int(meta.get("time_downsample", 10))
    time_prior_weight = float(args.time_prior_weight) if bool(checkpoint.get("has_time_prior", True)) else 0.0
    inference_config = InferenceConfig(
        time_downsample=time_downsample,
        objectness_threshold=float(args.objectness_threshold),
        peak_threshold=float(args.peak_threshold),
        min_visible_channels=int(args.min_visible_channels),
        max_tracks=int(args.max_predicted_tracks),
        speed_norm_kmh=float(speed_norm_kmh),
        use_viterbi_decoder=not bool(args.no_viterbi_decoder),
        decoder_mode="argmax" if bool(args.no_viterbi_decoder) else str(args.decoder_mode),
        viterbi_beam_size=int(args.viterbi_beam_size),
        time_prior_weight=float(time_prior_weight),
        global_conflict_penalty=float(args.global_conflict_penalty),
        conflict_mode=str(args.conflict_mode),
        duplicate_overlap_ratio=float(args.duplicate_overlap_ratio),
        min_unique_support_channels=int(args.min_unique_support_channels),
        extra_candidate_slots=int(args.extra_candidate_slots),
        candidate_objectness_floor=float(args.candidate_objectness_floor),
        viterbi_topk=int(args.viterbi_topk),
        viterbi_candidate_threshold=float(args.viterbi_candidate_threshold),
        viterbi_speed_min_kmh=float(args.viterbi_speed_min_kmh),
        viterbi_speed_max_kmh=float(args.viterbi_speed_max_kmh),
        viterbi_max_skip_channels=int(args.viterbi_max_skip_channels),
        viterbi_point_bonus=float(args.viterbi_point_bonus),
        viterbi_skip_penalty=float(args.viterbi_skip_penalty),
        viterbi_speed_penalty=float(args.viterbi_speed_penalty),
        viterbi_smoothness_penalty=float(args.viterbi_smoothness_penalty),
        viterbi_inertia_penalty=float(args.viterbi_inertia_penalty),
        viterbi_slope_memory=float(args.viterbi_slope_memory),
    )
    summary_path = out_dir / "summary.json"
    sample_summary_path = out_dir / "sample_summary.csv"
    prediction_diagnostics_path = out_dir / "prediction_diagnostics.csv"
    pred_csv_path = out_dir / "predicted_tracks.csv"
    gt_csv_path = out_dir / "ground_truth_tracks.csv"
    plots_dir = out_dir / "plots"
    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_weights: defaultdict[str, float] = defaultdict(float)
    loss_sums: defaultdict[str, float] = defaultdict(float)
    loss_weights: defaultdict[str, float] = defaultdict(float)
    filtered_pred_total = 0
    filtered_gt_total = 0
    filtered_count_abs_error = 0.0
    filtered_count_exact = 0
    decoded_pair_total = 0
    decoded_direction_bad = 0
    decoded_speed_bad = 0
    decoded_triple_total = 0
    decoded_smooth_bad = 0
    decoded_peak_use_total = 0
    decoded_peak_conflict_total = 0
    duplicate_removed_total = 0
    close_pair_kept_total = 0
    raw_candidate_total = 0
    fusion_added_point_total = 0
    fusion_input_track_total = 0
    fusion_output_track_total = 0
    filtered_det_sums: dict[str, float] = defaultdict(float)
    sample_count = 0
    csv_sample_count = 0
    plot_sample_count = 0
    batch_count = 0
    t0 = time.perf_counter()
    pred_fields = ["sample_index", "pred_track_id", "slot", "rank", "score", "direction", "speed_kmh", "channel", "peak_index", "time_norm", "peak_amp", "peak_prob"]
    gt_fields = ["sample_index", "gt_track_id", "direction", "channel", "peak_index", "time_norm"]
    sample_fields = [
        "sample_index",
        "gt_count",
        "raw_candidate_count",
        "pred_count",
        "duplicate_removed_count",
        "close_pair_kept_count",
        "fusion_added_point_count",
        "max_objectness",
        "mean_objectness",
        "top_score",
    ]
    diag_fields = ["sample_index", "slot", "rank", "score", "direction", "point_count", "action", "reason"]
    with sample_summary_path.open("w", newline="", encoding="utf-8") as sample_fp, pred_csv_path.open("w", newline="", encoding="utf-8") as pred_fp, prediction_diagnostics_path.open("w", newline="", encoding="utf-8") as diag_fp:
        sample_writer = csv.DictWriter(sample_fp, fieldnames=sample_fields)
        pred_writer = csv.DictWriter(pred_fp, fieldnames=pred_fields)
        diag_writer = csv.DictWriter(diag_fp, fieldnames=diag_fields)
        sample_writer.writeheader()
        pred_writer.writeheader()
        diag_writer.writeheader()
        gt_fp = None
        gt_writer: Optional[csv.DictWriter] = None
        if not bool(args.no_ground_truth_csv):
            gt_fp = gt_csv_path.open("w", newline="", encoding="utf-8")
            gt_writer = csv.DictWriter(gt_fp, fieldnames=gt_fields)
            gt_writer.writeheader()
        try:
            model.eval()
            with torch.no_grad():
                for sample_indices, x_cpu, targets_cpu in _iter_batches(data_dir, shards, batch_size=int(args.batch_size), max_samples=int(args.max_samples)):
                    batch_n = int(x_cpu.shape[0])
                    non_blocking = str(device).startswith("cuda")
                    x = x_cpu.to(device=device, non_blocking=non_blocking)
                    targets = move_targets_to_device(targets_cpu, device, non_blocking=non_blocking)
                    outputs = model(x, targets["peak_time"], targets["peak_amp"], targets["peak_valid"])
                    _, loss_metrics = peak_slot_set_loss(
                        outputs,
                        targets,
                        no_object_weight=float(args.no_object_weight),
                        none_weight=float(args.none_weight),
                        matcher=str(args.matcher),
                        close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                        close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                        close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                        close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                        collect_metrics=True,
                    )
                    det_metrics = peak_slot_detection_metrics(
                        outputs,
                        targets,
                        objectness_threshold=float(args.objectness_threshold),
                        point_threshold=float(args.metric_point_threshold),
                        matcher=str(args.matcher),
                        close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                        close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                        close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                        close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                    )
                    det_metrics.update(
                        peak_slot_physics_metrics(
                            outputs,
                            targets,
                            objectness_threshold=float(args.objectness_threshold),
                            peak_threshold=float(args.peak_threshold),
                            speed_min_kmh=float(args.viterbi_speed_min_kmh),
                            speed_max_kmh=float(args.viterbi_speed_max_kmh),
                            time_downsample=time_downsample,
                            fs=fs,
                            dx_m=dx_m,
                        )
                    )
                    _weighted_add(loss_sums, loss_weights, loss_metrics, batch_n)
                    _weighted_add(metric_sums, metric_weights, det_metrics, batch_n)
                    outputs_cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}
                    obj_cpu = torch.sigmoid(outputs_cpu["objectness_logits"])
                    for b, sample_index in enumerate(sample_indices):
                        conflict_diagnostics: dict[str, Any] = {}
                        predictions = _active_predictions(
                            outputs_cpu,
                            targets_cpu,
                            b,
                            objectness_threshold=float(args.objectness_threshold),
                            peak_threshold=float(args.peak_threshold),
                            min_visible_channels=int(args.min_visible_channels),
                            max_predicted_tracks=int(args.max_predicted_tracks),
                            speed_norm_kmh=float(speed_norm_kmh),
                            fs=fs,
                            dx_m=dx_m,
                            time_downsample=time_downsample,
                            inference_config=inference_config,
                            sample_index=int(sample_index),
                            diagnostics=conflict_diagnostics,
                        )
                        predictions = _filter_predictions_by_direction(predictions, str(args.prediction_direction_filter))
                        fusion_diagnostics: dict[str, Any] = {}
                        predictions, fusion_diagnostics = _apply_graph_fusion_to_predictions(
                            predictions,
                            x_cpu[b, 0],
                            targets_cpu,
                            b,
                            fs=fs,
                            dx_m=dx_m,
                            time_downsample=time_downsample,
                            args=args,
                        )
                        raw_candidate_count = int(conflict_diagnostics.get("raw_candidate_count", len(predictions)))
                        duplicate_removed_count = int(conflict_diagnostics.get("duplicate_removed_count", 0))
                        close_pair_kept_count = int(conflict_diagnostics.get("close_pair_kept_count", 0))
                        fusion_added_point_count = int(fusion_diagnostics.get("fusion_added_point_count", 0))
                        raw_candidate_total += raw_candidate_count
                        duplicate_removed_total += duplicate_removed_count
                        close_pair_kept_total += close_pair_kept_count
                        fusion_added_point_total += fusion_added_point_count
                        fusion_input_track_total += int(fusion_diagnostics.get("fusion_input_track_count", len(predictions)))
                        fusion_output_track_total += int(fusion_diagnostics.get("fusion_output_track_count", len(predictions)))
                        for row in conflict_diagnostics.get("rows", []):
                            diag_writer.writerow(row)
                        gt_count = int(targets_cpu["gt_valid"][b].sum().item())
                        pred_count = int(len(predictions))
                        decoded_counts = _decoded_physics_counts(
                            predictions,
                            targets_cpu,
                            b,
                            window_seconds=float(meta.get("window_seconds", 1.0)),
                            dx_m=dx_m,
                            speed_min_kmh=float(args.viterbi_speed_min_kmh),
                            speed_max_kmh=float(args.viterbi_speed_max_kmh),
                        )
                        decoded_pair_total += int(decoded_counts["pair_count"])
                        decoded_direction_bad += int(decoded_counts["direction_bad"])
                        decoded_speed_bad += int(decoded_counts["speed_bad"])
                        decoded_triple_total += int(decoded_counts["triple_count"])
                        decoded_smooth_bad += int(decoded_counts["smooth_bad"])
                        peak_use_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
                        for pred in predictions:
                            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"]):
                                peak_use_counts[(int(ch), int(peak_idx))] += 1
                        decoded_peak_use_total += sum(peak_use_counts.values())
                        decoded_peak_conflict_total += sum(max(0, count - 1) for count in peak_use_counts.values())
                        filtered_gt_total += gt_count
                        filtered_pred_total += pred_count
                        filtered_count_abs_error += abs(pred_count - gt_count)
                        filtered_count_exact += int(pred_count == gt_count)
                        filtered_det_counts = _filtered_detection_counts(
                            predictions,
                            targets_cpu,
                            b,
                            point_threshold=float(args.metric_point_threshold),
                            min_visible_channels=int(args.min_visible_channels),
                        )
                        for key, value in filtered_det_counts.items():
                            filtered_det_sums[key] += float(value)
                        top_score = float(predictions[0]["score"]) if predictions else float("nan")
                        sample_writer.writerow(
                            {
                                "sample_index": int(sample_index),
                                "gt_count": gt_count,
                                "raw_candidate_count": raw_candidate_count,
                                "pred_count": pred_count,
                                "duplicate_removed_count": duplicate_removed_count,
                                "close_pair_kept_count": close_pair_kept_count,
                                "fusion_added_point_count": fusion_added_point_count,
                                "max_objectness": f"{float(obj_cpu[b].max().item()):.6f}",
                                "mean_objectness": f"{float(obj_cpu[b].mean().item()):.6f}",
                                "top_score": "" if not math.isfinite(top_score) else f"{top_score:.6f}",
                            }
                        )
                        if int(args.max_csv_samples) > 0 and csv_sample_count < int(args.max_csv_samples):
                            _write_prediction_rows(pred_writer, sample_index=int(sample_index), predictions=predictions, targets_cpu=targets_cpu, batch_index=b)
                            if gt_writer is not None:
                                _write_ground_truth_rows(gt_writer, sample_index=int(sample_index), targets_cpu=targets_cpu, batch_index=b)
                            csv_sample_count += 1
                        if int(args.plot_samples) > 0 and plot_sample_count < int(args.plot_samples):
                            plot_predictions = _filter_predictions_by_direction(predictions, str(args.plot_direction_filter))
                            _plot_sample_overlay(
                                plots_dir / f"sample_{int(sample_index):06d}.png",
                                heatmap=x_cpu[b, 0],
                                sample_index=int(sample_index),
                                predictions=plot_predictions,
                                targets_cpu=targets_cpu,
                                batch_index=b,
                                window_seconds=float(meta.get("window_seconds", 1.0)),
                                dx_m=float(meta.get("dx_m", 0.0)),
                                plot_style=str(args.plot_style),
                                dpi=int(args.plot_dpi),
                            )
                            plot_sample_count += 1
                    sample_count += batch_n
                    batch_count += 1
        finally:
            if gt_fp is not None:
                gt_fp.close()
    filtered_count_mae = float(filtered_count_abs_error / max(1, sample_count))
    filtered_count_acc = float(filtered_count_exact / max(1, sample_count))
    filtered_tp = float(filtered_det_sums["tp"])
    filtered_pred = float(filtered_det_sums["pred"])
    filtered_gt = float(filtered_det_sums["gt"])
    filtered_precision = filtered_tp / max(1.0, filtered_pred)
    filtered_recall = filtered_tp / max(1.0, filtered_gt)
    filtered_f1 = 2.0 * filtered_precision * filtered_recall / max(1e-12, filtered_precision + filtered_recall)
    summary = {
        "mode": "peak_slot_tensor_shard_prediction",
        "data_dir": str(data_dir),
        "model": str(Path(args.model).expanduser()),
        "model_family": str(checkpoint.get("model_family", "peak_slot")),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "device": device,
        "speed_norm_kmh": float(speed_norm_kmh),
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "thresholds": {
            "objectness_threshold": float(args.objectness_threshold),
            "peak_threshold": float(args.peak_threshold),
            "min_visible_channels": int(args.min_visible_channels),
            "metric_point_threshold": float(args.metric_point_threshold),
        },
        "decoder": {
            "use_viterbi_decoder": bool(inference_config.use_viterbi_decoder),
            "decoder_mode": str(inference_config.decoder_mode),
            "viterbi_topk": int(inference_config.viterbi_topk),
            "viterbi_beam_size": int(inference_config.viterbi_beam_size),
            "time_prior_weight": float(inference_config.time_prior_weight),
            "global_conflict_penalty": float(inference_config.global_conflict_penalty),
            "conflict_mode": str(inference_config.conflict_mode),
            "duplicate_overlap_ratio": float(inference_config.duplicate_overlap_ratio),
            "min_unique_support_channels": int(inference_config.min_unique_support_channels),
            "extra_candidate_slots": int(inference_config.extra_candidate_slots),
            "candidate_objectness_floor": float(inference_config.candidate_objectness_floor),
            "viterbi_candidate_threshold": float(inference_config.viterbi_candidate_threshold),
            "viterbi_speed_min_kmh": float(inference_config.viterbi_speed_min_kmh),
            "viterbi_speed_max_kmh": float(inference_config.viterbi_speed_max_kmh),
            "viterbi_max_skip_channels": int(inference_config.viterbi_max_skip_channels),
            "viterbi_inertia_penalty": float(inference_config.viterbi_inertia_penalty),
            "viterbi_slope_memory": float(inference_config.viterbi_slope_memory),
        },
        "fusion": {
            "fusion_mode": str(args.fusion_mode),
            "fusion_min_seed_channels": int(args.fusion_min_seed_channels),
            "fusion_extend_left": bool(args.fusion_extend_left),
            "fusion_extend_right": bool(args.fusion_extend_right),
            "fusion_graph_prominence": float(args.fusion_graph_prominence),
            "fusion_graph_min_peak_distance": int(args.fusion_graph_min_peak_distance),
            "fusion_graph_max_skip_channels": int(args.fusion_graph_max_skip_channels),
            "fusion_min_added_channels": int(args.fusion_min_added_channels),
            "fusion_nms_tolerance_samples": int(args.fusion_nms_tolerance_samples),
            "fusion_bridge_search_radius_samples": int(args.fusion_bridge_search_radius_samples),
            "fusion_input_track_count": int(fusion_input_track_total),
            "fusion_output_track_count": int(fusion_output_track_total),
            "fusion_added_point_count": int(fusion_added_point_total),
        },
        "filters": {
            "prediction_direction_filter": str(args.prediction_direction_filter),
            "plot_direction_filter": str(args.plot_direction_filter),
        },
        "loss_metrics": _weighted_mean(loss_sums, loss_weights),
        "batch_mean_detection_metrics": _weighted_mean(metric_sums, metric_weights),
        "filtered_count_metrics": {
            "raw_candidate_count": int(raw_candidate_total),
            "pred_count": int(filtered_pred_total),
            "gt_count": int(filtered_gt_total),
            "count_mae": filtered_count_mae,
            "count_acc": filtered_count_acc,
            "duplicate_removed_count": int(duplicate_removed_total),
            "close_pair_kept_count": int(close_pair_kept_total),
        },
        "filtered_detection_metrics": {
            "track_tp": filtered_tp,
            "pred_count": filtered_pred,
            "gt_count": filtered_gt,
            "track_precision": filtered_precision,
            "track_recall": filtered_recall,
            "track_f1": filtered_f1,
            "time_mae_norm": float(filtered_det_sums["time_error_sum"] / max(1.0, filtered_det_sums["time_error_count"])),
            "matched_common_channels_mean": float(filtered_det_sums["matched_common_channels"] / max(1.0, filtered_tp)),
            "track_recall_forward": float(filtered_det_sums["matched_forward"] / max(1.0, filtered_det_sums["gt_forward"])),
            "track_recall_reverse": float(filtered_det_sums["matched_reverse"] / max(1.0, filtered_det_sums["gt_reverse"])),
            "track_recall_short_visible": float(filtered_det_sums["matched_short"] / max(1.0, filtered_det_sums["gt_short"])),
            "track_recall_long_visible": float(filtered_det_sums["matched_long"] / max(1.0, filtered_det_sums["gt_long"])),
        },
        "filtered_physics_metrics": {
            "pair_count": int(decoded_pair_total),
            "direction_violation_rate": float(decoded_direction_bad / max(1, decoded_pair_total)),
            "speed_window_violation_rate": float(decoded_speed_bad / max(1, decoded_pair_total)),
            "smoothness_violation_rate": float(decoded_smooth_bad / max(1, decoded_triple_total)),
            "peak_conflict_rate": float(decoded_peak_conflict_total / max(1, decoded_peak_use_total)),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "sample_summary_csv": str(sample_summary_path),
            "prediction_diagnostics_csv": str(prediction_diagnostics_path),
            "predicted_tracks_csv": str(pred_csv_path),
            "ground_truth_tracks_csv": None if bool(args.no_ground_truth_csv) else str(gt_csv_path),
            "plots_dir": str(plots_dir) if int(args.plot_samples) > 0 else None,
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: samples={sample_count}, filtered_pred={filtered_pred_total}, gt={filtered_gt_total}, "
        f"count_mae={filtered_count_mae:.2f}, out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
