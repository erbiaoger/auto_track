from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.vehicle_peak_set_transformer import (
    DecodedPeakTrack,
    PeakSetInferenceConfig,
    _build_fused_peak_track,
    _fused_anchor_choice,
    _refine_decoded_track,
    _softmax_numpy,
    _track_line_distance,
    auto_torch_device,
    json_ready,
    load_checkpoint_model,
)


@dataclass
class RealPeakSample:
    sample_index: int
    shard_index: int
    local_index: int
    x: torch.Tensor
    full_time: torch.Tensor | None
    full_valid: torch.Tensor | None
    gt_valid: torch.Tensor | None
    peak_time: torch.Tensor | None
    peak_amp: torch.Tensor | None
    peak_valid: torch.Tensor | None
    peak_index: torch.Tensor | None


@dataclass
class AnchorPathStep:
    choice: int | None
    time_norm: float
    amp: float
    node_score: float
    observed: bool
    fused_score: float


class RealPeakShardDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).expanduser()
        meta_path = self.dataset_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"missing dataset meta: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.shard_paths = [self.dataset_dir / str(name) for name in self.meta.get("shards", [])]
        if not self.shard_paths:
            raise ValueError(f"no shards listed in {meta_path}")
        meta_sizes = self.meta.get("shard_sizes", [])
        if len(meta_sizes) == len(self.shard_paths):
            self.shard_sizes = [int(size) for size in meta_sizes]
        else:
            self.shard_sizes = []
            for shard_path in self.shard_paths:
                payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
                self.shard_sizes.append(int(payload["x"].shape[0]))
        self.total = int(sum(self.shard_sizes))
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return self.total

    def _resolve(self, global_index: int) -> tuple[int, int]:
        idx = int(global_index)
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        acc = 0
        for shard_idx, shard_size in enumerate(self.shard_sizes):
            next_acc = acc + int(shard_size)
            if idx < next_acc:
                return shard_idx, idx - acc
            acc = next_acc
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(str(self.shard_paths[shard_idx]), map_location="cpu", weights_only=False)
        return self._cache[shard_idx]

    @staticmethod
    def _target_field(payload: dict[str, Any], name: str) -> torch.Tensor | None:
        value = payload.get(name, None)
        if torch.is_tensor(value):
            return value
        targets = payload.get("targets", None)
        if isinstance(targets, dict):
            value = targets.get(name, None)
            if torch.is_tensor(value):
                return value
        return None

    def __getitem__(self, index: int) -> RealPeakSample:
        shard_idx, local_idx = self._resolve(int(index))
        payload = self._load_shard(shard_idx)
        targets = payload.get("targets", {}) if isinstance(payload.get("targets", None), dict) else {}
        full_time = targets.get("full_time", None)
        full_valid = targets.get("full_valid", None)
        gt_valid = targets.get("gt_valid", None)
        peak_time = self._target_field(payload, "peak_time")
        peak_amp = self._target_field(payload, "peak_amp")
        peak_valid = self._target_field(payload, "peak_valid")
        peak_index = self._target_field(payload, "peak_index")
        return RealPeakSample(
            sample_index=int(index),
            shard_index=int(shard_idx),
            local_index=int(local_idx),
            x=payload["x"][local_idx].to(torch.float32),
            full_time=full_time[local_idx].to(torch.float32) if torch.is_tensor(full_time) else None,
            full_valid=full_valid[local_idx].to(torch.float32) if torch.is_tensor(full_valid) else None,
            gt_valid=gt_valid[local_idx].to(torch.bool) if torch.is_tensor(gt_valid) else None,
            peak_time=peak_time[local_idx].to(torch.float32) if peak_time is not None else None,
            peak_amp=peak_amp[local_idx].to(torch.float32) if peak_amp is not None else None,
            peak_valid=peak_valid[local_idx].to(torch.bool) if peak_valid is not None else None,
            peak_index=peak_index[local_idx].to(torch.long) if peak_index is not None else None,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict vehicle peak-set tracks on exported real-data peak shards.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Real exported shard dataset directory")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, cpu, mps, auto")
    parser.add_argument("--sample-index", type=int, default=0, help="First sample index")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples; 0 means all remaining samples")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--plot-samples", type=int, default=10, help="Number of per-sample PNG overlays to write")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI")
    parser.add_argument("--plot-style", default="waveform", choices=["waveform", "heatmap"], help="Overlay base style")
    parser.add_argument("--objectness-threshold", type=float, default=0.25, help="Minimum query objectness")
    parser.add_argument("--complete-valid-threshold", type=float, default=0.40, help="Minimum completed-channel probability")
    parser.add_argument("--observed-threshold", type=float, default=0.45, help="Minimum observed-channel probability")
    parser.add_argument(
        "--use-observed-valid-in-decode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use observed_valid_logits as a decode gate/score. Default is off so the auxiliary head cannot suppress recall.",
    )
    parser.add_argument(
        "--observed-valid-soft-floor",
        type=float,
        default=0.6,
        help="Soft observed decode floor. With 0.6, peak scores are multiplied by 0.6 + 0.4 * observed_valid.",
    )
    parser.add_argument("--min-visible-channels", type=int, default=5, help="Minimum channels in a decoded track")
    parser.add_argument("--min-peak-support-channels", type=int, default=3, help="Minimum nearby detected peak candidates required to keep a track")
    parser.add_argument("--min-peak-support-ratio", type=float, default=0.15, help="Minimum fraction of completed channels with nearby detected peaks")
    parser.add_argument("--fused-min-anchor-score", type=float, default=0.38, help="Minimum fused anchor score for a peak-guided point to count as observed support")
    parser.add_argument("--max-tracks", type=int, default=16, help="Maximum tracks per sample before de-duplication")
    parser.add_argument("--dedup-tolerance-samples", type=int, default=30, help="Line de-duplication tolerance")
    parser.add_argument("--dedup-line-tolerance-s", type=float, default=0.8, help="Strict line-distance tolerance in seconds for duplicate tracks")
    parser.add_argument("--dedup-loose-line-tolerance-s", type=float, default=1.5, help="Loose line-distance tolerance in seconds when speed is also similar")
    parser.add_argument("--dedup-speed-tolerance-kmh", type=float, default=10.0, help="Speed tolerance for loose duplicate removal")
    parser.add_argument("--min-track-observed-ratio", type=float, default=0.0, help="Minimum fraction of track points supported by observed peak anchors")
    parser.add_argument("--min-track-total-score", type=float, default=0.0, help="Minimum summed point score for a decoded track")
    parser.add_argument(
        "--anchor-path-refine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use continuous peak-path decoding for peak-guided tracks",
    )
    parser.add_argument("--anchor-path-top-k", type=int, default=5, help="Number of observed candidates kept per channel for continuous decoding")
    parser.add_argument(
        "--anchor-path-transition-weight",
        type=float,
        default=8.0,
        help="Penalty for per-step time jumps away from the fitted baseline",
    )
    parser.add_argument(
        "--anchor-path-second-diff-weight",
        type=float,
        default=2.5,
        help="Penalty for curvature in the selected peak path",
    )
    parser.add_argument(
        "--anchor-path-baseline-weight",
        type=float,
        default=1.5,
        help="Penalty for deviating from the fitted completed-time baseline",
    )
    parser.add_argument(
        "--anchor-path-robust-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit a robust line from peak candidates before dynamic-programming anchor selection",
    )
    parser.add_argument(
        "--anchor-path-robust-baseline-weight",
        type=float,
        default=2.5,
        help="Extra penalty for deviating from the robust peak-candidate baseline",
    )
    parser.add_argument(
        "--anchor-path-robust-baseline-tolerance-s",
        type=float,
        default=2.0,
        help="Inlier tolerance used when fitting the robust peak-candidate baseline",
    )
    parser.add_argument(
        "--anchor-path-robust-baseline-min-support",
        type=int,
        default=4,
        help="Minimum candidate-supported channels required to use the robust peak-candidate baseline",
    )
    parser.add_argument(
        "--anchor-path-max-adjacent-jump-s",
        type=float,
        default=8.0,
        help="Hard cap on adjacent time jumps allowed during continuous anchor search",
    )
    parser.add_argument(
        "--observed-outlier-repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downgrade observed anchor points that deviate too far from the fitted line",
    )
    parser.add_argument("--observed-outlier-residual-s", type=float, default=4.0, help="Residual threshold for observed outlier repair")
    parser.add_argument("--observed-outlier-adjacent-s", type=float, default=6.0, help="Adjacent residual-jump threshold for observed outlier repair")
    parser.add_argument("--snap-to-candidates", action=argparse.BooleanOptionalAction, default=True, help="Snap observed points to nearest detected peak candidate")
    parser.add_argument("--snap-tolerance-s", type=float, default=0.25, help="Maximum peak-candidate snap distance in seconds")
    parser.add_argument("--postprocess", action=argparse.BooleanOptionalAction, default=True, help="Filter and smooth decoded tracks")
    parser.add_argument("--postprocess-refit-missing", action=argparse.BooleanOptionalAction, default=True, help="Refit missing/completed points to a robust line")
    parser.add_argument("--postprocess-refit-observed", action=argparse.BooleanOptionalAction, default=False, help="Also refit observed anchor points to the robust line")
    parser.add_argument(
        "--postprocess-repair-outlier-points",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repair or drop per-point trajectory outliers before whole-track validation",
    )
    parser.add_argument("--postprocess-drop-point-residual-s", type=float, default=8.0, help="Drop a point when its line residual exceeds this threshold")
    parser.add_argument("--postprocess-repair-point-residual-s", type=float, default=4.0, help="Refit a point when its line residual exceeds this threshold")
    parser.add_argument("--postprocess-drop-point-second-diff-s", type=float, default=6.0, help="Drop/repair a point causing a large local second-difference jump")
    parser.add_argument(
        "--postprocess-extend-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extend incomplete tracks along the fitted speed line into missing head/tail/interior channels",
    )
    parser.add_argument("--postprocess-extend-max-channels", type=int, default=8, help="Maximum channels to extend beyond each track end")
    parser.add_argument("--postprocess-extend-min-observed", type=int, default=4, help="Minimum observed points needed before line-extension completion")
    parser.add_argument(
        "--postprocess-validate-observed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate line residuals on observed peak-supported points when available instead of rejecting tracks because of bad completed points",
    )
    parser.add_argument("--speed-min-kmh", type=float, default=40.0, help="Minimum accepted postprocessed track speed")
    parser.add_argument("--speed-max-kmh", type=float, default=180.0, help="Maximum accepted postprocessed track speed")
    parser.add_argument("--segment-speed-min-kmh", type=float, default=20.0, help="Minimum accepted adjacent segment speed")
    parser.add_argument("--segment-speed-max-kmh", type=float, default=240.0, help="Maximum accepted adjacent segment speed")
    parser.add_argument("--max-line-residual-s", type=float, default=8.0, help="Maximum robust-line residual for a decoded track")
    parser.add_argument("--max-adjacent-residual-s", type=float, default=6.0, help="Maximum adjacent residual jump after robust-line fitting")
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _collate_real_samples(items: list[RealPeakSample]) -> dict[str, Any]:
    raw_windows = [item.x[0] if item.x.ndim == 3 else item.x for item in items]
    return {
        "sample_index": [item.sample_index for item in items],
        "shard_index": [item.shard_index for item in items],
        "local_index": [item.local_index for item in items],
        "x": torch.stack([item.x for item in items], dim=0),
        "full_time": torch.stack([item.full_time for item in items], dim=0) if items and items[0].full_time is not None else None,
        "full_valid": torch.stack([item.full_valid for item in items], dim=0) if items and items[0].full_valid is not None else None,
        "gt_valid": torch.stack([item.gt_valid for item in items], dim=0) if items and items[0].gt_valid is not None else None,
        "peak_time": torch.stack([item.peak_time for item in items], dim=0) if items and items[0].peak_time is not None else None,
        "peak_amp": torch.stack([item.peak_amp for item in items], dim=0) if items and items[0].peak_amp is not None else None,
        "peak_valid": torch.stack([item.peak_valid for item in items], dim=0) if items and items[0].peak_valid is not None else None,
        "peak_index": torch.stack([item.peak_index for item in items], dim=0) if items and items[0].peak_index is not None else None,
        "raw_window": torch.stack(raw_windows, dim=0) if items else None,
    }


def _sample_indices(total: int, start: int, count: int) -> list[int]:
    start_idx = int(max(0, start))
    if start_idx >= int(total):
        return []
    if int(count) <= 0:
        end = int(total)
    else:
        end = min(int(total), start_idx + int(count))
    return list(range(start_idx, end))


def _window_start(meta: dict[str, Any], sample_index: int) -> tuple[int | None, float | None]:
    sample_idx = int(sample_index)
    starts = meta.get("window_start_samples", [])
    start_seconds = meta.get("window_start_seconds", [])
    sample_start = int(starts[sample_idx]) if isinstance(starts, list) and sample_idx < len(starts) else None
    second_start = float(start_seconds[sample_idx]) if isinstance(start_seconds, list) and sample_idx < len(start_seconds) else None
    return sample_start, second_start


def _nearest_candidate(
    *,
    ch: int,
    t_norm: float,
    peak_time: np.ndarray | None,
    peak_amp: np.ndarray | None,
    peak_valid: np.ndarray | None,
    window_seconds: float,
    tolerance_s: float,
) -> tuple[float, float, bool]:
    if peak_time is None or peak_valid is None:
        return float(t_norm), 0.0, False
    valid = np.asarray(peak_valid[int(ch)], dtype=bool)
    if valid.size == 0 or not bool(valid.any()):
        return float(t_norm), 0.0, False
    times = np.asarray(peak_time[int(ch)], dtype=np.float32)
    candidates = times[valid]
    if candidates.size == 0:
        return float(t_norm), 0.0, False
    distances = np.abs(candidates - float(t_norm))
    best = int(np.argmin(distances))
    if float(distances[best]) * float(window_seconds) > float(tolerance_s):
        return float(t_norm), 0.0, False
    if peak_amp is not None:
        amps = np.asarray(peak_amp[int(ch)], dtype=np.float32)[valid]
        amp = float(amps[best]) if amps.size > best else 0.0
    else:
        amp = 0.0
    return float(candidates[best]), amp, True


def _fit_complete_time_baseline(
    point_channels: list[int],
    complete_time: np.ndarray,
    x_axis_m: np.ndarray,
    *,
    window_seconds: float,
    fs: float,
) -> tuple[float, float] | None:
    points: list[TrackPoint] = []
    n_samples = int(max(1, round(float(window_seconds) * float(fs))))
    for ch in point_channels:
        t_norm = float(np.clip(float(complete_time[int(ch)]), 0.0, 1.0))
        t_idx = int(round(t_norm * float(max(1, n_samples - 1))))
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(max(1e-9, fs)),
                offset_m=float(x_axis_m[int(ch)]),
                amp=0.0,
                score=1.0,
            )
        )
    return _fit_track_line(points)


def _score_fused_anchor_candidates(
    *,
    anchor_logits: np.ndarray,
    peak_time: np.ndarray,
    peak_amp: np.ndarray,
    peak_valid: np.ndarray,
    raw_window: np.ndarray | None,
    channel_index: int,
    complete_time: float,
    window_seconds: float,
    cfg: PeakSetInferenceConfig,
    top_k: int,
) -> list[AnchorPathStep]:
    valid = np.asarray(peak_valid, dtype=bool)
    if valid.size == 0 or not bool(valid.any()):
        return []

    probs = _softmax_numpy(np.asarray(anchor_logits, dtype=np.float32))
    valid_idx = np.where(valid)[0]
    amp_values = np.abs(np.asarray(peak_amp, dtype=np.float32))[valid_idx]
    amp_denom = max(1e-6, float(np.max(amp_values)) if amp_values.size > 0 else 1.0)
    time_tol = max(1e-6, float(cfg.fused_time_tolerance_s))
    raw_row = None if raw_window is None else np.asarray(raw_window[int(channel_index)], dtype=np.float32)
    raw_denom = max(1e-6, float(np.max(np.abs(raw_row))) if raw_row is not None and raw_row.size > 0 else 1.0)

    candidates: list[AnchorPathStep] = []
    for choice in valid_idx:
        cand_time = float(np.clip(float(peak_time[choice]), 0.0, 1.0))
        prob = float(probs[choice]) if choice < int(probs.shape[0]) else 0.0
        if raw_row is not None and raw_row.size > 0:
            cand_idx = int(round(cand_time * float(max(1, raw_row.shape[0] - 1))))
            cand_idx = int(max(0, min(raw_row.shape[0] - 1, cand_idx)))
            signal = float(np.clip(abs(float(raw_row[cand_idx])) / raw_denom, 0.0, 1.0))
        else:
            signal = float(np.clip(abs(float(peak_amp[choice])) / amp_denom, 0.0, 1.0))
        time_score = float(np.exp(-abs(cand_time - float(complete_time)) * float(window_seconds) / time_tol))
        fused = (
            float(cfg.fused_anchor_weight) * prob
            + float(cfg.fused_signal_weight) * signal
            + float(cfg.fused_time_weight) * time_score
        )
        candidates.append(
            AnchorPathStep(
                choice=int(choice),
                time_norm=float(cand_time),
                amp=float(peak_amp[choice]),
                node_score=float(fused),
                observed=True,
                fused_score=float(fused),
            )
        )

    candidates.sort(key=lambda item: float(item.fused_score), reverse=True)
    if int(top_k) > 0:
        candidates = candidates[: int(top_k)]
    return candidates


def _fit_robust_anchor_baseline(
    *,
    point_channels: list[int],
    candidate_options: list[list[AnchorPathStep]],
    complete_time: np.ndarray,
    complete_prob: np.ndarray,
    window_seconds: float,
    tolerance_s: float,
    min_support: int,
) -> tuple[np.ndarray, int] | None:
    anchors: list[tuple[int, float, float]] = []
    complete_time = np.asarray(complete_time, dtype=np.float32)
    complete_prob = np.asarray(complete_prob, dtype=np.float32)
    for ch, options in zip(point_channels, candidate_options):
        observed = [opt for opt in options if bool(opt.observed)]
        if not observed:
            continue
        best = max(observed, key=lambda item: float(item.fused_score))
        score = float(best.fused_score) * float(complete_prob[int(ch)])
        anchors.append((int(ch), float(best.time_norm) * float(window_seconds), score))

    if len(anchors) < int(min_support):
        return None

    channels = np.asarray([item[0] for item in anchors], dtype=np.float64)
    times_s = np.asarray([item[1] for item in anchors], dtype=np.float64)
    scores = np.asarray([max(1e-6, item[2]) for item in anchors], dtype=np.float64)
    complete_s = np.asarray(
        [float(np.clip(float(complete_time[int(ch)]), 0.0, 1.0)) * float(window_seconds) for ch in channels],
        dtype=np.float64,
    )

    best_score = -np.inf
    best_fit: tuple[float, float] | None = None
    best_support = 0
    tol = float(max(1e-6, tolerance_s))
    n = len(anchors)
    min_gap = max(3, n // 4)
    pair_indices: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + min_gap, n):
            dc = float(channels[j] - channels[i])
            if abs(dc) < 1e-6:
                continue
            pair_indices.append((float(scores[i] + scores[j]), i, j))
    pair_indices.sort(key=lambda item: item[0], reverse=True)
    for _pair_score, i, j in pair_indices[:96]:
        dc = float(channels[j] - channels[i])
        slope = float((times_s[j] - times_s[i]) / dc)
        intercept = float(times_s[i] - slope * channels[i])
        pred = slope * channels + intercept
        resid = np.abs(times_s - pred)
        prior_resid = np.abs(complete_s - pred)
        inlier = resid <= tol
        support = int(np.sum(inlier))
        if support < int(min_support):
            continue
        weighted_support = float(np.sum(scores[inlier] * (1.0 - np.minimum(resid[inlier] / tol, 1.0))))
        prior_penalty = float(np.median(np.minimum(prior_resid, tol * 2.0))) / max(tol, 1e-6)
        score = weighted_support - 0.35 * prior_penalty
        if score > best_score:
            best_score = score
            best_fit = (slope, intercept)
            best_support = support

    if best_fit is None:
        return None

    slope, intercept = best_fit
    pred_all_s = np.asarray([slope * float(ch) + intercept for ch in point_channels], dtype=np.float64)
    pred_norm = np.clip(pred_all_s / float(max(1e-9, window_seconds)), 0.0, 1.0)
    return pred_norm, int(best_support)


def _fit_track_line(points: list[TrackPoint]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    ch = np.asarray([int(p.ch_idx) for p in points], dtype=np.float64)
    ts = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
    if ch.size < 2 or float(np.ptp(ch)) <= 0.0:
        return None
    if ch.size >= 4:
        slope, intercept = np.polyfit(ch, ts, deg=1)
        residual = ts - (slope * ch + intercept)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        scale = max(0.5, 1.4826 * mad)
        keep = np.abs(residual) <= 3.5 * scale
        if int(np.sum(keep)) >= 2:
            ch = ch[keep]
            ts = ts[keep]
    slope, intercept = np.polyfit(ch, ts, deg=1)
    return float(slope), float(intercept)


def _track_segment_speeds_kmh(points: list[TrackPoint], dx_m: float) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((0,), dtype=np.float64)
    ordered = sorted(points, key=lambda p: int(p.ch_idx))
    ch = np.asarray([int(p.ch_idx) for p in ordered], dtype=np.float64)
    ts = np.asarray([float(p.time_s) for p in ordered], dtype=np.float64)
    dch = np.diff(ch)
    dt = np.diff(ts)
    valid = (np.abs(dt) > 1e-9) & (np.abs(dch) > 0.0)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.float64)
    return np.abs(dch[valid]) * float(dx_m) / np.abs(dt[valid]) * 3.6


def _refit_point_from_line(point: TrackPoint, *, slope: float, intercept: float, fs: float, n_samples: int, observed: bool, score: float | None = None) -> TrackPoint:
    time_s = float(slope * float(point.ch_idx) + intercept)
    time_s = float(np.clip(time_s, 0.0, float(max(1, n_samples - 1)) / float(max(1e-9, fs))))
    t_idx = int(round(time_s * float(fs)))
    t_idx = int(np.clip(t_idx, 0, max(0, int(n_samples) - 1)))
    return TrackPoint(
        ch_idx=int(point.ch_idx),
        t_idx=t_idx,
        time_s=time_s,
        offset_m=float(point.offset_m),
        amp=float(point.amp) if bool(observed) else 0.0,
        score=float(point.score if score is None else score),
    )


def _make_completed_point(
    *,
    ch_idx: int,
    slope: float,
    intercept: float,
    fs: float,
    dx_m: float,
    n_samples: int,
    score: float,
) -> TrackPoint | None:
    time_s = float(slope * float(ch_idx) + intercept)
    max_time_s = float(max(1, n_samples - 1)) / float(max(1e-9, fs))
    if time_s < 0.0 or time_s > max_time_s:
        return None
    t_idx = int(round(time_s * float(fs)))
    t_idx = int(np.clip(t_idx, 0, max(0, int(n_samples) - 1)))
    return TrackPoint(
        ch_idx=int(ch_idx),
        t_idx=t_idx,
        time_s=float(t_idx) / float(max(1e-9, fs)),
        offset_m=float(ch_idx) * float(dx_m),
        amp=0.0,
        score=float(score),
    )


def _repair_observed_outliers(
    item: DecodedPeakTrack,
    *,
    fs: float,
    dx_m: float,
    n_samples: int,
    observed_outlier_residual_s: float,
    observed_outlier_adjacent_s: float,
    refit_missing: bool,
    refit_observed: bool,
) -> DecodedPeakTrack | None:
    points = sorted(item.track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return None

    observed = np.asarray(item.observed_valid, dtype=bool)
    if observed.size != len(points):
        observed = np.zeros((len(points),), dtype=bool)
    observed_idx = np.where(observed)[0]
    if int(observed_idx.size) < 3:
        return item

    fit_points = [points[idx] for idx in observed_idx.tolist()]
    fit = _fit_track_line(fit_points)
    if fit is None:
        return item
    slope, intercept = fit
    if abs(float(slope)) <= 1e-9:
        return item

    ch_obs = np.asarray([int(points[idx].ch_idx) for idx in observed_idx], dtype=np.float64)
    ts_obs = np.asarray([float(points[idx].time_s) for idx in observed_idx], dtype=np.float64)
    resid = ts_obs - (slope * ch_obs + intercept)
    outlier_mask = np.abs(resid) > float(observed_outlier_residual_s)
    if resid.size >= 2:
        jump = np.abs(np.diff(resid))
        for offset, jump_val in enumerate(jump, start=1):
            if float(jump_val) <= float(observed_outlier_adjacent_s):
                continue
            left = int(observed_idx[offset - 1])
            right = int(observed_idx[offset])
            if abs(float(resid[offset])) >= abs(float(resid[offset - 1])):
                outlier_mask[offset] = True
            else:
                outlier_mask[offset - 1] = True

    if not bool(np.any(outlier_mask)):
        return item

    repaired_observed = observed.copy()
    outlier_indices = np.zeros((len(points),), dtype=bool)
    for pos, is_outlier in enumerate(np.asarray(outlier_mask, dtype=bool)):
        if not bool(is_outlier):
            continue
        idx = int(observed_idx[pos])
        outlier_indices[idx] = True

    repaired_scores: list[float] = []
    repaired_amps: list[float] = []
    for idx, point in enumerate(points):
        score = float(point.score)
        amp = float(point.amp)
        if bool(observed[idx]) and bool(outlier_indices[idx]):
            repaired_observed[idx] = False
            amp = 0.0
            score = float(point.score) * 0.35
        repaired_scores.append(float(score))
        repaired_amps.append(float(amp))

    remaining_fit_points = [p for p, keep in zip(points, repaired_observed) if bool(keep)]
    if len(remaining_fit_points) >= 2:
        fit = _fit_track_line(remaining_fit_points)
        if fit is None:
            return item
        slope, intercept = fit
    else:
        return item

    refined_points: list[TrackPoint] = []
    for idx, point in enumerate(points):
        is_observed = bool(repaired_observed[idx])
        use_fit = bool(refit_observed) or (bool(refit_missing) and not is_observed)
        if use_fit:
            time_s = float(slope * float(point.ch_idx) + intercept)
            time_s = float(np.clip(time_s, 0.0, float(max(1, n_samples - 1)) / float(max(1e-9, fs))))
            t_idx = int(round(time_s * float(fs)))
            t_idx = int(np.clip(t_idx, 0, max(0, int(n_samples) - 1)))
            amp = float(repaired_amps[idx]) if is_observed else 0.0
            score = float(repaired_scores[idx])
        else:
            t_idx = int(point.t_idx)
            time_s = float(point.time_s)
            amp = float(repaired_amps[idx])
            score = float(repaired_scores[idx])
        refined_points.append(
            TrackPoint(
                ch_idx=int(point.ch_idx),
                t_idx=t_idx,
                time_s=time_s,
                offset_m=float(point.offset_m),
                amp=amp,
                score=score,
            )
        )

    out_speed = abs(float(dx_m) / float(slope)) * 3.6
    return DecodedPeakTrack(
        track=Track(
            track_id=int(item.track.track_id),
            direction="forward" if slope >= 0.0 else "reverse",
            points=refined_points,
            total_score=float(sum(float(p.score) for p in refined_points)),
            mean_speed_kmh=float(out_speed),
        ),
        objectness=float(item.objectness),
        query_index=int(item.query_index),
        channel_indices=np.asarray([int(p.ch_idx) for p in refined_points], dtype=np.int64),
        complete_valid=np.asarray(item.complete_valid, dtype=np.float32),
        observed_valid=repaired_observed.astype(np.float32),
        point_times_norm=np.asarray([float(p.time_s) / max(1e-9, float(n_samples) / float(fs)) for p in refined_points], dtype=np.float32),
    )


def _build_continuous_peak_guided_track(
    *,
    objectness: float,
    query_index: int,
    direction: int,
    complete_prob: np.ndarray,
    observed_prob: np.ndarray | None,
    complete_time: np.ndarray,
    anchor_logits: np.ndarray,
    peak_time: np.ndarray,
    peak_amp: np.ndarray,
    peak_valid: np.ndarray,
    raw_window: np.ndarray | None,
    fs: float,
    x_axis_m: np.ndarray,
    cfg: PeakSetInferenceConfig,
    window_seconds: float,
    top_k: int,
    transition_weight: float,
    second_diff_weight: float,
    baseline_weight: float,
    robust_baseline: bool,
    robust_baseline_weight: float,
    robust_baseline_tolerance_s: float,
    robust_baseline_min_support: int,
    max_adjacent_jump_s: float,
) -> DecodedPeakTrack | None:
    complete_mask = np.asarray(complete_prob, dtype=np.float32) >= float(cfg.complete_valid_threshold)
    if int(np.sum(complete_mask)) < int(cfg.min_visible_channels):
        return None

    point_channels = np.where(complete_mask)[0].tolist()
    if len(point_channels) < 2:
        return None

    baseline_fit = _fit_complete_time_baseline(
        point_channels,
        np.asarray(complete_time, dtype=np.float32),
        np.asarray(x_axis_m, dtype=np.float32),
        window_seconds=float(window_seconds),
        fs=float(fs),
    )
    if baseline_fit is None:
        baseline_times = np.asarray([float(np.clip(float(complete_time[ch]), 0.0, 1.0)) for ch in point_channels], dtype=np.float64)
    else:
        slope, intercept = baseline_fit
        baseline_times = np.asarray(
            [float(np.clip((slope * float(ch) + intercept) / float(max(1e-9, window_seconds)), 0.0, 1.0)) for ch in point_channels],
            dtype=np.float64,
        )
    baseline_dt = np.diff(baseline_times) if baseline_times.size >= 2 else np.zeros((0,), dtype=np.float64)
    max_jump_norm = float(max_adjacent_jump_s) / float(max(1e-9, window_seconds))

    options: list[list[AnchorPathStep]] = []
    for idx, ch in enumerate(point_channels):
        channel_options: list[AnchorPathStep] = []
        missing_time = float(np.clip(float(complete_time[ch]), 0.0, 1.0))
        missing_score = float(objectness) * float(complete_prob[ch]) * float(cfg.fused_missing_point_scale)
        observed_score = float(observed_prob[ch]) if observed_prob is not None else 1.0
        observed_floor = float(np.clip(float(getattr(cfg, "observed_valid_soft_floor", 0.6)), 0.0, 1.0))
        observed_soft = observed_floor + (1.0 - observed_floor) * float(np.clip(observed_score, 0.0, 1.0))
        channel_options.append(
            AnchorPathStep(
                choice=None,
                time_norm=missing_time,
                amp=0.0,
                node_score=missing_score,
                observed=False,
                fused_score=missing_score,
            )
        )
        candidates = _score_fused_anchor_candidates(
            anchor_logits=np.asarray(anchor_logits[ch], dtype=np.float32),
            peak_time=np.asarray(peak_time[ch], dtype=np.float32),
            peak_amp=np.asarray(peak_amp[ch], dtype=np.float32),
            peak_valid=np.asarray(peak_valid[ch], dtype=bool),
            raw_window=raw_window,
            channel_index=int(ch),
            complete_time=float(missing_time),
            window_seconds=float(window_seconds),
            cfg=cfg,
            top_k=int(top_k),
        )
        for cand in candidates:
            if float(cand.fused_score) <= 0.0:
                continue
            channel_options.append(
                AnchorPathStep(
                    choice=int(cand.choice) if cand.choice is not None else None,
                    time_norm=float(cand.time_norm),
                    amp=float(cand.amp),
                    node_score=float(objectness) * float(complete_prob[ch]) * float(observed_soft) * max(float(cand.fused_score), 0.25),
                    observed=True,
                    fused_score=float(cand.fused_score) * float(observed_soft),
                )
            )
        options.append(channel_options)

    robust_times = None
    if bool(robust_baseline):
        robust_result = _fit_robust_anchor_baseline(
            point_channels=point_channels,
            candidate_options=options,
            complete_time=np.asarray(complete_time, dtype=np.float32),
            complete_prob=np.asarray(complete_prob, dtype=np.float32),
            window_seconds=float(window_seconds),
            tolerance_s=float(robust_baseline_tolerance_s),
            min_support=int(robust_baseline_min_support),
        )
        if robust_result is not None:
            robust_times, _support = robust_result

    def baseline_penalty(opt: AnchorPathStep, idx: int) -> float:
        penalty = float(baseline_weight) * abs(float(opt.time_norm) - float(baseline_times[idx]))
        if robust_times is not None:
            penalty += float(robust_baseline_weight) * abs(float(opt.time_norm) - float(robust_times[idx]))
        return float(penalty)

    if len(options) == 1:
        best_idx = int(np.argmax([float(opt.node_score) - baseline_penalty(opt, 0) for opt in options[0]]))
        chosen = [best_idx]
    else:
        first_scores = np.asarray(
            [float(opt.node_score) - baseline_penalty(opt, 0) for opt in options[0]],
            dtype=np.float64,
        )
        second_scores = np.full((len(options[0]), len(options[1])), -np.inf, dtype=np.float64)
        second_back = np.zeros((len(options[0]), len(options[1])), dtype=np.int32)
        expected_dt = float(baseline_dt[0]) if baseline_dt.size >= 1 else 0.0
        dc01 = max(1e-6, float(point_channels[1] - point_channels[0]))
        for p_idx, prev_opt in enumerate(options[0]):
            for c_idx, cur_opt in enumerate(options[1]):
                jump = abs((float(cur_opt.time_norm) - float(prev_opt.time_norm)) - expected_dt)
                if jump > max_jump_norm:
                    continue
                score = (
                    first_scores[p_idx]
                    + float(cur_opt.node_score)
                    - baseline_penalty(cur_opt, 1)
                    - float(transition_weight) * jump
                )
                second_scores[p_idx, c_idx] = score
                second_back[p_idx, c_idx] = p_idx

        dp_prev = second_scores
        backptrs: list[np.ndarray] = [second_back]
        for idx in range(2, len(options)):
            prev_opts = options[idx - 1]
            cur_opts = options[idx]
            prevprev_opts = options[idx - 2]
            dp_cur = np.full((len(prev_opts), len(cur_opts)), -np.inf, dtype=np.float64)
            back_cur = np.zeros((len(prev_opts), len(cur_opts)), dtype=np.int32)
            expected_dt = float(baseline_dt[idx - 1]) if idx - 1 < baseline_dt.size else 0.0
            dc_prev = max(1e-6, float(point_channels[idx - 1] - point_channels[idx - 2]))
            dc_cur = max(1e-6, float(point_channels[idx] - point_channels[idx - 1]))
            for p_idx, prev_opt in enumerate(prev_opts):
                for c_idx, cur_opt in enumerate(cur_opts):
                    best_score = -np.inf
                    best_pp = 0
                    jump = abs((float(cur_opt.time_norm) - float(prev_opt.time_norm)) - expected_dt)
                    if jump > max_jump_norm:
                        continue
                    slope_curr = (float(cur_opt.time_norm) - float(prev_opt.time_norm)) / dc_cur
                    for pp_idx, prevprev_opt in enumerate(prevprev_opts):
                        prev_score = float(dp_prev[pp_idx, p_idx])
                        if not np.isfinite(prev_score):
                            continue
                        slope_prev = (float(prev_opt.time_norm) - float(prevprev_opt.time_norm)) / dc_prev
                        sec_pen = float(second_diff_weight) * abs(float(slope_curr) - float(slope_prev))
                        score = (
                            prev_score
                            + float(cur_opt.node_score)
                            - baseline_penalty(cur_opt, idx)
                            - float(transition_weight) * jump
                            - sec_pen
                        )
                        if score > best_score:
                            best_score = float(score)
                            best_pp = int(pp_idx)
                    if np.isfinite(best_score):
                        dp_cur[p_idx, c_idx] = best_score
                        back_cur[p_idx, c_idx] = best_pp
            dp_prev = dp_cur
            backptrs.append(back_cur)

        end_pos = np.unravel_index(int(np.argmax(dp_prev)), dp_prev.shape)
        if not np.isfinite(float(dp_prev[end_pos])):
            return None
        chosen = [0] * len(options)
        chosen[-2] = int(end_pos[0])
        chosen[-1] = int(end_pos[1])
        for idx in range(len(options) - 1, 1, -1):
            back = backptrs[idx - 1]
            pp = int(back[int(chosen[idx - 1]), int(chosen[idx])])
            chosen[idx - 2] = pp

    selected_steps = [options[idx][choice_idx] for idx, choice_idx in enumerate(chosen)]
    points: list[TrackPoint] = []
    observed_mask: list[bool] = []
    point_times_norm: list[float] = []
    anchor_count = 0
    n_samples = int(round(float(window_seconds) * float(fs)))
    n_samples = max(1, n_samples)
    for idx, ch in enumerate(point_channels):
        step = selected_steps[idx]
        if bool(step.observed):
            t_norm = float(np.clip(float(step.time_norm), 0.0, 1.0))
            amp = float(step.amp)
            score = float(step.node_score)
            is_observed = True
            anchor_count += 1
        else:
            t_norm = float(np.clip(float(step.time_norm), 0.0, 1.0))
            amp = 0.0
            score = float(step.node_score)
            is_observed = False
        t_idx = int(round(t_norm * float(max(1, n_samples - 1))))
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(max(1e-9, fs)),
                offset_m=float(x_axis_m[int(ch)]),
                amp=float(amp),
                score=float(score),
            )
        )
        observed_mask.append(bool(is_observed))
        point_times_norm.append(float(t_norm))

    if anchor_count < int(cfg.min_anchor_support_channels):
        return None
    anchor_ratio = float(anchor_count) / float(max(1, len(point_channels)))
    if float(cfg.min_anchor_support_ratio) > 0.0 and anchor_ratio < float(cfg.min_anchor_support_ratio):
        return None

    points = sorted(points, key=lambda p: int(p.ch_idx))
    mean_speed = float("nan")
    if len(points) >= 2:
        ts = np.asarray([p.time_s for p in points], dtype=np.float64)
        chs = np.asarray([p.ch_idx for p in points], dtype=np.float64)
        dt = np.diff(ts)
        dch = np.diff(chs)
        valid = np.abs(dt) > 1e-9
        if np.any(valid):
            dx = float(x_axis_m[1] - x_axis_m[0] if len(x_axis_m) > 1 else 20.0)
            speed_mps = np.abs(dch[valid]) * dx / np.abs(dt[valid])
            mean_speed = float(3.6 * np.mean(speed_mps))

    track = Track(
        track_id=0,
        direction="forward" if int(direction) == 0 else "reverse",
        points=points,
        total_score=float(sum(p.score for p in points)),
        mean_speed_kmh=mean_speed,
    )
    decoded = DecodedPeakTrack(
        track=track,
        objectness=float(objectness),
        query_index=int(query_index),
        channel_indices=np.asarray([int(p.ch_idx) for p in points], dtype=np.int64),
        complete_valid=np.asarray(complete_mask[point_channels], dtype=np.float32),
        observed_valid=np.asarray(observed_mask, dtype=np.float32),
        point_times_norm=np.asarray(point_times_norm, dtype=np.float32),
    )
    return decoded


def _postprocess_track(
    item: DecodedPeakTrack,
    *,
    fs: float,
    dx_m: float,
    n_samples: int,
    speed_min_kmh: float,
    speed_max_kmh: float,
    segment_speed_min_kmh: float,
    segment_speed_max_kmh: float,
    max_line_residual_s: float,
    max_adjacent_residual_s: float,
    refit_missing: bool,
    refit_observed: bool,
    validate_observed_only: bool,
    min_visible_channels: int,
    n_channels: int,
    repair_outlier_points: bool,
    drop_point_residual_s: float,
    repair_point_residual_s: float,
    drop_point_second_diff_s: float,
    extend_missing: bool,
    extend_max_channels: int,
    extend_min_observed: int,
) -> DecodedPeakTrack | None:
    points = sorted(item.track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return None

    observed = np.asarray(item.observed_valid, dtype=bool)
    if observed.size != len(points):
        observed = np.zeros((len(points),), dtype=bool)
    fit_points = [p for p, keep in zip(points, observed) if bool(keep)]
    if len(fit_points) < 2:
        fit_points = points
    fit = _fit_track_line(fit_points)
    if fit is None:
        return None
    slope, intercept = fit
    if abs(float(slope)) <= 1e-9:
        return None

    repaired_observed = observed.copy()
    drop_mask = np.zeros((len(points),), dtype=bool)
    repair_mask = np.zeros((len(points),), dtype=bool)
    if bool(repair_outlier_points) and len(points) >= 3:
        all_ch = np.asarray([int(p.ch_idx) for p in points], dtype=np.float64)
        all_ts = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
        all_residual = all_ts - (slope * all_ch + intercept)
        abs_residual = np.abs(all_residual)
        drop_mask |= abs_residual > float(drop_point_residual_s)
        repair_mask |= abs_residual > float(repair_point_residual_s)

        if all_residual.size >= 2:
            jumps = np.abs(np.diff(all_residual))
            for offset, jump_val in enumerate(jumps, start=1):
                if float(jump_val) <= float(max_adjacent_residual_s):
                    continue
                left = offset - 1
                right = offset
                target = right if abs_residual[right] >= abs_residual[left] else left
                repair_mask[target] = True
        if all_ts.size >= 3:
            second = np.abs(np.diff(all_ts, n=2))
            for center_offset, value in enumerate(second, start=1):
                if float(value) <= float(drop_point_second_diff_s):
                    continue
                nearby = [center_offset - 1, center_offset, center_offset + 1]
                target = max(nearby, key=lambda idx: float(abs_residual[idx]))
                if float(abs_residual[target]) >= float(drop_point_residual_s):
                    drop_mask[target] = True
                else:
                    repair_mask[target] = True

        # A point can be close to the fitted line while still creating an
        # impossible adjacent speed jump.  Check only observed anchors here;
        # completed points are already model-derived line points and should
        # not be used to decide whether a real peak is an outlier.  When an
        # adjacent observed segment falls outside the physically allowed
        # segment-speed range, downgrade the endpoint with the larger line
        # residual and reconstruct it from the robust line below.
        observed_indices = np.flatnonzero(repaired_observed & ~drop_mask)
        if observed_indices.size >= 2:
            for left_pos, right_pos in zip(observed_indices[:-1], observed_indices[1:]):
                left_ch = float(all_ch[left_pos])
                right_ch = float(all_ch[right_pos])
                delta_ch = abs(right_ch - left_ch)
                delta_t = abs(float(all_ts[right_pos] - all_ts[left_pos]))
                if delta_ch <= 0.0:
                    continue
                segment_speed = float("inf") if delta_t <= 1e-9 else delta_ch * float(dx_m) / delta_t * 3.6
                if not np.isfinite(segment_speed) or segment_speed < float(segment_speed_min_kmh) or segment_speed > float(segment_speed_max_kmh):
                    target = int(right_pos) if abs_residual[right_pos] >= abs_residual[left_pos] else int(left_pos)
                    repair_mask[target] = True
        repair_mask &= ~drop_mask

        kept_for_refit = [p for idx, p in enumerate(points) if not bool(drop_mask[idx])]
        kept_observed_for_refit = [p for idx, p in enumerate(points) if not bool(drop_mask[idx]) and bool(repaired_observed[idx]) and not bool(repair_mask[idx])]
        refit_source = kept_observed_for_refit if len(kept_observed_for_refit) >= 2 else kept_for_refit
        refit = _fit_track_line(refit_source)
        if refit is not None:
            slope, intercept = refit
        for idx in range(len(points)):
            if bool(repair_mask[idx]):
                repaired_observed[idx] = False

    if bool(np.any(drop_mask)):
        original_indices = [idx for idx in range(len(points)) if not bool(drop_mask[idx])]
        points = [p for idx, p in enumerate(points) if not bool(drop_mask[idx])]
        repaired_observed = np.asarray([keep for idx, keep in enumerate(repaired_observed) if not bool(drop_mask[idx])], dtype=bool)
        repair_mask = np.asarray([keep for idx, keep in enumerate(repair_mask) if not bool(drop_mask[idx])], dtype=bool)
        if len(points) < int(min_visible_channels):
            return None
        fit_points = [p for p, keep in zip(points, repaired_observed) if bool(keep)]
        if len(fit_points) < 2:
            fit_points = points
        fit = _fit_track_line(fit_points)
        if fit is None:
            return None
        slope, intercept = fit
        if abs(float(slope)) <= 1e-9:
            return None

    fit_speed = abs(float(dx_m) / float(slope)) * 3.6
    if fit_speed < float(speed_min_kmh) or fit_speed > float(speed_max_kmh):
        return None

    working_points: list[TrackPoint] = []
    working_observed: list[bool] = []
    working_complete: list[float] = []
    original_complete_all = np.asarray(item.complete_valid, dtype=np.float32).reshape(-1)
    if "original_indices" in locals() and original_complete_all.size == len(drop_mask):
        original_complete = original_complete_all[np.asarray(original_indices, dtype=np.int64)]
    else:
        original_complete = original_complete_all
    if original_complete.size != len(points):
        original_complete = np.ones((len(points),), dtype=np.float32)

    for idx, point in enumerate(points):
        is_observed = bool(repaired_observed[idx])
        use_fit = bool(refit_observed) or bool(repair_mask[idx] if idx < repair_mask.size else False) or (bool(refit_missing) and not is_observed)
        if use_fit:
            score = float(point.score) * (0.35 if not is_observed else 1.0)
            refined = _refit_point_from_line(
                point,
                slope=float(slope),
                intercept=float(intercept),
                fs=float(fs),
                n_samples=int(n_samples),
                observed=is_observed,
                score=score,
            )
        else:
            refined = TrackPoint(
                ch_idx=int(point.ch_idx),
                t_idx=int(point.t_idx),
                time_s=float(point.time_s),
                offset_m=float(point.offset_m),
                amp=float(point.amp),
                score=float(point.score),
            )
        working_points.append(refined)
        working_observed.append(bool(is_observed))
        working_complete.append(float(original_complete[idx]))

    if bool(extend_missing):
        observed_count = int(np.sum(np.asarray(working_observed, dtype=bool)))
        if observed_count >= int(extend_min_observed):
            existing = {int(p.ch_idx) for p in working_points}
            min_ch = min(existing)
            max_ch = max(existing)
            max_extend = int(max(0, extend_max_channels))
            start_ch = max(0, min_ch - max_extend)
            end_ch = min(int(n_channels) - 1, max_ch + max_extend)
            base_score = float(np.median([float(p.score) for p in working_points])) * 0.25 if working_points else 0.01
            for ch_idx in range(start_ch, end_ch + 1):
                if ch_idx in existing:
                    continue
                completed = _make_completed_point(
                    ch_idx=int(ch_idx),
                    slope=float(slope),
                    intercept=float(intercept),
                    fs=float(fs),
                    dx_m=float(dx_m),
                    n_samples=int(n_samples),
                    score=max(0.01, base_score),
                )
                if completed is None:
                    continue
                working_points.append(completed)
                working_observed.append(False)
                working_complete.append(1.0)
                existing.add(int(ch_idx))

    ordered = sorted(zip(working_points, working_observed, working_complete), key=lambda item_: int(item_[0].ch_idx))
    refined_points = [item_[0] for item_ in ordered]
    refined_observed = np.asarray([item_[1] for item_ in ordered], dtype=bool)
    refined_complete = np.asarray([item_[2] for item_ in ordered], dtype=np.float32)
    if len(refined_points) < int(min_visible_channels):
        return None

    validation_points = [p for p, keep in zip(refined_points, refined_observed) if bool(keep)] if bool(validate_observed_only) else refined_points
    if len(validation_points) < 2:
        validation_points = refined_points
    ch = np.asarray([int(p.ch_idx) for p in validation_points], dtype=np.float64)
    ts = np.asarray([float(p.time_s) for p in validation_points], dtype=np.float64)
    pred = slope * ch + intercept
    residual = ts - pred
    if residual.size and float(np.max(np.abs(residual))) > float(max_line_residual_s):
        return None
    if residual.size >= 2 and float(np.max(np.abs(np.diff(residual)))) > float(max_adjacent_residual_s):
        return None

    seg_speed = _track_segment_speeds_kmh(validation_points, dx_m=float(dx_m))
    if seg_speed.size:
        if float(np.nanmedian(seg_speed)) < float(speed_min_kmh) or float(np.nanmedian(seg_speed)) > float(speed_max_kmh):
            return None
        bad_segments = (seg_speed < float(segment_speed_min_kmh)) | (seg_speed > float(segment_speed_max_kmh))
        if float(np.mean(bad_segments.astype(np.float32))) > 0.25:
            return None

    out_speed = fit_speed
    return DecodedPeakTrack(
        track=Track(
            track_id=int(item.track.track_id),
            direction="forward" if slope >= 0.0 else "reverse",
            points=refined_points,
            total_score=float(sum(float(p.score) for p in refined_points)),
            mean_speed_kmh=float(out_speed),
        ),
        objectness=float(item.objectness),
        query_index=int(item.query_index),
        channel_indices=np.asarray([int(p.ch_idx) for p in refined_points], dtype=np.int64),
        complete_valid=refined_complete,
        observed_valid=refined_observed.astype(np.float32),
        point_times_norm=np.asarray([float(p.time_s) / max(1e-9, float(n_samples) / float(fs)) for p in refined_points], dtype=np.float32),
    )


def _deduplicate_tracks_for_real_predict(
    decoded: list[DecodedPeakTrack],
    *,
    strict_line_tolerance_s: float,
    loose_line_tolerance_s: float,
    speed_tolerance_kmh: float,
) -> list[DecodedPeakTrack]:
    kept: list[DecodedPeakTrack] = []
    strict_tol = float(max(0.0, strict_line_tolerance_s))
    loose_tol = float(max(strict_tol, loose_line_tolerance_s))
    speed_tol = float(max(0.0, speed_tolerance_kmh))
    for item in sorted(decoded, key=lambda tr: float(tr.objectness), reverse=True):
        duplicate = False
        for existing in kept:
            a = item.track
            b = existing.track
            a_ch = {int(p.ch_idx) for p in a.points}
            b_ch = {int(p.ch_idx) for p in b.points}
            common = len(a_ch & b_ch)
            ratio = common / float(max(1, min(len(a_ch), len(b_ch))))
            line_distance = _track_line_distance(a, b)
            speed_diff = abs(float(a.mean_speed_kmh) - float(b.mean_speed_kmh))
            if (ratio >= 0.75 and line_distance < strict_tol) or (
                ratio >= 0.60 and line_distance < loose_tol and speed_diff < speed_tol
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def _track_observed_ratio(item: DecodedPeakTrack) -> float:
    observed = np.asarray(item.observed_valid, dtype=np.float32).reshape(-1)
    if observed.size == 0:
        return 0.0
    return float(np.mean(observed > 0.5))


def _passes_track_quality_gate(
    item: DecodedPeakTrack,
    *,
    min_observed_ratio: float,
    min_total_score: float,
) -> bool:
    if float(min_observed_ratio) > 0.0 and _track_observed_ratio(item) < float(min_observed_ratio):
        return False
    if float(min_total_score) > 0.0 and float(item.track.total_score) < float(min_total_score):
        return False
    return True


def _decode_batch_outputs(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    cfg: PeakSetInferenceConfig,
    meta: dict[str, Any],
    snap_to_candidates: bool,
    snap_tolerance_s: float,
    min_peak_support_channels: int,
    min_peak_support_ratio: float,
    postprocess: bool,
    postprocess_refit_missing: bool,
    postprocess_refit_observed: bool,
    speed_min_kmh: float,
    speed_max_kmh: float,
    segment_speed_min_kmh: float,
    segment_speed_max_kmh: float,
    max_line_residual_s: float,
    max_adjacent_residual_s: float,
    postprocess_validate_observed_only: bool,
    postprocess_repair_outlier_points: bool,
    postprocess_drop_point_residual_s: float,
    postprocess_repair_point_residual_s: float,
    postprocess_drop_point_second_diff_s: float,
    postprocess_extend_missing: bool,
    postprocess_extend_max_channels: int,
    postprocess_extend_min_observed: int,
    dedup_line_tolerance_s: float,
    dedup_loose_line_tolerance_s: float,
    dedup_speed_tolerance_kmh: float,
    min_track_observed_ratio: float,
    min_track_total_score: float,
    anchor_path_refine: bool,
    anchor_path_top_k: int,
    anchor_path_transition_weight: float,
    anchor_path_second_diff_weight: float,
    anchor_path_baseline_weight: float,
    anchor_path_robust_baseline: bool,
    anchor_path_robust_baseline_weight: float,
    anchor_path_robust_baseline_tolerance_s: float,
    anchor_path_robust_baseline_min_support: int,
    anchor_path_max_adjacent_jump_s: float,
    observed_outlier_repair: bool,
    observed_outlier_residual_s: float,
    observed_outlier_adjacent_s: float,
) -> list[list[DecodedPeakTrack]]:
    objectness = torch.sigmoid(outputs["objectness_logits"]).detach().cpu().numpy()
    complete_prob = torch.sigmoid(outputs["complete_valid_logits"]).detach().cpu().numpy()
    direction = torch.argmax(outputs["direction_logits"], dim=-1).detach().cpu().numpy()
    speed_pred = outputs["speed"].detach().cpu().numpy()

    is_peak_guided = "anchor_peak_logits" in outputs and "complete_time" in outputs
    if is_peak_guided:
        anchor_logits = outputs["anchor_peak_logits"].detach().cpu().numpy()
        complete_time = outputs["complete_time"].detach().cpu().numpy()
        observed_prob = (
            torch.sigmoid(outputs["observed_valid_logits"]).detach().cpu().numpy()
            if "observed_valid_logits" in outputs and bool(getattr(cfg, "use_observed_valid_in_decode", False))
            else np.ones_like(complete_prob, dtype=np.float32)
        )
        peak_time_batch = batch["peak_time"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_time")) else outputs["peak_time"].detach().cpu().numpy()
        peak_amp_batch = batch["peak_amp"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_amp")) else outputs["peak_amp"].detach().cpu().numpy()
        peak_valid_batch = batch["peak_valid"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_valid")) else outputs["peak_valid"].detach().cpu().numpy()
        raw_window_batch = batch["raw_window"].detach().cpu().numpy() if torch.is_tensor(batch.get("raw_window")) else None
    else:
        observed_prob = torch.sigmoid(outputs["observed_logits"]).detach().cpu().numpy()
        peak_pred = outputs["peak_time"].detach().cpu().numpy()
        peak_time_batch = batch["peak_time"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_time")) else None
        peak_amp_batch = batch["peak_amp"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_amp")) else None
        peak_valid_batch = batch["peak_valid"].detach().cpu().numpy() if torch.is_tensor(batch.get("peak_valid")) else None
        raw_window_batch = batch["raw_window"].detach().cpu().numpy() if torch.is_tensor(batch.get("raw_window")) else None

    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    window_seconds = float(meta.get("window_seconds", 120.0))
    window_samples = int(meta.get("window_samples", round(window_seconds * fs)))
    speed_norm_kmh = float(meta.get("speed_norm_kmh", 150.0))
    n_channels = int((outputs["complete_valid_logits"].shape[-1] if is_peak_guided else peak_pred.shape[-1]))
    x_axis_m = np.arange(n_channels, dtype=np.float32) * dx_m

    decoded_batch: list[list[DecodedPeakTrack]] = []
    for b in range(int(objectness.shape[0])):
        order = np.argsort(objectness[b])[::-1]
        sample_tracks: list[DecodedPeakTrack] = []
        for q_idx in order[: int(max(1, cfg.max_tracks))]:
            obj = float(objectness[b, q_idx])
            if obj < float(cfg.objectness_threshold):
                continue
            if bool(is_peak_guided) and bool(anchor_path_refine):
                decoded_item = _build_continuous_peak_guided_track(
                    objectness=obj,
                    query_index=int(q_idx),
                    direction=int(direction[b, q_idx]),
                    complete_prob=np.asarray(complete_prob[b, q_idx], dtype=np.float32),
                    observed_prob=np.asarray(observed_prob[b, q_idx], dtype=np.float32),
                    complete_time=np.asarray(complete_time[b, q_idx], dtype=np.float32),
                    anchor_logits=np.asarray(anchor_logits[b, q_idx], dtype=np.float32),
                    peak_time=np.asarray(peak_time_batch[b], dtype=np.float32),
                    peak_amp=np.asarray(peak_amp_batch[b], dtype=np.float32),
                    peak_valid=np.asarray(peak_valid_batch[b], dtype=bool),
                    raw_window=np.asarray(raw_window_batch[b], dtype=np.float32) if raw_window_batch is not None else None,
                    fs=float(fs),
                    x_axis_m=np.asarray(x_axis_m, dtype=np.float32),
                    cfg=cfg,
                    window_seconds=float(window_seconds),
                    top_k=int(anchor_path_top_k),
                    transition_weight=float(anchor_path_transition_weight),
                    second_diff_weight=float(anchor_path_second_diff_weight),
                    baseline_weight=float(anchor_path_baseline_weight),
                    robust_baseline=bool(anchor_path_robust_baseline),
                    robust_baseline_weight=float(anchor_path_robust_baseline_weight),
                    robust_baseline_tolerance_s=float(anchor_path_robust_baseline_tolerance_s),
                    robust_baseline_min_support=int(anchor_path_robust_baseline_min_support),
                    max_adjacent_jump_s=float(anchor_path_max_adjacent_jump_s),
                )
                if decoded_item is None:
                    continue
                if bool(observed_outlier_repair):
                    repaired = _repair_observed_outliers(
                        decoded_item,
                        fs=float(fs),
                        dx_m=float(dx_m),
                        n_samples=int(window_samples),
                        observed_outlier_residual_s=float(observed_outlier_residual_s),
                        observed_outlier_adjacent_s=float(observed_outlier_adjacent_s),
                        refit_missing=bool(postprocess_refit_missing),
                        refit_observed=bool(postprocess_refit_observed),
                    )
                    if repaired is not None:
                        decoded_item = repaired
                sample_tracks.append(decoded_item)
            else:
                complete_mask = complete_prob[b, q_idx] >= float(cfg.complete_valid_threshold)
                if int(np.sum(complete_mask)) < int(cfg.min_visible_channels):
                    continue
                point_channels = np.where(complete_mask)[0].tolist()
                points: list[TrackPoint] = []
                observed_mask: list[bool] = []
                point_times_norm: list[float] = []
                peak_support_count = 0
                for ch in point_channels:
                    amp = 0.0
                    t_norm = float(np.clip(peak_pred[b, q_idx, ch], 0.0, 1.0))
                    snapped = False
                    is_observed = False
                    if bool(snap_to_candidates):
                        snapped_t_norm, snapped_amp, snapped = _nearest_candidate(
                            ch=int(ch),
                            t_norm=t_norm,
                            peak_time=peak_time_batch[b] if peak_time_batch is not None else None,
                            peak_amp=peak_amp_batch[b] if peak_amp_batch is not None else None,
                            peak_valid=peak_valid_batch[b] if peak_valid_batch is not None else None,
                            window_seconds=window_seconds,
                            tolerance_s=float(snap_tolerance_s),
                        )
                        if bool(snapped):
                            t_norm = float(snapped_t_norm)
                            amp = float(snapped_amp)
                            peak_support_count += 1
                        is_observed = bool(snapped and observed_prob[b, q_idx, ch] >= float(cfg.anchor_threshold))
                    score = float(obj * complete_prob[b, q_idx, ch] * (observed_prob[b, q_idx, ch] if is_observed else 0.35))
                    t_idx = int(round(t_norm * float(max(1, window_samples - 1))))
                    points.append(
                        TrackPoint(
                            ch_idx=int(ch),
                            t_idx=int(t_idx),
                            time_s=float(t_idx) / float(max(1e-9, fs)),
                            offset_m=float(x_axis_m[int(ch)]),
                            amp=float(amp),
                            score=score,
                        )
                    )
                    observed_mask.append(bool(is_observed))
                    point_times_norm.append(float(t_norm))
                if len(points) < int(cfg.min_visible_channels):
                    continue
                support_ratio = float(peak_support_count) / float(max(1, len(points)))
                if int(min_peak_support_channels) > 0 and int(peak_support_count) < int(min_peak_support_channels):
                    continue
                if float(min_peak_support_ratio) > 0.0 and support_ratio < float(min_peak_support_ratio):
                    continue
                points = sorted(points, key=lambda p: int(p.ch_idx))
                ts = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
                chs = np.asarray([int(p.ch_idx) for p in points], dtype=np.float64)
                mean_speed = float(speed_pred[b, q_idx] * speed_norm_kmh)
                if len(points) >= 2:
                    dt = np.diff(ts)
                    dch = np.diff(chs)
                    valid = np.abs(dt) > 1e-9
                    if np.any(valid):
                        speed_mps = np.abs(dch[valid]) * dx_m / np.abs(dt[valid])
                        mean_speed = float(3.6 * np.mean(speed_mps))
                track = Track(
                    track_id=int(len(sample_tracks)),
                    direction="forward" if int(direction[b, q_idx]) == 0 else "reverse",
                    points=points,
                    total_score=float(sum(float(p.score) for p in points)),
                    mean_speed_kmh=mean_speed,
                )
                sample_tracks.append(
                    DecodedPeakTrack(
                        track=track,
                        objectness=obj,
                        query_index=int(q_idx),
                        channel_indices=np.asarray(point_channels, dtype=np.int64),
                        complete_valid=np.asarray(complete_mask[point_channels], dtype=np.float32),
                        observed_valid=np.asarray(observed_mask, dtype=np.float32),
                        point_times_norm=np.asarray(point_times_norm, dtype=np.float32),
                    )
                )
        if bool(postprocess):
            postprocessed: list[DecodedPeakTrack] = []
            for item in sample_tracks:
                kept = _postprocess_track(
                    item,
                    fs=float(fs),
                    dx_m=float(dx_m),
                    n_samples=int(window_samples),
                    speed_min_kmh=float(speed_min_kmh),
                    speed_max_kmh=float(speed_max_kmh),
                    segment_speed_min_kmh=float(segment_speed_min_kmh),
                    segment_speed_max_kmh=float(segment_speed_max_kmh),
                    max_line_residual_s=float(max_line_residual_s),
                    max_adjacent_residual_s=float(max_adjacent_residual_s),
                    refit_missing=bool(postprocess_refit_missing),
                    refit_observed=bool(postprocess_refit_observed),
                    validate_observed_only=bool(postprocess_validate_observed_only),
                    min_visible_channels=int(cfg.min_visible_channels),
                    n_channels=int(n_channels),
                    repair_outlier_points=bool(postprocess_repair_outlier_points),
                    drop_point_residual_s=float(postprocess_drop_point_residual_s),
                    repair_point_residual_s=float(postprocess_repair_point_residual_s),
                    drop_point_second_diff_s=float(postprocess_drop_point_second_diff_s),
                    extend_missing=bool(postprocess_extend_missing),
                    extend_max_channels=int(postprocess_extend_max_channels),
                    extend_min_observed=int(postprocess_extend_min_observed),
                )
                if kept is not None:
                    postprocessed.append(kept)
            sample_tracks = postprocessed
        if float(min_track_observed_ratio) > 0.0 or float(min_track_total_score) > 0.0:
            sample_tracks = [
                item
                for item in sample_tracks
                if _passes_track_quality_gate(
                    item,
                    min_observed_ratio=float(min_track_observed_ratio),
                    min_total_score=float(min_track_total_score),
                )
            ]
        decoded = _deduplicate_tracks_for_real_predict(
            sample_tracks,
            strict_line_tolerance_s=float(dedup_line_tolerance_s),
            loose_line_tolerance_s=float(dedup_loose_line_tolerance_s),
            speed_tolerance_kmh=float(dedup_speed_tolerance_kmh),
        )
        for track_id, item in enumerate(decoded):
            item.track.track_id = int(track_id)
        decoded_batch.append(decoded)
    return decoded_batch


def _track_to_json(item: DecodedPeakTrack) -> dict[str, Any]:
    return {
        "track_id": int(item.track.track_id),
        "query_index": int(item.query_index),
        "objectness": float(item.objectness),
        "direction": str(item.track.direction),
        "mean_speed_kmh": float(item.track.mean_speed_kmh),
        "total_score": float(item.track.total_score),
        "observed_ratio": _track_observed_ratio(item),
        "points": [
            {
                "channel": int(point.ch_idx),
                "time_s": float(point.time_s),
                "sample_index": int(point.t_idx),
                "offset_m": float(point.offset_m),
                "amp": float(point.amp),
                "score": float(point.score),
                "observed": bool(item.observed_valid[idx] > 0.5) if idx < len(item.observed_valid) else False,
            }
            for idx, point in enumerate(sorted(item.track.points, key=lambda p: int(p.ch_idx)))
        ],
    }


def _draw_overlay(
    path: Path,
    *,
    x: torch.Tensor,
    full_time: torch.Tensor | None,
    full_valid: torch.Tensor | None,
    peak_time: torch.Tensor | None,
    peak_amp: torch.Tensor | None,
    peak_valid: torch.Tensor | None,
    tracks: list[DecodedPeakTrack],
    sample_index: int,
    meta: dict[str, Any],
    dpi: int,
    plot_style: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    feature = x[0].detach().cpu().numpy() if x.ndim == 3 else x.detach().cpu().numpy()
    window_seconds = float(meta.get("window_seconds", 120.0))
    dx_m = float(meta.get("dx_m", 100.0))
    n_ch, n_t = int(feature.shape[0]), int(feature.shape[1])
    finite = feature[np.isfinite(feature)]
    vmax = max(float(np.quantile(np.abs(finite), 0.995)), 1e-6) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    plot_style = str(plot_style).lower()
    if plot_style == "waveform":
        if dx_m > 0.0:
            x_axis = np.arange(n_ch, dtype=np.float64) * dx_m * 1e-3
            x_label = "Offset [km]"
        else:
            x_axis = np.arange(n_ch, dtype=np.float64)
            x_label = "Channel index"
        t_axis = np.linspace(0.0, window_seconds, n_t, dtype=np.float64)
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        wiggle_amp = 0.27 * spacing
        clip_ratio = 1.35
        for ch in range(n_ch):
            ratio = np.clip(feature[ch].astype(np.float64) / max(vmax, 1e-12), -clip_ratio, clip_ratio)
            ax.plot(x_axis[ch] + ratio * wiggle_amp, t_axis, color="0.45", linewidth=0.8, alpha=0.9)
        pad = 0.4 * spacing
        x_min_plot = float(x_axis[0] - pad)
        x_max_plot = float(x_axis[-1] + pad)
        x_span_full = max(1e-6, x_max_plot - x_min_plot)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(0.0, window_seconds)
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        im = None
    else:
        im = ax.imshow(
            feature.T,
            aspect="auto",
            origin="lower",
            cmap="magma",
            extent=(-0.5, float(n_ch) - 0.5, 0.0, window_seconds),
        )
        x_axis = np.arange(n_ch, dtype=np.float64)
        x_max_plot = float(n_ch) - 0.5
        x_span_full = max(1e-6, float(n_ch))
    gt_count = 0
    if full_time is not None and full_valid is not None:
        ft = full_time.detach().cpu().numpy()
        fv = full_valid.detach().cpu().numpy() > 0.5
        gt_color = "#b8b8b8"
        for idx in range(int(ft.shape[0])):
            mask = fv[idx]
            if int(mask.sum()) < 2:
                continue
            chs = np.where(mask)[0]
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) if int(ch) < len(x_axis) else float(ch) for ch in chs]
            else:
                xs = chs.tolist()
            ys = (ft[idx, mask] * window_seconds).astype(float)
            ax.plot(xs, ys, color=gt_color, linestyle="--", linewidth=0.85, alpha=0.9, zorder=8)
            ax.scatter(
                xs,
                ys,
                s=10,
                c=gt_color,
                alpha=0.95,
                marker="o",
                linewidths=0.0,
                zorder=9,
                label="GT" if gt_count == 0 else None,
            )
            gt_count += 1

    if peak_time is not None and peak_valid is not None:
        pt = peak_time.detach().cpu().numpy()
        pv = peak_valid.detach().cpu().numpy().astype(bool)
        pa = peak_amp.detach().cpu().numpy() if peak_amp is not None else np.ones_like(pt, dtype=np.float32)
        xs: list[int] = []
        ys: list[float] = []
        amps: list[float] = []
        for ch in range(int(pt.shape[0])):
            valid = pv[ch]
            if not bool(valid.any()):
                continue
            xs.extend([int(ch)] * int(valid.sum()))
            ys.extend((pt[ch, valid] * window_seconds).astype(float).tolist())
            amps.extend(pa[ch, valid].astype(float).tolist())
        if xs:
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) for ch in xs]
            sizes = 8.0 + 6.0 * np.clip(np.asarray(amps, dtype=np.float32), 0.0, 1.0)
            ax.scatter(
                xs,
                ys,
                s=np.maximum(sizes, 8.0),
                c="#c0c0c0",
                alpha=0.45,
                marker="o",
                linewidths=0.0,
                zorder=4,
                label="Peak candidates" if gt_count == 0 and len(tracks) == 0 else None,
            )

    cmap = plt.get_cmap("tab20", max(1, len(tracks)))
    pred_label_added = False
    for idx, item in enumerate(tracks):
        points = sorted(item.track.points, key=lambda p: int(p.ch_idx))
        if plot_style == "waveform":
            xs = [float(p.offset_m) * 1e-3 if dx_m > 0.0 else float(p.ch_idx) for p in points]
        else:
            xs = [int(p.ch_idx) for p in points]
        ys = [float(p.time_s) for p in points]
        color = cmap(idx % max(1, cmap.N))
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.8 if plot_style == "waveform" else 2.0,
            alpha=0.95,
            zorder=4,
            label="Prediction" if not pred_label_added else None,
        )
        observed = np.asarray(item.observed_valid, dtype=bool)
        obs_x = [x0 for x0, keep in zip(xs, observed) if bool(keep)]
        obs_y = [y0 for y0, keep in zip(ys, observed) if bool(keep)]
        miss_x = [x0 for x0, keep in zip(xs, observed) if not bool(keep)]
        miss_y = [y0 for y0, keep in zip(ys, observed) if not bool(keep)]
        if obs_x:
            ax.scatter(obs_x, obs_y, s=12, marker="s", color=color, linewidths=0.8, zorder=5)
        if miss_x:
            ax.scatter(miss_x, miss_y, s=22, marker="o", facecolors="none", edgecolors=color, linewidths=1.0, zorder=5)
        speed_kmh = float(item.track.mean_speed_kmh)
        if math.isfinite(speed_kmh) and xs and ys:
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
                zorder=6,
            )
        pred_label_added = True
    ax.set_title(f"Vehicle peak-set prediction overlay, sample {sample_index}  Pred={len(tracks)}")
    if plot_style != "waveform":
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Time (s)")
        ax.invert_yaxis()
    if pred_label_added:
        ax.legend(loc="upper right", frameon=True)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized amplitude")
    fig.tight_layout()
    fig.savefig(str(path), dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(str(args.device))
    dataset = RealPeakShardDataset(Path(args.dataset_dir))
    selected_indices = _sample_indices(len(dataset), int(args.sample_index), int(args.num_samples))
    if not selected_indices:
        raise SystemExit("no samples selected")

    model, checkpoint = load_checkpoint_model(Path(args.model).expanduser(), device=device)
    model.eval()
    meta = dict(dataset.meta)
    if int(meta.get("n_channels", model.config.n_channels)) != int(model.config.n_channels):
        raise SystemExit(f"channel mismatch: model expects {model.config.n_channels}, dataset has {meta.get('n_channels')}")
    if int(meta.get("in_channels", model.config.in_channels)) != int(model.config.in_channels):
        raise SystemExit(f"input-channel mismatch: model expects {model.config.in_channels}, dataset has {meta.get('in_channels')}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    cfg = PeakSetInferenceConfig(
        objectness_threshold=float(args.objectness_threshold),
        complete_valid_threshold=float(args.complete_valid_threshold),
        anchor_threshold=float(args.observed_threshold),
        min_visible_channels=int(args.min_visible_channels),
        max_tracks=int(args.max_tracks),
        dedup_tolerance_samples=int(args.dedup_tolerance_samples),
        fused_min_anchor_score=float(args.fused_min_anchor_score),
        use_observed_valid_in_decode=bool(args.use_observed_valid_in_decode),
        observed_valid_soft_floor=float(args.observed_valid_soft_floor),
    )
    subset = torch.utils.data.Subset(dataset, selected_indices)
    loader = DataLoader(
        subset,
        batch_size=int(max(1, args.batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_real_samples,
    )

    prediction_path = out_dir / "predictions.jsonl"
    sample_summaries: list[dict[str, Any]] = []
    plotted = 0
    with prediction_path.open("w", encoding="utf-8") as fh:
        for batch in loader:
            x = batch["x"].to(device=device, dtype=torch.float32)
            with torch.inference_mode():
                outputs = model(x)
            decoded_batch = _decode_batch_outputs(
                outputs,
                batch,
                cfg=cfg,
                meta=meta,
                snap_to_candidates=bool(args.snap_to_candidates),
                snap_tolerance_s=float(args.snap_tolerance_s),
                min_peak_support_channels=int(args.min_peak_support_channels),
                min_peak_support_ratio=float(args.min_peak_support_ratio),
                postprocess=bool(args.postprocess),
                postprocess_refit_missing=bool(args.postprocess_refit_missing),
                postprocess_refit_observed=bool(args.postprocess_refit_observed),
                speed_min_kmh=float(args.speed_min_kmh),
                speed_max_kmh=float(args.speed_max_kmh),
                segment_speed_min_kmh=float(args.segment_speed_min_kmh),
                segment_speed_max_kmh=float(args.segment_speed_max_kmh),
                max_line_residual_s=float(args.max_line_residual_s),
                max_adjacent_residual_s=float(args.max_adjacent_residual_s),
                postprocess_validate_observed_only=bool(args.postprocess_validate_observed_only),
                postprocess_repair_outlier_points=bool(args.postprocess_repair_outlier_points),
                postprocess_drop_point_residual_s=float(args.postprocess_drop_point_residual_s),
                postprocess_repair_point_residual_s=float(args.postprocess_repair_point_residual_s),
                postprocess_drop_point_second_diff_s=float(args.postprocess_drop_point_second_diff_s),
                postprocess_extend_missing=bool(args.postprocess_extend_missing),
                postprocess_extend_max_channels=int(args.postprocess_extend_max_channels),
                postprocess_extend_min_observed=int(args.postprocess_extend_min_observed),
                dedup_line_tolerance_s=float(args.dedup_line_tolerance_s),
                dedup_loose_line_tolerance_s=float(args.dedup_loose_line_tolerance_s),
                dedup_speed_tolerance_kmh=float(args.dedup_speed_tolerance_kmh),
                min_track_observed_ratio=float(args.min_track_observed_ratio),
                min_track_total_score=float(args.min_track_total_score),
                anchor_path_refine=bool(args.anchor_path_refine),
                anchor_path_top_k=int(args.anchor_path_top_k),
                anchor_path_transition_weight=float(args.anchor_path_transition_weight),
                anchor_path_second_diff_weight=float(args.anchor_path_second_diff_weight),
                anchor_path_baseline_weight=float(args.anchor_path_baseline_weight),
                anchor_path_robust_baseline=bool(args.anchor_path_robust_baseline),
                anchor_path_robust_baseline_weight=float(args.anchor_path_robust_baseline_weight),
                anchor_path_robust_baseline_tolerance_s=float(args.anchor_path_robust_baseline_tolerance_s),
                anchor_path_robust_baseline_min_support=int(args.anchor_path_robust_baseline_min_support),
                anchor_path_max_adjacent_jump_s=float(args.anchor_path_max_adjacent_jump_s),
                observed_outlier_repair=bool(args.observed_outlier_repair),
                observed_outlier_residual_s=float(args.observed_outlier_residual_s),
                observed_outlier_adjacent_s=float(args.observed_outlier_adjacent_s),
            )
            for row, tracks in enumerate(decoded_batch):
                sample_index = int(batch["sample_index"][row])
                start_sample, start_second = _window_start(meta, sample_index)
                raw_window = batch["raw_window"][row]
                overlay_path: Path | None = None
                if plotted < int(max(0, args.plot_samples)):
                    overlay_path = overlay_dir / f"real_vehicle_peak_set_overlay_{sample_index:06d}.png"
                    _draw_overlay(
                        overlay_path,
                        x=raw_window if torch.is_tensor(raw_window) else batch["x"][row],
                        full_time=batch["full_time"][row] if torch.is_tensor(batch.get("full_time")) else None,
                        full_valid=batch["full_valid"][row] if torch.is_tensor(batch.get("full_valid")) else None,
                        peak_time=batch["peak_time"][row] if torch.is_tensor(batch.get("peak_time")) else None,
                        peak_amp=batch["peak_amp"][row] if torch.is_tensor(batch.get("peak_amp")) else None,
                        peak_valid=batch["peak_valid"][row] if torch.is_tensor(batch.get("peak_valid")) else None,
                        tracks=tracks,
                        sample_index=sample_index,
                        meta=meta,
                        dpi=int(args.plot_dpi),
                        plot_style=str(args.plot_style),
                    )
                    plotted += 1
                record = {
                    "sample_index": sample_index,
                    "shard_index": int(batch["shard_index"][row]),
                    "local_index": int(batch["local_index"][row]),
                    "window_start_sample": start_sample,
                    "window_start_seconds": start_second,
                    "track_count": int(len(tracks)),
                    "tracks": [_track_to_json(item) for item in tracks],
                    "overlay": str(overlay_path) if overlay_path is not None else None,
                }
                fh.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")
                sample_summaries.append(
                    {
                        "sample_index": sample_index,
                        "window_start_seconds": start_second,
                        "track_count": int(len(tracks)),
                        "overlay": str(overlay_path) if overlay_path is not None else None,
                    }
                )

    summary = {
        "model": str(Path(args.model).expanduser()),
        "dataset_dir": str(Path(args.dataset_dir).expanduser()),
        "device": device,
        "sample_index": int(args.sample_index),
        "num_samples": int(len(selected_indices)),
        "prediction_jsonl": str(prediction_path),
        "out_dir": str(out_dir),
        "plot_samples": int(plotted),
        "plot_style": str(args.plot_style),
        "thresholds": {
            "objectness": float(args.objectness_threshold),
            "complete_valid": float(args.complete_valid_threshold),
            "observed": float(args.observed_threshold),
            "min_visible_channels": int(args.min_visible_channels),
            "min_peak_support_channels": int(args.min_peak_support_channels),
            "min_peak_support_ratio": float(args.min_peak_support_ratio),
            "fused_min_anchor_score": float(args.fused_min_anchor_score),
            "max_tracks": int(args.max_tracks),
            "dedup_line_tolerance_s": float(args.dedup_line_tolerance_s),
            "dedup_loose_line_tolerance_s": float(args.dedup_loose_line_tolerance_s),
            "dedup_speed_tolerance_kmh": float(args.dedup_speed_tolerance_kmh),
            "min_track_observed_ratio": float(args.min_track_observed_ratio),
            "min_track_total_score": float(args.min_track_total_score),
        },
        "snap_to_candidates": bool(args.snap_to_candidates),
        "snap_tolerance_s": float(args.snap_tolerance_s),
        "postprocess": {
            "enabled": bool(args.postprocess),
            "refit_missing": bool(args.postprocess_refit_missing),
            "refit_observed": bool(args.postprocess_refit_observed),
            "validate_observed_only": bool(args.postprocess_validate_observed_only),
            "speed_min_kmh": float(args.speed_min_kmh),
            "speed_max_kmh": float(args.speed_max_kmh),
            "segment_speed_min_kmh": float(args.segment_speed_min_kmh),
            "segment_speed_max_kmh": float(args.segment_speed_max_kmh),
            "max_line_residual_s": float(args.max_line_residual_s),
            "max_adjacent_residual_s": float(args.max_adjacent_residual_s),
            "repair_outlier_points": bool(args.postprocess_repair_outlier_points),
            "drop_point_residual_s": float(args.postprocess_drop_point_residual_s),
            "repair_point_residual_s": float(args.postprocess_repair_point_residual_s),
            "drop_point_second_diff_s": float(args.postprocess_drop_point_second_diff_s),
            "extend_missing": bool(args.postprocess_extend_missing),
            "extend_max_channels": int(args.postprocess_extend_max_channels),
            "extend_min_observed": int(args.postprocess_extend_min_observed),
        },
        "dataset_meta": {
            "num_samples": int(meta.get("num_samples", len(dataset))),
            "fs": float(meta.get("fs", 1000.0)),
            "dx_m": float(meta.get("dx_m", 100.0)),
            "window_seconds": float(meta.get("window_seconds", 120.0)),
            "stride_seconds": float(meta.get("stride_seconds", 0.0)),
            "source_file": meta.get("source_file"),
            "source_array": meta.get("source_array"),
        },
        "checkpoint_dataset_config": checkpoint.get("dataset_config", {}),
        "samples": sample_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(json_ready(summary), ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
