"""Trajectory energy network for vehicle track extraction.

This is a clean replacement direction for the old slot-based lines:

- predict a dense trajectory energy map instead of a fixed slot set
- decode one vehicle with graph search / Viterbi
- smooth with a Kalman filter
- remove duplicates with trajectory-level NMS / Hungarian matching

The model is intentionally small and interpretable. It is designed for the
50-channel, 100 m spacing, 70-90 km/h regime described in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from torch import nn

from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, extract_all


def _robust_scale(data: np.ndarray) -> float:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite.size == 0:
        return 1.0
    abs_vals = np.abs(finite)
    q995 = float(np.quantile(abs_vals, 0.995))
    rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
    return max(q995, 3.0 * rms, 1e-6)


def prepare_window_input(data_window: np.ndarray, time_downsample: int, clip_ratio: float = 1.35) -> torch.Tensor:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = _robust_scale(arr_ds)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    return torch.from_numpy(raw[None, :, :].astype(np.float32, copy=False))


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    base_dim: int = 32
    hidden_dim: int = 96
    dropout: float = 0.1


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    decoder_profile: str = "strict"
    scene_active_ratio_threshold: float = 0.033
    min_visible_channels: int = 3
    min_track_channels: int = 3
    min_track_score: float = 1.0
    edge_min_track_channels: int = 3
    peak_prominence: float = 0.02
    peak_min_height: float = 0.04
    peak_distance_samples: int = 8
    max_skip_channels: int = 8
    candidate_topk_per_channel: int = 24
    seed_threshold: float = 0.15
    suppression_channel_radius: int = 2
    suppression_time_radius: int = 2500
    dedup_time_tol: int = 1000
    dedup_channel_overlap: int = 3
    dedup_overlap_ratio: float = 0.7
    speed_norm_kmh: float = 150.0
    kalman_process_pos: float = 0.02
    kalman_process_vel: float = 0.002
    kalman_measurement: float = 0.08
    point_bonus: float = 0.85


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int] = (1, 1), dropout: float = 0.0):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class TrajectoryEnergyNet(nn.Module):
    """Predict a dense trajectory energy map and coarse motion heads."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        d = int(c.base_dim)
        h = int(c.hidden_dim)
        self.enc1 = ConvBlock(int(c.in_channels), d, dropout=float(c.dropout))
        self.enc2 = ConvBlock(d, d * 2, stride=(1, 2), dropout=float(c.dropout))
        self.enc3 = ConvBlock(d * 2, h, stride=(2, 2), dropout=float(c.dropout))
        self.bottleneck = ConvBlock(h, h, dropout=float(c.dropout))
        self.up2 = UpBlock(h, d * 2, d * 2, dropout=float(c.dropout))
        self.up1 = UpBlock(d * 2, d, d, dropout=float(c.dropout))
        self.energy_head = nn.Sequential(
            nn.Conv2d(d, max(8, d // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(8, d // 2), 1, kernel_size=1),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, int(c.n_channels)),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 2),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(d, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        feat = self.bottleneck(enc3)
        feat = self.up2(feat, enc2)
        feat = self.up1(feat, enc1)
        pooled = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return {
            "energy_logits": self.energy_head(feat).squeeze(1),
            "visibility_logits": self.visibility_head(pooled),
            "direction_logits": self.direction_head(pooled),
            "speed": self.speed_head(pooled).squeeze(-1),
        }


def build_energy_target(
    time: torch.Tensor,
    visibility: torch.Tensor,
    *,
    n_channels: int,
    time_bins: int,
    sigma_ch: float = 0.8,
    sigma_t: float = 2.0,
) -> torch.Tensor:
    time = time.to(torch.float32)
    visibility = visibility.to(torch.float32)
    ch_axis = torch.arange(int(n_channels), dtype=torch.float32).view(-1, 1)
    t_axis = torch.arange(int(time_bins), dtype=torch.float32).view(1, -1)
    mask = torch.zeros((int(n_channels), int(time_bins)), dtype=torch.float32)
    for ch in torch.where(visibility > 0.5)[0].tolist():
        t_ds = float(np.clip(float(time[ch].item()) * float(max(1, time_bins - 1)), 0.0, float(max(0, time_bins - 1))))
        gc = torch.exp(-0.5 * ((ch_axis - float(ch)) / float(max(1e-6, sigma_ch))) ** 2)
        gt = torch.exp(-0.5 * ((t_axis - t_ds) / float(max(1e-6, sigma_t))) ** 2)
        mask = torch.maximum(mask, gc * gt)
    return mask.clamp(0.0, 1.0)


def _expected_time_from_energy(energy_logits: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    logits = energy_logits.to(torch.float32)
    probs = torch.softmax(logits / float(max(1e-3, temperature)), dim=-1)
    bins = torch.linspace(0.0, 1.0, int(logits.shape[-1]), device=logits.device, dtype=probs.dtype)
    return torch.sum(probs * bins.view(1, 1, -1), dim=-1)


def trajectory_energy_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_energy: torch.Tensor,
    target_visibility: torch.Tensor,
    target_time: Optional[torch.Tensor] = None,
    target_direction: Optional[torch.Tensor] = None,
    target_speed: Optional[torch.Tensor] = None,
    energy_weight: float = 1.0,
    visibility_weight: float = 0.4,
    direction_weight: float = 0.2,
    speed_weight: float = 0.2,
    time_weight: float = 0.3,
    speed_norm_kmh: float = 150.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    energy_logits = outputs["energy_logits"]
    energy_tgt = target_energy.to(energy_logits.device, dtype=energy_logits.dtype)
    vis_tgt = target_visibility.to(energy_logits.device, dtype=energy_logits.dtype)
    loss_energy = F.binary_cross_entropy_with_logits(energy_logits, energy_tgt, reduction="mean")
    visibility_logits = outputs["visibility_logits"]
    loss_visibility = F.binary_cross_entropy_with_logits(visibility_logits, vis_tgt, reduction="mean")
    loss_direction = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if target_direction is not None:
        loss_direction = F.cross_entropy(outputs["direction_logits"], target_direction.to(energy_logits.device, dtype=torch.long), reduction="mean")
    loss_speed = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if target_speed is not None:
        speed_tgt = target_speed.to(energy_logits.device, dtype=energy_logits.dtype)
        speed_pred = torch.sigmoid(outputs["speed"]) * float(speed_norm_kmh)
        loss_speed = F.smooth_l1_loss(speed_pred, speed_tgt * float(speed_norm_kmh), reduction="mean")
    pred_time = _expected_time_from_energy(energy_logits)
    visible_mask = vis_tgt > 0.5
    loss_time = torch.zeros((), device=energy_logits.device, dtype=energy_logits.dtype)
    if target_time is not None and bool(visible_mask.any()):
        time_tgt = target_time.to(energy_logits.device, dtype=energy_logits.dtype)
        loss_time = F.smooth_l1_loss(pred_time[visible_mask], time_tgt[visible_mask], reduction="mean")
    total = (
        float(energy_weight) * loss_energy
        + float(visibility_weight) * loss_visibility
        + float(direction_weight) * loss_direction
        + float(speed_weight) * loss_speed
        + float(time_weight) * loss_time
    )
    return total, {
        "loss_energy": loss_energy.detach(),
        "loss_visibility": loss_visibility.detach(),
        "loss_direction": loss_direction.detach(),
        "loss_speed": loss_speed.detach(),
        "loss_time": loss_time.detach(),
    }


def score_trajectory_window(outputs: dict[str, torch.Tensor], *, speed_norm_kmh: float = 150.0) -> dict[str, float]:
    energy = torch.sigmoid(outputs["energy_logits"].detach())
    visibility = torch.sigmoid(outputs["visibility_logits"].detach())
    peak = float(torch.max(energy).item())
    mean = float(torch.mean(energy).item())
    vis = float(torch.mean(visibility).item())
    speed_kmh = float(torch.sigmoid(outputs["speed"].detach()).item() * float(speed_norm_kmh))
    return {
        "confidence": float(max(0.0, peak - mean) * 0.7 + vis * 0.3),
        "energy_peak": peak,
        "energy_mean": mean,
        "visibility_mean": vis,
        "speed_kmh": speed_kmh,
    }


def _kalman_smooth_times(times: np.ndarray, visible: np.ndarray) -> np.ndarray:
    idx = np.where(visible > 0.5)[0]
    if idx.size < 2:
        return times
    x = idx.astype(np.float32)
    y = times[idx].astype(np.float32)
    state = np.array([float(y[0]), float((y[-1] - y[0]) / max(1.0, x[-1] - x[0]))], dtype=np.float32)
    cov = np.diag([1.0, 1.0]).astype(np.float32)
    q = np.diag([0.01, 0.001]).astype(np.float32)
    r = np.array([[0.08]], dtype=np.float32)
    H = np.array([[1.0, 0.0]], dtype=np.float32)
    smoothed = times.copy().astype(np.float32)
    last_ch = int(idx[0])
    for ch in range(int(idx[0]), int(idx[-1]) + 1):
        if ch > last_ch:
            dt = float(ch - last_ch)
            Fm = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            state = Fm @ state
            cov = Fm @ cov @ Fm.T + q
            last_ch = ch
        if visible[ch] > 0.5:
            z = np.array([[float(times[ch])]], dtype=np.float32)
            yk = z - H @ state
            S = H @ cov @ H.T + r
            K = cov @ H.T @ np.linalg.inv(S)
            state = state + (K @ yk).reshape(-1)
            cov = (np.eye(2, dtype=np.float32) - K @ H) @ cov
            smoothed[ch] = float(state[0])
    return np.clip(smoothed, 0.0, 1.0)


def _path_overlap(a: Track, b: Track, *, tol_samples: int, min_overlap_channels: int) -> tuple[float, int]:
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
    groups: list[list[Track]] = []
    for tr in ordered:
        placed = False
        for group in groups:
            ratio, common = _path_overlap(tr, group[0], tol_samples=int(tol_samples), min_overlap_channels=int(min_overlap_channels))
            if common >= int(min_overlap_channels) and ratio >= float(min_overlap_ratio):
                group.append(tr)
                placed = True
                break
        if not placed:
            groups.append([tr])
    kept = [max(group, key=lambda tr: tr.total_score) for group in groups]
    return sorted(kept, key=lambda tr: tr.total_score, reverse=True)


def _graph_config_from_inference(cfg: InferenceConfig) -> ExtractorConfig:
    return ExtractorConfig(
        prominence=float(max(1e-4, cfg.peak_prominence)),
        min_peak_distance=int(max(1, cfg.peak_distance_samples)),
        max_skip_channels=int(max(1, cfg.max_skip_channels)),
        edge_relax_enabled=True,
        edge_min_track_channels=int(max(2, cfg.edge_min_track_channels)),
        min_track_channels=int(max(2, cfg.min_track_channels)),
        min_track_score=float(max(1.0, cfg.min_track_score)),
        nms_time_radius=int(max(1, cfg.dedup_time_tol)),
        nms_channel_radius=int(max(0, cfg.suppression_channel_radius)),
        max_tracks=128,
    )


def _smooth_track_with_kalman(track: Track, *, n_time_bins: int) -> Track:
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return track
    ts = np.array([float(p.t_idx) for p in points], dtype=np.float32)
    vis = np.ones_like(ts, dtype=np.float32)
    smoothed = _kalman_smooth_times(ts / float(max(1, int(n_time_bins) - 1)), vis)
    new_points: list[TrackPoint] = []
    for point, t_norm in zip(points, smoothed):
        t_idx = int(round(float(np.clip(t_norm, 0.0, 1.0)) * float(max(1, int(n_time_bins) - 1))))
        new_points.append(
            TrackPoint(
                ch_idx=int(point.ch_idx),
                t_idx=int(t_idx),
                time_s=float(point.time_s),
                offset_m=float(point.offset_m),
                amp=float(point.amp),
                score=float(point.score),
            )
        )
    return Track(
        track_id=int(track.track_id),
        direction=str(track.direction),
        points=new_points,
        total_score=float(track.total_score),
        mean_speed_kmh=float(track.mean_speed_kmh),
    )


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    ordered = sorted(points, key=lambda p: int(p.ch_idx))
    times = np.array([float(p.time_s) for p in ordered], dtype=np.float64)
    chs = np.array([float(p.ch_idx) for p in ordered], dtype=np.float64)
    dt = np.diff(times)
    dch = np.diff(chs)
    valid = np.abs(dt) > 1e-9
    if not np.any(valid):
        return float("nan")
    speed_mps = np.abs(dch[valid]) * float(dx_m) / np.abs(dt[valid])
    return float(3.6 * np.mean(speed_mps))


def _scene_active_ratio(energy: np.ndarray, *, threshold: float = 0.02) -> float:
    arr = np.asarray(energy, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr >= float(threshold)))


def _select_decoder_profile(cfg: InferenceConfig, energy: np.ndarray) -> InferenceConfig:
    profile = str(cfg.decoder_profile).strip().lower()
    if profile not in {"auto", "balanced", "strict", "recall"}:
        profile = "auto"
    if profile == "auto":
        active_ratio = _scene_active_ratio(energy, threshold=0.02)
        profile = "strict" if active_ratio >= float(cfg.scene_active_ratio_threshold) else "recall"
    if profile == "strict":
        return replace(
            cfg,
            min_visible_channels=int(max(2, cfg.min_visible_channels + 1)),
            min_track_channels=int(max(2, cfg.min_track_channels + 1)),
            min_track_score=float(max(0.6, cfg.min_track_score * 1.15)),
            seed_threshold=float(max(0.02, cfg.seed_threshold * 1.15)),
            peak_prominence=float(max(1e-4, cfg.peak_prominence * 1.1)),
            peak_min_height=float(max(1e-4, cfg.peak_min_height * 1.1)),
            dedup_overlap_ratio=float(min(0.82, cfg.dedup_overlap_ratio + 0.05)),
        )
    if profile == "recall":
        return replace(
            cfg,
            min_visible_channels=int(max(2, cfg.min_visible_channels - 1)),
            min_track_channels=int(max(2, cfg.min_track_channels - 1)),
            min_track_score=float(max(0.5, cfg.min_track_score * 0.9)),
            seed_threshold=float(max(0.01, cfg.seed_threshold * 0.85)),
            peak_prominence=float(max(1e-4, cfg.peak_prominence * 0.85)),
            peak_min_height=float(max(1e-4, cfg.peak_min_height * 0.85)),
            dedup_overlap_ratio=float(max(0.45, cfg.dedup_overlap_ratio - 0.04)),
        )
    return cfg


def _seed_peaks(energy: np.ndarray, *, config: InferenceConfig) -> list[tuple[int, int, float]]:
    filt = maximum_filter(energy, size=(3, 11), mode="nearest")
    mask = (energy >= filt) & (energy >= float(config.seed_threshold))
    ys, xs = np.where(mask)
    items = [(int(y), int(x), float(energy[y, x])) for y, x in zip(ys, xs)]
    items.sort(key=lambda item: item[2], reverse=True)
    return items


def _expected_step_bounds(
    *,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
) -> tuple[float, float]:
    vmin_mps = max(1e-6, float(vmin_kmh) / 3.6)
    vmax_mps = max(1e-6, float(vmax_kmh) / 3.6)
    if str(direction).lower() == "forward":
        low = float(dx_m) / vmax_mps * float(fs)
        high = float(dx_m) / vmin_mps * float(fs)
    elif str(direction).lower() == "reverse":
        low = -float(dx_m) / vmin_mps * float(fs)
        high = -float(dx_m) / vmax_mps * float(fs)
    else:
        raise ValueError("direction must be forward or reverse")
    return float(low), float(high)


def _sample_line_track(
    energy: np.ndarray,
    *,
    slope: float,
    intercept: float,
    search_radius: int,
    min_visible_channels: int,
) -> tuple[list[TrackPoint], float]:
    n_ch, n_t = energy.shape
    points: list[TrackPoint] = []
    total = 0.0
    radius = int(max(1, search_radius))
    for ch in range(n_ch):
        pred = float(intercept) + float(slope) * float(ch)
        if not np.isfinite(pred):
            continue
        center = int(round(pred))
        if center < 0 or center >= n_t:
            continue
        lo = max(0, center - radius)
        hi = min(n_t, center + radius + 1)
        if lo >= hi:
            continue
        local = energy[ch, lo:hi]
        rel = int(np.argmax(local))
        t_idx = int(lo + rel)
        score = float(local[rel])
        if score <= 0.0:
            continue
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx),
                offset_m=float(ch),
                amp=float(score),
                score=float(score),
            )
        )
        total += float(score)
    if len(points) < int(max(2, min_visible_channels)):
        return [], 0.0
    return points, float(total / float(max(1, len(points))))


def _channel_peak_candidates(
    energy: np.ndarray,
    *,
    config: InferenceConfig,
) -> list[list[tuple[int, float]]]:
    n_ch, n_t = energy.shape
    candidates: list[list[tuple[int, float]]] = []
    for ch in range(n_ch):
        row = energy[ch]
        peaks, _props = find_peaks(
            row,
            height=float(config.peak_min_height),
            prominence=float(config.peak_prominence),
            distance=int(config.peak_distance_samples),
        )
        if peaks.size == 0:
            peaks = np.argsort(row)[-max(1, int(config.candidate_topk_per_channel)) :]
        peaks = np.unique(np.clip(peaks, 0, n_t - 1))
        amps = row[peaks]
        order = np.argsort(amps)[-int(config.candidate_topk_per_channel) :]
        peaks = peaks[order]
        amps = amps[order]
        pairs = [(int(t), float(a)) for t, a in zip(peaks.tolist(), amps.tolist()) if float(a) > 0.0]
        candidates.append(pairs)
    return candidates


def _line_hough_candidates(
    energy: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: InferenceConfig,
    max_slopes: int = 96,
    intercept_bin: float = 10.0,
    top_bins_per_slope: int = 4,
) -> list[tuple[float, float, float]]:
    step_low, step_high = _expected_step_bounds(
        fs=float(fs),
        dx_m=float(dx_m),
        direction=str(direction),
        vmin_kmh=float(vmin_kmh),
        vmax_kmh=float(vmax_kmh),
    )
    if not np.isfinite(step_low) or not np.isfinite(step_high):
        return []
    if step_low > step_high:
        step_low, step_high = step_high, step_low
    slopes = np.linspace(float(step_low), float(step_high), num=int(max(8, max_slopes)), dtype=np.float32)
    candidates_by_ch = _channel_peak_candidates(energy, config=config)
    n_ch, _n_t = energy.shape
    intercept_bin = float(max(1.0, intercept_bin))
    results: list[tuple[float, float, float]] = []
    for slope in slopes:
        votes: dict[int, float] = {}
        support: dict[int, int] = {}
        for ch in range(n_ch):
            for t_idx, amp in candidates_by_ch[ch]:
                intercept = float(t_idx) - float(slope) * float(ch)
                bin_idx = int(round(intercept / intercept_bin))
                votes[bin_idx] = votes.get(bin_idx, 0.0) + float(max(0.0, amp))
                support[bin_idx] = support.get(bin_idx, 0) + 1
        if not votes:
            continue
        ranked = sorted(votes.items(), key=lambda item: item[1], reverse=True)[: int(max(1, top_bins_per_slope))]
        for bin_idx, vote in ranked:
            if support.get(bin_idx, 0) < int(max(2, config.min_visible_channels)):
                continue
            intercept = float(bin_idx) * intercept_bin
            results.append((float(slope), float(intercept), float(vote)))
    results.sort(key=lambda item: item[2], reverse=True)
    return results


def _extract_tracks_by_hough_scan(
    energy: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: InferenceConfig,
    max_tracks: int,
) -> list[Track]:
    if int(max_tracks) <= 0:
        return []
    work = np.clip(np.asarray(energy, dtype=np.float32), 0.0, None).copy()
    n_ch, n_t = work.shape
    search_radius = int(max(3, min(28, round(float(np.mean([float(n_t) / max(1, n_ch), 8.0])) * 0.25))))
    suppress_ch = int(max(1, config.suppression_channel_radius))
    suppress_t = int(max(24, min(int(config.suppression_time_radius), int(max(32, round(float(config.peak_distance_samples) * 4.0))))))
    min_accept_len = int(max(4, config.min_visible_channels))
    min_accept_score = float(max(0.0, config.seed_threshold * 0.5))
    candidate_infos: list[tuple[float, float, float, list[TrackPoint]]] = []
    for _pass in range(3):
        candidates = _line_hough_candidates(
            work,
            fs=float(fs),
            dx_m=float(dx_m),
            direction=str(direction),
            vmin_kmh=float(vmin_kmh),
            vmax_kmh=float(vmax_kmh),
            config=config,
            max_slopes=192,
            intercept_bin=float(max(2.0, config.peak_distance_samples * 0.25)),
            top_bins_per_slope=8,
        )
        if not candidates:
            break
        for slope, intercept, vote in candidates[: min(len(candidates), 192)]:
            points, score = _sample_line_track(
                work,
                slope=float(slope),
                intercept=float(intercept),
                search_radius=int(search_radius),
                min_visible_channels=int(min_accept_len),
            )
            if len(points) < int(min_accept_len) or score <= float(min_accept_score):
                continue
            candidate_infos.append((float(score), float(vote), float(slope), points))
        if not candidate_infos:
            break
        seed = max(candidate_infos, key=lambda item: item[0] * (0.7 + 0.3 * item[1]))
        for point in seed[3]:
            ch = int(point.ch_idx)
            t = int(point.t_idx)
            lo_ch = max(0, ch - suppress_ch)
            hi_ch = min(work.shape[0], ch + suppress_ch + 1)
            lo_t = max(0, t - suppress_t)
            hi_t = min(work.shape[1], t + suppress_t + 1)
            work[lo_ch:hi_ch, lo_t:hi_t] *= 0.12

    tracks: list[Track] = []
    for track_id, (score, vote, slope, points) in enumerate(
        sorted(candidate_infos, key=lambda item: item[0] * (0.7 + 0.3 * item[1]), reverse=True)[: int(max_tracks)]
    ):
        fitted_points: list[TrackPoint] = []
        for point in points:
            fitted_points.append(
                TrackPoint(
                    ch_idx=int(point.ch_idx),
                    t_idx=int(point.t_idx),
                    time_s=float(point.t_idx) / float(max(1e-6, fs)),
                    offset_m=float(point.ch_idx) * float(dx_m),
                    amp=float(point.amp),
                    score=float(point.score),
                )
            )
        tracks.append(
            Track(
                track_id=int(track_id),
                direction=str(direction),
                points=fitted_points,
                total_score=float(score + 0.01 * vote + 1e-4 * abs(float(slope))),
                mean_speed_kmh=float(_mean_speed_kmh(fitted_points, float(dx_m))),
            )
        )
    return tracks


def _extract_tracks_by_line_scan(
    energy: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: InferenceConfig,
    max_tracks: int,
) -> list[Track]:
    if int(max_tracks) <= 0:
        return []
    step_low, step_high = _expected_step_bounds(
        fs=float(fs),
        dx_m=float(dx_m),
        direction=str(direction),
        vmin_kmh=float(vmin_kmh),
        vmax_kmh=float(vmax_kmh),
    )
    if not np.isfinite(step_low) or not np.isfinite(step_high):
        return []
    if step_low > step_high:
        step_low, step_high = step_high, step_low
    slopes = np.linspace(float(step_low), float(step_high), num=48, dtype=np.float32)
    work = np.clip(np.asarray(energy, dtype=np.float32), 0.0, None).copy()
    seed_items = _seed_peaks(work, config=config)
    if not seed_items:
        return []
    seed_items = seed_items[: min(len(seed_items), 96)]
    tracks: list[Track] = []
    suppression_radius = int(max(6, round(float(np.mean(np.abs(slopes))) * 0.06)))
    search_radius = int(max(3, min(24, round(float(np.mean(np.abs(slopes))) * 0.04))))
    min_accept_len = int(max(10, int(config.min_visible_channels) * 2))
    min_accept_score = float(max(float(config.seed_threshold), 0.25))

    for track_id in range(int(max_tracks)):
        best: tuple[float, float, list[TrackPoint]] | None = None
        best_score = float("-inf")
        for seed_ch, seed_t, seed_score in seed_items:
            if seed_score <= 0.0:
                continue
            for slope in slopes:
                intercept = float(seed_t) - float(slope) * float(seed_ch)
                points, score = _sample_line_track(
                    work,
                    slope=float(slope),
                    intercept=float(intercept),
                    search_radius=int(search_radius),
                    min_visible_channels=int(config.min_visible_channels),
                )
                if len(points) < int(max(2, config.min_visible_channels)):
                    continue
                coverage = float(len(points)) / float(max(1, work.shape[0]))
                score = float(score) * (0.5 + 0.5 * coverage)
                if score > best_score:
                    best_score = float(score)
                    best = (float(slope), float(intercept), points)
        if best is None or best_score <= min_accept_score:
            break
        slope, intercept, points = best
        if len(points) < min_accept_len:
            break
        fitted_points: list[TrackPoint] = []
        for point in points:
            fitted_points.append(
                TrackPoint(
                    ch_idx=int(point.ch_idx),
                    t_idx=int(point.t_idx),
                    time_s=float(point.t_idx) / float(max(1e-6, fs)),
                    offset_m=float(point.ch_idx) * float(dx_m),
                    amp=float(point.amp),
                    score=float(point.score),
                )
            )
        tracks.append(
            Track(
                track_id=int(track_id),
                direction=str(direction),
                points=fitted_points,
                total_score=float(best_score),
                mean_speed_kmh=float(3.6 * float(dx_m) / max(1e-6, float(np.mean(np.abs(slopes)))) * float(fs)),
            )
        )
        for point in fitted_points:
            ch = int(point.ch_idx)
            t = int(point.t_idx)
            lo = max(0, t - suppression_radius)
            hi = min(work.shape[1], t + suppression_radius + 1)
            work[ch, lo:hi] *= 0.15
        seed_items = _seed_peaks(work, config=config)
        seed_items = seed_items[: min(len(seed_items), 96)]
        if not seed_items:
            break
    return tracks


def _decode_path(
    energy: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    direction: str,
    speed_kmh: float,
    config: InferenceConfig,
    anchor_ch: int = 0,
    anchor_t: Optional[int] = None,
) -> tuple[list[TrackPoint], float]:
    n_ch, n_t = energy.shape
    candidate_t = []
    for ch in range(n_ch):
        row = energy[ch]
        peaks, props = find_peaks(
            row,
            height=float(config.peak_min_height),
            prominence=float(config.peak_prominence),
            distance=int(config.peak_distance_samples),
        )
        if peaks.size == 0:
            peaks = np.argsort(row)[-max(1, int(config.candidate_topk_per_channel)) :]
        peaks = np.unique(np.clip(peaks, 0, n_t - 1))
        amps = row[peaks]
        order = np.argsort(amps)[-int(config.candidate_topk_per_channel) :]
        take = np.sort(peaks[order])
        candidate_t.append([(int(t), float(row[int(t)])) for t in take])
    if not candidate_t or not candidate_t[0]:
        return [], 0.0

    v_ms = max(1e-3, float(speed_kmh) / 3.6)
    expected_step = float(dx_m) / v_ms * float(fs) / float(max(1, config.time_downsample))
    if str(direction).lower() == "reverse":
        expected_step = -expected_step

    seed_ch = int(max(0, min(n_ch - 1, int(anchor_ch))))
    start_candidates = candidate_t[seed_ch]
    if anchor_t is not None:
        start_candidates = sorted(start_candidates, key=lambda item: abs(int(item[0]) - int(anchor_t)))
    start_t, start_score = start_candidates[0]
    search_tol = float(max(float(config.peak_distance_samples) * 4.0, abs(float(expected_step)) * 0.4, 12.0))
    selected: dict[int, tuple[int, float]] = {seed_ch: (int(start_t), float(start_score))}
    total_score = float(np.log(max(1e-6, float(start_score))) + float(config.point_bonus))

    prev_t = int(start_t)
    for ch in range(seed_ch + 1, n_ch):
        predicted = float(prev_t) + float(expected_step)
        cand = candidate_t[ch]
        if not cand:
            continue
        best_item: Optional[tuple[int, float]] = None
        best_cost = float("inf")
        for t, score in cand:
            diff = abs(float(t) - predicted)
            if diff > search_tol:
                continue
            cost = diff - 40.0 * float(score)
            if cost < best_cost:
                best_cost = cost
                best_item = (int(t), float(score))
        if best_item is None:
            continue
        selected[ch] = best_item
        total_score += float(np.log(max(1e-6, float(best_item[1]))) + float(config.point_bonus) - 0.0015 * abs(float(best_item[0]) - predicted))
        prev_t = int(best_item[0])

    next_t = int(start_t)
    for ch in range(seed_ch - 1, -1, -1):
        predicted = float(next_t) - float(expected_step)
        cand = candidate_t[ch]
        if not cand:
            continue
        best_item = None
        best_cost = float("inf")
        for t, score in cand:
            diff = abs(float(t) - predicted)
            if diff > search_tol:
                continue
            cost = diff - 40.0 * float(score)
            if cost < best_cost:
                best_cost = cost
                best_item = (int(t), float(score))
        if best_item is None:
            continue
        selected[ch] = best_item
        total_score += float(np.log(max(1e-6, float(best_item[1]))) + float(config.point_bonus) - 0.0015 * abs(float(best_item[0]) - predicted))
        next_t = int(best_item[0])

    if len(selected) < int(max(2, config.min_visible_channels)):
        return [], float(total_score)

    path = [(int(ch), int(t_score[0]), float(t_score[1])) for ch, t_score in sorted(selected.items(), key=lambda item: item[0])]
    points: list[TrackPoint] = []
    ts = np.array([p[1] for p in path], dtype=np.float32)
    vis = np.ones_like(ts, dtype=np.float32)
    smoothed_ts = _kalman_smooth_times(ts / float(max(1, n_t - 1)), vis)
    for (ch, t, score), t_sm in zip(path, smoothed_ts):
        t_sm_samples = float(t_sm) * float(max(1, n_t - 1))
        points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(round(t_sm_samples)),
                time_s=float(t_sm_samples) * float(config.time_downsample) / float(fs),
                offset_m=float(ch) * float(dx_m),
                amp=float(score),
                score=float(score),
            )
        )
    return points, float(total_score)


def extract_vehicle_tracks_from_energy(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[InferenceConfig | dict[str, Any]] = None,
) -> list[Track]:
    cfg = config if isinstance(config, InferenceConfig) else InferenceConfig(**config) if isinstance(config, dict) else InferenceConfig()
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must have shape [n_channel, n_sample]")
    energy = np.clip(arr, 0.0, None).astype(np.float32, copy=False)
    cfg = _select_decoder_profile(cfg, energy)
    fs_eff = float(fs) / float(max(1, int(cfg.time_downsample)))
    graph_cfg = _graph_config_from_inference(cfg)
    graph_cfg = replace(
        graph_cfg,
        max_tracks=int(max(1, graph_cfg.max_tracks)),
        min_track_channels=int(max(2, graph_cfg.min_track_channels)),
        min_track_score=float(max(0.5, graph_cfg.min_track_score)),
        max_skip_channels=int(max(1, graph_cfg.max_skip_channels)),
        prominence=float(max(1e-4, min(graph_cfg.prominence, 0.08))),
        min_peak_distance=int(max(1, min(graph_cfg.min_peak_distance, 12))),
        edge_relax_enabled=True,
    )

    direction_mode = str(direction).lower().strip()
    if direction_mode not in {"forward", "reverse", "both"}:
        raise ValueError("direction must be forward, reverse, or both")
    dir_list = ["forward", "reverse"] if direction_mode == "both" else [direction_mode]

    tracks: list[Track] = []
    for dir_name in dir_list:
        tracks.extend(
            extract_all(
                energy,
                fs=float(fs_eff),
                dx_m=float(dx_m),
                direction=str(dir_name),
                vmin_kmh=float(vmin_kmh),
                vmax_kmh=float(vmax_kmh),
                config=graph_cfg,
            )
        )

    residual = energy.copy()
    if tracks:
        n_t = residual.shape[1]
        for tr in tracks:
            for point in tr.points:
                ch = int(point.ch_idx)
                t = int(point.t_idx)
                lo = max(0, t - int(max(6, round(0.08 * n_t))))
                hi = min(n_t, t + int(max(6, round(0.08 * n_t))) + 1)
                residual[ch, lo:hi] *= 0.15

    hough_tracks = _extract_tracks_by_hough_scan(
        residual,
        fs=float(fs_eff),
        dx_m=float(dx_m),
        direction=direction_mode if direction_mode != "both" else "forward",
        vmin_kmh=float(vmin_kmh),
        vmax_kmh=float(vmax_kmh),
        config=cfg,
        max_tracks=int(max(0, graph_cfg.max_tracks - len(tracks))),
    )
    if direction_mode == "both":
        hough_tracks.extend(
            _extract_tracks_by_hough_scan(
                residual,
                fs=float(fs_eff),
                dx_m=float(dx_m),
                direction="reverse",
                vmin_kmh=float(vmin_kmh),
                vmax_kmh=float(vmax_kmh),
                config=cfg,
                max_tracks=int(max(0, graph_cfg.max_tracks - len(tracks))),
            )
        )
    tracks.extend(hough_tracks)

    supplement = _extract_tracks_by_line_scan(
        residual,
        fs=float(fs_eff),
        dx_m=float(dx_m),
        direction=direction_mode if direction_mode != "both" else "forward",
        vmin_kmh=float(vmin_kmh),
        vmax_kmh=float(vmax_kmh),
        config=cfg,
        max_tracks=int(max(0, graph_cfg.max_tracks - len(tracks))),
    )
    if direction_mode == "both":
        supplement.extend(
            _extract_tracks_by_line_scan(
                residual,
                fs=float(fs_eff),
                dx_m=float(dx_m),
                direction="reverse",
                vmin_kmh=float(vmin_kmh),
                vmax_kmh=float(vmax_kmh),
                config=cfg,
                max_tracks=int(max(0, graph_cfg.max_tracks - len(tracks))),
            )
        )
    tracks.extend(supplement)

    if not tracks:
        return []

    smoothed = [_smooth_track_with_kalman(tr, n_time_bins=int(energy.shape[1])) for tr in tracks]
    return _deduplicate_tracks(
        smoothed,
        tol_samples=int(cfg.dedup_time_tol),
        min_overlap_channels=int(cfg.dedup_channel_overlap),
        min_overlap_ratio=float(cfg.dedup_overlap_ratio),
    )
