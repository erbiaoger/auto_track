"""Query-based multi-vehicle trajectory set predictor.

This is a fresh direction for the vehicle-reconstruction task:

- one forward pass emits a fixed-size set of trajectory hypotheses
- Hungarian matching assigns each hypothesis to one ground-truth vehicle
- the decoder is allowed to use graph/Kalman/NMS style post-processing

The model predicts per-query per-channel time and visibility, which is a much
more direct instance-level formulation than the older heatmap/slot routes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.core.track_extractor_graph import ExtractorConfig, extract_all
from autotrack.dl.trajectory_energy_model import ConvBlock, UpBlock, _smooth_track_with_kalman


@dataclass
class VehicleSetModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    base_dim: int = 32
    hidden_dim: int = 96
    query_dim: int = 128
    num_queries: int = 32
    decoder_layers: int = 2
    decoder_heads: int = 4
    dropout: float = 0.1
    speed_norm_kmh: float = 150.0
    min_visible_channels: int = 3
    objectness_threshold: float = 0.35
    visibility_threshold: float = 0.5
    dedup_time_tol: int = 800
    dedup_channel_overlap: int = 3
    dedup_overlap_ratio: float = 0.6
    kalman_smooth: bool = True
    max_output_tracks: Optional[int] = None


def _as_config(config: Optional[VehicleSetModelConfig | dict[str, Any]]) -> VehicleSetModelConfig:
    if config is None:
        return VehicleSetModelConfig()
    if isinstance(config, VehicleSetModelConfig):
        return config
    if isinstance(config, dict):
        defaults = VehicleSetModelConfig()
        return VehicleSetModelConfig(
            n_channels=int(config.get("n_channels", defaults.n_channels)),
            in_channels=int(config.get("in_channels", defaults.in_channels)),
            base_dim=int(config.get("base_dim", defaults.base_dim)),
            hidden_dim=int(config.get("hidden_dim", defaults.hidden_dim)),
            query_dim=int(config.get("query_dim", defaults.query_dim)),
            num_queries=int(config.get("num_queries", defaults.num_queries)),
            decoder_layers=int(config.get("decoder_layers", defaults.decoder_layers)),
            decoder_heads=int(config.get("decoder_heads", defaults.decoder_heads)),
            dropout=float(config.get("dropout", defaults.dropout)),
            speed_norm_kmh=float(config.get("speed_norm_kmh", defaults.speed_norm_kmh)),
            min_visible_channels=int(config.get("min_visible_channels", defaults.min_visible_channels)),
            objectness_threshold=float(config.get("objectness_threshold", defaults.objectness_threshold)),
            visibility_threshold=float(config.get("visibility_threshold", defaults.visibility_threshold)),
            dedup_time_tol=int(config.get("dedup_time_tol", defaults.dedup_time_tol)),
            dedup_channel_overlap=int(config.get("dedup_channel_overlap", defaults.dedup_channel_overlap)),
            dedup_overlap_ratio=float(config.get("dedup_overlap_ratio", defaults.dedup_overlap_ratio)),
            kalman_smooth=bool(config.get("kalman_smooth", defaults.kalman_smooth)),
            max_output_tracks=(
                None
                if config.get("max_output_tracks", defaults.max_output_tracks) in {None, "", "none", "None"}
                else int(config.get("max_output_tracks", defaults.max_output_tracks))
            ),
        )
    raise TypeError("config must be VehicleSetModelConfig / dict / None")


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


class VehicleSetNet(nn.Module):
    def __init__(self, config: Optional[VehicleSetModelConfig] = None):
        super().__init__()
        self.config = config or VehicleSetModelConfig()
        c = self.config
        base = int(c.base_dim)
        hidden = int(c.hidden_dim)
        query_dim = int(c.query_dim)
        self.enc1 = ConvBlock(int(c.in_channels), base, dropout=float(c.dropout))
        self.enc2 = ConvBlock(base, base * 2, stride=(1, 2), dropout=float(c.dropout))
        self.enc3 = ConvBlock(base * 2, hidden, stride=(2, 2), dropout=float(c.dropout))
        self.bottleneck = ConvBlock(hidden, hidden, dropout=float(c.dropout))
        self.up2 = UpBlock(hidden, base * 2, base * 2, dropout=float(c.dropout))
        self.up1 = UpBlock(base * 2, base, base, dropout=float(c.dropout))
        self.token_proj = nn.Sequential(
            nn.Conv2d(base, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden // 4)), hidden),
            nn.GELU(),
        )
        self.pos_proj = nn.Sequential(
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.query_embed = nn.Embedding(int(c.num_queries), hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.decoder_heads),
            dim_feedforward=query_dim,
            dropout=float(c.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(c.decoder_layers))
        self.norm = nn.LayerNorm(hidden)
        self.objectness_head = nn.Linear(hidden, 1)
        self.direction_head = nn.Linear(hidden, 2)
        self.speed_head = nn.Linear(hidden, 1)
        self.visibility_head = nn.Linear(hidden, int(c.n_channels))
        self.time_head = nn.Linear(hidden, int(c.n_channels))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        feat = self.bottleneck(enc3)
        feat = self.up2(feat, enc2)
        feat = self.up1(feat, enc1)
        feat = self.token_proj(feat)
        b, hidden, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)
        yy = torch.linspace(0.0, 1.0, h, device=x.device, dtype=tokens.dtype)
        xx = torch.linspace(0.0, 1.0, w, device=x.device, dtype=tokens.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        pos = torch.stack(
            [
                grid_y,
                grid_x,
                torch.sin(2.0 * torch.pi * grid_y),
                torch.cos(2.0 * torch.pi * grid_y),
                torch.sin(2.0 * torch.pi * grid_x),
                torch.cos(2.0 * torch.pi * grid_x),
            ],
            dim=-1,
        ).reshape(-1, 6)
        tokens = tokens + self.pos_proj(pos).unsqueeze(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(tgt=queries, memory=tokens)
        decoded = self.norm(decoded)
        return {
            "objectness_logits": self.objectness_head(decoded).squeeze(-1),
            "direction_logits": self.direction_head(decoded),
            "speed": self.speed_head(decoded).squeeze(-1),
            "visibility_logits": self.visibility_head(decoded),
            "time_logits": self.time_head(decoded),
        }


def _sample_tracks_from_target(target: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    time = target["time"].to(torch.float32)
    visibility = target["visibility"].to(torch.float32)
    direction = target["direction"].to(torch.long)
    speed = target["speed"].to(torch.float32)
    gt_valid = target.get("gt_valid")
    if gt_valid is None:
        gt_valid = torch.ones((time.shape[0],), dtype=torch.bool, device=time.device)
    tracks: list[dict[str, torch.Tensor]] = []
    for g_idx in torch.where(gt_valid.to(torch.bool))[0].tolist():
        tracks.append(
            {
                "time": time[g_idx],
                "visibility": visibility[g_idx],
                "direction": direction[g_idx],
                "speed": speed[g_idx],
            }
        )
    return tracks


def _track_cost(pred: dict[str, torch.Tensor], gt: dict[str, torch.Tensor], *, speed_norm_kmh: float) -> float:
    device = pred["time_logits"].device
    gt_vis = gt["visibility"].to(device=device, dtype=torch.float32)
    visible = gt_vis > 0.5
    if not bool(visible.any()):
        return 1e6
    pred_time = torch.sigmoid(pred["time_logits"]).to(device=device, dtype=torch.float32)
    pred_vis_logits = pred["visibility_logits"].to(device=device, dtype=torch.float32)
    time_cost = torch.mean(torch.abs(pred_time[visible] - gt["time"].to(device=device, dtype=torch.float32)[visible]))
    vis_cost = F.binary_cross_entropy_with_logits(pred_vis_logits, gt_vis, reduction="mean")
    dir_cost = F.cross_entropy(
        pred["direction_logits"].unsqueeze(0),
        gt["direction"].to(device=device, dtype=torch.long).unsqueeze(0),
        reduction="mean",
    )
    speed_pred = torch.sigmoid(pred["speed"]) * float(speed_norm_kmh)
    speed_cost = F.smooth_l1_loss(speed_pred, gt["speed"].to(device=device, dtype=torch.float32), reduction="mean") / float(speed_norm_kmh)
    obj_cost = 1.0 - torch.sigmoid(pred["objectness_logits"])
    return float(4.0 * time_cost.item() + 2.0 * vis_cost.item() + 0.8 * dir_cost.item() + 0.8 * speed_cost.item() + 0.5 * obj_cost.item())


def _hungarian_assign(outputs: dict[str, torch.Tensor], target_tracks: list[dict[str, torch.Tensor]], *, speed_norm_kmh: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    pred_count = int(outputs["objectness_logits"].shape[0])
    gt_count = int(len(target_tracks))
    if pred_count == 0 or gt_count == 0:
        return [], list(range(pred_count)), list(range(gt_count))
    cost = np.zeros((pred_count, gt_count), dtype=np.float32)
    for p_idx in range(pred_count):
        pred = {key: outputs[key][p_idx] for key in outputs.keys()}
        for g_idx, gt in enumerate(target_tracks):
            cost[p_idx, g_idx] = _track_cost(pred, gt, speed_norm_kmh=float(speed_norm_kmh))
    row_ind, col_ind = linear_sum_assignment(cost)
    pairs = [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]
    used_pred = {int(r) for r in row_ind.tolist()}
    used_gt = {int(c) for c in col_ind.tolist()}
    unmatched_pred = [idx for idx in range(pred_count) if idx not in used_pred]
    unmatched_gt = [idx for idx in range(gt_count) if idx not in used_gt]
    return pairs, unmatched_pred, unmatched_gt


def vehicle_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    *,
    speed_norm_kmh: float = 150.0,
    objectness_weight: float = 0.4,
    time_weight: float = 6.0,
    visibility_weight: float = 3.0,
    direction_weight: float = 0.4,
    speed_weight: float = 0.4,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = torch.zeros((), dtype=torch.float32, device=outputs["objectness_logits"].device)
    objectness_losses: list[torch.Tensor] = []
    time_losses: list[torch.Tensor] = []
    visibility_losses: list[torch.Tensor] = []
    direction_losses: list[torch.Tensor] = []
    speed_losses: list[torch.Tensor] = []

    batch = int(outputs["objectness_logits"].shape[0])
    for b_idx in range(batch):
        sample_out = {key: value[b_idx] for key, value in outputs.items()}
        sample_targets = _sample_tracks_from_target(targets[b_idx])
        pairs, unmatched_pred, _unmatched_gt = _hungarian_assign(sample_out, sample_targets, speed_norm_kmh=float(speed_norm_kmh))
        for p_idx, g_idx in pairs:
            pred = {key: sample_out[key][p_idx] for key in sample_out.keys()}
            gt = sample_targets[g_idx]
            device = pred["time_logits"].device
            gt_time = gt["time"].to(device=device, dtype=torch.float32)
            gt_vis = gt["visibility"].to(device=device, dtype=torch.float32)
            gt_dir = gt["direction"].to(device=device, dtype=torch.long)
            gt_speed = gt["speed"].to(device=device, dtype=torch.float32)
            visible = gt_vis > 0.5
            if bool(visible.any()):
                time_losses.append(F.smooth_l1_loss(torch.sigmoid(pred["time_logits"])[visible], gt_time[visible], reduction="mean"))
            visibility_losses.append(F.binary_cross_entropy_with_logits(pred["visibility_logits"], gt_vis, reduction="mean"))
            direction_losses.append(F.cross_entropy(pred["direction_logits"].unsqueeze(0), gt_dir.unsqueeze(0), reduction="mean"))
            speed_pred = torch.sigmoid(pred["speed"]) * float(speed_norm_kmh)
            speed_losses.append(F.smooth_l1_loss(speed_pred, gt_speed, reduction="mean") / float(speed_norm_kmh))
            objectness_losses.append(F.binary_cross_entropy_with_logits(pred["objectness_logits"].unsqueeze(0), torch.ones(1, device=pred["objectness_logits"].device), reduction="mean"))
        for p_idx in unmatched_pred:
            pred = {key: sample_out[key][p_idx] for key in sample_out.keys()}
            objectness_losses.append(F.binary_cross_entropy_with_logits(pred["objectness_logits"].unsqueeze(0), torch.zeros(1, device=pred["objectness_logits"].device), reduction="mean"))

    if objectness_losses:
        total = total + float(objectness_weight) * torch.stack(objectness_losses).mean()
    if time_losses:
        total = total + float(time_weight) * torch.stack(time_losses).mean()
    if visibility_losses:
        total = total + float(visibility_weight) * torch.stack(visibility_losses).mean()
    if direction_losses:
        total = total + float(direction_weight) * torch.stack(direction_losses).mean()
    if speed_losses:
        total = total + float(speed_weight) * torch.stack(speed_losses).mean()
    metrics = {
        "loss_objectness": torch.stack(objectness_losses).mean().detach() if objectness_losses else torch.tensor(0.0),
        "loss_time": torch.stack(time_losses).mean().detach() if time_losses else torch.tensor(0.0),
        "loss_visibility": torch.stack(visibility_losses).mean().detach() if visibility_losses else torch.tensor(0.0),
        "loss_direction": torch.stack(direction_losses).mean().detach() if direction_losses else torch.tensor(0.0),
        "loss_speed": torch.stack(speed_losses).mean().detach() if speed_losses else torch.tensor(0.0),
    }
    return total, metrics


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


def refine_track_with_raw_window(
    track: Track,
    raw_window: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    search_radius: int = 6000,
) -> Track:
    raw = np.abs(np.asarray(raw_window, dtype=np.float32))
    if raw.ndim != 2 or not track.points:
        return track
    n_time = int(raw.shape[1])
    radius = int(max(1, search_radius))
    refined: list[TrackPoint] = []
    for point in sorted(track.points, key=lambda p: int(p.ch_idx)):
        ch = int(np.clip(int(point.ch_idx), 0, raw.shape[0] - 1))
        t_pred = int(np.clip(int(point.t_idx), 0, n_time - 1))
        lo = max(0, t_pred - radius)
        hi = min(n_time, t_pred + radius + 1)
        local = raw[ch, lo:hi]
        if local.size == 0:
            refined.append(point)
            continue
        rel = int(np.argmax(local))
        t_idx = int(lo + rel)
        amp = float(local[rel])
        refined.append(
            TrackPoint(
                ch_idx=int(point.ch_idx),
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(max(1e-6, fs)),
                offset_m=float(point.offset_m),
                amp=amp,
                score=max(float(point.score), amp),
            )
        )
    refined_track = Track(
        track_id=int(track.track_id),
        direction=str(track.direction),
        points=refined,
        total_score=float(track.total_score),
        mean_speed_kmh=float(_mean_speed_kmh(refined, float(dx_m))),
    )
    return refined_track


def _graph_candidate_tracks_from_raw_window(
    raw_window: np.ndarray,
    *,
    fs: float,
    dx_m: float,
    single_vehicle_mode: bool = False,
) -> list[Track]:
    raw = np.asarray(raw_window, dtype=np.float32)
    is_single = bool(single_vehicle_mode)
    cfg = ExtractorConfig(
        use_template_enhancement=False,
        enhance_decimate=1,
        prominence=0.06 if is_single else 0.08,
        min_peak_distance=150,
        max_skip_channels=10,
        lambda_speed=2.0,
        lambda_prediction=1.3,
        lambda_skip=0.45,
        speed_change_tolerance_kmh=24.0,
        speed_penalty_power=0.85,
        speed_penalty_cap=2.5,
        prediction_tolerance_ratio=0.22,
        prediction_tolerance_min_seconds=0.05,
        min_track_channels=3 if is_single else 4,
        min_track_score=2.0 if is_single else 2.5,
        edge_relax_enabled=True,
        edge_min_track_channels=3 if is_single else 4,
        edge_time_margin_seconds=12.0,
        edge_min_score_scale=0.2 if is_single else 0.25,
        nms_time_radius=500 if is_single else 600,
        nms_channel_radius=1,
        max_tracks=16 if is_single else 48,
        max_peaks_per_channel=120 if is_single else 180,
        k_best_per_node=4,
    )
    tracks: list[Track] = []
    for direction in ("forward", "reverse"):
        tracks.extend(extract_all(raw, float(fs), float(dx_m), direction, 70.0, 90.0, config=cfg))
    tracks.sort(key=lambda tr: tr.total_score, reverse=True)
    return tracks


def _decode_query_prior_tracks(
    outputs: dict[str, torch.Tensor],
    *,
    raw_time_bins: int,
    fs: float,
    dx_m: float,
    cfg: VehicleSetModelConfig,
) -> list[Track]:
    objectness = torch.sigmoid(outputs["objectness_logits"]).detach().cpu()
    visibility = torch.sigmoid(outputs["visibility_logits"]).detach().cpu()
    time_norm = torch.sigmoid(outputs["time_logits"]).detach().cpu()
    direction = outputs["direction_logits"].detach().cpu()
    speed = torch.sigmoid(outputs["speed"]).detach().cpu() * float(cfg.speed_norm_kmh)
    tracks: list[Track] = []
    for q_idx in torch.argsort(objectness, descending=True).tolist():
        obj_score = float(objectness[q_idx].item())
        if obj_score < float(cfg.objectness_threshold):
            continue
        vis = visibility[q_idx]
        time_row = time_norm[q_idx]
        dir_idx = int(torch.argmax(direction[q_idx]).item())
        direction_name = "forward" if dir_idx == 0 else "reverse"
        selected = torch.where(vis >= float(cfg.visibility_threshold))[0].tolist()
        if len(selected) < int(cfg.min_visible_channels):
            topk = torch.topk(vis, k=min(int(cfg.min_visible_channels), int(vis.shape[0]))).indices.tolist()
            selected = sorted(set(int(item) for item in topk))
        points: list[TrackPoint] = []
        for ch in sorted(selected):
            t_idx = int(round(float(time_row[int(ch)].item()) * float(max(1, int(raw_time_bins) - 1))))
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=int(t_idx),
                    time_s=float(t_idx) / float(max(1e-6, fs)),
                    offset_m=float(ch) * float(dx_m),
                    amp=float(vis[int(ch)].item()),
                    score=float(vis[int(ch)].item()),
                )
            )
        if len(points) < int(cfg.min_visible_channels):
            continue
        track = Track(
            track_id=int(len(tracks)),
            direction=direction_name,
            points=points,
            total_score=float(obj_score * (0.6 + 0.4 * float(vis.mean().item()))),
            mean_speed_kmh=float(speed[q_idx].item()),
        )
        if bool(cfg.kalman_smooth):
            track = _smooth_track_with_kalman(track, n_time_bins=int(raw_time_bins))
        tracks.append(track)
    return tracks


def _best_track_match(
    source: Track,
    candidates: list[Track],
    *,
    tol_samples: int,
    min_overlap_channels: int,
) -> tuple[int, float, int]:
    best_idx = -1
    best_ratio = 0.0
    best_common = 0
    for idx, other in enumerate(candidates):
        ratio, common = _track_overlap(source, other, tol_samples=int(tol_samples), min_overlap_channels=int(min_overlap_channels))
        if common < int(min_overlap_channels):
            continue
        if ratio > best_ratio or (abs(ratio - best_ratio) <= 1e-6 and common > best_common):
            best_idx = int(idx)
            best_ratio = float(ratio)
            best_common = int(common)
    return best_idx, float(best_ratio), int(best_common)


def _prefer_track(a: Track, b: Track) -> Track:
    if float(b.total_score) > float(a.total_score):
        return b
    if float(b.total_score) < float(a.total_score):
        return a
    if len(b.points) > len(a.points):
        return b
    return a


def _blend_track_score(track: Track, query_track: Track, *, overlap_ratio: float) -> Track:
    blended = Track(
        track_id=int(track.track_id),
        direction=str(track.direction),
        points=list(track.points),
        total_score=float(0.72 * float(track.total_score) + 0.28 * float(query_track.total_score) * (0.5 + 0.5 * float(overlap_ratio))),
        mean_speed_kmh=float(track.mean_speed_kmh),
    )
    return blended


def decode_vehicle_set_tracks(
    outputs: dict[str, torch.Tensor],
    *,
    raw_time_bins: int,
    fs: float,
    dx_m: float,
    raw_window: Optional[np.ndarray] = None,
    snap_search_radius: int = 6000,
    config: Optional[VehicleSetModelConfig | dict[str, Any]] = None,
) -> list[Track]:
    cfg = _as_config(config)
    max_tracks = int(cfg.max_output_tracks) if cfg.max_output_tracks is not None and int(cfg.max_output_tracks) > 0 else None
    query_tracks = _decode_query_prior_tracks(
        outputs,
        raw_time_bins=int(raw_time_bins),
        fs=float(fs),
        dx_m=float(dx_m),
        cfg=cfg,
    )
    tracks: list[Track] = []
    if raw_window is not None:
        graph_tracks = _graph_candidate_tracks_from_raw_window(
            raw_window,
            fs=float(fs),
            dx_m=float(dx_m),
            single_vehicle_mode=bool(cfg.max_output_tracks is not None and int(cfg.max_output_tracks) == 1),
        )
        if graph_tracks:
            graph_refined: list[Track] = []
            for track in graph_tracks:
                new_track = track
                if bool(cfg.kalman_smooth):
                    new_track = _smooth_track_with_kalman(new_track, n_time_bins=int(raw_time_bins))
                new_track = refine_track_with_raw_window(new_track, raw_window, fs=float(fs), dx_m=float(dx_m), search_radius=int(snap_search_radius))
                graph_refined.append(new_track)
            matched_query: set[int] = set()
            scored_tracks: list[Track] = []
            for graph_track in graph_refined:
                match_idx, match_ratio, match_common = _best_track_match(
                    graph_track,
                    query_tracks,
                    tol_samples=int(cfg.dedup_time_tol),
                    min_overlap_channels=int(cfg.dedup_channel_overlap),
                )
                if match_idx >= 0:
                    matched_query.add(int(match_idx))
                    query_track = query_tracks[match_idx]
                    if match_common >= int(cfg.dedup_channel_overlap):
                        graph_track = _blend_track_score(graph_track, query_track, overlap_ratio=float(match_ratio))
                        if float(query_track.total_score) > float(graph_track.total_score) and match_ratio >= float(cfg.dedup_overlap_ratio):
                            graph_track = _prefer_track(graph_track, query_track)
                scored_tracks.append(graph_track)
            tracks = scored_tracks
            if not tracks:
                tracks = list(query_tracks)
            elif max_tracks is not None and len(tracks) < int(max_tracks):
                for idx, query_track in enumerate(query_tracks):
                    if idx in matched_query:
                        continue
                    if float(query_track.total_score) < max(1.0, float(cfg.objectness_threshold) * 10.0):
                        continue
                    tracks.append(query_track)
                    if len(tracks) >= int(max_tracks):
                        break
        elif query_tracks:
            tracks = list(query_tracks)
    elif query_tracks:
        tracks = list(query_tracks)

    ordered = sorted(tracks, key=lambda tr: tr.total_score, reverse=True)
    kept: list[Track] = []
    for tr in ordered:
        keep = True
        for other in kept:
            ratio, common = _track_overlap(tr, other, tol_samples=int(cfg.dedup_time_tol), min_overlap_channels=int(cfg.dedup_channel_overlap))
            if common >= int(cfg.dedup_channel_overlap) and ratio >= float(cfg.dedup_overlap_ratio):
                keep = False
                break
        if keep:
            kept.append(tr)
    for idx, track in enumerate(kept):
        track.track_id = int(idx)
        track.mean_speed_kmh = float(_mean_speed_kmh(track.points, float(dx_m)))
    if cfg.max_output_tracks is not None and int(cfg.max_output_tracks) > 0:
        kept = kept[: int(cfg.max_output_tracks)]
    return kept


def save_checkpoint(
    path: str | Path,
    model: VehicleSetNet,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: VehicleSetModelConfig,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "vehicle_set",
        "model_state_dict": model.state_dict(),
        "model_config": asdict(model_config),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(checkpoint_path)


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[VehicleSetNet, dict[str, Any]]:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = VehicleSetModelConfig(**dict(checkpoint.get("model_config", {})))
    model = VehicleSetNet(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint
