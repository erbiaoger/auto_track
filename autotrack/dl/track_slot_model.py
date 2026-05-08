"""Track-slot instance model for DAS vehicle trajectory recognition.

This module predicts a fixed set of trajectory slots. Each slot represents one
candidate vehicle and directly outputs one time value and one visibility value
per DAS channel. Unlike the query-mask model, it never materializes per-slot
`[channel, time]` masks during training or inference.
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
from autotrack.dl.trajectory_set_model import (
    LABEL_TO_DIRECTION,
    WindowDatasetConfig,
    auto_torch_device,
    prepare_window_input,
)


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    max_tracks: int = 96
    hidden_dim: int = 128
    num_heads: int = 4
    decoder_layers: int = 2
    pooled_channels: int = 8
    pooled_time: int = 128
    dropout: float = 0.1


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.5
    visibility_threshold: float = 0.5
    min_visible_channels: int = 3
    max_tracks: int = 96
    dedup_tolerance_samples: int = 180
    dedup_min_overlap_channels: int = 3
    speed_norm_kmh: float = 150.0
    clip_ratio: float = 1.35


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrackSlotPredictor(nn.Module):
    """Predict fixed trajectory slots; valid high-objectness slots become cars."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.backbone = nn.Sequential(
            ConvBlock(int(c.in_channels), 32, stride=(1, 2)),
            ConvBlock(32, 64, stride=(1, 2)),
            ConvBlock(64, hidden, stride=(2, 2)),
            ConvBlock(hidden, hidden, stride=(2, 2)),
        )
        self.pool_size = (int(c.pooled_channels), int(c.pooled_time))
        token_count = int(c.pooled_channels) * int(c.pooled_time)
        self.pos_embed = nn.Parameter(torch.randn(1, token_count, hidden) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(c.decoder_layers))
        self.slot_embed = nn.Embedding(int(c.max_tracks), hidden)
        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(c.n_channels)),
        )
        self.visibility_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))

    def forward(self, x: torch.Tensor, targets: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        del targets
        feat = F.interpolate(
            self.backbone(x),
            size=self.pool_size,
            mode="bilinear",
            align_corners=False,
        )
        memory = feat.flatten(2).transpose(1, 2)
        memory = memory + self.pos_embed[:, : memory.shape[1], :]
        query = self.slot_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        hs = self.decoder(tgt=query, memory=memory)
        return {
            "num_regular_queries": int(self.config.max_tracks),
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "time": torch.sigmoid(self.time_head(hs)),
            "visibility_logits": self.visibility_head(hs),
        }


def _greedy_match_cost(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_count = int(cost.shape[0])
    g_count = int(cost.shape[1])
    k = int(min(q_count, g_count))
    if k <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    work = cost.clone()
    large = torch.finfo(work.dtype).max
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for _ in range(k):
        idx = torch.argmin(work)
        r = torch.div(idx, g_count, rounding_mode="floor").long()
        c = (idx - r * g_count).long()
        rows.append(r)
        cols.append(c)
        work[r, :] = large
        work[:, c] = large
    return torch.stack(rows), torch.stack(cols)


def _match_single(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    b: int,
    *,
    matcher: str,
    w_time: float,
    w_vis: float,
    w_obj: float,
    w_dir: float,
    w_speed: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    gt_valid = targets["gt_valid"][b].to(device=device, dtype=torch.bool)
    g_count = int(gt_valid.sum().item())
    if g_count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    pred_time = outputs["time"][b, :q_count].detach()
    pred_vis = torch.sigmoid(outputs["visibility_logits"][b, :q_count].detach())
    pred_obj = torch.sigmoid(outputs["objectness_logits"][b, :q_count].detach())
    pred_dir = torch.softmax(outputs["direction_logits"][b, :q_count].detach(), dim=-1)
    pred_speed = outputs["speed"][b, :q_count].detach()

    gt_time = targets["time"][b, gt_valid].to(device=device, dtype=torch.float32)
    gt_vis = targets["visibility"][b, gt_valid].to(device=device, dtype=torch.float32)
    gt_dir = targets["direction"][b, gt_valid].to(device=device, dtype=torch.long)
    gt_speed = targets["speed"][b, gt_valid].to(device=device, dtype=torch.float32)

    vis_weight = gt_vis[None, :, :]
    denom = torch.clamp(vis_weight.sum(dim=-1), min=1.0)
    time_cost = (torch.abs(pred_time[:, None, :] - gt_time[None, :, :]) * vis_weight).sum(dim=-1) / denom
    vis_cost = torch.abs(pred_vis[:, None, :] - gt_vis[None, :, :]).mean(dim=-1)
    dir_cost = -pred_dir[:, gt_dir]
    speed_cost = torch.abs(pred_speed[:, None] - gt_speed[None, :])
    obj_cost = -pred_obj[:, None]
    cost = (
        float(w_time) * time_cost
        + float(w_vis) * vis_cost
        + float(w_obj) * obj_cost
        + float(w_dir) * dir_cost
        + float(w_speed) * speed_cost
    )

    if str(matcher).lower() == "greedy":
        return _greedy_match_cost(cost)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


def _weighted_smooth_l1(pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, gt, reduction="none") * weight
    return loss.sum() / torch.clamp(weight.sum(), min=1.0)


def track_slot_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.05,
    matcher: str = "hungarian",
    collect_metrics: bool = True,
    w_time: float = 5.0,
    w_vis: float = 1.0,
    w_obj: float = 0.75,
    w_dir: float = 0.5,
    w_speed: float = 0.25,
    time_loss_weight: float = 8.0,
    visibility_loss_weight: float = 1.0,
    direction_loss_weight: float = 0.5,
    speed_loss_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    q_count = int(outputs["num_regular_queries"])
    obj_target = torch.zeros((batch_size, q_count), dtype=torch.float32, device=device)
    obj_weight = torch.full((batch_size, q_count), float(no_object_weight), dtype=torch.float32, device=device)

    matched_b: list[torch.Tensor] = []
    matched_q: list[torch.Tensor] = []
    matched_g: list[torch.Tensor] = []
    matched_total = 0
    gt_total = int(targets["gt_valid"].sum().detach().cpu())

    for b in range(batch_size):
        rows, cols = _match_single(
            outputs,
            targets,
            b,
            matcher=str(matcher),
            w_time=float(w_time),
            w_vis=float(w_vis),
            w_obj=float(w_obj),
            w_dir=float(w_dir),
            w_speed=float(w_speed),
        )
        if rows.numel() == 0:
            continue
        obj_target[b, rows] = 1.0
        obj_weight[b, rows] = 1.0
        matched_total += int(rows.numel())
        matched_b.append(torch.full_like(rows, b))
        matched_q.append(rows)
        matched_g.append(cols)

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss_obj = F.binary_cross_entropy_with_logits(
        outputs["objectness_logits"][:, :q_count],
        obj_target,
        weight=obj_weight,
        reduction="mean",
    )

    if matched_b:
        b_sel = torch.cat(matched_b)
        q_sel = torch.cat(matched_q)
        g_sel = torch.cat(matched_g)
        gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
        mapped_items = []
        for bi, gi in zip(b_sel.tolist(), g_sel.tolist()):
            valid_idx = torch.where(gt_valid[int(bi)])[0]
            mapped_items.append(valid_idx[int(gi)])
        mapped_g = torch.stack(mapped_items)

        pred_time = outputs["time"][b_sel, q_sel]
        pred_vis_logits = outputs["visibility_logits"][b_sel, q_sel]
        pred_dir = outputs["direction_logits"][b_sel, q_sel]
        pred_speed = outputs["speed"][b_sel, q_sel]
        gt_time = targets["time"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_vis = targets["visibility"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_dir = targets["direction"][b_sel, mapped_g].to(device=device, dtype=torch.long)
        gt_speed = targets["speed"][b_sel, mapped_g].to(device=device, dtype=torch.float32)

        loss_time = _weighted_smooth_l1(pred_time, gt_time, gt_vis)
        loss_vis = F.binary_cross_entropy_with_logits(pred_vis_logits, gt_vis, reduction="mean")
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")
    else:
        loss_time = zero
        loss_vis = zero
        loss_dir = zero
        loss_speed = zero

    total = (
        loss_obj
        + float(time_loss_weight) * loss_time
        + float(visibility_loss_weight) * loss_vis
        + float(direction_loss_weight) * loss_dir
        + float(speed_loss_weight) * loss_speed
    )
    if not collect_metrics:
        return total, {}

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_time": float(loss_time.detach().cpu()),
        "loss_vis": float(loss_vis.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(torch.max(obj_prob).detach().cpu()),
        "mean_objectness": float(torch.mean(obj_prob).detach().cpu()),
    }
    return total, metrics


def track_slot_detection_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    objectness_threshold: float = 0.5,
    point_threshold: float = 0.05,
    matcher: str = "hungarian",
) -> dict[str, float]:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    active = obj >= float(objectness_threshold)
    pred_total = int(active.sum().detach().cpu())
    gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    gt_total = int(gt_valid.sum().detach().cpu())

    good_total = 0
    count_abs_error = 0.0
    count_exact = 0
    time_errors: list[float] = []
    for b in range(int(obj.shape[0])):
        pred_count = int(active[b].sum().item())
        gt_count = int(gt_valid[b].sum().item())
        count_abs_error += abs(pred_count - gt_count)
        count_exact += int(pred_count == gt_count)
        if gt_count <= 0:
            continue
        rows, cols = _match_single(
            outputs,
            targets,
            b,
            matcher=str(matcher),
            w_time=5.0,
            w_vis=1.0,
            w_obj=0.75,
            w_dir=0.5,
            w_speed=0.25,
        )
        if rows.numel() == 0:
            continue
        valid_idx = torch.where(gt_valid[b])[0]
        for r, c in zip(rows.tolist(), cols.tolist()):
            if not bool(active[b, r]):
                continue
            gi = int(valid_idx[c].item())
            vis = targets["visibility"][b, gi].to(device=device, dtype=torch.float32) > 0.5
            if not torch.any(vis):
                continue
            err = torch.mean(torch.abs(outputs["time"][b, r, vis] - targets["time"][b, gi, vis].to(device))).detach()
            err_f = float(err.cpu())
            time_errors.append(err_f)
            if err_f <= float(point_threshold):
                good_total += 1

    precision = float(good_total / max(1, pred_total))
    recall = float(good_total / max(1, gt_total))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    batch_size = int(obj.shape[0])
    return {
        "track_precision": precision,
        "track_recall": recall,
        "track_f1": f1,
        "track_tp": float(good_total),
        "pred_count": float(pred_total),
        "gt_count": float(gt_total),
        "count_mae": float(count_abs_error / max(1, batch_size)),
        "count_acc": float(count_exact / max(1, batch_size)),
        "time_mae_norm": float(np.mean(time_errors)) if time_errors else float("nan"),
    }


def _local_speed_series(points: list[TrackPoint]) -> list[float]:
    if not points:
        return []
    speeds = [float("nan")] * len(points)
    for i, point in enumerate(points):
        vals = []
        if i > 0:
            prev = points[i - 1]
            vals.append(3.6 * abs(point.offset_m - prev.offset_m) / max(1e-9, abs(point.time_s - prev.time_s)))
        if i + 1 < len(points):
            nxt = points[i + 1]
            vals.append(3.6 * abs(nxt.offset_m - point.offset_m) / max(1e-9, abs(nxt.time_s - point.time_s)))
        vals = [v for v in vals if np.isfinite(v)]
        speeds[i] = float(np.mean(vals)) if vals else float("nan")
    return speeds


def _track_stats(track_id: int, direction: str, points: list[TrackPoint]) -> Track:
    points_sorted = sorted(points, key=lambda p: p.ch_idx)
    speeds = [v for v in _local_speed_series(points_sorted) if np.isfinite(v)]
    mean_speed = float(np.mean(speeds)) if speeds else float("nan")
    total_score = float(np.sum([p.score for p in points_sorted]))
    return Track(
        track_id=int(track_id),
        direction=direction,
        points=points_sorted,
        total_score=total_score,
        mean_speed_kmh=mean_speed,
    )


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
    return [_track_stats(i, track.direction, track.points) for i, track in enumerate(kept)]


def predict_tracks_from_window(
    model: TrackSlotPredictor,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[InferenceConfig] = None,
    device: Optional[str] = None,
) -> list[Track]:
    cfg = config or InferenceConfig()
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    if arr.shape[0] != int(model.config.n_channels):
        raise ValueError(f"Model expects {model.config.n_channels} channels, but input has {arr.shape[0]} channels")

    resolved_device = device or next(model.parameters()).device
    x = prepare_window_input(
        arr,
        int(cfg.time_downsample),
        clip_ratio=float(cfg.clip_ratio),
        input_mode="raw" if int(model.config.in_channels) == 1 else "raw_abs",
    ).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(x.to(resolved_device))

    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    visibility = torch.sigmoid(outputs["visibility_logits"][0]).detach().cpu().numpy()
    time_norm = outputs["time"][0].detach().cpu().numpy()
    dirs = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    n_samples = int(arr.shape[1])
    max_tracks = min(int(cfg.max_tracks), int(obj.shape[0]))
    order = np.argsort(obj)[::-1][:max_tracks]
    tracks: list[Track] = []
    for q_idx in order.tolist():
        score = float(obj[q_idx])
        if score < float(cfg.objectness_threshold):
            continue
        chs = np.where(visibility[q_idx] >= float(cfg.visibility_threshold))[0]
        if chs.size < int(cfg.min_visible_channels):
            continue
        points: list[TrackPoint] = []
        for ch in chs.tolist():
            t_idx = int(round(float(np.clip(time_norm[q_idx, ch], 0.0, 1.0)) * float(max(1, n_samples - 1))))
            t_idx = int(max(0, min(n_samples - 1, t_idx)))
            amp = float(abs(arr[int(ch), t_idx]))
            if int(ch) < len(x_axis_m):
                offset = float(x_axis_m[int(ch)])
            else:
                offset = float(ch)
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=t_idx,
                    time_s=float(t_idx) / float(fs),
                    offset_m=offset,
                    amp=amp,
                    score=score * float(visibility[q_idx, ch]),
                )
            )
        if len(points) < int(cfg.min_visible_channels):
            continue
        tracks.append(_track_stats(len(tracks), LABEL_TO_DIRECTION.get(int(dirs[q_idx]), "forward"), points))

    return _deduplicate_tracks(
        tracks,
        tol_samples=int(cfg.dedup_tolerance_samples),
        min_overlap=int(cfg.dedup_min_overlap_channels),
    )


def save_checkpoint(
    path: str | Path,
    model: TrackSlotPredictor,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    dataset_config: WindowDatasetConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "track_slot",
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "dataset_config": asdict(dataset_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(checkpoint_path)


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[TrackSlotPredictor, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = TrackSlotPredictor(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
