"""Query mask instance model for DAS trajectory extraction.

This module keeps the DETR-style query decoder but predicts one instance mask
per query over the `[channel, time_downsampled]` grid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.trajectory_set_model import (
    LABEL_TO_DIRECTION,
    WindowDatasetConfig,
    _refine_t_idx,
    _robust_scale,
    auto_torch_device,
    load_sac_matrix,
    load_tracks_json,
    prepare_window_input,
)


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    max_queries: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    decoder_layers: int = 2
    pooled_channels: int = 8
    pooled_time: int = 128
    dropout: float = 0.1
    match_time_bins: int = 320


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.35
    visibility_threshold: float = 0.5
    min_visible_channels: int = 3
    refine_radius_samples: int = 120
    max_tracks: int = 128
    dedup_tolerance_samples: int = 180
    mask_iou_dedup_threshold: float = 0.65
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


class QueryMaskInstancePredictor(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.backbone = nn.Sequential(
            ConvBlock(int(c.in_channels), 32, stride=(1, 2)),
            ConvBlock(32, 64, stride=(1, 2)),
            ConvBlock(64, hidden, stride=(1, 2)),
        )
        self.pool_size = (int(c.pooled_channels), int(c.pooled_time))
        self.pos_embed = nn.Parameter(torch.randn(1, int(c.pooled_channels) * int(c.pooled_time), hidden) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(c.decoder_layers))
        self.query_embed = nn.Embedding(int(c.max_queries), hidden)

        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

        self.pixel_proj = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.query_mask_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))

    def forward(self, x: torch.Tensor, targets: Optional[dict[str, torch.Tensor]] = None) -> dict[str, torch.Tensor]:
        del targets
        feat = self.backbone(x)
        feat = F.interpolate(feat, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        memory = F.adaptive_avg_pool2d(feat, output_size=self.pool_size).flatten(2).transpose(1, 2)
        memory = memory + self.pos_embed[:, : memory.shape[1], :]

        query = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        hs = self.decoder(tgt=query, memory=memory)

        query_feat = self.query_mask_proj(hs)
        pixel_feat = self.pixel_proj(feat)
        mask_logits = torch.einsum("bqh,bhct->bqct", query_feat, pixel_feat)

        return {
            "num_regular_queries": int(self.config.max_queries),
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "mask_logits": mask_logits,
        }


def _ensure_gt_masks(
    target: dict[str, torch.Tensor],
    *,
    n_channels: int,
    time_bins: int,
    time_downsample: int,
    sigma_ch: float = 0.8,
    sigma_t: float = 2.0,
) -> torch.Tensor:
    if "gt_masks" in target:
        out = target["gt_masks"].to(torch.float32)
        if out.ndim == 3:
            return out
    time = target["time"].to(torch.float32)
    vis = target["visibility"].to(torch.float32)
    n_gt = int(time.shape[0])
    if n_gt <= 0:
        return torch.zeros((0, n_channels, time_bins), dtype=torch.float32)

    ch_axis = torch.arange(n_channels, dtype=torch.float32).view(-1, 1)
    t_axis = torch.arange(time_bins, dtype=torch.float32).view(1, -1)
    out = torch.zeros((n_gt, n_channels, time_bins), dtype=torch.float32)
    denom_t = float(max(1, time_downsample))
    for g in range(n_gt):
        mask = torch.zeros((n_channels, time_bins), dtype=torch.float32)
        chs = torch.where(vis[g] > 0.5)[0]
        for ch in chs.tolist():
            t_full = float(time[g, ch].item() * max(1, (time_bins * time_downsample) - 1))
            t_ds = float(np.clip(t_full / denom_t, 0.0, float(max(0, time_bins - 1))))
            gc = torch.exp(-0.5 * ((ch_axis - float(ch)) / float(max(1e-6, sigma_ch))) ** 2)
            gt = torch.exp(-0.5 * ((t_axis - t_ds) / float(max(1e-6, sigma_t))) ** 2)
            mask = torch.maximum(mask, gc * gt)
        out[g] = mask.clamp(0.0, 1.0)
    return out


def mask_trajectory_collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    *,
    n_channels: int,
    time_downsample: int,
    sigma_ch: float = 0.8,
    sigma_t: float = 2.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in batch], dim=0)
    batch_size = len(batch)
    time_bins = int(xs.shape[-1])
    max_gt = max((int(item[1]["time"].shape[0]) for item in batch), default=0)

    time = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    visibility = torch.zeros((batch_size, max_gt, n_channels), dtype=torch.float32)
    direction = torch.zeros((batch_size, max_gt), dtype=torch.long)
    speed = torch.zeros((batch_size, max_gt), dtype=torch.float32)
    track_id = torch.full((batch_size, max_gt), -1, dtype=torch.long)
    gt_valid = torch.zeros((batch_size, max_gt), dtype=torch.bool)
    gt_masks = torch.zeros((batch_size, max_gt, n_channels, time_bins), dtype=torch.float32)

    for b, (_x, target) in enumerate(batch):
        n_gt = int(target["time"].shape[0])
        if n_gt <= 0:
            continue
        copy_ch = min(n_channels, int(target["time"].shape[1]))
        time[b, :n_gt, :copy_ch] = target["time"][:, :copy_ch].to(torch.float32)
        visibility[b, :n_gt, :copy_ch] = target["visibility"][:, :copy_ch].to(torch.float32)
        direction[b, :n_gt] = target["direction"].to(torch.long)
        speed[b, :n_gt] = target["speed"].to(torch.float32)
        if "track_id" in target:
            track_id[b, :n_gt] = target["track_id"].to(torch.long)
        gt_valid[b, :n_gt] = True

        masks = _ensure_gt_masks(
            target,
            n_channels=n_channels,
            time_bins=time_bins,
            time_downsample=time_downsample,
            sigma_ch=sigma_ch,
            sigma_t=sigma_t,
        )
        gt_masks[b, :n_gt] = masks[:n_gt]

    targets = {
        "time": time,
        "visibility": visibility,
        "direction": direction,
        "speed": speed,
        "track_id": track_id,
        "gt_valid": gt_valid,
        "gt_masks": gt_masks,
        "gt_count": gt_valid.sum(dim=1).to(torch.long),
    }
    return xs, targets


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


def _dice_from_prob(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inter = torch.sum(pred * gt, dim=(-1, -2))
    denom = torch.sum(pred, dim=(-1, -2)) + torch.sum(gt, dim=(-1, -2))
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _match_single(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    b: int,
    *,
    matcher: str,
    w_bce: float,
    w_dice: float,
    w_obj: float,
    w_dir: float,
    w_speed: float,
    match_time_bins: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["objectness_logits"].device
    regular_q = int(outputs["num_regular_queries"])
    gt_valid = targets["gt_valid"][b].to(device=device, dtype=torch.bool)
    g_count = int(gt_valid.sum().item())
    if g_count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    pred_masks = torch.sigmoid(outputs["mask_logits"][b, :regular_q])
    pred_obj = torch.sigmoid(outputs["objectness_logits"][b, :regular_q])
    pred_dir = torch.softmax(outputs["direction_logits"][b, :regular_q], dim=-1)
    pred_speed = outputs["speed"][b, :regular_q]

    gt_masks = targets["gt_masks"][b, gt_valid].to(device=device, dtype=torch.float32)
    gt_dir = targets["direction"][b, gt_valid].to(device=device, dtype=torch.long)
    gt_speed = targets["speed"][b, gt_valid].to(device=device, dtype=torch.float32)

    bins = int(max(8, min(match_time_bins, pred_masks.shape[-1])))
    pred_pool = F.adaptive_avg_pool2d(pred_masks, output_size=(pred_masks.shape[-2], bins))
    gt_pool = F.adaptive_avg_pool2d(gt_masks, output_size=(gt_masks.shape[-2], bins))

    pred_e = pred_pool[:, None, :, :]
    gt_e = gt_pool[None, :, :, :]
    bce = F.binary_cross_entropy(pred_e.expand(-1, g_count, -1, -1), gt_e.expand(int(pred_pool.shape[0]), -1, -1, -1), reduction="none").mean(dim=(-1, -2))
    dice = _dice_from_prob(pred_e, gt_e)
    dir_cost = -pred_dir[:, gt_dir]
    speed_cost = torch.abs(pred_speed[:, None] - gt_speed[None, :])
    obj_cost = -pred_obj[:, None]

    cost = float(w_bce) * bce + float(w_dice) * dice + float(w_obj) * obj_cost + float(w_dir) * dir_cost + float(w_speed) * speed_cost

    if str(matcher).lower() == "greedy":
        return _greedy_match_cost(cost)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


def _duplicate_mask_loss(outputs: dict[str, torch.Tensor], regular_q: int) -> torch.Tensor:
    obj = torch.sigmoid(outputs["objectness_logits"][:, :regular_q])
    masks = torch.sigmoid(outputs["mask_logits"][:, :regular_q])
    if regular_q <= 1:
        return torch.zeros((), dtype=torch.float32, device=obj.device)
    triu = torch.triu(torch.ones((regular_q, regular_q), dtype=torch.bool, device=obj.device), diagonal=1)
    inter = torch.einsum("bqct,bkct->bqk", masks, masks)
    area = torch.sum(masks, dim=(-1, -2))
    union = area[:, :, None] + area[:, None, :] - inter
    iou = inter / torch.clamp(union, min=1e-6)
    pair_obj = obj[:, :, None] * obj[:, None, :]
    return (iou[:, triu] * pair_obj[:, triu]).mean()


def query_mask_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.05,
    duplicate_loss_weight: float = 0.1,
    denoising_loss_weight: float = 0.0,
    matcher: str = "greedy",
    collect_metrics: bool = True,
    epoch: int = 1,
    warmup_epochs: int = 8,
    w_bce: float = 2.0,
    w_dice: float = 3.0,
    w_obj: float = 0.5,
    w_dir: float = 0.3,
    w_speed: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    regular_q = int(outputs["num_regular_queries"])

    obj_target = torch.zeros((batch_size, regular_q), dtype=torch.float32, device=device)
    obj_weight = torch.full((batch_size, regular_q), float(no_object_weight), dtype=torch.float32, device=device)

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
            w_bce=float(w_bce),
            w_dice=float(w_dice),
            w_obj=float(w_obj),
            w_dir=float(w_dir),
            w_speed=float(w_speed),
            match_time_bins=int(getattr(outputs.get("config", None), "match_time_bins", 320) if False else 320),
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
        outputs["objectness_logits"][:, :regular_q],
        obj_target,
        weight=obj_weight,
        reduction="mean",
    )

    if matched_b:
        b_sel = torch.cat(matched_b)
        q_sel = torch.cat(matched_q)
        g_sel = torch.cat(matched_g)
        gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)

        pred_masks = outputs["mask_logits"][b_sel, q_sel]
        pred_dir = outputs["direction_logits"][b_sel, q_sel]
        pred_speed = outputs["speed"][b_sel, q_sel]

        gt_masks_all = []
        gt_dir_all = []
        gt_speed_all = []
        for bi, gi in zip(b_sel.tolist(), g_sel.tolist()):
            valid_idx = torch.where(gt_valid[bi])[0]
            mapped = int(valid_idx[gi].item())
            gt_masks_all.append(targets["gt_masks"][bi, mapped])
            gt_dir_all.append(targets["direction"][bi, mapped])
            gt_speed_all.append(targets["speed"][bi, mapped])
        gt_masks = torch.stack(gt_masks_all, dim=0).to(device=device, dtype=torch.float32)
        gt_dir = torch.stack(gt_dir_all, dim=0).to(device=device, dtype=torch.long)
        gt_speed = torch.stack(gt_speed_all, dim=0).to(device=device, dtype=torch.float32)

        loss_mask_bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction="mean")
        loss_mask_dice = _dice_from_prob(torch.sigmoid(pred_masks), gt_masks).mean()
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")
    else:
        loss_mask_bce = zero
        loss_mask_dice = zero
        loss_dir = zero
        loss_speed = zero

    use_aux = int(epoch) > int(max(0, warmup_epochs))
    loss_duplicate = _duplicate_mask_loss(outputs, regular_q) if (use_aux and float(duplicate_loss_weight) > 0.0) else zero
    loss_dn = zero if not use_aux else zero

    total = (
        loss_obj
        + 6.0 * loss_mask_bce
        + 6.0 * loss_mask_dice
        + 0.5 * loss_dir
        + 0.5 * loss_speed
        + float(duplicate_loss_weight) * loss_duplicate
        + float(denoising_loss_weight) * loss_dn
    )
    if not collect_metrics:
        return total, {}

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :regular_q])
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_time": float(loss_mask_bce.detach().cpu()),
        "loss_vis": float(loss_mask_dice.detach().cpu()),
        "loss_mask_bce": float(loss_mask_bce.detach().cpu()),
        "loss_mask_dice": float(loss_mask_dice.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_duplicate": float(loss_duplicate.detach().cpu()),
        "loss_dn": float(loss_dn.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(torch.max(obj_prob).detach().cpu()),
        "mean_objectness": float(torch.mean(obj_prob).detach().cpu()),
    }
    return total, metrics


def query_mask_detection_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    objectness_threshold: float = 0.5,
    iou_threshold: float = 0.35,
    matcher: str = "greedy",
) -> dict[str, float]:
    del matcher
    device = outputs["objectness_logits"].device
    regular_q = int(outputs["num_regular_queries"])
    obj = torch.sigmoid(outputs["objectness_logits"][:, :regular_q])
    active = obj >= float(objectness_threshold)

    pred_total = int(active.sum().detach().cpu())
    gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    gt_total = int(gt_valid.sum().detach().cpu())

    good_total = 0
    count_abs_error = 0.0
    count_exact = 0
    point_like_errors: list[float] = []
    time_like_errors: list[float] = []

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
            matcher="greedy",
            w_bce=2.0,
            w_dice=3.0,
            w_obj=0.5,
            w_dir=0.3,
            w_speed=0.2,
            match_time_bins=320,
        )
        if rows.numel() == 0:
            continue
        valid_idx = torch.where(gt_valid[b])[0]
        for r, c in zip(rows.tolist(), cols.tolist()):
            if not bool(active[b, r]):
                continue
            gi = int(valid_idx[c].item())
            pred_prob = torch.sigmoid(outputs["mask_logits"][b, r])
            gt_mask = targets["gt_masks"][b, gi].to(device=device, dtype=torch.float32)
            inter = torch.sum(pred_prob * gt_mask)
            union = torch.sum(pred_prob) + torch.sum(gt_mask) - inter
            iou = float((inter / torch.clamp(union, min=1e-6)).detach().cpu())
            if iou >= float(iou_threshold):
                good_total += 1
            point_like_errors.append(1.0 - iou)

            pred_t = torch.argmax(pred_prob, dim=-1).to(torch.float32)
            gt_t = torch.argmax(gt_mask, dim=-1).to(torch.float32)
            vis = torch.max(gt_mask, dim=-1).values > 0.1
            if torch.any(vis):
                time_like_errors.append(float(torch.mean(torch.abs(pred_t[vis] - gt_t[vis])).detach().cpu() / max(1.0, float(pred_prob.shape[-1]))))

    precision = float(good_total / max(1, pred_total))
    recall = float(good_total / max(1, gt_total))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    point_mae = float(np.mean(point_like_errors)) if point_like_errors else float("nan")
    time_mae = float(np.mean(time_like_errors)) if time_like_errors else float("nan")
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
        "point_mae_norm": point_mae,
        "time_mae_norm": time_mae,
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


def _track_overlap(track_a: Track, track_b: Track, tol_samples: int) -> int:
    a = {int(p.ch_idx): int(p.t_idx) for p in track_a.points}
    b = {int(p.ch_idx): int(p.t_idx) for p in track_b.points}
    overlap = 0
    for ch in set(a) & set(b):
        if abs(a[ch] - b[ch]) <= int(tol_samples):
            overlap += 1
    return overlap


def _deduplicate_tracks_with_mask(
    tracks_with_masks: list[tuple[Track, np.ndarray]],
    tol_samples: int,
    iou_thr: float,
) -> list[Track]:
    kept: list[tuple[Track, np.ndarray]] = []
    for tr, m in sorted(tracks_with_masks, key=lambda item: item[0].total_score, reverse=True):
        duplicate = False
        for ex, exm in kept:
            overlap = _track_overlap(ex, tr, tol_samples)
            ratio = overlap / max(1, min(len(ex.points), len(tr.points)))
            inter = float(np.logical_and(exm, m).sum())
            union = float(np.logical_or(exm, m).sum())
            iou = inter / max(1.0, union)
            if ratio >= 0.6 and iou >= float(iou_thr):
                duplicate = True
                break
        if not duplicate:
            kept.append((tr, m))
    return [_track_stats(i, tr.direction, tr.points) for i, (tr, _m) in enumerate(kept)]


def predict_tracks_from_window(
    model: QueryMaskInstancePredictor,
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
    input_mode = "raw_abs" if int(getattr(model.config, "in_channels", 1)) == 2 else "raw"
    x = prepare_window_input(
        arr,
        int(cfg.time_downsample),
        float(cfg.clip_ratio),
        input_mode=input_mode,
    ).unsqueeze(0).to(resolved_device)

    with torch.inference_mode():
        outputs = model(x)

    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    masks = torch.sigmoid(outputs["mask_logits"][0]).detach().cpu().numpy()
    direction_label = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    order = np.argsort(obj)[::-1]
    n_channels = int(arr.shape[0])
    n_samples = int(arr.shape[1])
    tracks_with_masks: list[tuple[Track, np.ndarray]] = []

    for q_idx in order[: int(max(1, cfg.max_tracks))]:
        score_obj = float(obj[q_idx])
        if score_obj < float(cfg.objectness_threshold):
            continue
        mask_prob = masks[q_idx]
        points: list[TrackPoint] = []
        mask_bin = mask_prob >= float(cfg.visibility_threshold)
        for ch in range(n_channels):
            row = mask_prob[ch]
            vmax = float(np.max(row))
            if vmax < float(cfg.visibility_threshold):
                continue
            t_ds = int(np.argmax(row))
            t_idx_raw = int(round(t_ds * int(max(1, cfg.time_downsample))))
            t_idx_raw = int(max(0, min(n_samples - 1, t_idx_raw)))
            t_idx = _refine_t_idx(arr, ch, t_idx_raw, int(cfg.refine_radius_samples))
            point = TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(fs),
                offset_m=float(x_axis_m[ch]),
                amp=float(abs(arr[ch, t_idx])),
                score=float(score_obj * vmax),
            )
            points.append(point)
        if len(points) < int(cfg.min_visible_channels):
            continue
        direction = LABEL_TO_DIRECTION.get(int(direction_label[q_idx]), "forward")
        tracks_with_masks.append((_track_stats(len(tracks_with_masks), direction, points), mask_bin))

    return _deduplicate_tracks_with_mask(
        tracks_with_masks,
        tol_samples=int(cfg.dedup_tolerance_samples),
        iou_thr=float(cfg.mask_iou_dedup_threshold),
    )


def save_checkpoint(
    path: str | Path,
    model: QueryMaskInstancePredictor,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    dataset_config: WindowDatasetConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "query_masks",
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "dataset_config": asdict(dataset_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, str(Path(path).expanduser()))


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[QueryMaskInstancePredictor, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = QueryMaskInstancePredictor(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
