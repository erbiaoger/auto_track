from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from autotrack.core.track_extractor_graph import Track, TrackPoint
from vehicle_peak_set_tracker.simple_vehicle_peak_dataset import (
    detect_peak_candidates,
    prepare_peakset_input,
    targets_to_batched_peakset,
)


@dataclass
class PeakSetModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    max_queries: int = 32
    peak_candidates: int = 64
    hidden_dim: int = 128
    num_heads: int = 4
    encoder_layers: int = 2
    decoder_layers: int = 2
    pooled_time: int = 96
    dropout: float = 0.1
    trajectory_time_bins: int = 50
    residual_completion: bool = True
    max_residual_norm: float = 0.005
    max_base_slope_norm: float = 0.75
    max_base_curve_norm: float = 0.35


@dataclass
class PeakSetInferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.35
    complete_valid_threshold: float = 0.50
    anchor_threshold: float = 0.50
    min_visible_channels: int = 5
    min_anchor_support_channels: int = 3
    min_anchor_support_ratio: float = 0.15
    speed_min_kmh: float = 60.0
    speed_max_kmh: float = 90.0
    refine_radius_samples: int = 80
    max_tracks: int = 32
    dedup_tolerance_samples: int = 30
    clip_ratio: float = 1.35
    graph_refine: bool = False
    fused_anchor_weight: float = 0.55
    fused_signal_weight: float = 0.25
    fused_time_weight: float = 0.20
    fused_time_tolerance_s: float = 0.25
    fused_min_anchor_score: float = 0.38
    fused_missing_point_scale: float = 0.35
    use_observed_valid_in_decode: bool = False
    observed_valid_soft_floor: float = 0.6


@dataclass
class DecodedPeakTrack:
    track: Track
    objectness: float
    query_index: int
    channel_indices: np.ndarray
    complete_valid: np.ndarray
    observed_valid: np.ndarray
    point_times_norm: np.ndarray


class _ConvBlock(nn.Module):
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


class VehiclePeakSetTransformer(nn.Module):
    def __init__(self, config: Optional[PeakSetModelConfig] = None):
        super().__init__()
        self.config = config or PeakSetModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.backbone = nn.Sequential(
            _ConvBlock(int(c.in_channels), 32, stride=(1, 2)),
            _ConvBlock(32, 64, stride=(1, 2)),
            _ConvBlock(64, hidden, stride=(1, 2)),
            _ConvBlock(hidden, hidden, stride=(1, 1)),
        )
        self.channel_embed = nn.Parameter(torch.randn(1, int(c.n_channels), 1, hidden) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, 1, int(c.pooled_time), hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(c.encoder_layers))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(c.decoder_layers))
        self.query_embed = nn.Embedding(int(c.max_queries), hidden)

        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.channel_pair_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.peak_time_head = nn.Linear(hidden, 1)
        self.complete_valid_head = nn.Linear(hidden, 1)
        self.observed_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C_in, n_channels, n_samples]")
        feat = self.backbone(x)
        feat = F.interpolate(feat, size=(int(self.config.n_channels), int(self.config.pooled_time)), mode="bilinear", align_corners=False)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat + self.channel_embed[:, : feat.shape[1], :, :] + self.time_embed[:, :, : feat.shape[2], :]
        memory = self.encoder(feat.view(x.shape[0], -1, feat.shape[-1]))
        query = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        hs = self.decoder(tgt=query, memory=memory)

        channel_embed = self.channel_embed[:, : int(self.config.n_channels), 0, :].unsqueeze(1)
        query_expand = hs.unsqueeze(2).expand(-1, -1, int(self.config.n_channels), -1)
        channel_expand = channel_embed.expand(x.shape[0], hs.shape[1], -1, -1)
        pair = self.channel_pair_head(torch.cat([query_expand, channel_expand], dim=-1))
        peak_time = torch.sigmoid(self.peak_time_head(pair).squeeze(-1))
        complete_valid_logits = self.complete_valid_head(pair).squeeze(-1)
        observed_logits = self.observed_head(pair).squeeze(-1)

        return {
            "num_regular_queries": int(self.config.max_queries),
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "peak_time": peak_time,
            "complete_valid_logits": complete_valid_logits,
            "observed_logits": observed_logits,
        }


def _regular_query_count(outputs: dict[str, Any]) -> int:
    return int(outputs.get("num_regular_queries", int(outputs["objectness_logits"].shape[1])))


def _target_at(targets: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor], batch_idx: int) -> dict[str, torch.Tensor]:
    if isinstance(targets, dict):
        out: dict[str, torch.Tensor] = {}
        for key, value in targets.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) > batch_idx:
                out[key] = value[batch_idx]
            elif torch.is_tensor(value):
                out[key] = value
        return out
    return targets[batch_idx]


def _target_valid_count(target: dict[str, torch.Tensor]) -> int:
    if "gt_valid" in target:
        return int(target["gt_valid"].sum().item())
    return int(target["full_time"].shape[0])


def _target_valid_mask(target: dict[str, torch.Tensor], device: torch.device | str) -> torch.Tensor:
    if "gt_valid" in target:
        return target["gt_valid"].to(device=device, dtype=torch.bool)
    return torch.ones((int(target["full_time"].shape[0]),), dtype=torch.bool, device=device)


def _valid_rank_to_gt_index(gt_valid: torch.Tensor) -> torch.Tensor:
    if gt_valid.ndim != 2:
        raise ValueError("gt_valid must have shape [B, G]")
    device = gt_valid.device
    bsz, max_gt = int(gt_valid.shape[0]), int(gt_valid.shape[1])
    mapped = torch.full((bsz, max_gt), -1, dtype=torch.long, device=device)
    for b in range(bsz):
        valid = torch.where(gt_valid[b])[0]
        if valid.numel() == 0:
            continue
        mapped[b, : int(valid.numel())] = valid.to(dtype=torch.long)
    return mapped


def _target_attrs(
    target: dict[str, torch.Tensor],
    *,
    n_channels: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _target_valid_mask(target, device)
    full_time = target["full_time"].to(device=device, dtype=torch.float32)
    full_valid = target["full_valid"].to(device=device, dtype=torch.float32)
    observed = target["observed_visibility"].to(device=device, dtype=torch.float32)
    direction = target["direction"].to(device=device, dtype=torch.long).clamp(0, 1)
    speed = target["speed"].to(device=device, dtype=torch.float32)
    if mask.numel() != int(full_time.shape[0]):
        mask = mask[: int(full_time.shape[0])]
    return full_time[mask], full_valid[mask], observed[mask], direction[mask], speed[mask]


def _greedy_match_cost(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    device = cost.device
    q_count = int(cost.shape[0])
    gt_count = int(cost.shape[1])
    match_count = int(min(q_count, gt_count))
    if match_count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty
    work = cost.float().clone()
    large = torch.finfo(work.dtype).max
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for _ in range(match_count):
        flat_idx = torch.argmin(work)
        row = torch.div(flat_idx, gt_count, rounding_mode="floor").long()
        col = (flat_idx - row * gt_count).long()
        rows.append(row)
        cols.append(col)
        work[row, :] = large
        work[:, col] = large
    return torch.stack(rows).to(dtype=torch.long), torch.stack(cols).to(dtype=torch.long)


def _match_single(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch_idx: int,
    matcher: str = "hungarian",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["objectness_logits"].device
    n_gt = _target_valid_count(target)
    if n_gt == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    regular_q = _regular_query_count(outputs)
    pred_obj = torch.sigmoid(outputs["objectness_logits"][batch_idx, :regular_q])
    pred_peak = outputs["peak_time"][batch_idx, :regular_q]
    pred_complete = torch.sigmoid(outputs["complete_valid_logits"][batch_idx, :regular_q])
    pred_observed = torch.sigmoid(outputs["observed_logits"][batch_idx, :regular_q])
    pred_dir = torch.softmax(outputs["direction_logits"][batch_idx, :regular_q], dim=-1)
    pred_speed = outputs["speed"][batch_idx, :regular_q]

    gt_full_time, gt_full_valid, gt_observed, gt_dir, gt_speed = _target_attrs(
        target,
        n_channels=int(outputs["peak_time"].shape[-1]),
        device=device,
    )
    ch = torch.arange(int(gt_full_time.shape[-1]), device=device, dtype=torch.float32)
    ch = ch / float(max(1, int(gt_full_time.shape[-1]) - 1))

    diff = torch.abs(pred_peak[:, None, :] - gt_full_time[None, :, :])
    denom = torch.clamp(gt_full_valid.sum(dim=-1), min=1.0)[None, :]
    time_cost = (diff * gt_full_valid[None, :, :]).sum(dim=-1) / denom
    complete_cost = F.binary_cross_entropy_with_logits(
        outputs["complete_valid_logits"][batch_idx, :regular_q, :][:, None, :].expand(-1, int(gt_full_valid.shape[0]), -1),
        gt_full_valid[None, :, :].expand(regular_q, -1, -1),
        reduction="none",
    ).mean(dim=-1)
    observed_cost = F.binary_cross_entropy_with_logits(
        outputs["observed_logits"][batch_idx, :regular_q, :][:, None, :].expand(-1, int(gt_observed.shape[0]), -1),
        gt_observed[None, :, :].expand(regular_q, -1, -1),
        reduction="none",
    ).mean(dim=-1)
    dir_cost = -pred_dir[:, gt_dir]
    speed_cost = torch.abs(pred_speed[:, None] - gt_speed[None, :])
    obj_cost = -pred_obj[:, None]
    cost = 5.0 * time_cost + 1.0 * complete_cost + 1.0 * observed_cost + 0.5 * dir_cost + 0.25 * speed_cost + 0.75 * obj_cost

    if str(matcher).lower() == "greedy":
        rows, cols = _greedy_match_cost(cost)
        return rows.to(device=device), cols.to(device=device)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(rows, dtype=torch.long, device=device), torch.as_tensor(cols, dtype=torch.long, device=device)


def _linearity_penalty(pred_peak: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device = pred_peak.device
    if pred_peak.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    penalties: list[torch.Tensor] = []
    x = torch.arange(int(pred_peak.shape[-1]), device=device, dtype=torch.float32)
    x = x / float(max(1, int(pred_peak.shape[-1]) - 1))
    for b in range(int(pred_peak.shape[0])):
        mask = valid_mask[b] > 0.5
        if int(mask.sum().item()) < 2:
            continue
        y = pred_peak[b, mask]
        x_sel = x[mask]
        x_mean = x_sel.mean()
        y_mean = y.mean()
        x_centered = x_sel - x_mean
        denom = torch.clamp((x_centered * x_centered).sum(), min=1e-6)
        slope = ((x_centered * (y - y_mean)).sum()) / denom
        intercept = y_mean - slope * x_mean
        fit = slope * x_sel + intercept
        penalties.append(F.smooth_l1_loss(y, fit, reduction="mean"))
    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()


def _smoothness_penalty(pred_peak: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device = pred_peak.device
    if pred_peak.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    penalties: list[torch.Tensor] = []
    for b in range(int(pred_peak.shape[0])):
        mask = valid_mask[b] > 0.5
        if int(mask.sum().item()) < 3:
            continue
        y = pred_peak[b, mask]
        if int(y.numel()) < 3:
            continue
        second = y[2:] - 2.0 * y[1:-1] + y[:-2]
        penalties.append(torch.mean(torch.abs(second)))
    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    fill_value = -1e4 if logits.dtype in {torch.float16, torch.bfloat16} else -1e9
    masked_logits = logits.masked_fill(~mask, fill_value)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(dtype=probs.dtype)
    denom = torch.clamp(probs.sum(dim=dim, keepdim=True), min=1e-6)
    return probs / denom


def _expected_anchor_time(
    pred_anchor_logits: torch.Tensor,
    peak_time: torch.Tensor,
    peak_valid: torch.Tensor,
) -> torch.Tensor:
    if pred_anchor_logits.ndim != 3:
        raise ValueError("pred_anchor_logits must have shape [N, C, K]")
    if peak_time.ndim != 3 or peak_valid.ndim != 3:
        raise ValueError("peak_time and peak_valid must have shape [N, C, K]")
    k_count = int(min(pred_anchor_logits.shape[-1], peak_time.shape[-1], peak_valid.shape[-1]))
    if k_count <= 0:
        return torch.zeros(pred_anchor_logits.shape[:-1], dtype=torch.float32, device=pred_anchor_logits.device)
    logits = pred_anchor_logits[..., :k_count]
    time = peak_time[..., :k_count]
    mask = peak_valid[..., :k_count].to(dtype=torch.bool)
    probs = _masked_softmax(logits, mask, dim=-1)
    return torch.sum(probs * time.to(dtype=probs.dtype), dim=-1)


def _masked_second_difference_penalty(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device = values.device
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    penalties: list[torch.Tensor] = []
    for b in range(int(values.shape[0])):
        mask = valid_mask[b] > 0.5
        if int(mask.sum().item()) < 3:
            continue
        y = values[b, mask]
        if int(y.numel()) < 3:
            continue
        second = y[2:] - 2.0 * y[1:-1] + y[:-2]
        penalties.append(torch.mean(torch.abs(second)))
    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()


def _masked_first_difference_penalty(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device = values.device
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    penalties: list[torch.Tensor] = []
    for b in range(int(values.shape[0])):
        mask = valid_mask[b] > 0.5
        if int(mask.sum().item()) < 2:
            continue
        y = values[b, mask]
        if int(y.numel()) < 2:
            continue
        first = y[1:] - y[:-1]
        penalties.append(torch.mean(torch.abs(first)))
    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()


def _masked_slope_variation_penalty(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    device = values.device
    if values.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    penalties: list[torch.Tensor] = []
    for b in range(int(values.shape[0])):
        mask = valid_mask[b] > 0.5
        if int(mask.sum().item()) < 3:
            continue
        y = values[b, mask]
        if int(y.numel()) < 3:
            continue
        slopes = y[1:] - y[:-1]
        if int(slopes.numel()) < 2:
            continue
        slope_mean = slopes.mean()
        penalties.append(torch.mean(torch.abs(slopes - slope_mean)))
    if not penalties:
        return torch.zeros((), dtype=torch.float32, device=device)
    return torch.stack(penalties).mean()


def _duplicate_query_penalty(outputs: dict[str, torch.Tensor], regular_q: int, *, distance_tau: float = 0.08) -> torch.Tensor:
    device = outputs["objectness_logits"].device
    q_count = int(max(0, min(regular_q, int(outputs["objectness_logits"].shape[1]))))
    if q_count <= 1:
        return torch.zeros((), dtype=torch.float32, device=device)

    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    peak = outputs["peak_time"][:, :q_count]
    valid = torch.sigmoid(outputs["complete_valid_logits"][:, :q_count]).detach()
    pair_mask = torch.triu(torch.ones((q_count, q_count), dtype=torch.bool, device=device), diagonal=1)
    pair_valid = valid[:, :, None, :] * valid[:, None, :, :]
    denom = torch.clamp(pair_valid.sum(dim=-1), min=1.0)
    distance = (torch.abs(peak[:, :, None, :] - peak[:, None, :, :]) * pair_valid).sum(dim=-1) / denom
    similarity = torch.exp(-distance.detach() / float(max(1e-6, distance_tau)))
    pair_obj = obj[:, :, None] * obj[:, None, :]
    penalty = pair_obj[:, pair_mask] * similarity[:, pair_mask]
    if penalty.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)
    return penalty.mean()


def peak_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.02,
    duplicate_weight: float = 0.20,
    linearity_weight: float = 0.4,
    smoothness_weight: float = 0.15,
    matcher: str = "hungarian",
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    max_queries = _regular_query_count(outputs)

    if isinstance(targets, dict) and "gt_valid" in targets:
        target_dict = targets
    elif isinstance(targets, dict) and "full_time" in targets and targets["full_time"].ndim == 3:
        target_dict = dict(targets)
        if "gt_valid" not in target_dict:
            target_dict["gt_valid"] = torch.ones(
                (int(target_dict["full_time"].shape[0]), int(target_dict["full_time"].shape[1])),
                dtype=torch.bool,
                device=target_dict["full_time"].device,
            )
    else:
        target_dict = targets_to_batched_peakset(targets if isinstance(targets, Sequence) else [targets])  # type: ignore[arg-type]

    gt_valid = target_dict["gt_valid"].to(device=device, dtype=torch.bool)
    obj_target = torch.zeros((batch_size, max_queries), dtype=torch.float32, device=device)
    obj_weight = torch.full((batch_size, max_queries), float(no_object_weight), dtype=torch.float32, device=device)

    matched_pred_peak: list[torch.Tensor] = []
    matched_gt_peak: list[torch.Tensor] = []
    matched_gt_full_valid: list[torch.Tensor] = []
    matched_pred_complete: list[torch.Tensor] = []
    matched_pred_observed: list[torch.Tensor] = []
    matched_gt_observed: list[torch.Tensor] = []
    matched_pred_dir: list[torch.Tensor] = []
    matched_gt_dir: list[torch.Tensor] = []
    matched_pred_speed: list[torch.Tensor] = []
    matched_gt_speed: list[torch.Tensor] = []
    matched_total = 0
    gt_total = 0

    for b in range(batch_size):
        target = _target_at(target_dict, b)
        gt_total += _target_valid_count(target)
        rows, cols = _match_single(outputs, target, b, matcher=matcher)
        matched_total += int(rows.numel())
        if rows.numel() > 0:
            obj_target[b, rows] = 1.0
            obj_weight[b, rows] = 1.0
        if rows.numel() == 0:
            continue
        gt_full_time, gt_full_valid, gt_observed, gt_dir, gt_speed = _target_attrs(
            target,
            n_channels=int(outputs["peak_time"].shape[-1]),
            device=device,
        )
        matched_pred_peak.append(outputs["peak_time"][b, rows])
        matched_gt_peak.append(gt_full_time[cols])
        matched_gt_full_valid.append(gt_full_valid[cols])
        matched_pred_complete.append(outputs["complete_valid_logits"][b, rows])
        matched_pred_observed.append(outputs["observed_logits"][b, rows])
        matched_gt_observed.append(gt_observed[cols])
        matched_pred_dir.append(outputs["direction_logits"][b, rows])
        matched_gt_dir.append(gt_dir[cols])
        matched_pred_speed.append(outputs["speed"][b, rows])
        matched_gt_speed.append(gt_speed[cols])

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss_obj = F.binary_cross_entropy_with_logits(
        outputs["objectness_logits"][:, :max_queries],
        obj_target,
        weight=obj_weight,
        reduction="mean",
    )

    if matched_pred_peak:
        pred_peak = torch.cat(matched_pred_peak, dim=0)
        gt_peak = torch.cat(matched_gt_peak, dim=0)
        gt_full_valid = torch.cat(matched_gt_full_valid, dim=0)
        pred_complete = torch.cat(matched_pred_complete, dim=0)
        pred_observed = torch.cat(matched_pred_observed, dim=0)
        gt_observed = torch.cat(matched_gt_observed, dim=0)
        pred_dir = torch.cat(matched_pred_dir, dim=0)
        gt_dir = torch.cat(matched_gt_dir, dim=0)
        pred_speed = torch.cat(matched_pred_speed, dim=0)
        gt_speed = torch.cat(matched_gt_speed, dim=0)

        valid_mask = gt_full_valid > 0.5
        loss_peak = F.smooth_l1_loss(pred_peak[valid_mask], gt_peak[valid_mask], reduction="mean") if torch.any(valid_mask) else zero
        loss_complete = F.binary_cross_entropy_with_logits(pred_complete, gt_full_valid, reduction="mean")
        loss_observed = F.binary_cross_entropy_with_logits(pred_observed, gt_observed, reduction="mean")
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")
        loss_linearity = _linearity_penalty(pred_peak, gt_full_valid) if float(linearity_weight) > 0.0 else zero
        loss_smoothness = _smoothness_penalty(pred_peak, gt_full_valid) if float(smoothness_weight) > 0.0 else zero
    else:
        loss_peak = zero
        loss_complete = zero
        loss_observed = zero
        loss_dir = zero
        loss_speed = zero
        loss_linearity = zero
        loss_smoothness = zero
    loss_duplicate = _duplicate_query_penalty(outputs, max_queries) if float(duplicate_weight) > 0.0 else zero

    total = (
        loss_obj
        + 8.0 * loss_peak
        + 1.0 * loss_complete
        + 1.0 * loss_observed
        + 0.5 * loss_dir
        + 0.5 * loss_speed
        + float(duplicate_weight) * loss_duplicate
        + float(linearity_weight) * loss_linearity
        + float(smoothness_weight) * loss_smoothness
    )
    if not collect_metrics:
        return total, {}

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :max_queries])
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_peak": float(loss_peak.detach().cpu()),
        "loss_complete": float(loss_complete.detach().cpu()),
        "loss_observed": float(loss_observed.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_duplicate": float(loss_duplicate.detach().cpu()),
        "loss_linearity": float(loss_linearity.detach().cpu()),
        "loss_smoothness": float(loss_smoothness.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "mean_objectness": float(torch.mean(obj_prob).detach().cpu()),
        "max_objectness": float(torch.max(obj_prob).detach().cpu()),
    }
    return total, metrics


def auto_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: PeakSetModelConfig,
    dataset_config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "dataset_config": dict(dataset_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_type": type(model).__name__,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, str(Path(path).expanduser()))


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[nn.Module, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config_payload = dict(checkpoint.get("model_config", {}))
    model_config_payload.setdefault("residual_completion", False)
    model_config_payload.setdefault("max_residual_norm", 0.005)
    model_config_payload.setdefault("max_base_slope_norm", 0.75)
    model_config_payload.setdefault("max_base_curve_norm", 0.35)
    model_config = PeakSetModelConfig(**model_config_payload)
    model_type = str(checkpoint.get("model_type", "VehiclePeakSetTransformer"))
    if model_type == "PeakGuidedVehicleSetTransformer":
        model = PeakGuidedVehicleSetTransformer(model_config).to(resolved_device)
    else:
        model = VehiclePeakSetTransformer(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()
    return model, checkpoint


def _refine_peak_index(data: np.ndarray, ch: int, predicted_t_idx: int, radius: int) -> int:
    n_samples = int(data.shape[1])
    center = int(max(0, min(n_samples - 1, predicted_t_idx)))
    rad = int(max(0, radius))
    if rad <= 0:
        return center
    left = max(0, center - rad)
    right = min(n_samples - 1, center + rad)
    local = np.abs(data[int(ch), left : right + 1])
    if local.size == 0:
        return center
    return int(left + int(np.argmax(local)))


def _track_line_fit(points: list[TrackPoint]) -> tuple[float, float] | None:
    if len(points) < 2:
        return None
    ch = np.asarray([int(p.ch_idx) for p in points], dtype=np.float64)
    t = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
    if np.ptp(ch) < 1e-9:
        return None
    slope, intercept = np.polyfit(ch, t, deg=1)
    return float(slope), float(intercept)


def _track_line_distance(a: Track, b: Track) -> float:
    fa = _track_line_fit(a.points)
    fb = _track_line_fit(b.points)
    if fa is None or fb is None:
        return float("inf")
    slope_a, intercept_a = fa
    slope_b, intercept_b = fb
    center_a = 0.5 * (float(min(int(p.ch_idx) for p in a.points)) + float(max(int(p.ch_idx) for p in a.points)))
    center_b = 0.5 * (float(min(int(p.ch_idx) for p in b.points)) + float(max(int(p.ch_idx) for p in b.points)))
    center = 0.5 * (center_a + center_b)
    pred_a = slope_a * center + intercept_a
    pred_b = slope_b * center + intercept_b
    return abs(pred_a - pred_b) + 0.2 * abs(slope_a - slope_b)


def _weighted_line_fit(ch_idx: np.ndarray, t_idx: np.ndarray, weights: np.ndarray) -> tuple[float, float] | None:
    if ch_idx.size < 2 or t_idx.size < 2:
        return None
    w = np.asarray(weights, dtype=np.float64)
    if w.size != ch_idx.size:
        w = np.ones_like(ch_idx, dtype=np.float64)
    w = np.clip(w, 1e-3, None)
    total = float(np.sum(w))
    if total <= 1e-9:
        return None
    ch_mean = float(np.sum(w * ch_idx) / total)
    t_mean = float(np.sum(w * t_idx) / total)
    ch_centered = ch_idx - ch_mean
    denom = float(np.sum(w * ch_centered * ch_centered))
    if denom <= 1e-9:
        return None
    slope = float(np.sum(w * ch_centered * (t_idx - t_mean)) / denom)
    intercept = float(t_mean - slope * ch_mean)
    return slope, intercept


def _robust_line_fit(
    points: list[TrackPoint],
    *,
    fs: float,
    dx_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    direction_hint: str = "forward",
) -> tuple[float, float, str] | None:
    if len(points) < 2:
        return None
    ch_idx = np.asarray([int(p.ch_idx) for p in points], dtype=np.float64)
    t_idx = np.asarray([int(p.t_idx) for p in points], dtype=np.float64)
    weights = np.asarray([max(0.1, float(p.score)) for p in points], dtype=np.float64)
    fit = _weighted_line_fit(ch_idx, t_idx, weights)
    if fit is None:
        return None
    slope, intercept = fit

    if len(points) >= 4:
        residual = t_idx - (slope * ch_idx + intercept)
        mad = float(np.median(np.abs(residual - np.median(residual))))
        scale = max(3.0, 1.4826 * mad)
        keep = np.abs(residual) <= 3.5 * scale
        if int(np.sum(keep)) >= 2 and int(np.sum(keep)) < len(points):
            fit = _weighted_line_fit(ch_idx[keep], t_idx[keep], weights[keep])
            if fit is not None:
                slope, intercept = fit

    min_step = float(fs) * float(dx_m) / max(1e-6, float(speed_max_kmh) / 3.6)
    max_step = float(fs) * float(dx_m) / max(1e-6, float(speed_min_kmh) / 3.6)
    slope_mag = float(np.clip(abs(slope), min_step, max_step))
    inferred_direction = "forward" if slope >= 0.0 else "reverse"
    if abs(slope) < 1e-6:
        inferred_direction = str(direction_hint)
    slope = slope_mag if inferred_direction == "forward" else -slope_mag
    return float(slope), float(intercept), inferred_direction


def _clamp_prediction_to_bounds(
    predicted: int,
    *,
    lower: int,
    upper: int,
) -> int:
    if lower > upper:
        lower, upper = upper, lower
    return int(max(lower, min(upper, int(predicted))))


def _build_fused_peak_track(
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
) -> DecodedPeakTrack | None:
    complete_mask = np.asarray(complete_prob, dtype=np.float32) >= float(cfg.complete_valid_threshold)
    if int(np.sum(complete_mask)) < int(cfg.min_visible_channels):
        return None

    point_channels = np.where(complete_mask)[0].tolist()
    points: list[TrackPoint] = []
    observed_mask: list[bool] = []
    point_times_norm: list[float] = []
    anchor_count = 0
    n_samples = int(round(float(window_seconds) * float(fs)))
    n_samples = max(1, n_samples)
    for ch in point_channels:
        observed_score = float(observed_prob[ch]) if observed_prob is not None else 1.0
        observed_floor = float(np.clip(float(getattr(cfg, "observed_valid_soft_floor", 0.6)), 0.0, 1.0))
        observed_soft = observed_floor + (1.0 - observed_floor) * float(np.clip(observed_score, 0.0, 1.0))
        choice, fused_score, anchor_prob, signal_score, time_score = _fused_anchor_choice(
            anchor_logits=anchor_logits[ch],
            peak_time=peak_time[ch],
            peak_amp=peak_amp[ch],
            peak_valid=peak_valid[ch],
            raw_window=raw_window,
            channel_index=int(ch),
            complete_time=float(np.clip(float(complete_time[ch]), 0.0, 1.0)),
            window_seconds=float(window_seconds),
            cfg=cfg,
        )
        fused_score = float(fused_score) * float(observed_soft)
        if choice is not None and float(fused_score) >= float(cfg.fused_min_anchor_score):
            t_norm = float(np.clip(float(peak_time[ch, choice]), 0.0, 1.0))
            amp = float(peak_amp[ch, choice])
            score = float(objectness * float(complete_prob[ch]) * max(float(fused_score), 0.25))
            is_observed = True
            anchor_count += 1
        else:
            t_norm = float(np.clip(float(complete_time[ch]), 0.0, 1.0))
            amp = 0.0
            score = float(objectness * float(complete_prob[ch]) * float(cfg.fused_missing_point_scale))
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

    anchor_ratio = float(anchor_count) / float(max(1, len(point_channels)))
    if int(anchor_count) < int(cfg.min_anchor_support_channels):
        return None
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
    return DecodedPeakTrack(
        track=track,
        objectness=float(objectness),
        query_index=int(query_index),
        channel_indices=np.asarray(point_channels, dtype=np.int64),
        complete_valid=np.asarray(complete_mask[point_channels], dtype=np.float32),
        observed_valid=np.asarray(observed_mask, dtype=np.float32),
        point_times_norm=np.asarray(point_times_norm, dtype=np.float32),
    )


def _refine_decoded_track(
    item: DecodedPeakTrack,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    cfg: PeakSetInferenceConfig,
) -> DecodedPeakTrack:
    if not bool(cfg.graph_refine):
        return item

    track = item.track
    points = list(sorted(track.points, key=lambda p: int(p.ch_idx)))
    if len(points) < 2:
        return item

    observed_mask = np.asarray(item.observed_valid, dtype=bool)
    observed_points = [pt for pt, keep in zip(points, observed_mask) if bool(keep)]
    fit_points = observed_points if len(observed_points) >= 2 else points
    raw_times = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
    raw_span = float(np.ptp(raw_times)) if raw_times.size > 1 else 0.0
    fit = _robust_line_fit(
        fit_points,
        fs=float(fs),
        dx_m=float(x_axis_m[1] - x_axis_m[0] if len(x_axis_m) > 1 else 20.0),
        speed_min_kmh=float(cfg.speed_min_kmh),
        speed_max_kmh=float(cfg.speed_max_kmh),
        direction_hint=str(track.direction),
    )
    if fit is None:
        return item

    slope, intercept, inferred_direction = fit
    n_samples = int(data_window.shape[1])
    search_radius = int(max(0, cfg.refine_radius_samples))
    if len(observed_points) >= 3:
        search_radius = max(search_radius, int(round(abs(slope) * 0.12)))
    max_step = float(fs) * float(x_axis_m[1] - x_axis_m[0] if len(x_axis_m) > 1 else 20.0) / max(1e-6, float(cfg.speed_min_kmh) / 3.6)
    if search_radius > 0:
        search_radius = int(min(search_radius, max(12, int(round(max_step * 0.25)))))

    refined_points: list[TrackPoint] = []
    for point, is_observed in zip(points, observed_mask):
        ch = int(point.ch_idx)
        predicted = int(round(slope * float(ch) + intercept))
        predicted = _clamp_prediction_to_bounds(predicted, lower=0, upper=n_samples - 1)
        if bool(is_observed) and search_radius > 0:
            t_idx = _refine_peak_index(data_window, ch, predicted, search_radius)
            t_idx = _clamp_prediction_to_bounds(t_idx, lower=0, upper=n_samples - 1)
            amp = float(abs(data_window[ch, t_idx]))
            score = float(point.score * (0.65 + 0.35 * min(1.0, amp / (np.max(np.abs(data_window[ch])) + 1e-6))))
        elif bool(is_observed):
            t_idx = predicted
            amp = float(abs(data_window[ch, t_idx]))
            score = float(point.score)
        else:
            t_idx = predicted
            amp = 0.0
            score = float(point.score * 0.35)
        refined_points.append(
            TrackPoint(
                ch_idx=ch,
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(max(1e-9, fs)),
                offset_m=float(x_axis_m[ch]),
                amp=amp,
                score=score,
            )
        )

    refined_points = sorted(refined_points, key=lambda p: int(p.ch_idx))
    if len(refined_points) >= 2:
        ts = np.asarray([float(p.time_s) for p in refined_points], dtype=np.float64)
        chs = np.asarray([int(p.ch_idx) for p in refined_points], dtype=np.float64)
        dt = np.diff(ts)
        dch = np.diff(chs)
        valid = np.abs(dt) > 1e-9
        mean_speed_kmh = float("nan")
        if np.any(valid):
            speed_mps = np.abs(dch[valid]) * float(x_axis_m[1] - x_axis_m[0] if len(x_axis_m) > 1 else 20.0) / np.abs(dt[valid])
            mean_speed_kmh = float(3.6 * np.mean(speed_mps))
    else:
        mean_speed_kmh = float("nan")

    total_score = float(sum(pt.score for pt in refined_points))
    direction = "forward" if inferred_direction == "forward" else "reverse"

    refined_times = np.asarray([float(p.time_s) for p in refined_points], dtype=np.float64)
    refined_span = float(np.ptp(refined_times)) if refined_times.size > 1 else 0.0
    clamp_count = int(
        sum(
            int(p.t_idx <= 0 or p.t_idx >= n_samples - 1)
            for p in refined_points
        )
    )
    clamp_ratio = float(clamp_count) / float(max(1, len(refined_points)))
    speed_floor = max(10.0, float(cfg.speed_min_kmh) * 0.5)
    speed_ceil = max(float(cfg.speed_max_kmh) * 1.8, float(cfg.speed_max_kmh) + 40.0)
    if (
        len(refined_points) < len(points)
        or refined_span < 0.4 * raw_span
        or clamp_ratio > 0.25
        or not np.isfinite(mean_speed_kmh)
        or mean_speed_kmh < speed_floor
        or mean_speed_kmh > speed_ceil
    ):
        return item

    return DecodedPeakTrack(
        track=Track(
            track_id=int(track.track_id),
            direction=direction,
            points=refined_points,
            total_score=total_score,
            mean_speed_kmh=mean_speed_kmh,
        ),
        objectness=float(item.objectness),
        query_index=int(item.query_index),
        channel_indices=np.asarray(item.channel_indices, dtype=np.int64),
        complete_valid=np.asarray(item.complete_valid, dtype=np.float32),
        observed_valid=np.asarray(item.observed_valid, dtype=np.float32),
        point_times_norm=np.asarray(item.point_times_norm, dtype=np.float32),
    )


def _deduplicate_tracks(decoded: list[DecodedPeakTrack], tol_samples: int) -> list[DecodedPeakTrack]:
    kept: list[DecodedPeakTrack] = []
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
            if (ratio >= 0.7 and line_distance < 3.0) or (ratio >= 0.55 and line_distance < 5.0 and speed_diff < 12.0):
                duplicate = True
                break
        if not duplicate:
            kept.append(item)
    return kept


def decode_vehicle_peak_tracks(
    model: VehiclePeakSetTransformer,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[PeakSetInferenceConfig] = None,
    device: Optional[str] = None,
) -> list[DecodedPeakTrack]:
    cfg = config or PeakSetInferenceConfig()
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    if arr.shape[0] != int(model.config.n_channels):
        raise ValueError(f"Model expects {model.config.n_channels} channels, got {arr.shape[0]}")
    resolved_device = device or next(model.parameters()).device
    x = prepare_peakset_input(arr, int(cfg.time_downsample), float(cfg.clip_ratio), input_mode="raw" if int(model.config.in_channels) == 1 else "raw_abs")
    if int(model.config.in_channels) == 1 and x.shape[0] != 1:
        x = x[:1]
    x = x.unsqueeze(0).to(resolved_device)

    with torch.inference_mode():
        outputs = model(x)

    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    complete_prob = torch.sigmoid(outputs["complete_valid_logits"][0]).detach().cpu().numpy()
    observed_prob = torch.sigmoid(outputs["observed_logits"][0]).detach().cpu().numpy()
    peak_time = outputs["peak_time"][0].detach().cpu().numpy()
    direction = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    order = np.argsort(obj)[::-1]
    tracks: list[DecodedPeakTrack] = []
    n_samples = int(arr.shape[1])
    for q_idx in order[: int(max(1, cfg.max_tracks))]:
        if float(obj[q_idx]) < float(cfg.objectness_threshold):
            continue
        complete_mask = complete_prob[q_idx] >= float(cfg.complete_valid_threshold)
        if int(np.sum(complete_mask)) < int(cfg.min_visible_channels):
            continue

        point_channels = np.where(complete_mask)[0].tolist()
        points: list[TrackPoint] = []
        observed_mask = []
        point_times_norm = []
        complete_scores = []
        observed_scores = []
        for ch in point_channels:
            t_norm = float(np.clip(peak_time[q_idx, ch], 0.0, 1.0))
            t_idx_pred = int(round(t_norm * float(max(1, n_samples - 1))))
            is_observed = bool(observed_prob[q_idx, ch] >= float(cfg.anchor_threshold))
            if is_observed:
                t_idx = _refine_peak_index(arr, int(ch), t_idx_pred, int(cfg.refine_radius_samples))
                amp = float(abs(arr[int(ch), t_idx]))
                score = float(obj[q_idx] * complete_prob[q_idx, ch] * observed_prob[q_idx, ch])
            else:
                t_idx = t_idx_pred
                amp = 0.0
                score = float(obj[q_idx] * complete_prob[q_idx, ch] * 0.35)
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=int(t_idx),
                    time_s=float(t_idx) / float(max(1e-9, fs)),
                    offset_m=float(x_axis_m[int(ch)]),
                    amp=amp,
                    score=score,
                )
            )
            observed_mask.append(is_observed)
            point_times_norm.append(t_norm)
            complete_scores.append(float(complete_prob[q_idx, ch]))
            observed_scores.append(float(observed_prob[q_idx, ch]))

        if len(points) < int(cfg.min_visible_channels):
            continue
        points = sorted(points, key=lambda p: p.ch_idx)
        if len(points) < int(cfg.min_visible_channels):
            continue
        mean_speed = float("nan")
        if len(points) >= 2:
            ts = np.asarray([p.time_s for p in points], dtype=np.float64)
            chs = np.asarray([p.ch_idx for p in points], dtype=np.float64)
            dt = np.diff(ts)
            dch = np.diff(chs)
            valid = np.abs(dt) > 1e-9
            if np.any(valid):
                speed_mps = np.abs(dch[valid]) * float(x_axis_m[1] - x_axis_m[0] if len(x_axis_m) > 1 else 20.0) / np.abs(dt[valid])
                mean_speed = float(3.6 * np.mean(speed_mps))
        track = Track(
            track_id=int(len(tracks)),
            direction="forward" if int(direction[q_idx]) == 0 else "reverse",
            points=points,
            total_score=float(sum(p.score for p in points)),
            mean_speed_kmh=mean_speed,
        )
        tracks.append(
            DecodedPeakTrack(
                track=track,
                objectness=float(obj[q_idx]),
                query_index=int(q_idx),
                channel_indices=np.asarray(point_channels, dtype=np.int64),
                complete_valid=np.asarray(complete_mask[point_channels], dtype=np.float32),
                observed_valid=np.asarray(observed_mask, dtype=np.float32),
                point_times_norm=np.asarray(point_times_norm, dtype=np.float32),
            )
        )

    if bool(cfg.graph_refine):
        tracks = [
            _refine_decoded_track(item, arr, fs=float(fs), x_axis_m=np.asarray(x_axis_m, dtype=np.float32), cfg=cfg)
            for item in tracks
        ]
    return _deduplicate_tracks(tracks, int(cfg.dedup_tolerance_samples))


def predict_tracks_from_window(
    model: VehiclePeakSetTransformer,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[PeakSetInferenceConfig] = None,
    device: Optional[str] = None,
) -> list[Track]:
    return [item.track for item in decode_vehicle_peak_tracks(model, data_window, fs, x_axis_m, config=config, device=device)]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _resolve_peak_inputs(
    x: torch.Tensor,
    peak_time: torch.Tensor | None,
    peak_amp: torch.Tensor | None,
    peak_valid: torch.Tensor | None,
    peak_index: torch.Tensor | None,
    *,
    peak_candidates: int,
    time_downsample: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if peak_time is not None and peak_amp is not None and peak_valid is not None and peak_index is not None:
        return peak_time, peak_amp, peak_valid, peak_index
    if x.ndim != 4:
        raise ValueError("x must have shape [B, C_in, n_channels, n_samples]")
    peak_time_list: list[torch.Tensor] = []
    peak_amp_list: list[torch.Tensor] = []
    peak_valid_list: list[torch.Tensor] = []
    peak_index_list: list[torch.Tensor] = []
    for item in x.detach().cpu():
        pt, pa, pv, pi = detect_peak_candidates(
            item.numpy(),
            fs=1000.0,
            time_downsample=int(time_downsample),
            candidates_per_channel=int(peak_candidates),
            min_distance_s=0.15,
            min_height=0.02,
            prominence=0.02,
        )
        peak_time_list.append(pt)
        peak_amp_list.append(pa)
        peak_valid_list.append(pv)
        peak_index_list.append(pi)
    return (
        torch.stack(peak_time_list, dim=0).to(x.device),
        torch.stack(peak_amp_list, dim=0).to(x.device),
        torch.stack(peak_valid_list, dim=0).to(x.device),
        torch.stack(peak_index_list, dim=0).to(x.device),
    )


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    arr = np.asarray(logits, dtype=np.float32)
    if arr.size == 0:
        return arr
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    denom = float(np.sum(exp))
    if denom <= 1e-9:
        return np.full_like(arr, 1.0 / float(max(1, arr.size)), dtype=np.float32)
    return exp / denom


def _fused_anchor_choice(
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
) -> tuple[int | None, float, float, float, float]:
    valid = np.asarray(peak_valid, dtype=bool)
    if valid.size == 0 or not bool(valid.any()):
        return None, 0.0, 0.0, 0.0, 0.0

    probs = _softmax_numpy(np.asarray(anchor_logits, dtype=np.float32))
    valid_idx = np.where(valid)[0]
    amp_values = np.abs(np.asarray(peak_amp, dtype=np.float32))[valid_idx]
    amp_denom = max(1e-6, float(np.max(amp_values)) if amp_values.size > 0 else 1.0)
    time_tol = max(1e-6, float(cfg.fused_time_tolerance_s))
    best_choice: int | None = None
    best_score = -float("inf")
    best_prob = 0.0
    best_signal = 0.0
    best_time_score = 0.0
    raw_row = None if raw_window is None else np.asarray(raw_window[int(channel_index)], dtype=np.float32)
    raw_denom = max(1e-6, float(np.max(np.abs(raw_row))) if raw_row is not None and raw_row.size > 0 else 1.0)
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
        if fused > best_score:
            best_choice = int(choice)
            best_score = float(fused)
            best_prob = prob
            best_signal = signal
            best_time_score = time_score
    return best_choice, float(best_score), float(best_prob), float(best_signal), float(best_time_score)


class PeakGuidedVehicleSetTransformer(nn.Module):
    def __init__(self, config: Optional[PeakSetModelConfig] = None):
        super().__init__()
        self.config = config or PeakSetModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.backbone = nn.Sequential(
            _ConvBlock(int(c.in_channels), 32, stride=(1, 2)),
            _ConvBlock(32, 64, stride=(1, 2)),
            _ConvBlock(64, hidden, stride=(1, 2)),
            _ConvBlock(hidden, hidden, stride=(1, 1)),
        )
        self.channel_embed = nn.Parameter(torch.randn(1, int(c.n_channels), 1, hidden) * 0.02)
        self.time_embed = nn.Parameter(torch.randn(1, 1, int(c.pooled_time), hidden) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(c.encoder_layers))
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(c.decoder_layers))
        self.query_embed = nn.Embedding(int(c.max_queries), hidden)
        self.complete_pair_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.anchor_context_head = nn.Sequential(
            nn.LayerNorm(hidden * 2 + 2),
            nn.Linear(hidden * 2 + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.anchor_candidate_head = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.complete_time_head = nn.Linear(hidden, 1)
        self.trajectory_param_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.residual_time_head = nn.Linear(hidden, 1)
        self.complete_valid_head = nn.Linear(hidden, 1)
        self.observed_valid_head = nn.Linear(hidden, 1)
        self.anchor_none_head = nn.Linear(hidden, 1)

    def forward(
        self,
        x: torch.Tensor,
        peak_time: torch.Tensor | None = None,
        peak_amp: torch.Tensor | None = None,
        peak_valid: torch.Tensor | None = None,
        peak_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, C_in, n_channels, n_samples]")
        feat = self.backbone(x)
        feat = F.interpolate(feat, size=(int(self.config.n_channels), int(self.config.pooled_time)), mode="bilinear", align_corners=False)
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat + self.channel_embed[:, : feat.shape[1], :, :] + self.time_embed[:, :, : feat.shape[2], :]
        memory = self.encoder(feat.view(x.shape[0], -1, feat.shape[-1]))
        query = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        hs = self.decoder(tgt=query, memory=memory)

        peak_time, peak_amp, peak_valid, peak_index = _resolve_peak_inputs(
            x,
            peak_time,
            peak_amp,
            peak_valid,
            peak_index,
            peak_candidates=int(self.config.peak_candidates),
            time_downsample=10,
        )
        peak_time = peak_time.to(device=x.device, dtype=torch.float32)
        peak_amp = peak_amp.to(device=x.device, dtype=torch.float32)
        peak_valid = peak_valid.to(device=x.device, dtype=torch.bool)
        peak_index = peak_index.to(device=x.device, dtype=torch.long)

        bsz = int(x.shape[0])
        n_ch = int(self.config.n_channels)
        n_peak = int(peak_time.shape[-1])
        channel_embed = self.channel_embed[:, :n_ch, 0, :].unsqueeze(1)
        query_expand = hs.unsqueeze(2).expand(-1, -1, n_ch, -1)
        channel_expand = channel_embed.expand(bsz, hs.shape[1], -1, -1)

        complete_pair = self.complete_pair_head(torch.cat([query_expand, channel_expand], dim=-1))
        ch_norm = torch.linspace(0.0, 1.0, n_ch, device=x.device, dtype=hs.dtype).view(1, 1, n_ch)
        ch_center = ch_norm - 0.5
        if bool(getattr(self.config, "residual_completion", True)):
            trajectory_params = self.trajectory_param_head(hs)
            base_center = torch.sigmoid(trajectory_params[..., 0]).unsqueeze(-1)
            base_slope = float(getattr(self.config, "max_base_slope_norm", 0.75)) * torch.tanh(trajectory_params[..., 1]).unsqueeze(-1)
            base_curve = float(getattr(self.config, "max_base_curve_norm", 0.35)) * torch.tanh(trajectory_params[..., 2]).unsqueeze(-1)
            curve_basis = ch_center.square() - float((0.5**2) / 3.0)
            base_time = base_center + base_slope * ch_center + base_curve * curve_basis
            residual_time = float(getattr(self.config, "max_residual_norm", 0.005)) * torch.tanh(
                self.residual_time_head(complete_pair).squeeze(-1)
            )
            complete_time = torch.clamp(base_time + residual_time, 0.0, 1.0)
        else:
            trajectory_params = torch.zeros((bsz, hs.shape[1], 3), device=x.device, dtype=hs.dtype)
            base_time = torch.sigmoid(self.complete_time_head(complete_pair).squeeze(-1))
            residual_time = torch.zeros_like(base_time)
            complete_time = base_time
        complete_valid_logits = self.complete_valid_head(complete_pair).squeeze(-1)
        complete_valid_prob = torch.sigmoid(complete_valid_logits)
        observed_valid_logits = self.observed_valid_head(complete_pair).squeeze(-1)
        observed_valid_prob = torch.sigmoid(observed_valid_logits)

        peak_time_exp = peak_time[:, None, :, :].expand(bsz, hs.shape[1], n_ch, n_peak)
        peak_amp_exp = peak_amp[:, None, :, :].expand(bsz, hs.shape[1], n_ch, n_peak)
        peak_valid_exp = peak_valid[:, None, :, :].expand(bsz, hs.shape[1], n_ch, n_peak).to(dtype=complete_time.dtype)
        ch_pos = torch.linspace(0.0, 1.0, n_ch, device=x.device, dtype=complete_time.dtype).view(1, 1, n_ch, 1)
        ch_pos_exp = ch_pos.expand(bsz, hs.shape[1], n_ch, n_peak)
        complete_time_exp = complete_time.unsqueeze(-1).expand(-1, -1, -1, n_peak)
        delta_t = peak_time_exp - complete_time_exp

        anchor_context = self.anchor_context_head(
            torch.cat(
                [
                    query_expand,
                    channel_expand,
                    complete_time.unsqueeze(-1),
                    complete_valid_prob.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        candidate_tokens = self.anchor_candidate_head(
            torch.stack(
                [
                    peak_time_exp,
                    peak_amp_exp,
                    peak_valid_exp,
                    ch_pos_exp,
                    delta_t,
                    delta_t.abs(),
                ],
                dim=-1,
            )
        )
        anchor_logits = torch.einsum("bqch,bqckh->bqck", anchor_context, candidate_tokens) / float(max(1, anchor_context.shape[-1])) ** 0.5
        anchor_logits = anchor_logits.masked_fill(~peak_valid[:, None, :, :], -1e4)
        anchor_none_logits = self.anchor_none_head(anchor_context).squeeze(-1).unsqueeze(-1)
        anchor_peak_logits = torch.cat([anchor_logits, anchor_none_logits], dim=-1)

        return {
            "num_regular_queries": int(self.config.max_queries),
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "complete_time": complete_time,
            "base_time": base_time,
            "residual_time": residual_time,
            "trajectory_params": trajectory_params,
            "complete_valid_logits": complete_valid_logits,
            "observed_valid_logits": observed_valid_logits,
            "anchor_peak_logits": anchor_peak_logits,
            "peak_time": peak_time,
            "peak_amp": peak_amp,
            "peak_valid": peak_valid,
            "peak_index": peak_index,
        }


def _match_complete_single(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch_idx: int,
    matcher: str = "hungarian",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["objectness_logits"].device
    n_gt = _target_valid_count(target)
    if n_gt == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    regular_q = _regular_query_count(outputs)
    pred_obj = torch.sigmoid(outputs["objectness_logits"][batch_idx, :regular_q])
    pred_time = outputs["complete_time"][batch_idx, :regular_q]
    pred_complete = torch.sigmoid(outputs["complete_valid_logits"][batch_idx, :regular_q])
    pred_observed_logits = outputs.get("observed_valid_logits")
    gt_full_time, gt_full_valid, gt_observed, gt_dir, gt_speed = _target_attrs(
        target,
        n_channels=int(outputs["complete_time"].shape[-1]),
        device=device,
    )
    diff = torch.abs(pred_time[:, None, :] - gt_full_time[None, :, :])
    denom = torch.clamp(gt_full_valid.sum(dim=-1), min=1.0)[None, :]
    time_cost = (diff * gt_full_valid[None, :, :]).sum(dim=-1) / denom
    complete_cost = F.binary_cross_entropy_with_logits(
        outputs["complete_valid_logits"][batch_idx, :regular_q, :][:, None, :].expand(-1, int(gt_full_valid.shape[0]), -1),
        gt_full_valid[None, :, :].expand(regular_q, -1, -1),
        reduction="none",
    ).mean(dim=-1)
    if pred_observed_logits is not None:
        observed_cost = F.binary_cross_entropy_with_logits(
            pred_observed_logits[batch_idx, :regular_q, :][:, None, :].expand(-1, int(gt_observed.shape[0]), -1),
            gt_observed[None, :, :].expand(regular_q, -1, -1),
            reduction="none",
        ).mean(dim=-1)
    else:
        observed_cost = torch.zeros_like(complete_cost)
    obj_cost = -pred_obj[:, None]
    cost = 5.0 * time_cost + 1.0 * complete_cost + 1.0 * observed_cost + 0.75 * obj_cost
    if str(matcher).lower() == "greedy":
        rows, cols = _greedy_match_cost(cost)
        return rows.to(device=device), cols.to(device=device)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(rows, dtype=torch.long, device=device), torch.as_tensor(cols, dtype=torch.long, device=device)


def peak_guided_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.02,
    none_weight: float = 0.15,
    matcher: str = "hungarian",
    collect_metrics: bool = True,
    complete_time_weight: float = 8.0,
    complete_valid_weight: float = 1.0,
    missing_complete_time_weight: float = 0.0,
    missing_complete_valid_weight: float = 0.0,
    observed_valid_weight: float = 1.0,
    anchor_weight: float = 1.0,
    anchor_time_weight: float = 0.5,
    inertia_weight: float = 0.25,
    line_weight: float = 0.4,
    jump_weight: float = 0.20,
    slope_variation_weight: float = 0.30,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    max_queries = _regular_query_count(outputs)
    if isinstance(targets, dict) and "gt_valid" in targets:
        target_dict = targets
    else:
        target_dict = targets_to_batched_peakset(targets if isinstance(targets, Sequence) else [targets])  # type: ignore[arg-type]

    gt_valid = target_dict["gt_valid"].to(device=device, dtype=torch.bool)
    missing_channel_mask = target_dict.get("missing_channel_mask")
    if missing_channel_mask is None:
        missing_channel_mask = torch.zeros(
            (batch_size, int(target_dict["full_time"].shape[1]), int(target_dict["full_time"].shape[2])),
            dtype=torch.float32,
            device=device,
        )
    else:
        missing_channel_mask = missing_channel_mask.to(device=device, dtype=torch.float32)
    obj_target = torch.zeros((batch_size, max_queries), dtype=torch.float32, device=device)
    obj_weight = torch.full((batch_size, max_queries), float(no_object_weight), dtype=torch.float32, device=device)

    matched_b: list[torch.Tensor] = []
    matched_q: list[torch.Tensor] = []
    matched_g: list[torch.Tensor] = []
    matched_anchor_time: list[torch.Tensor] = []
    matched_anchor_valid: list[torch.Tensor] = []
    matched_total = 0
    gt_total = 0

    for b in range(batch_size):
        target = _target_at(target_dict, b)
        gt_total += _target_valid_count(target)
        rows, cols = _match_complete_single(outputs, target, b, matcher=matcher)
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
        outputs["objectness_logits"][:, :max_queries],
        obj_target,
        weight=obj_weight,
        reduction="mean",
    )

    if matched_b:
        b_sel = torch.cat(matched_b)
        q_sel = torch.cat(matched_q)
        g_sel = torch.cat(matched_g)
        mapped_g = _valid_rank_to_gt_index(gt_valid)[b_sel, g_sel]
        gt_full_time = target_dict["full_time"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_full_valid = target_dict["full_valid"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_observed = target_dict["observed_visibility"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_dir = target_dict["direction"][b_sel, mapped_g].to(device=device, dtype=torch.long)
        gt_missing = missing_channel_mask[b_sel, mapped_g]
        pred_time = outputs["complete_time"][b_sel, q_sel]
        pred_complete_valid_logits = outputs["complete_valid_logits"][b_sel, q_sel]
        pred_observed_valid_logits = outputs["observed_valid_logits"][b_sel, q_sel]
        pred_anchor_logits = outputs["anchor_peak_logits"][b_sel, q_sel]
        peak_time = target_dict["peak_time"][b_sel].to(device=device, dtype=torch.float32)
        peak_valid = target_dict["peak_valid"][b_sel].to(device=device, dtype=torch.bool)
        valid_mask = gt_full_valid > 0.5
        observed_valid_mask = (gt_observed > 0.5) & valid_mask
        completed_valid_mask = (gt_observed <= 0.5) & valid_mask
        loss_complete_time = (
            F.smooth_l1_loss(pred_time[completed_valid_mask], gt_full_time[completed_valid_mask], reduction="mean")
            if torch.any(completed_valid_mask)
            else zero
        )
        loss_complete_valid = F.binary_cross_entropy_with_logits(pred_complete_valid_logits, gt_full_valid, reduction="mean")
        loss_observed_valid = F.binary_cross_entropy_with_logits(pred_observed_valid_logits, gt_observed, reduction="mean")
        missing_valid_mask = (gt_missing > 0.5) & (gt_full_valid > 0.5)
        if torch.any(missing_valid_mask):
            loss_missing_complete_time = F.smooth_l1_loss(
                pred_time[missing_valid_mask],
                gt_full_time[missing_valid_mask],
                reduction="mean",
            )
            loss_missing_complete_valid = F.binary_cross_entropy_with_logits(
                pred_complete_valid_logits[missing_valid_mask],
                gt_full_valid[missing_valid_mask],
                reduction="mean",
            )
        else:
            loss_missing_complete_time = zero
            loss_missing_complete_valid = zero

        gt_anchor_index = target_dict["gt_peak_index"][b_sel, mapped_g].to(device=device, dtype=torch.long)
        none_index = int(pred_anchor_logits.shape[-1] - 1)
        anchor_flat = pred_anchor_logits.reshape(-1, none_index + 1)
        target_flat = gt_anchor_index.reshape(-1).clamp(0, none_index)
        ce = F.cross_entropy(anchor_flat, target_flat, reduction="none").view_as(gt_anchor_index)
        anchor_weight_map = torch.where(gt_anchor_index == none_index, torch.full_like(ce, float(none_weight)), torch.ones_like(ce))
        loss_anchor = (ce * anchor_weight_map).sum() / torch.clamp(anchor_weight_map.sum(), min=1.0)

        peak_time = target_dict["peak_time"][b_sel].to(device=device, dtype=torch.float32)
        gt_anchor_time = torch.zeros_like(gt_full_time)
        valid_anchor_mask = gt_anchor_index != none_index
        if torch.any(valid_anchor_mask):
            safe_index = gt_anchor_index.clamp(0, max(0, none_index - 1)).unsqueeze(-1)
            gathered = peak_time.gather(-1, safe_index).squeeze(-1)
            gt_anchor_time = torch.where(valid_anchor_mask, gathered, gt_anchor_time)
        observed_anchor_mask = valid_anchor_mask & observed_valid_mask
        loss_anchor_time = (
            F.smooth_l1_loss(pred_time[observed_anchor_mask], gt_anchor_time[observed_anchor_mask], reduction="mean")
            if torch.any(observed_anchor_mask)
            else zero
        )
        anchor_time_expect = _expected_anchor_time(pred_anchor_logits, peak_time, peak_valid)
        matched_anchor_time.append(anchor_time_expect)
        matched_anchor_valid.append(observed_anchor_mask.to(dtype=torch.float32))

        vis = gt_full_valid > 0.5
        pair_vis = vis[:, 1:] * vis[:, :-1]
        dt = pred_time[:, 1:] - pred_time[:, :-1]
        sign = torch.where(gt_dir[:, None] == 0, torch.ones_like(dt), -torch.ones_like(dt))
        loss_mono = (torch.relu(-(dt * sign)) * pair_vis).sum() / torch.clamp(pair_vis.sum(), min=1.0)
        loss_jump = _masked_first_difference_penalty(pred_time, gt_full_valid) if torch.any(vis) else zero
        loss_slope_variation = _masked_slope_variation_penalty(pred_time, gt_full_valid) if torch.any(vis) else zero
    else:
        loss_complete_time = zero
        loss_complete_valid = zero
        loss_observed_valid = zero
        loss_anchor = zero
        loss_anchor_time = zero
        loss_mono = zero
        loss_jump = zero
        loss_slope_variation = zero
    if matched_anchor_time:
        anchor_time_stack = torch.cat(matched_anchor_time, dim=0)
        anchor_valid_stack = torch.cat(matched_anchor_valid, dim=0)
        loss_inertia = _masked_second_difference_penalty(anchor_time_stack, anchor_valid_stack) if float(inertia_weight) > 0.0 else zero
    else:
        loss_inertia = zero
    total = (
        loss_obj
        + float(complete_time_weight) * loss_complete_time
        + float(complete_valid_weight) * loss_complete_valid
        + float(missing_complete_time_weight) * loss_missing_complete_time
        + float(missing_complete_valid_weight) * loss_missing_complete_valid
        + float(observed_valid_weight) * loss_observed_valid
        + float(anchor_weight) * loss_anchor
        + float(anchor_time_weight) * loss_anchor_time
        + float(inertia_weight) * loss_inertia
        + float(jump_weight) * loss_jump
        + float(slope_variation_weight) * loss_slope_variation
        + float(line_weight) * loss_mono
    )
    if not collect_metrics:
        return total, {}

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :max_queries])
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_complete_time": float(loss_complete_time.detach().cpu()),
        "loss_complete_valid": float(loss_complete_valid.detach().cpu()),
        "loss_observed_valid": float(loss_observed_valid.detach().cpu()),
        "loss_missing_complete_time": float(loss_missing_complete_time.detach().cpu()),
        "loss_missing_complete_valid": float(loss_missing_complete_valid.detach().cpu()),
        "loss_anchor": float(loss_anchor.detach().cpu()),
        "loss_anchor_time": float(loss_anchor_time.detach().cpu()),
        "loss_inertia": float(loss_inertia.detach().cpu()),
        "loss_monotonic": float(loss_mono.detach().cpu()),
        "loss_jump": float(loss_jump.detach().cpu()),
        "loss_slope_variation": float(loss_slope_variation.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(torch.max(obj_prob).detach().cpu()),
        "mean_objectness": float(torch.mean(obj_prob).detach().cpu()),
    }
    return total, metrics


def decode_peak_guided_vehicle_tracks(
    model: PeakGuidedVehicleSetTransformer,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[PeakSetInferenceConfig] = None,
    device: Optional[str] = None,
    *,
    peak_time: torch.Tensor | None = None,
    peak_amp: torch.Tensor | None = None,
    peak_valid: torch.Tensor | None = None,
    peak_index: torch.Tensor | None = None,
) -> list[DecodedPeakTrack]:
    cfg = config or PeakSetInferenceConfig()
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    if arr.shape[0] != int(model.config.n_channels):
        raise ValueError(f"Model expects {model.config.n_channels} channels, got {arr.shape[0]}")
    resolved_device = device or next(model.parameters()).device
    x = prepare_peakset_input(arr, int(cfg.time_downsample), float(cfg.clip_ratio), input_mode="raw" if int(model.config.in_channels) == 1 else "raw_abs")
    if int(model.config.in_channels) == 1 and x.shape[0] != 1:
        x = x[:1]
    x = x.unsqueeze(0).to(resolved_device)
    peak_time, peak_amp, peak_valid, peak_index = _resolve_peak_inputs(
        x,
        peak_time,
        peak_amp,
        peak_valid,
        peak_index,
        peak_candidates=int(model.config.peak_candidates),
        time_downsample=int(cfg.time_downsample),
    )
    peak_time = peak_time.to(resolved_device)
    peak_amp = peak_amp.to(resolved_device)
    peak_valid = peak_valid.to(resolved_device)
    peak_index = peak_index.to(resolved_device)

    with torch.inference_mode():
        outputs = model(x, peak_time=peak_time, peak_amp=peak_amp, peak_valid=peak_valid, peak_index=peak_index)

    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    complete_valid_prob = torch.sigmoid(outputs["complete_valid_logits"][0]).detach().cpu().numpy()
    observed_valid_prob = (
        torch.sigmoid(outputs["observed_valid_logits"][0]).detach().cpu().numpy()
        if "observed_valid_logits" in outputs and bool(getattr(cfg, "use_observed_valid_in_decode", False))
        else np.ones_like(complete_valid_prob, dtype=np.float32)
    )
    complete_time = outputs["complete_time"][0].detach().cpu().numpy()
    anchor_logits = outputs["anchor_peak_logits"][0].detach().cpu().numpy()
    peak_time_np = outputs["peak_time"][0].detach().cpu().numpy()
    peak_amp_np = outputs["peak_amp"][0].detach().cpu().numpy()
    peak_valid_np = outputs["peak_valid"][0].detach().cpu().numpy()
    direction = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    order = np.argsort(obj)[::-1]
    tracks: list[DecodedPeakTrack] = []
    for q_idx in order[: int(max(1, cfg.max_tracks))]:
        if float(obj[q_idx]) < float(cfg.objectness_threshold):
            continue
        item = _build_fused_peak_track(
            objectness=float(obj[q_idx]),
            query_index=int(q_idx),
            direction=int(direction[q_idx]),
            complete_prob=complete_valid_prob[q_idx],
            observed_prob=observed_valid_prob[q_idx],
            complete_time=complete_time[q_idx],
            anchor_logits=anchor_logits[q_idx],
            peak_time=peak_time_np,
            peak_amp=peak_amp_np,
            peak_valid=peak_valid_np,
            raw_window=arr,
            fs=float(fs),
            x_axis_m=np.asarray(x_axis_m, dtype=np.float32),
            cfg=cfg,
            window_seconds=float(arr.shape[1]) / float(max(1e-9, fs)),
        )
        if item is None:
            continue
        item.track.track_id = int(len(tracks))
        tracks.append(item)

    if bool(cfg.graph_refine):
        tracks = [
            _refine_decoded_track(item, arr, fs=float(fs), x_axis_m=np.asarray(x_axis_m, dtype=np.float32), cfg=cfg)
            for item in tracks
        ]
    return _deduplicate_tracks(tracks, int(cfg.dedup_tolerance_samples))
