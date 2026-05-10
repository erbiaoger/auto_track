"""PeakSlotNet for peak-candidate vehicle trajectory recognition.

Purpose:
    Predict vehicle instance slots by selecting per-channel peak candidates.
    Unlike TrackSlotNet, this model never outputs an arbitrary time for a
    channel. Each slot either chooses one detected peak candidate on that
    channel or chooses the final `none` class.

Example:
    uv run python -m autotrack.dl.train_peak_slot \
        --data-dir datasets/peak_slot/train \
        --out-dir models/peak_slot_cuda \
        --device cuda \
        --amp on

Outputs:
    objectness_logits [B, Q]
    direction_logits  [B, Q, 2]
    speed             [B, Q]
    peak_logits       [B, Q, C, K + 1], where K is candidate count and K is none.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from torch import nn

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.trajectory_set_model import (
    LABEL_TO_DIRECTION,
    WindowDatasetConfig,
    auto_torch_device,
    prepare_window_input,
)


@dataclass
class PeakDetectionConfig:
    candidates_per_channel: int = 64
    min_distance_s: float = 0.5
    min_height: float = 0.02
    prominence: float = 0.02
    match_tolerance_s: float = 0.25


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    max_tracks: int = 96
    peak_candidates: int = 64
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
    peak_threshold: float = 0.4
    min_visible_channels: int = 3
    max_tracks: int = 96
    dedup_tolerance_samples: int = 180
    dedup_min_overlap_channels: int = 3
    speed_norm_kmh: float = 150.0
    clip_ratio: float = 1.35
    peak_candidates: int = 64
    peak_min_distance_s: float = 0.5
    peak_min_height: float = 0.02
    peak_prominence: float = 0.02
    use_viterbi_decoder: bool = True
    viterbi_topk: int = 16
    viterbi_candidate_threshold: float = 0.01
    viterbi_speed_min_kmh: float = 60.0
    viterbi_speed_max_kmh: float = 100.0
    viterbi_max_skip_channels: int = 4
    viterbi_point_bonus: float = 3.0
    viterbi_skip_penalty: float = 2.0
    viterbi_speed_penalty: float = 1.0
    viterbi_smoothness_penalty: float = 0.6
    viterbi_inertia_penalty: float = 2.5
    viterbi_slope_memory: float = 0.75
    viterbi_fallback_speed_kmh: float = 80.0
    physics_smooth_tolerance_s: float = 2.0


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


def detect_peak_candidates_from_tensor(
    heatmap: torch.Tensor,
    *,
    fs: float,
    time_downsample: int,
    window_samples: int,
    config: PeakDetectionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Detect peak candidates for one normalized heatmap [C, T_down]."""
    arr = heatmap.detach().cpu().to(torch.float32).numpy()
    if arr.ndim != 2:
        raise ValueError("heatmap must have shape [channel, time]")
    n_ch, t_down = int(arr.shape[0]), int(arr.shape[1])
    k_count = int(config.candidates_per_channel)
    peak_time = torch.zeros((n_ch, k_count), dtype=torch.float32)
    peak_amp = torch.zeros((n_ch, k_count), dtype=torch.float32)
    peak_valid = torch.zeros((n_ch, k_count), dtype=torch.bool)
    peak_index = torch.full((n_ch, k_count), -1, dtype=torch.long)
    distance = int(max(1, round(float(config.min_distance_s) * float(fs) / float(max(1, time_downsample)))))
    for ch in range(n_ch):
        row = np.abs(arr[ch]).astype(np.float32, copy=False)
        peaks, props = find_peaks(
            row,
            height=float(config.min_height),
            prominence=float(config.prominence),
            distance=distance,
        )
        if peaks.size == 0:
            peaks, props = find_peaks(row, distance=distance)
        if peaks.size == 0:
            continue
        amps = row[peaks].astype(np.float32, copy=False)
        prominences = props.get("prominences", amps).astype(np.float32, copy=False)
        score = amps + 0.1 * prominences
        if peaks.size > k_count:
            keep = np.argsort(score)[-k_count:]
            peaks = peaks[keep]
            amps = amps[keep]
        order = np.argsort(peaks)
        peaks = peaks[order]
        amps = amps[order]
        take = min(k_count, int(peaks.size))
        idx = torch.as_tensor(peaks[:take], dtype=torch.long)
        peak_index[ch, :take] = idx
        peak_time[ch, :take] = (
            idx.to(torch.float32) * float(max(1, time_downsample)) / float(max(1, window_samples - 1))
        ).clamp(0.0, 1.0)
        peak_amp[ch, :take] = torch.as_tensor(amps[:take], dtype=torch.float32)
        peak_valid[ch, :take] = True
    return peak_time, peak_amp, peak_valid, peak_index


class PeakSlotPredictor(nn.Module):
    """Predict fixed vehicle slots that select per-channel peak candidates."""

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
        self.channel_embed = nn.Embedding(int(c.n_channels), hidden)
        self.peak_encoder = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.slot_peak_proj = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        self.none_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(
        self,
        x: torch.Tensor,
        peak_time: torch.Tensor,
        peak_amp: torch.Tensor,
        peak_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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

        bsz, n_ch, k_count = int(peak_time.shape[0]), int(peak_time.shape[1]), int(peak_time.shape[2])
        ch_pos = torch.linspace(0.0, 1.0, n_ch, device=peak_time.device, dtype=peak_time.dtype)
        ch_pos = ch_pos.view(1, n_ch, 1).expand(bsz, n_ch, k_count)
        valid_f = peak_valid.to(dtype=peak_time.dtype)
        peak_feat = torch.stack([peak_time, peak_amp, valid_f, ch_pos], dim=-1)
        peak_tokens = self.peak_encoder(peak_feat)
        ch_ids = torch.arange(n_ch, device=peak_time.device, dtype=torch.long)
        peak_tokens = peak_tokens + self.channel_embed(ch_ids).view(1, n_ch, 1, -1)

        slot_peak = self.slot_peak_proj(hs)
        peak_scores = torch.einsum("bqh,bckh->bqck", slot_peak, peak_tokens) / float(max(1, slot_peak.shape[-1])) ** 0.5
        peak_scores = peak_scores.masked_fill(~peak_valid[:, None, :, :], -1e4)
        none_context = hs[:, :, None, :] + self.channel_embed(ch_ids).view(1, 1, n_ch, -1)
        none_scores = self.none_head(none_context).squeeze(-1).unsqueeze(-1)
        peak_logits = torch.cat([peak_scores, none_scores], dim=-1)
        return {
            "num_regular_queries": int(self.config.max_tracks),
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "peak_logits": peak_logits,
        }


def _expected_peak_time(logits: torch.Tensor, peak_time: torch.Tensor, peak_valid: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits[..., :-1], dim=-1) * peak_valid[:, None, :, :].to(logits.dtype)
    denom = torch.clamp(probs.sum(dim=-1), min=1e-6)
    return (probs * peak_time[:, None, :, :].to(logits.dtype)).sum(dim=-1) / denom


def _greedy_match_cost(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q_count, g_count = int(cost.shape[0]), int(cost.shape[1])
    k = min(q_count, g_count)
    if k <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    work = cost.clone()
    large = torch.finfo(work.dtype).max
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for _ in range(k):
        idx = torch.argmin(work)
        row = torch.div(idx, g_count, rounding_mode="floor").long()
        col = (idx - row * g_count).long()
        rows.append(row)
        cols.append(col)
        work[row, :] = large
        work[:, col] = large
    return torch.stack(rows), torch.stack(cols)


def _match_single(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    b: int,
    *,
    matcher: str,
    w_peak: float,
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
    logp = torch.log_softmax(outputs["peak_logits"][b, :q_count].detach(), dim=-1)
    gt_peak = targets["gt_peak_index"][b, gt_valid].to(device=device, dtype=torch.long)
    gt_vis = targets["visibility"][b, gt_valid].to(device=device, dtype=torch.float32)
    gather_idx = gt_peak.clamp(0, logp.shape[-1] - 1)
    logp_expanded = logp[:, None, :, :].expand(q_count, g_count, -1, -1)
    nll = -logp_expanded.gather(-1, gather_idx[None, :, :, None].expand(q_count, g_count, -1, 1)).squeeze(-1)
    peak_cost = (nll * gt_vis[None, :, :]).sum(dim=-1) / torch.clamp(gt_vis.sum(dim=-1)[None, :], min=1.0)
    pred_obj = torch.sigmoid(outputs["objectness_logits"][b, :q_count].detach())
    pred_dir = torch.softmax(outputs["direction_logits"][b, :q_count].detach(), dim=-1)
    pred_speed = outputs["speed"][b, :q_count].detach()
    gt_dir = targets["direction"][b, gt_valid].to(device=device, dtype=torch.long)
    gt_speed = targets["speed"][b, gt_valid].to(device=device, dtype=torch.float32)
    cost = (
        float(w_peak) * peak_cost
        - float(w_obj) * pred_obj[:, None]
        - float(w_dir) * pred_dir[:, gt_dir]
        + float(w_speed) * torch.abs(pred_speed[:, None] - gt_speed[None, :])
    )
    if str(matcher).lower() == "greedy":
        return _greedy_match_cost(cost)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(rows, dtype=torch.long, device=device), torch.as_tensor(cols, dtype=torch.long, device=device)


def peak_slot_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    no_object_weight: float = 0.15,
    none_weight: float = 0.35,
    matcher: str = "hungarian",
    collect_metrics: bool = True,
    w_peak: float = 3.0,
    w_obj: float = 1.0,
    w_dir: float = 0.2,
    w_speed: float = 0.1,
    peak_loss_weight: float = 1.0,
    count_loss_weight: float = 0.05,
    direction_loss_weight: float = 0.5,
    speed_loss_weight: float = 0.25,
    monotonic_loss_weight: float = 0.2,
    smoothness_loss_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    q_count = int(outputs["num_regular_queries"])
    none_index = int(outputs["peak_logits"].shape[-1] - 1)
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
            matcher=matcher,
            w_peak=w_peak,
            w_obj=w_obj,
            w_dir=w_dir,
            w_speed=w_speed,
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
        pred_peak_logits = outputs["peak_logits"][b_sel, q_sel]
        gt_peak = targets["gt_peak_index"][b_sel, mapped_g].to(device=device, dtype=torch.long).clamp(0, none_index)
        ce = F.cross_entropy(pred_peak_logits.reshape(-1, none_index + 1), gt_peak.reshape(-1), reduction="none").view_as(gt_peak)
        ce_weight = torch.where(
            gt_peak == none_index,
            torch.full_like(ce, float(none_weight)),
            torch.ones_like(ce),
        )
        loss_peak = (ce * ce_weight).sum() / torch.clamp(ce_weight.sum(), min=1.0)
        pred_dir = outputs["direction_logits"][b_sel, q_sel]
        pred_speed = outputs["speed"][b_sel, q_sel]
        gt_dir = targets["direction"][b_sel, mapped_g].to(device=device, dtype=torch.long)
        gt_speed = targets["speed"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")

        exp_time = _expected_peak_time(outputs["peak_logits"], targets["peak_time"].to(device), targets["peak_valid"].to(device))
        pred_time = exp_time[b_sel, q_sel]
        gt_vis = targets["visibility"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        pair_vis = gt_vis[:, 1:] * gt_vis[:, :-1]
        dt = pred_time[:, 1:] - pred_time[:, :-1]
        sign = torch.where(gt_dir[:, None] == 0, torch.ones_like(dt), -torch.ones_like(dt))
        loss_mono = (torch.relu(-(dt * sign)) * pair_vis).sum() / torch.clamp(pair_vis.sum(), min=1.0)
        if pred_time.shape[1] >= 3:
            tri_vis = gt_vis[:, 2:] * gt_vis[:, 1:-1] * gt_vis[:, :-2]
            d2 = pred_time[:, 2:] - 2.0 * pred_time[:, 1:-1] + pred_time[:, :-2]
            smooth_raw = F.smooth_l1_loss(d2, torch.zeros_like(d2), reduction="none")
            loss_smooth = (smooth_raw * tri_vis).sum() / torch.clamp(tri_vis.sum(), min=1.0)
        else:
            loss_smooth = zero
    else:
        loss_peak = zero
        loss_dir = zero
        loss_speed = zero
        loss_mono = zero
        loss_smooth = zero

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    gt_count = targets["gt_valid"].to(device=device, dtype=torch.float32).sum(dim=1)
    loss_count = F.smooth_l1_loss(obj_prob.sum(dim=1), gt_count, reduction="mean")
    total = (
        loss_obj
        + float(peak_loss_weight) * loss_peak
        + float(count_loss_weight) * loss_count
        + float(direction_loss_weight) * loss_dir
        + float(speed_loss_weight) * loss_speed
        + float(monotonic_loss_weight) * loss_mono
        + float(smoothness_loss_weight) * loss_smooth
    )
    if not collect_metrics:
        return total, {}
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_peak": float(loss_peak.detach().cpu()),
        "loss_count": float(loss_count.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_monotonic": float(loss_mono.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(torch.max(obj_prob).detach().cpu()),
        "mean_objectness": float(torch.mean(obj_prob).detach().cpu()),
    }
    return total, metrics


def peak_slot_detection_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    objectness_threshold: float = 0.5,
    point_threshold: float = 0.05,
    matcher: str = "hungarian",
) -> dict[str, float]:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    none_index = int(outputs["peak_logits"].shape[-1] - 1)
    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    active = obj >= float(objectness_threshold)
    selected = torch.argmax(outputs["peak_logits"][:, :q_count], dim=-1)
    pred_total = int(active.sum().detach().cpu())
    gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    gt_total = int(gt_valid.sum().detach().cpu())
    good_total = 0
    count_abs_error = 0.0
    count_exact = 0
    time_errors: list[float] = []
    peak_time = targets["peak_time"].to(device=device, dtype=torch.float32)
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
            matcher=matcher,
            w_peak=3.0,
            w_obj=1.0,
            w_dir=0.2,
            w_speed=0.1,
        )
        valid_idx = torch.where(gt_valid[b])[0]
        for r, c in zip(rows.tolist(), cols.tolist()):
            if not bool(active[b, r]):
                continue
            gi = int(valid_idx[c].item())
            vis = targets["visibility"][b, gi].to(device=device, dtype=torch.float32) > 0.5
            if not torch.any(vis):
                continue
            pred_idx = selected[b, r].clamp(0, none_index)
            valid_pred = vis & (pred_idx < none_index)
            if not torch.any(valid_pred):
                continue
            pred_t = peak_time[b].gather(1, pred_idx[:, None].clamp(0, none_index - 1)).squeeze(1)
            gt_idx = targets["gt_peak_index"][b, gi].to(device=device, dtype=torch.long).clamp(0, none_index - 1)
            gt_t = peak_time[b].gather(1, gt_idx[:, None]).squeeze(1)
            err = torch.mean(torch.abs(pred_t[valid_pred] - gt_t[valid_pred])).detach()
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


def _direction_sign(direction_label: int) -> float:
    return 1.0 if LABEL_TO_DIRECTION.get(int(direction_label), "forward") == "forward" else -1.0


def _slot_peak_candidates(
    peak_prob: np.ndarray,
    peak_valid: torch.Tensor,
    peak_index: torch.Tensor,
    *,
    peak_threshold: float,
    topk: int,
) -> list[list[dict[str, float | int]]]:
    candidates: list[list[dict[str, float | int]]] = []
    n_ch = int(peak_prob.shape[0])
    none_idx = int(peak_prob.shape[1] - 1)
    for ch in range(n_ch):
        ch_items: list[dict[str, float | int]] = []
        for k in range(none_idx):
            if not bool(peak_valid[ch, k]):
                continue
            prob = float(peak_prob[ch, k])
            if prob < float(peak_threshold):
                continue
            ch_items.append(
                {
                    "ch": int(ch),
                    "peak_idx": int(k),
                    "prob": prob,
                    "logp": float(np.log(max(prob, 1e-12))),
                    "t_down": int(peak_index[ch, k].item()),
                }
            )
        ch_items.sort(key=lambda item: float(item["logp"]), reverse=True)
        candidates.append(ch_items[: max(1, int(topk))])
    return candidates


def _transition_score(
    prev: dict[str, float | int | None],
    cur: dict[str, float | int],
    *,
    x_axis_m: np.ndarray,
    time_downsample: int,
    fs: float,
    direction_label: int,
    speed_min_kmh: float,
    speed_max_kmh: float,
    reference_speed_kmh: float,
    skip_penalty: float,
    speed_penalty: float,
    smoothness_penalty: float,
    inertia_penalty: float,
    slope_memory: float,
) -> tuple[bool, float, float]:
    prev_ch = int(prev["ch"])  # type: ignore[arg-type]
    cur_ch = int(cur["ch"])
    ch_gap = cur_ch - prev_ch
    if ch_gap <= 0:
        return False, 0.0, 0.0
    prev_t = int(prev["t_down"]) * int(time_downsample) / float(fs)  # type: ignore[arg-type]
    cur_t = int(cur["t_down"]) * int(time_downsample) / float(fs)
    dt_signed = cur_t - prev_t
    if _direction_sign(int(direction_label)) * dt_signed < 0.0:
        return False, 0.0, 0.0
    if cur_ch < len(x_axis_m) and prev_ch < len(x_axis_m):
        dx = abs(float(x_axis_m[cur_ch]) - float(x_axis_m[prev_ch]))
    else:
        dx = float(ch_gap) * 100.0
    if dx <= 1e-9:
        return False, 0.0, 0.0
    dt_abs = abs(dt_signed)
    if dt_abs <= 1e-9:
        return False, 0.0, 0.0
    speed_kmh = 3.6 * dx / dt_abs
    if speed_kmh < float(speed_min_kmh) or speed_kmh > float(speed_max_kmh):
        return False, 0.0, 0.0
    ref_speed = float(reference_speed_kmh)
    if not np.isfinite(ref_speed) or ref_speed <= 1e-6:
        ref_speed = 0.5 * (float(speed_min_kmh) + float(speed_max_kmh))
    trans = -float(skip_penalty) * float(max(0, ch_gap - 1))
    trans -= float(speed_penalty) * abs(speed_kmh - ref_speed) / max(1e-6, float(speed_max_kmh) - float(speed_min_kmh))
    slope = dt_signed / dx
    prev_slope = prev.get("slope")
    if prev_slope is not None and np.isfinite(float(prev_slope)):
        ref_slope = 3.6 / max(1e-6, ref_speed)
        prev_slope_f = float(prev_slope)
        trans -= float(smoothness_penalty) * abs(slope - prev_slope_f) / max(1e-6, ref_slope)
        expected_dt = prev_slope_f * dx
        inertia_scale = max(1e-6, abs(expected_dt), ref_slope * dx)
        trans -= float(inertia_penalty) * abs(dt_signed - expected_dt) / inertia_scale
        memory = min(0.98, max(0.0, float(slope_memory)))
        slope = memory * prev_slope_f + (1.0 - memory) * slope
    return True, float(trans), float(slope)


def decode_peak_slot_path(
    peak_prob: np.ndarray,
    peak_time: torch.Tensor,
    peak_valid: torch.Tensor,
    peak_index: torch.Tensor,
    *,
    direction_label: int,
    predicted_speed_kmh: float,
    x_axis_m: np.ndarray,
    time_downsample: int,
    fs: float,
    config: InferenceConfig,
) -> list[dict[str, float | int]]:
    """Decode one slot as a physically consistent peak path.

    The model still supplies per-channel peak probabilities. This function is a
    non-differentiable inference-time Viterbi pass that replaces independent
    per-channel argmax with a speed-window-constrained path search.
    """
    del peak_time
    candidates = _slot_peak_candidates(
        peak_prob,
        peak_valid,
        peak_index,
        peak_threshold=float(config.viterbi_candidate_threshold),
        topk=int(config.viterbi_topk),
    )
    states: list[list[dict[str, float | int | None]]] = []
    ref_speed = float(predicted_speed_kmh)
    if not np.isfinite(ref_speed) or ref_speed <= 1e-6:
        ref_speed = float(config.viterbi_fallback_speed_kmh)
    for ch, ch_candidates in enumerate(candidates):
        ch_states: list[dict[str, float | int | None]] = []
        for item in ch_candidates:
            best_score = float(item["logp"]) + float(config.viterbi_point_bonus)
            best_prev_ch: Optional[int] = None
            best_prev_idx: Optional[int] = None
            best_slope: Optional[float] = None
            start_ch = max(0, ch - int(config.viterbi_max_skip_channels))
            for prev_ch in range(start_ch, ch):
                for prev_idx, prev in enumerate(states[prev_ch]):
                    ok, trans, slope = _transition_score(
                        prev,
                        item,
                        x_axis_m=x_axis_m,
                        time_downsample=int(time_downsample),
                        fs=float(fs),
                        direction_label=int(direction_label),
                        speed_min_kmh=float(config.viterbi_speed_min_kmh),
                        speed_max_kmh=float(config.viterbi_speed_max_kmh),
                        reference_speed_kmh=ref_speed,
                        skip_penalty=float(config.viterbi_skip_penalty),
                        speed_penalty=float(config.viterbi_speed_penalty),
                        smoothness_penalty=float(config.viterbi_smoothness_penalty),
                        inertia_penalty=float(config.viterbi_inertia_penalty),
                        slope_memory=float(config.viterbi_slope_memory),
                    )
                    if not ok:
                        continue
                    cand_score = float(prev["score"]) + float(item["logp"]) + float(config.viterbi_point_bonus) + trans
                    if cand_score > best_score:
                        best_score = cand_score
                        best_prev_ch = int(prev_ch)
                        best_prev_idx = int(prev_idx)
                        best_slope = float(slope)
            state = dict(item)
            state.update({"score": best_score, "prev_ch": best_prev_ch, "prev_idx": best_prev_idx, "slope": best_slope})
            ch_states.append(state)
        states.append(ch_states)
    best: Optional[tuple[int, int, dict[str, float | int | None]]] = None
    for ch, ch_states in enumerate(states):
        for idx, state in enumerate(ch_states):
            if best is None or float(state["score"]) > float(best[2]["score"]):
                best = (ch, idx, state)
    if best is None:
        return []
    path_rev: list[dict[str, float | int | None]] = []
    ch, idx, state = best
    while True:
        path_rev.append(state)
        prev_ch = state.get("prev_ch")
        prev_idx = state.get("prev_idx")
        if prev_ch is None or prev_idx is None:
            break
        ch = int(prev_ch)
        idx = int(prev_idx)
        state = states[ch][idx]
    path = list(reversed(path_rev))
    return [
        {
            "ch": int(item["ch"]),  # type: ignore[arg-type]
            "peak_idx": int(item["peak_idx"]),  # type: ignore[arg-type]
            "prob": float(item["prob"]),  # type: ignore[arg-type]
        }
        for item in path
    ]


def argmax_peak_slot_path(
    peak_prob: np.ndarray,
    peak_valid: torch.Tensor,
    *,
    peak_threshold: float,
) -> list[dict[str, float | int]]:
    none_idx = int(peak_prob.shape[-1] - 1)
    path: list[dict[str, float | int]] = []
    for ch in range(int(peak_prob.shape[0])):
        choice = int(np.argmax(peak_prob[ch]))
        if choice >= none_idx:
            continue
        if not bool(peak_valid[ch, choice]):
            continue
        prob = float(peak_prob[ch, choice])
        if prob < float(peak_threshold):
            continue
        path.append({"ch": int(ch), "peak_idx": int(choice), "prob": prob})
    return path


def peak_slot_physics_metrics(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    objectness_threshold: float = 0.5,
    peak_threshold: float = 0.4,
    speed_min_kmh: float = 60.0,
    speed_max_kmh: float = 100.0,
    time_downsample: int = 10,
    fs: float = 1000.0,
    dx_m: float = 100.0,
    smooth_tolerance_s: float = 2.0,
) -> dict[str, float]:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    peak_prob = torch.softmax(outputs["peak_logits"][:, :q_count], dim=-1)
    selected = torch.argmax(peak_prob, dim=-1)
    selected_prob = torch.max(peak_prob, dim=-1).values
    dirs = torch.argmax(outputs["direction_logits"][:, :q_count], dim=-1)
    peak_index = targets["peak_index"].to(device=device, dtype=torch.long)
    peak_valid = targets["peak_valid"].to(device=device, dtype=torch.bool)
    pair_total = 0
    direction_bad = 0
    speed_bad = 0
    triple_total = 0
    smooth_bad = 0
    active_tracks = 0
    for b in range(int(obj.shape[0])):
        active_slots = torch.where(obj[b] >= float(objectness_threshold))[0].tolist()
        for q in active_slots:
            points: list[tuple[int, float]] = []
            for ch in range(int(peak_prob.shape[2])):
                k = int(selected[b, q, ch].item())
                if k >= int(peak_prob.shape[-1] - 1):
                    continue
                if not bool(peak_valid[b, ch, k]):
                    continue
                if float(selected_prob[b, q, ch].item()) < float(peak_threshold):
                    continue
                t = int(peak_index[b, ch, k].item()) * int(time_downsample) / float(fs)
                points.append((int(ch), float(t)))
            if len(points) < 2:
                continue
            active_tracks += 1
            sign = _direction_sign(int(dirs[b, q].item()))
            for (ch0, t0), (ch1, t1) in zip(points[:-1], points[1:]):
                pair_total += 1
                dt = t1 - t0
                if sign * dt < 0.0:
                    direction_bad += 1
                dx = abs(float(ch1 - ch0)) * float(dx_m)
                speed = 3.6 * dx / max(1e-9, abs(dt))
                if speed < float(speed_min_kmh) or speed > float(speed_max_kmh):
                    speed_bad += 1
            for (_, t0), (_, t1), (_, t2) in zip(points[:-2], points[1:-1], points[2:]):
                triple_total += 1
                if abs(float(t2 - 2.0 * t1 + t0)) > float(smooth_tolerance_s):
                    smooth_bad += 1
    return {
        "physics_tracks": float(active_tracks),
        "physics_pair_count": float(pair_total),
        "direction_violation_rate": float(direction_bad / max(1, pair_total)),
        "speed_window_violation_rate": float(speed_bad / max(1, pair_total)),
        "smoothness_violation_rate": float(smooth_bad / max(1, triple_total)),
    }


def _local_speed_series(points: list[TrackPoint]) -> list[float]:
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
    return Track(
        track_id=int(track_id),
        direction=direction,
        points=points_sorted,
        total_score=float(np.sum([p.score for p in points_sorted])),
        mean_speed_kmh=float(np.mean(speeds)) if speeds else float("nan"),
    )


def _deduplicate_tracks(tracks: list[Track], tol_samples: int, min_overlap: int) -> list[Track]:
    kept: list[Track] = []
    for track in sorted(tracks, key=lambda item: item.total_score, reverse=True):
        tmap = {int(p.ch_idx): int(p.t_idx) for p in track.points}
        duplicate = False
        for existing in kept:
            emap = {int(p.ch_idx): int(p.t_idx) for p in existing.points}
            common = sorted(set(tmap) & set(emap))
            if len(common) < int(min_overlap):
                continue
            diffs = np.array([abs(tmap[ch] - emap[ch]) for ch in common], dtype=np.float64)
            if float(np.median(diffs)) <= float(tol_samples):
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return [_track_stats(i, track.direction, track.points) for i, track in enumerate(kept)]


def predict_tracks_from_window(
    model: PeakSlotPredictor,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[InferenceConfig] = None,
    device: Optional[str] = None,
) -> list[Track]:
    cfg = config or InferenceConfig()
    arr = np.asarray(data_window, dtype=np.float32)
    x = prepare_window_input(
        arr,
        int(cfg.time_downsample),
        clip_ratio=float(cfg.clip_ratio),
        input_mode="raw" if int(model.config.in_channels) == 1 else "raw_abs",
    )
    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(model.config.peak_candidates),
        min_distance_s=float(cfg.peak_min_distance_s),
        min_height=float(cfg.peak_min_height),
        prominence=float(cfg.peak_prominence),
    )
    peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates_from_tensor(
        x[0],
        fs=float(fs),
        time_downsample=int(cfg.time_downsample),
        window_samples=int(arr.shape[1]),
        config=peak_cfg,
    )
    resolved_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(
            x.unsqueeze(0).to(resolved_device),
            peak_time.unsqueeze(0).to(resolved_device),
            peak_amp.unsqueeze(0).to(resolved_device),
            peak_valid.unsqueeze(0).to(resolved_device),
        )
    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    peak_prob = torch.softmax(outputs["peak_logits"][0], dim=-1).detach().cpu().numpy()
    dirs = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()
    speeds = outputs["speed"][0].detach().cpu().numpy()
    max_tracks = min(int(cfg.max_tracks), int(obj.shape[0]))
    order = np.argsort(obj)[::-1][:max_tracks]
    tracks: list[Track] = []
    for q_idx in order.tolist():
        score = float(obj[q_idx])
        if score < float(cfg.objectness_threshold):
            continue
        if bool(cfg.use_viterbi_decoder):
            decoded = decode_peak_slot_path(
                peak_prob[q_idx],
                peak_time,
                peak_valid,
                peak_index,
                direction_label=int(dirs[q_idx]),
                predicted_speed_kmh=float(speeds[q_idx]) * float(cfg.speed_norm_kmh),
                x_axis_m=np.asarray(x_axis_m, dtype=np.float64),
                time_downsample=int(cfg.time_downsample),
                fs=float(fs),
                config=cfg,
            )
        else:
            decoded = argmax_peak_slot_path(peak_prob[q_idx], peak_valid, peak_threshold=float(cfg.peak_threshold))
        points: list[TrackPoint] = []
        for item in decoded:
            ch = int(item["ch"])
            choice = int(item["peak_idx"])
            prob = float(item["prob"])
            t_idx = int(peak_index[ch, choice].item()) * int(cfg.time_downsample)
            t_idx = int(max(0, min(int(arr.shape[1]) - 1, t_idx)))
            offset = float(x_axis_m[int(ch)]) if int(ch) < len(x_axis_m) else float(ch)
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=t_idx,
                    time_s=float(t_idx) / float(fs),
                    offset_m=offset,
                    amp=float(abs(arr[int(ch), t_idx])),
                    score=score * prob,
                )
            )
        if len(points) >= int(cfg.min_visible_channels):
            tracks.append(_track_stats(len(tracks), LABEL_TO_DIRECTION.get(int(dirs[q_idx]), "forward"), points))
    return _deduplicate_tracks(
        tracks,
        tol_samples=int(cfg.dedup_tolerance_samples),
        min_overlap=int(cfg.dedup_min_overlap_channels),
    )


def save_checkpoint(
    path: str | Path,
    model: PeakSlotPredictor,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    dataset_config: WindowDatasetConfig,
    epoch: int,
    metrics: dict[str, float],
    dataset_meta: Optional[dict[str, Any]] = None,
) -> None:
    payload: dict[str, Any] = {
        "model_family": "peak_slot",
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "dataset_config": asdict(dataset_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if dataset_meta is not None:
        payload["dataset_meta"] = dict(dataset_meta)
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(checkpoint_path)


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[PeakSlotPredictor, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = PeakSlotPredictor(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
