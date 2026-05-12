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
    objectness_threshold: float = 0.35
    peak_threshold: float = 0.4
    min_visible_channels: int = 2
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
    decoder_mode: str = "beam_global"
    viterbi_beam_size: int = 8
    time_prior_weight: float = 2.0
    global_conflict_penalty: float = 2.0
    extra_candidate_slots: int = 16
    candidate_objectness_floor: float = 0.05
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
        self.time_prior_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))
        self.visibility_prior_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))

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
            "time_prior": torch.sigmoid(self.time_prior_head(hs)),
            "visibility_prior_logits": self.visibility_prior_head(hs),
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


def _auction_match_cost(cost: torch.Tensor, *, eps: float = 1e-3, max_iter: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate one-to-one assignment with a parallel torch auction loop.

    This is intended as a GPU-friendly alternative to SciPy Hungarian during
    training. It is not guaranteed to return the exact global optimum, but it
    avoids converting the cost matrix to CPU NumPy and avoids per-bidder
    `.item()` synchronization.
    """
    q_count, g_count = int(cost.shape[0]), int(cost.shape[1])
    if q_count <= 0 or g_count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=cost.device)
        return empty, empty
    if q_count >= g_count:
        values = -cost.transpose(0, 1).detach()  # bidders: GT, items: slots
        bidder_count, item_count = int(values.shape[0]), int(values.shape[1])
        assigned_item = _parallel_auction_assign(values, eps=eps, max_iter=max_iter)
        cols = torch.where(assigned_item >= 0)[0]
        rows = assigned_item[cols]
        return rows.to(torch.long), cols.to(torch.long)

    values = -cost.detach()  # bidders: slots, items: GT
    assigned_item = _parallel_auction_assign(values, eps=eps, max_iter=max_iter)
    rows = torch.where(assigned_item >= 0)[0]
    cols = assigned_item[rows]
    return rows.to(torch.long), cols.to(torch.long)


def _parallel_auction_assign(values: torch.Tensor, *, eps: float = 1e-3, max_iter: Optional[int] = None) -> torch.Tensor:
    bidder_count, item_count = int(values.shape[0]), int(values.shape[1])
    device = values.device
    if bidder_count <= 0 or item_count <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    if item_count == 1:
        return torch.zeros((bidder_count,), dtype=torch.long, device=device)
    prices = torch.zeros((item_count,), dtype=values.dtype, device=device)
    owner = torch.full((item_count,), -1, dtype=torch.long, device=device)
    assigned_item = torch.full((bidder_count,), -1, dtype=torch.long, device=device)
    bidder_ids = torch.arange(bidder_count, dtype=torch.long, device=device)
    min_score = torch.finfo(values.dtype).min
    limit = int(max_iter or max(8, min(64, bidder_count * 3)))
    eps_value = torch.as_tensor(float(eps), dtype=values.dtype, device=device)
    for _ in range(limit):
        unassigned = assigned_item < 0
        scores = values - prices[None, :]
        scores = scores.masked_fill(~unassigned[:, None], min_score)
        top = torch.topk(scores, k=2, dim=1)
        best_item = top.indices[:, 0]
        bid_increment = top.values[:, 0] - top.values[:, 1] + eps_value
        bid_increment = torch.where(unassigned, bid_increment, torch.full_like(bid_increment, min_score))

        best_bid = torch.full((item_count,), min_score, dtype=values.dtype, device=device)
        best_bid.scatter_reduce_(0, best_item, bid_increment, reduce="amax", include_self=True)
        has_bid = best_bid > (min_score * 0.5)
        is_top_bid = unassigned & (bid_increment == best_bid.gather(0, best_item))

        winner_by_item = torch.full((item_count,), -1, dtype=torch.long, device=device)
        winner_candidate = torch.where(is_top_bid, bidder_ids, torch.full_like(bidder_ids, -1))
        winner_by_item.scatter_reduce_(0, best_item, winner_candidate, reduce="amax", include_self=True)
        won_items = torch.where(has_bid)[0]
        if won_items.numel() == 0:
            continue
        winners = winner_by_item[won_items]
        valid_winners = winners >= 0
        won_items = won_items[valid_winners]
        winners = winners[valid_winners]
        if winners.numel() == 0:
            continue
        previous = owner[won_items]
        prev_valid = previous >= 0
        assigned_item[previous[prev_valid]] = -1
        owner[won_items] = winners
        assigned_item[winners] = won_items
        prices[won_items] = prices[won_items] + best_bid[won_items]
    return assigned_item


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
    matcher_name = str(matcher).lower()
    if matcher_name == "independent":
        cols = torch.arange(g_count, dtype=torch.long, device=device)
        rows = torch.argmin(cost, dim=0).to(torch.long)
        return rows, cols
    if matcher_name == "greedy":
        return _greedy_match_cost(cost)
    if matcher_name == "auction":
        return _auction_match_cost(cost)
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return torch.as_tensor(rows, dtype=torch.long, device=device), torch.as_tensor(cols, dtype=torch.long, device=device)


def _independent_match_batch(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    w_peak: float,
    w_obj: float,
    w_dir: float,
    w_speed: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast approximate matcher: each GT independently picks its best slot.

    This keeps the assignment fully on the torch device and avoids the
    per-sample synchronization used by the exact/auction matchers. It does not
    enforce one-to-one slot ownership, so it is intended for high-throughput
    training rather than exact validation metrics.
    """
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    if int(gt_valid.numel()) <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty, empty

    logp = torch.log_softmax(outputs["peak_logits"][:, :q_count].detach(), dim=-1)
    batch_size, _, n_ch, k_plus_one = logp.shape
    max_gt = int(gt_valid.shape[1])
    gt_peak = targets["gt_peak_index"].to(device=device, dtype=torch.long).clamp(0, k_plus_one - 1)
    gt_vis = targets["visibility"].to(device=device, dtype=torch.float32)
    gather_idx = gt_peak[:, None, :, :, None].expand(batch_size, q_count, max_gt, n_ch, 1)
    nll = -logp[:, :, None, :, :].expand(-1, -1, max_gt, -1, -1).gather(-1, gather_idx).squeeze(-1)
    peak_cost = (nll * gt_vis[:, None, :, :]).sum(dim=-1) / torch.clamp(gt_vis.sum(dim=-1)[:, None, :], min=1.0)

    pred_obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count].detach())
    pred_dir = torch.softmax(outputs["direction_logits"][:, :q_count].detach(), dim=-1)
    pred_speed = outputs["speed"][:, :q_count].detach()
    gt_dir = targets["direction"].to(device=device, dtype=torch.long).clamp(0, pred_dir.shape[-1] - 1)
    gt_speed = targets["speed"].to(device=device, dtype=torch.float32)
    dir_score = pred_dir[:, :, None, :].expand(-1, -1, max_gt, -1).gather(
        -1,
        gt_dir[:, None, :, None].expand(batch_size, q_count, max_gt, 1),
    ).squeeze(-1)
    cost = (
        float(w_peak) * peak_cost
        - float(w_obj) * pred_obj[:, :, None]
        - float(w_dir) * dir_score
        + float(w_speed) * torch.abs(pred_speed[:, :, None] - gt_speed[:, None, :])
    )
    large = torch.finfo(cost.dtype).max
    cost = cost.masked_fill(~gt_valid[:, None, :], large)
    q_sel_all = torch.argmin(cost, dim=1)
    b_sel, g_sel = torch.where(gt_valid)
    q_sel = q_sel_all[b_sel, g_sel]
    return b_sel.to(torch.long), q_sel.to(torch.long), g_sel.to(torch.long)


def _slot_peak_competition_loss(outputs: dict[str, torch.Tensor], q_count: int) -> torch.Tensor:
    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    peak_prob = torch.softmax(outputs["peak_logits"][:, :q_count, :, :-1], dim=-1)
    weighted = peak_prob * obj[:, :, None, None]
    q = int(weighted.shape[1])
    if q <= 1:
        return torch.zeros((), dtype=weighted.dtype, device=weighted.device)
    sum_by_peak = weighted.sum(dim=1)
    offdiag_sum = (sum_by_peak.square() - weighted.square().sum(dim=1)).sum(dim=(1, 2))
    return offdiag_sum.mean() / float(q * (q - 1))


def _valid_rank_to_gt_index(gt_valid: torch.Tensor) -> torch.Tensor:
    batch_size, max_gt = int(gt_valid.shape[0]), int(gt_valid.shape[1])
    rank_map = torch.full((batch_size, max_gt), -1, dtype=torch.long, device=gt_valid.device)
    valid_rank = torch.cumsum(gt_valid.to(torch.long), dim=1) - 1
    batch_idx, gt_idx = torch.where(gt_valid)
    if batch_idx.numel() > 0:
        rank_map[batch_idx, valid_rank[batch_idx, gt_idx]] = gt_idx
    return rank_map


def _peak_switch_margin_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    b_sel: torch.Tensor,
    q_sel: torch.Tensor,
    mapped_g: torch.Tensor,
) -> torch.Tensor:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    none_index = int(outputs["peak_logits"].shape[-1] - 1)
    logp = torch.log_softmax(outputs["peak_logits"][:, :q_count], dim=-1)
    if b_sel.numel() == 0:
        return torch.zeros((), dtype=logp.dtype, device=device)
    matched_logp = logp[b_sel, q_sel]
    gt_peak = targets["gt_peak_index"][b_sel, mapped_g].to(device=device, dtype=torch.long)
    gt_vis = targets["visibility"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
    valid = (gt_vis > 0.5) & (gt_peak < none_index)
    if not torch.any(valid):
        return torch.zeros((), dtype=logp.dtype, device=device)
    good = matched_logp.gather(2, gt_peak.clamp(0, none_index)[:, :, None]).squeeze(-1)
    bad_logits = matched_logp[:, :, :none_index].clone()
    bad_logits.scatter_(2, gt_peak.clamp(0, none_index - 1)[:, :, None], -1e4)
    bad = torch.max(bad_logits, dim=-1).values
    margin = 0.75
    return torch.relu(margin + bad[valid] - good[valid]).mean()


def _slot_gt_peak_nll(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], q_count: int) -> torch.Tensor:
    device = outputs["objectness_logits"].device
    none_index = int(outputs["peak_logits"].shape[-1] - 1)
    logp = torch.log_softmax(outputs["peak_logits"][:, :q_count], dim=-1)
    gt_peak = targets["gt_peak_index"].to(device=device, dtype=torch.long).clamp(0, none_index)
    gt_vis = targets["visibility"].to(device=device, dtype=torch.float32)
    nll = -logp[:, :, None, :, :].expand(-1, -1, int(gt_peak.shape[1]), -1, -1).gather(
        -1,
        gt_peak[:, None, :, :, None].expand(-1, q_count, -1, -1, 1),
    ).squeeze(-1)
    return (nll * gt_vis[:, None, :, :]).sum(dim=-1) / torch.clamp(gt_vis.sum(dim=-1)[:, None, :], min=1.0)


def _gt_coverage_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    q_count: int,
    *,
    temperature: float,
) -> torch.Tensor:
    nll = _slot_gt_peak_nll(outputs, targets, q_count)
    valid = targets["gt_valid"].to(device=nll.device, dtype=torch.bool)
    has_vis = targets["visibility"].to(device=nll.device, dtype=torch.float32).sum(dim=-1) > 0.5
    valid = valid & has_vis
    if not torch.any(valid):
        return torch.zeros((), dtype=nll.dtype, device=nll.device)
    temp = max(1e-6, float(temperature))
    weights = torch.softmax(-nll / temp, dim=1)
    coverage = (weights * nll).sum(dim=1)
    return coverage[valid].mean()


def _close_pair_mask(
    targets: dict[str, torch.Tensor],
    *,
    window_seconds: float,
    min_common_channels: int,
    min_gap_s: float,
    max_gap_s: float,
) -> torch.Tensor:
    device = targets["gt_valid"].device
    gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    visibility = targets["visibility"].to(device=device, dtype=torch.float32) > 0.5
    peak_time = targets["peak_time"].to(device=device, dtype=torch.float32)
    gt_peak = targets["gt_peak_index"].to(device=device, dtype=torch.long)
    none_index = int(peak_time.shape[-1])
    gt_peak_clamped = gt_peak.clamp(0, max(0, none_index - 1))
    gt_time = peak_time[:, None, :, :].expand(-1, int(gt_peak.shape[1]), -1, -1).gather(3, gt_peak_clamped[:, :, :, None]).squeeze(3)
    visible = visibility & (gt_peak < none_index)
    common = visible[:, :, None, :] & visible[:, None, :, :]
    common_count = common.sum(dim=-1)
    gap_s = torch.abs(gt_time[:, :, None, :] - gt_time[:, None, :, :]) * float(window_seconds)
    mean_gap = (gap_s * common.to(gap_s.dtype)).sum(dim=-1) / torch.clamp(common_count.to(gap_s.dtype), min=1.0)
    pair_valid = gt_valid[:, :, None] & gt_valid[:, None, :]
    pair_valid = pair_valid & (common_count >= int(min_common_channels))
    pair_valid = pair_valid & (mean_gap >= float(min_gap_s)) & (mean_gap <= float(max_gap_s))
    g_count = int(gt_valid.shape[1])
    upper = torch.triu(torch.ones((g_count, g_count), dtype=torch.bool, device=device), diagonal=1)
    return pair_valid & upper[None, :, :]


def _close_pair_separation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    q_count: int,
    *,
    b_sel: torch.Tensor,
    q_sel: torch.Tensor,
    mapped_g: torch.Tensor,
    margin: float,
    window_seconds: float,
    min_common_channels: int,
    min_gap_s: float,
    max_gap_s: float,
) -> torch.Tensor:
    if b_sel.numel() == 0:
        return torch.zeros((), dtype=outputs["objectness_logits"].dtype, device=outputs["objectness_logits"].device)
    nll = _slot_gt_peak_nll(outputs, targets, q_count)
    close = _close_pair_mask(
        targets,
        window_seconds=float(window_seconds),
        min_common_channels=int(min_common_channels),
        min_gap_s=float(min_gap_s),
        max_gap_s=float(max_gap_s),
    )
    if not torch.any(close):
        return torch.zeros((), dtype=nll.dtype, device=nll.device)
    batch_size, max_gt = int(targets["gt_valid"].shape[0]), int(targets["gt_valid"].shape[1])
    gt_to_slot = torch.full((batch_size, max_gt), -1, dtype=torch.long, device=nll.device)
    gt_to_slot[b_sel, mapped_g] = q_sel
    b_pair, g1, g2 = torch.where(close)
    q1 = gt_to_slot[b_pair, g1]
    q2 = gt_to_slot[b_pair, g2]
    valid = (q1 >= 0) & (q2 >= 0) & (q1 != q2)
    if not torch.any(valid):
        return torch.zeros((), dtype=nll.dtype, device=nll.device)
    b_pair = b_pair[valid]
    g1 = g1[valid]
    g2 = g2[valid]
    q1 = q1[valid]
    q2 = q2[valid]
    loss_1 = torch.relu(float(margin) + nll[b_pair, q1, g1] - nll[b_pair, q1, g2])
    loss_2 = torch.relu(float(margin) + nll[b_pair, q2, g2] - nll[b_pair, q2, g1])
    return 0.5 * (loss_1 + loss_2).mean()


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
    object_loss_weight: float = 1.0,
    count_loss_weight: float = 0.05,
    direction_loss_weight: float = 0.5,
    speed_loss_weight: float = 0.25,
    monotonic_loss_weight: float = 1.0,
    smoothness_loss_weight: float = 0.2,
    time_prior_loss_weight: float = 2.0,
    visibility_prior_loss_weight: float = 0.5,
    slot_competition_loss_weight: float = 0.1,
    crossing_loss_weight: float = 0.2,
    gt_coverage_loss_weight: float = 0.5,
    gt_coverage_temperature: float = 0.2,
    close_pair_separation_loss_weight: float = 0.3,
    close_pair_margin: float = 0.5,
    close_pair_min_common_channels: int = 8,
    close_pair_min_gap_s: float = 0.15,
    close_pair_max_gap_s: float = 1.5,
    close_pair_window_seconds: float = 120.0,
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
    mapped_g: Optional[torch.Tensor] = None
    matched_total = 0
    gt_total = 0

    if str(matcher).lower() == "independent":
        b_sel, q_sel, mapped_g = _independent_match_batch(
            outputs,
            targets,
            w_peak=w_peak,
            w_obj=w_obj,
            w_dir=w_dir,
            w_speed=w_speed,
        )
        if b_sel.numel() > 0:
            obj_target[b_sel, q_sel] = 1.0
            obj_weight[b_sel, q_sel] = 1.0
            matched_total = int(q_sel.numel())
            matched_b.append(b_sel)
            matched_q.append(q_sel)
            matched_g.append(mapped_g)
    else:
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
        if mapped_g is None:
            mapped_g = _valid_rank_to_gt_index(gt_valid)[b_sel, g_sel]
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
        pred_time_prior = outputs["time_prior"][b_sel, q_sel]
        pred_vis_prior_logits = outputs["visibility_prior_logits"][b_sel, q_sel]
        gt_dir = targets["direction"][b_sel, mapped_g].to(device=device, dtype=torch.long)
        gt_speed = targets["speed"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        gt_time = targets["peak_time"][b_sel].to(device=device, dtype=torch.float32)
        gt_peak_for_time = gt_peak.clamp(0, none_index - 1)
        gt_time_prior = gt_time.gather(2, gt_peak_for_time[:, :, None]).squeeze(-1)
        gt_vis_for_prior = targets["visibility"][b_sel, mapped_g].to(device=device, dtype=torch.float32)
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")
        loss_time_prior = (F.smooth_l1_loss(pred_time_prior, gt_time_prior, reduction="none") * gt_vis_for_prior).sum()
        loss_time_prior = loss_time_prior / torch.clamp(gt_vis_for_prior.sum(), min=1.0)
        vis_prior_raw = F.binary_cross_entropy_with_logits(pred_vis_prior_logits, gt_vis_for_prior, reduction="none")
        loss_vis_prior = vis_prior_raw.mean()

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
        loss_crossing = (
            _peak_switch_margin_loss(outputs, targets, b_sel=b_sel, q_sel=q_sel, mapped_g=mapped_g)
            if float(crossing_loss_weight) > 0.0
            else zero
        )
        loss_close_pair_sep = (
            _close_pair_separation_loss(
                outputs,
                targets,
                q_count,
                b_sel=b_sel,
                q_sel=q_sel,
                mapped_g=mapped_g,
                margin=float(close_pair_margin),
                window_seconds=float(close_pair_window_seconds),
                min_common_channels=int(close_pair_min_common_channels),
                min_gap_s=float(close_pair_min_gap_s),
                max_gap_s=float(close_pair_max_gap_s),
            )
            if float(close_pair_separation_loss_weight) > 0.0
            else zero
        )
    else:
        loss_peak = zero
        loss_dir = zero
        loss_speed = zero
        loss_mono = zero
        loss_smooth = zero
        loss_time_prior = zero
        loss_vis_prior = zero
        loss_crossing = zero
        loss_close_pair_sep = zero

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    gt_count = targets["gt_valid"].to(device=device, dtype=torch.float32).sum(dim=1)
    loss_count = F.smooth_l1_loss(obj_prob.sum(dim=1), gt_count, reduction="mean")
    loss_competition = _slot_peak_competition_loss(outputs, q_count) if float(slot_competition_loss_weight) > 0.0 else zero
    loss_gt_coverage = (
        _gt_coverage_loss(outputs, targets, q_count, temperature=float(gt_coverage_temperature))
        if float(gt_coverage_loss_weight) > 0.0
        else zero
    )
    total = (
        float(object_loss_weight) * loss_obj
        + float(peak_loss_weight) * loss_peak
        + float(count_loss_weight) * loss_count
        + float(direction_loss_weight) * loss_dir
        + float(speed_loss_weight) * loss_speed
        + float(monotonic_loss_weight) * loss_mono
        + float(smoothness_loss_weight) * loss_smooth
        + float(time_prior_loss_weight) * loss_time_prior
        + float(visibility_prior_loss_weight) * loss_vis_prior
        + float(slot_competition_loss_weight) * loss_competition
        + float(crossing_loss_weight) * loss_crossing
        + float(gt_coverage_loss_weight) * loss_gt_coverage
        + float(close_pair_separation_loss_weight) * loss_close_pair_sep
    )
    if not collect_metrics:
        return total, {}
    gt_total = int(targets["gt_valid"].sum().detach().cpu())
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_peak": float(loss_peak.detach().cpu()),
        "loss_count": float(loss_count.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_monotonic": float(loss_mono.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_time_prior": float(loss_time_prior.detach().cpu()),
        "loss_visibility_prior": float(loss_vis_prior.detach().cpu()),
        "loss_competition": float(loss_competition.detach().cpu()),
        "loss_crossing": float(loss_crossing.detach().cpu()),
        "loss_gt_coverage": float(loss_gt_coverage.detach().cpu()),
        "loss_close_pair_separation": float(loss_close_pair_sep.detach().cpu()),
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
    close_pair_window_seconds: float = 120.0,
    close_pair_min_common_channels: int = 8,
    close_pair_min_gap_s: float = 0.15,
    close_pair_max_gap_s: float = 1.5,
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
    gt_forward = 0
    gt_reverse = 0
    good_forward = 0
    good_reverse = 0
    gt_short_visible = 0
    gt_long_visible = 0
    good_short_visible = 0
    good_long_visible = 0
    good_gt = torch.zeros_like(gt_valid, dtype=torch.bool)
    peak_time = targets["peak_time"].to(device=device, dtype=torch.float32)
    for b in range(int(obj.shape[0])):
        pred_count = int(active[b].sum().item())
        gt_count = int(gt_valid[b].sum().item())
        count_abs_error += abs(pred_count - gt_count)
        count_exact += int(pred_count == gt_count)
        if gt_count <= 0:
            continue
        for gi_total in torch.where(gt_valid[b])[0].tolist():
            dir_label = int(targets["direction"][b, gi_total].item())
            visible_count = int((targets["visibility"][b, gi_total] > 0.5).sum().item())
            if dir_label == 0:
                gt_forward += 1
            else:
                gt_reverse += 1
            if visible_count <= 8:
                gt_short_visible += 1
            else:
                gt_long_visible += 1
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
                good_gt[b, gi] = True
                dir_label = int(targets["direction"][b, gi].item())
                visible_count = int((targets["visibility"][b, gi] > 0.5).sum().item())
                if dir_label == 0:
                    good_forward += 1
                else:
                    good_reverse += 1
                if visible_count <= 8:
                    good_short_visible += 1
                else:
                    good_long_visible += 1
    close_mask = _close_pair_mask(
        {
            "gt_valid": gt_valid,
            "visibility": targets["visibility"].to(device=device),
            "peak_time": peak_time,
            "gt_peak_index": targets["gt_peak_index"].to(device=device),
        },
        window_seconds=float(close_pair_window_seconds),
        min_common_channels=int(close_pair_min_common_channels),
        min_gap_s=float(close_pair_min_gap_s),
        max_gap_s=float(close_pair_max_gap_s),
    )
    close_pair_total = int(close_mask.sum().detach().cpu())
    if close_pair_total > 0:
        close_b, close_g1, close_g2 = torch.where(close_mask)
        both_good = good_gt[close_b, close_g1] & good_gt[close_b, close_g2]
        one_or_more_good = good_gt[close_b, close_g1] | good_gt[close_b, close_g2]
        close_pair_both = int(both_good.sum().detach().cpu())
        close_pair_any = int(one_or_more_good.sum().detach().cpu())
    else:
        close_pair_both = 0
        close_pair_any = 0
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
        "track_recall_forward": float(good_forward / max(1, gt_forward)),
        "track_recall_reverse": float(good_reverse / max(1, gt_reverse)),
        "track_recall_short_visible": float(good_short_visible / max(1, gt_short_visible)),
        "track_recall_long_visible": float(good_long_visible / max(1, gt_long_visible)),
        "close_pair_gt_count": float(close_pair_total),
        "close_pair_recall": float(close_pair_both / max(1, close_pair_total)),
        "close_pair_both_detected_rate": float(close_pair_both / max(1, close_pair_total)),
        "close_pair_miss_rate": float((close_pair_total - close_pair_any) / max(1, close_pair_total)),
        "gt_forward_count": float(gt_forward),
        "gt_reverse_count": float(gt_reverse),
        "gt_short_visible_count": float(gt_short_visible),
        "gt_long_visible_count": float(gt_long_visible),
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
    peak_time: Optional[torch.Tensor] = None,
    time_prior: Optional[np.ndarray] = None,
    time_prior_weight: float = 0.0,
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
            logp = float(np.log(max(prob, 1e-12)))
            if time_prior is not None and peak_time is not None and ch < int(time_prior.shape[0]):
                prior_delta = abs(float(peak_time[ch, k].item()) - float(time_prior[ch]))
                logp -= float(time_prior_weight) * prior_delta
            ch_items.append(
                {
                    "ch": int(ch),
                    "peak_idx": int(k),
                    "prob": prob,
                    "logp": logp,
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


def _trace_viterbi_path(
    states: list[list[dict[str, float | int | None]]],
    best: tuple[int, int, dict[str, float | int | None]],
) -> list[dict[str, float | int]]:
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


def decode_peak_slot_paths(
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
    time_prior: Optional[np.ndarray] = None,
) -> list[dict[str, Any]]:
    """Decode one slot into a beam of physically consistent peak paths.

    The model still supplies per-channel peak probabilities. This function is a
    non-differentiable inference-time Viterbi pass. Keeping a small beam is
    important around crossings: the best local path can switch vehicles, while a
    slightly lower-scoring candidate can keep the previous velocity identity.
    """
    candidates = _slot_peak_candidates(
        peak_prob,
        peak_valid,
        peak_index,
        peak_threshold=float(config.viterbi_candidate_threshold),
        topk=int(config.viterbi_topk),
        peak_time=peak_time,
        time_prior=time_prior,
        time_prior_weight=float(config.time_prior_weight),
    )
    states: list[list[dict[str, float | int | None]]] = []
    ref_speed = float(predicted_speed_kmh)
    if not np.isfinite(ref_speed) or ref_speed <= 1e-6:
        ref_speed = float(config.viterbi_fallback_speed_kmh)
    beam_size = max(1, int(config.viterbi_beam_size))
    for ch, ch_candidates in enumerate(candidates):
        ch_states: list[dict[str, float | int | None]] = []
        for item in ch_candidates:
            alternatives: list[dict[str, float | int | None]] = []
            start_state = dict(item)
            start_state.update(
                {
                    "score": float(item["logp"]) + float(config.viterbi_point_bonus),
                    "prev_ch": None,
                    "prev_idx": None,
                    "slope": None,
                }
            )
            alternatives.append(start_state)
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
                    state = dict(item)
                    state.update(
                        {
                            "score": float(cand_score),
                            "prev_ch": int(prev_ch),
                            "prev_idx": int(prev_idx),
                            "slope": float(slope),
                        }
                    )
                    alternatives.append(state)
            alternatives.sort(key=lambda state: float(state["score"]), reverse=True)
            ch_states.extend(alternatives[:beam_size])
        ch_states.sort(key=lambda state: float(state["score"]), reverse=True)
        ch_states = ch_states[: max(beam_size, beam_size * max(1, int(config.viterbi_topk)))]
        states.append(ch_states)
    terminal: list[tuple[int, int, dict[str, float | int | None]]] = []
    for ch, ch_states in enumerate(states):
        for idx, state in enumerate(ch_states):
            terminal.append((ch, idx, state))
    terminal.sort(key=lambda item: float(item[2]["score"]), reverse=True)
    paths: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for best in terminal:
        path = _trace_viterbi_path(states, best)
        key = tuple((int(item["ch"]), int(item["peak_idx"])) for item in path)
        if not key or key in seen:
            continue
        seen.add(key)
        paths.append({"score": float(best[2]["score"]), "path": path})
        if len(paths) >= beam_size:
            break
    return paths


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
    time_prior: Optional[np.ndarray] = None,
) -> list[dict[str, float | int]]:
    """Decode one slot as the best physically consistent peak path."""
    paths = decode_peak_slot_paths(
        peak_prob,
        peak_time,
        peak_valid,
        peak_index,
        direction_label=direction_label,
        predicted_speed_kmh=float(predicted_speed_kmh),
        x_axis_m=x_axis_m,
        time_downsample=int(time_downsample),
        fs=float(fs),
        config=config,
        time_prior=time_prior,
    )
    if not paths:
        return []
    return list(paths[0]["path"])


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
    time_prior = outputs.get("time_prior")
    time_prior_np = (
        time_prior[0].detach().cpu().numpy()
        if torch.is_tensor(time_prior) and bool(getattr(model, "has_time_prior", True))
        else None
    )
    max_tracks = min(int(cfg.max_tracks), int(obj.shape[0]))
    order = np.argsort(obj)[::-1][:max_tracks]
    tracks: list[Track] = []
    for q_idx in order.tolist():
        score = float(obj[q_idx])
        if score < float(cfg.objectness_threshold):
            continue
        decoder_mode = str(cfg.decoder_mode).lower()
        path_options: list[list[dict[str, float | int]]] = []
        if bool(cfg.use_viterbi_decoder) and decoder_mode == "beam_global":
            decoded_options = decode_peak_slot_paths(
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
                time_prior=None if time_prior_np is None else time_prior_np[q_idx],
            )
            path_options = [list(item["path"]) for item in decoded_options]
        elif bool(cfg.use_viterbi_decoder) and decoder_mode != "argmax":
            path_options = [
                decode_peak_slot_path(
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
                    time_prior=None if time_prior_np is None else time_prior_np[q_idx],
                )
            ]
        else:
            path_options = [argmax_peak_slot_path(peak_prob[q_idx], peak_valid, peak_threshold=float(cfg.peak_threshold))]
        for decoded in path_options:
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
                tracks.append(_track_stats(int(q_idx), LABEL_TO_DIRECTION.get(int(dirs[q_idx]), "forward"), points))
    if str(cfg.decoder_mode).lower() == "beam_global":
        best_per_slot: list[Track] = []
        used_slots: set[int] = set()
        for track in sorted(tracks, key=lambda item: item.total_score, reverse=True):
            if int(track.track_id) in used_slots:
                continue
            best_per_slot.append(track)
            used_slots.add(int(track.track_id))
        tracks = best_per_slot
    if str(cfg.decoder_mode).lower() == "beam_global" and float(cfg.global_conflict_penalty) > 0.0:
        kept_tracks: list[Track] = []
        for track in sorted(tracks, key=lambda item: item.total_score, reverse=True):
            tmap = {int(p.ch_idx): int(p.t_idx) for p in track.points}
            conflict = False
            for existing in kept_tracks:
                emap = {int(p.ch_idx): int(p.t_idx) for p in existing.points}
                common = sorted(set(tmap) & set(emap))
                if len(common) < int(cfg.dedup_min_overlap_channels):
                    continue
                diffs = np.array([abs(tmap[ch] - emap[ch]) for ch in common], dtype=np.float64)
                if float(np.median(diffs)) <= float(cfg.dedup_tolerance_samples):
                    conflict = True
                    break
            if not conflict:
                kept_tracks.append(track)
        tracks = kept_tracks
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
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"PeakSlot checkpoint loaded with newly initialized keys: {missing}", flush=True)
    if unexpected:
        print(f"PeakSlot checkpoint ignored unexpected keys: {unexpected}", flush=True)
    checkpoint["has_time_prior"] = not any(str(key).startswith("time_prior_head") for key in missing)
    checkpoint["has_visibility_prior"] = not any(str(key).startswith("visibility_prior_head") for key in missing)
    model.has_time_prior = bool(checkpoint["has_time_prior"])  # type: ignore[attr-defined]
    model.eval()
    return model, checkpoint
