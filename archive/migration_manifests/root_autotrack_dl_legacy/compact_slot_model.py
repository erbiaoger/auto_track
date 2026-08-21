"""Compact slot-based vehicle trajectory model.

This is a new lightweight architecture for the 50-channel DAS task.
It keeps the set-prediction interface of the slot models, but replaces the
heavy 2D CNN + deep decoder stack with a channel-first encoder:

1. Encode each channel's time series with a small 1D temporal CNN.
2. Sweep the channel embeddings with a tiny bidirectional GRU.
3. Let a small set of learned slot queries attend to the channel context.
4. Predict one vehicle slot per query.

The output interface matches the existing shard datasets:

- objectness_logits: [B, Q]
- direction_logits:  [B, Q, 2]
- speed:             [B, Q]
- time:              [B, Q, C]
- visibility_logits: [B, Q, C]

The training loss still uses Hungarian matching, but the model is much smaller
than the previous TrackSlotNet / PeakSlot family.
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

from autotrack.core.single_vehicle_tracker import extract_single_vehicle_track
from autotrack.core.track_extractor_graph import Track, TrackPoint


LABEL_TO_DIRECTION = {0: "forward", 1: "reverse"}


def auto_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _robust_scale(data: np.ndarray) -> float:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite.size == 0:
        return 1.0
    abs_vals = np.abs(finite)
    q995 = float(np.quantile(abs_vals, 0.995))
    rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
    return max(q995, 3.0 * rms, 1e-6)


def prepare_window_input(
    data_window: np.ndarray,
    time_downsample: int,
    clip_ratio: float = 1.35,
    input_mode: str = "raw",
) -> torch.Tensor:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = _robust_scale(arr_ds)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    mode = str(input_mode).lower()
    if mode == "raw":
        features = raw[None, :, :].astype(np.float32, copy=False)
    elif mode == "raw_abs":
        abs_feat = np.clip(np.abs(arr_ds) / scale, 0.0, clip) / clip
        features = np.stack([raw, abs_feat], axis=0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported input_mode={input_mode!r}; expected raw or raw_abs")
    return torch.from_numpy(features)


def _refine_t_idx(data: np.ndarray, ch: int, t_idx: int, radius: int) -> int:
    n_samples = int(data.shape[1])
    center = int(max(0, min(n_samples - 1, t_idx)))
    rad = int(max(0, radius))
    if rad <= 0:
        return center
    left = max(0, center - rad)
    right = min(n_samples - 1, center + rad)
    local = np.abs(data[int(ch), left : right + 1])
    if local.size == 0:
        return center
    return int(left + int(np.argmax(local)))


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    max_tracks: int = 24
    hidden_dim: int = 96
    decoder_layers: int = 1
    pooled_channels: int = 1
    pooled_time: int = 1
    temporal_channels: int = 48
    temporal_kernel: int = 7
    temporal_stride: int = 2
    channel_gru_layers: int = 1
    num_heads: int = 4
    dropout: float = 0.1
    dx_m: float = 100.0
    window_seconds: float = 120.0
    speed_norm_kmh: float = 150.0
    line_slope_scale: float = 0.55
    line_curve_scale: float = 0.12


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.15
    visibility_threshold: float = 0.35
    min_visible_channels: int = 3
    max_tracks: int = 24
    candidate_objectness_floor: float = 0.05
    objectness_count_scale: float = 1.05
    refine_with_graph: bool = True
    refine_prior_sigma_ch: float = 1.0
    refine_prior_sigma_t: float = 4.0
    refine_prior_weight: float = 1.25
    refine_speed_margin_kmh: float = 18.0
    dedup_tolerance_samples: int = 180
    dedup_min_overlap_channels: int = 3
    speed_norm_kmh: float = 150.0
    clip_ratio: float = 1.35
    refine_radius_samples: int = 120
    kalman_smooth: bool = True


class TemporalEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, temporal_channels: int, kernel_size: int, stride: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, temporal_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(max(1, min(8, temporal_channels // 4)), temporal_channels),
            nn.GELU(),
            nn.Conv1d(temporal_channels, temporal_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.GroupNorm(max(1, min(8, temporal_channels // 4)), temporal_channels),
            nn.GELU(),
            nn.Conv1d(temporal_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, hidden_dim // 4)), hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SlotSelfBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(query, context, context, need_weights=False)
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TrackSlotPredictor(nn.Module):
    """Compact slot predictor for 50-channel DAS windows."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.temporal = TemporalEncoder(
            int(c.in_channels),
            hidden,
            int(c.temporal_channels),
            int(c.temporal_kernel),
            int(c.temporal_stride),
            float(c.dropout),
        )
        self.channel_pos = nn.Parameter(torch.randn(1, int(c.n_channels), hidden) * 0.02)
        self.channel_norm = nn.LayerNorm(hidden)
        self.channel_gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden // 2,
            num_layers=int(c.channel_gru_layers),
            batch_first=True,
            bidirectional=True,
        )
        self.channel_post = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(c.dropout)),
        )

        self.slot_embed = nn.Embedding(int(c.max_tracks), hidden)
        self.global_proj = nn.Linear(hidden, hidden)
        self.slot_block = SlotSelfBlock(hidden, int(c.num_heads), float(c.dropout))

        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.anchor_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.visibility_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))

    def _encode_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, I, C, T]")
        b, in_ch, n_ch, n_t = x.shape
        if n_ch != int(self.config.n_channels):
            raise ValueError(f"Model expects {self.config.n_channels} channels, but input has {n_ch}")
        chan = x.permute(0, 2, 1, 3).reshape(b * n_ch, in_ch, n_t)
        feat = self.temporal(chan).view(b, n_ch, -1)
        feat = self.channel_norm(feat + self.channel_pos[:, :n_ch, :])
        feat, _ = self.channel_gru(feat)
        feat = self.channel_post(feat)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor]] = None,
    ) -> dict[str, torch.Tensor]:
        del targets
        context = self._encode_channels(x)
        global_ctx = context.mean(dim=1, keepdim=True)
        query = self.slot_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        query = query + self.global_proj(global_ctx)
        slot = self.slot_block(query, context)
        anchor = torch.sigmoid(self.anchor_head(slot)).squeeze(-1)
        speed_norm = torch.sigmoid(self.speed_head(slot)).squeeze(-1)
        direction_prob = torch.softmax(self.direction_head(slot), dim=-1)
        direction_sign = direction_prob[..., 0] - direction_prob[..., 1]
        ch_idx = torch.arange(int(self.config.n_channels), device=x.device, dtype=slot.dtype)
        ch_offset_m = ch_idx * float(self.config.dx_m)
        speed_mps = torch.clamp(speed_norm * float(self.config.speed_norm_kmh) / 3.6, min=1e-3)
        delta_t_norm = ch_offset_m.view(1, 1, -1) / (speed_mps.unsqueeze(-1) * float(self.config.window_seconds))
        line = anchor.unsqueeze(-1) + direction_sign.unsqueeze(-1) * delta_t_norm
        return {
            "num_regular_queries": int(self.config.max_tracks),
            "objectness_logits": self.objectness_head(slot).squeeze(-1),
            "direction_logits": self.direction_head(slot),
            "speed": torch.sigmoid(self.speed_head(slot).squeeze(-1)),
            "time": torch.sigmoid(line).squeeze(-2),
            "visibility_logits": self.visibility_head(slot),
        }


def _targets_batch_size(targets: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor]) -> int:
    if isinstance(targets, dict):
        if "gt_valid" in targets:
            return int(targets["gt_valid"].shape[0])
        if "time" in targets and targets["time"].ndim >= 3:
            return int(targets["time"].shape[0])
    return len(targets)  # type: ignore[arg-type]


def _target_at(
    targets: Sequence[dict[str, torch.Tensor]] | dict[str, torch.Tensor],
    batch_idx: int,
) -> dict[str, torch.Tensor]:
    if isinstance(targets, dict):
        out: dict[str, torch.Tensor] = {}
        for key, value in targets.items():
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) > batch_idx:
                out[key] = value[batch_idx]
            elif torch.is_tensor(value):
                out[key] = value
        return out
    return targets[batch_idx]


def _target_valid_mask(target: dict[str, torch.Tensor], device: torch.device | str) -> torch.Tensor:
    if "gt_valid" in target:
        return target["gt_valid"].to(device=device, dtype=torch.bool)
    n_gt = int(target["time"].shape[0])
    return torch.ones((n_gt,), dtype=torch.bool, device=device)


def _target_valid_count(target: dict[str, torch.Tensor]) -> int:
    if "gt_valid" in target:
        return int(target["gt_valid"].sum().item())
    return int(target["time"].shape[0])


def _target_polyline_and_attrs(
    target: dict[str, torch.Tensor],
    *,
    n_channels: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mask = _target_valid_mask(target, device)
    gt_time = target["time"].to(device=device, dtype=torch.float32)
    gt_vis = target["visibility"].to(device=device, dtype=torch.float32)
    gt_dir = target["direction"].to(device=device, dtype=torch.long)
    gt_speed = target["speed"].to(device=device, dtype=torch.float32)
    if mask.numel() != int(gt_time.shape[0]):
        raise ValueError("gt_valid mask mismatch")
    return gt_time[mask], gt_vis[mask], gt_dir[mask], gt_speed[mask]


def _weighted_smooth_l1(pred: torch.Tensor, gt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, gt, reduction="none") * weight
    return loss.sum() / torch.clamp(weight.sum(), min=1.0)


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
    if "gt_valid" in targets:
        gt_valid = targets["gt_valid"][b].to(device=device, dtype=torch.bool)
    else:
        gt_valid = torch.ones((int(targets["time"].shape[1]),), dtype=torch.bool, device=device)
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
        q, g = cost.shape
        k = int(min(q, g))
        if k <= 0:
            empty = torch.empty((0,), dtype=torch.long, device=device)
            return empty, empty
        work = cost.clone()
        rows: list[torch.Tensor] = []
        cols: list[torch.Tensor] = []
        large = torch.finfo(work.dtype).max
        for _ in range(k):
            idx = torch.argmin(work)
            r = torch.div(idx, g, rounding_mode="floor").long()
            c = (idx - r * g).long()
            rows.append(r)
            cols.append(c)
            work[r, :] = large
            work[:, c] = large
        return torch.stack(rows), torch.stack(cols)

    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


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
    count_loss_weight: float = 0.05,
    monotonic_loss_weight: float = 0.5,
    smoothness_loss_weight: float = 0.1,
    visibility_negative_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    q_count = int(outputs["num_regular_queries"])
    obj_target = torch.zeros((batch_size, q_count), dtype=torch.float32, device=device)
    obj_weight = torch.full((batch_size, q_count), float(no_object_weight), dtype=torch.float32, device=device)
    if "gt_valid" in targets:
        gt_valid_all = targets["gt_valid"].to(device=device, dtype=torch.bool)
    else:
        gt_valid_all = torch.ones(
            (batch_size, int(targets["time"].shape[1])),
            dtype=torch.bool,
            device=device,
        )

    matched_b: list[torch.Tensor] = []
    matched_q: list[torch.Tensor] = []
    matched_g: list[torch.Tensor] = []
    matched_total = 0
    gt_total = int(gt_valid_all.sum().detach().cpu())

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
        gt_valid = gt_valid_all
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
        vis_raw = F.binary_cross_entropy_with_logits(pred_vis_logits, gt_vis, reduction="none")
        vis_weight = torch.where(gt_vis > 0.5, torch.ones_like(gt_vis), torch.full_like(gt_vis, float(visibility_negative_weight)))
        loss_vis = (vis_raw * vis_weight).sum() / torch.clamp(vis_weight.sum(), min=1.0)
        loss_dir = F.cross_entropy(pred_dir, gt_dir, reduction="mean")
        loss_speed = F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean")

        pair_vis = gt_vis[:, 1:] * gt_vis[:, :-1]
        dt = pred_time[:, 1:] - pred_time[:, :-1]
        sign = torch.where(gt_dir[:, None] == 0, torch.ones_like(dt), -torch.ones_like(dt))
        signed_dt = dt * sign
        loss_monotonic = (torch.relu(-signed_dt) * pair_vis).sum() / torch.clamp(pair_vis.sum(), min=1.0)

        if pred_time.shape[1] >= 3:
            tri_vis = gt_vis[:, 2:] * gt_vis[:, 1:-1] * gt_vis[:, :-2]
            d2 = pred_time[:, 2:] - 2.0 * pred_time[:, 1:-1] + pred_time[:, :-2]
            smooth_raw = F.smooth_l1_loss(d2, torch.zeros_like(d2), reduction="none")
            loss_smooth = (smooth_raw * tri_vis).sum() / torch.clamp(tri_vis.sum(), min=1.0)
        else:
            loss_smooth = zero
    else:
        loss_time = zero
        loss_vis = zero
        loss_dir = zero
        loss_speed = zero
        loss_monotonic = zero
        loss_smooth = zero

    obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    gt_count = gt_valid_all.to(device=device, dtype=torch.float32).sum(dim=1)
    pred_count_soft = obj_prob.sum(dim=1)
    loss_count = F.smooth_l1_loss(pred_count_soft, gt_count, reduction="mean")

    total = (
        loss_obj
        + float(time_loss_weight) * loss_time
        + float(visibility_loss_weight) * loss_vis
        + float(direction_loss_weight) * loss_dir
        + float(speed_loss_weight) * loss_speed
        + float(count_loss_weight) * loss_count
        + float(monotonic_loss_weight) * loss_monotonic
        + float(smoothness_loss_weight) * loss_smooth
    )
    if not collect_metrics:
        return total, {}

    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_time": float(loss_time.detach().cpu()),
        "loss_vis": float(loss_vis.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_count": float(loss_count.detach().cpu()),
        "loss_monotonic": float(loss_monotonic.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
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
    point_threshold: float = 0.12,
    matcher: str = "hungarian",
) -> dict[str, float]:
    device = outputs["objectness_logits"].device
    q_count = int(outputs["num_regular_queries"])
    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    active = obj >= float(objectness_threshold)
    pred_total = int(active.sum().detach().cpu())
    if "gt_valid" in targets:
        gt_valid = targets["gt_valid"].to(device=device, dtype=torch.bool)
    else:
        gt_valid = torch.ones(
            (int(obj.shape[0]), int(targets["time"].shape[1])),
            dtype=torch.bool,
            device=device,
        )
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


def _kalman_smooth_track(points: list[TrackPoint], process_var: float = 0.6, meas_var: float = 0.2) -> list[TrackPoint]:
    if len(points) < 3:
        return points
    ordered = sorted(points, key=lambda p: int(p.ch_idx))
    state = np.array([float(ordered[0].time_s), 0.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64) * 10.0
    out: list[TrackPoint] = []
    prev_ch = int(ordered[0].ch_idx)
    for p in ordered:
        ch = int(p.ch_idx)
        dt_ch = max(1, ch - prev_ch)
        f = np.array([[1.0, float(dt_ch)], [0.0, 1.0]], dtype=np.float64)
        q = np.array([[0.25 * dt_ch**2, 0.5 * dt_ch], [0.5 * dt_ch, 1.0]], dtype=np.float64) * float(process_var)
        state = f @ state
        cov = f @ cov @ f.T + q
        z = np.array([float(p.time_s)], dtype=np.float64)
        h = np.array([[1.0, 0.0]], dtype=np.float64)
        s = h @ cov @ h.T + np.array([[float(meas_var)]], dtype=np.float64)
        k = cov @ h.T @ np.linalg.inv(s)
        state = state + (k @ (z - h @ state)).reshape(-1)
        cov = (np.eye(2) - k @ h) @ cov
        out.append(
            TrackPoint(
                ch_idx=ch,
                t_idx=int(round(state[0])),
                time_s=float(state[0]),
                offset_m=float(p.offset_m),
                amp=float(p.amp),
                score=float(p.score),
            )
        )
        prev_ch = ch
    return out


def _track_prior_heatmap(track: Track, n_channels: int, n_samples: int, *, sigma_ch: float = 1.0, sigma_t: float = 4.0) -> np.ndarray:
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
    return [
        Track(
            track_id=int(i),
            direction=track.direction,
            points=sorted(track.points, key=lambda p: p.ch_idx),
            total_score=float(track.total_score),
            mean_speed_kmh=float(track.mean_speed_kmh),
        )
        for i, track in enumerate(kept)
    ]


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
    speed_kmh = outputs["speed"][0].detach().cpu().numpy() * float(cfg.speed_norm_kmh)

    n_samples = int(arr.shape[1])
    max_tracks = min(int(cfg.max_tracks), int(obj.shape[0]))
    keep_limit = int(round(float(np.sum(obj)) * float(cfg.objectness_count_scale)))
    keep_limit = int(np.clip(keep_limit, 1, max_tracks))
    floor = min(float(cfg.objectness_threshold), float(cfg.candidate_objectness_floor))
    pool = np.where(obj >= floor)[0]
    if pool.size == 0:
        pool = np.arange(obj.shape[0], dtype=np.int64)
    order = pool[np.argsort(obj[pool])[::-1]]
    if order.size < keep_limit:
        remaining = np.array([i for i in np.argsort(obj)[::-1].tolist() if i not in set(int(v) for v in order.tolist())], dtype=np.int64)
        order = np.concatenate([order, remaining], axis=0)
    order = order[:keep_limit]
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
            t_idx_raw = int(round(float(np.clip(time_norm[q_idx, ch], 0.0, 1.0)) * float(max(1, n_samples - 1))))
            t_idx_raw = int(max(0, min(n_samples - 1, t_idx_raw)))
            t_idx = _refine_t_idx(arr, int(ch), t_idx_raw, int(cfg.refine_radius_samples))
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
        if bool(cfg.kalman_smooth):
            points = _kalman_smooth_track(points)
        track = Track(
            track_id=len(tracks),
            direction=LABEL_TO_DIRECTION.get(int(dirs[q_idx]), "forward"),
            points=sorted(points, key=lambda p: p.ch_idx),
            total_score=float(score),
            mean_speed_kmh=float(speed_kmh[q_idx]),
        )
        if bool(cfg.refine_with_graph):
            prior = _track_prior_heatmap(
                track,
                int(arr.shape[0]),
                int(arr.shape[1]),
                sigma_ch=float(cfg.refine_prior_sigma_ch),
                sigma_t=float(cfg.refine_prior_sigma_t),
            )
            hint = np.full((int(arr.shape[0]),), np.nan, dtype=np.float32)
            for point in track.points:
                hint[int(point.ch_idx)] = float(point.time_s)
            dx_m = float(np.median(np.diff(np.asarray(x_axis_m, dtype=np.float32)))) if len(x_axis_m) > 1 else 1.0
            refined = extract_single_vehicle_track(
                arr,
                float(fs),
                float(dx_m),
                str(track.direction),
                max(1.0, float(speed_kmh[q_idx]) - float(cfg.refine_speed_margin_kmh)),
                float(speed_kmh[q_idx]) + float(cfg.refine_speed_margin_kmh),
                config={
                    "candidate_prominence": 0.22,
                    "candidate_min_distance": 180,
                    "candidate_max_peaks_per_channel": 32,
                    "max_skip_channels": 8,
                    "min_track_channels": int(cfg.min_visible_channels),
                    "min_track_score": 8.0,
                    "edge_relax_enabled": True,
                    "edge_min_track_channels": 4,
                    "edge_time_margin_seconds": 8.0,
                    "edge_min_score_scale": 0.5,
                    "kalman_bridge_gap_channels": 12,
                    "kalman_fill_missing": True,
                    "kalman_gate_seconds": 0.35,
                    "kalman_speed_gate_kmh": 30.0,
                    "direction": str(track.direction),
                    "candidate_hypotheses": 4,
                    "hypothesis_prior_weight": 4.0,
                },
                prior_heatmap=prior,
                prior_weight=float(cfg.refine_prior_weight),
                prior_time_hint=hint,
            )
            if refined:
                window_duration_s = float(arr.shape[1]) / float(fs)
                track = refined[0]
                fixed_points: list[TrackPoint] = []
                coarse_by_ch = {int(p.ch_idx): int(p.t_idx) for p in points}
                for point in track.points:
                    normalized_t_idx = int(
                        round(
                            float(np.clip(point.time_s / max(1e-9, window_duration_s), 0.0, 1.0))
                            * float(max(1, arr.shape[0] - 1))
                        )
                    )
                    fixed_points.append(
                        TrackPoint(
                            ch_idx=int(point.ch_idx),
                            t_idx=int(coarse_by_ch.get(int(point.ch_idx), normalized_t_idx)),
                            time_s=float(point.time_s),
                            offset_m=float(point.offset_m),
                            amp=float(point.amp),
                            score=float(point.score),
                        )
                    )
                track.points = fixed_points
                track.track_id = len(tracks)
                track.total_score = float(max(track.total_score, score))
        tracks.append(track)

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
        "model_family": "compact_slot",
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
