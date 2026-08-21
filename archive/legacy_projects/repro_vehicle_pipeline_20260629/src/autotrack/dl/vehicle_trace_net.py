"""Trajectory-centric single-vehicle network.

This module replaces the earlier heatmap-first design with a smaller model that
predicts a per-channel trajectory proposal directly, then derives a heatmap
prior from that proposal. The graph/Kalman/Hungarian tracker remains the final
decoder, but the network is now focused on the actual one-vehicle problem:

- predict a coherent channel-time curve
- predict visibility per channel
- predict coarse global motion cues
- hand the tracker a prior instead of forcing it to rediscover the track from
  a dense segmentation-like map
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from autotrack.core.single_vehicle_tracker import SingleVehicleTrackerConfig, extract_single_vehicle_track
from autotrack.core.track_extractor_graph import Track


def _auto_torch_device() -> str:
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
    if str(input_mode).lower() != "raw":
        raise ValueError("only raw input is supported")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = _robust_scale(arr_ds)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    features = raw[None, :, :].astype(np.float32, copy=False)
    return torch.from_numpy(features)


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    hidden_dim: int = 64
    pooled_channels: int = 16
    dropout: float = 0.08
    heatmap_sigma_t: float = 2.0


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    min_visible_channels: int = 4
    objectness_threshold: float = 0.35
    peak_threshold: float = 0.25
    speed_norm_kmh: float = 150.0
    prior_weight: float = 1.0
    low_confidence_prior_weight: float = 0.0
    low_confidence_objectness_threshold: float = 0.50
    low_confidence_heatmap_threshold: float = 0.10
    single_vehicle_tracker: SingleVehicleTrackerConfig = field(default_factory=SingleVehicleTrackerConfig)


def _as_inference_config(config: Optional[InferenceConfig | dict[str, Any]]) -> InferenceConfig:
    if config is None:
        return InferenceConfig()
    if isinstance(config, InferenceConfig):
        return config
    if isinstance(config, dict):
        defaults = InferenceConfig()
        tracker_cfg = config.get("single_vehicle_tracker", None)
        if isinstance(tracker_cfg, SingleVehicleTrackerConfig):
            tracker = tracker_cfg
        elif isinstance(tracker_cfg, dict):
            tracker = SingleVehicleTrackerConfig(**tracker_cfg)
        else:
            tracker = defaults.single_vehicle_tracker
        return InferenceConfig(
            time_downsample=int(config.get("time_downsample", defaults.time_downsample)),
            min_visible_channels=int(config.get("min_visible_channels", defaults.min_visible_channels)),
            objectness_threshold=float(config.get("objectness_threshold", defaults.objectness_threshold)),
            peak_threshold=float(config.get("peak_threshold", defaults.peak_threshold)),
            speed_norm_kmh=float(config.get("speed_norm_kmh", defaults.speed_norm_kmh)),
            prior_weight=float(config.get("prior_weight", defaults.prior_weight)),
            low_confidence_prior_weight=float(config.get("low_confidence_prior_weight", defaults.low_confidence_prior_weight)),
            low_confidence_objectness_threshold=float(
                config.get("low_confidence_objectness_threshold", defaults.low_confidence_objectness_threshold)
            ),
            low_confidence_heatmap_threshold=float(
                config.get("low_confidence_heatmap_threshold", defaults.low_confidence_heatmap_threshold)
            ),
            single_vehicle_tracker=tracker,
        )
    raise TypeError("config must be InferenceConfig / dict / None")


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


class VehicleTraceNet(nn.Module):
    """Lightweight proposal net for one-vehicle trajectory decoding."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        pooled = int(max(8, c.pooled_channels))

        self.stem = ConvBlock(int(c.in_channels), 24)
        self.enc2 = ConvBlock(24, 32, stride=(1, 2))
        self.enc3 = ConvBlock(32, hidden, stride=(1, 2))
        self.mix = ConvBlock(hidden, hidden)
        self.dropout = nn.Dropout2d(float(c.dropout))

        channel_latent_dim = hidden * 2
        self.channel_proj = nn.Sequential(
            nn.Conv1d(channel_latent_dim, pooled, kernel_size=1, bias=False),
            nn.GroupNorm(1, pooled),
            nn.GELU(),
        )
        self.visibility_head = nn.Sequential(
            nn.Linear(pooled, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(pooled, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.line_head = nn.Sequential(
            nn.Linear(channel_latent_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.objectness_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self._heatmap_sigma_t = float(max(1e-3, c.heatmap_sigma_t))

    def _channel_latent(self, feat: torch.Tensor, n_channels: int) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feat, output_size=(n_channels, 1)).squeeze(-1)
        mx = F.adaptive_max_pool2d(feat, output_size=(n_channels, 1)).squeeze(-1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, in_channels, channels, time]")
        feat = self.stem(x)
        feat = self.enc2(feat)
        feat = self.enc3(feat)
        feat = self.mix(feat)
        feat = self.dropout(feat)

        batch, _, n_channels, _ = feat.shape
        channel_latent = self._channel_latent(feat, int(n_channels))
        channel_latent = self.channel_proj(channel_latent).transpose(1, 2).contiguous()

        visibility_logits = self.visibility_head(channel_latent).squeeze(-1)
        trajectory_raw = self.trajectory_head(channel_latent).squeeze(-1)
        trajectory_time = torch.sigmoid(trajectory_raw)
        line_endpoints = torch.stack([trajectory_time[:, 0], trajectory_time[:, -1]], dim=-1)

        global_feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        objectness_logits = self.objectness_head(global_feat).squeeze(-1)
        direction_logits = self.direction_head(global_feat)
        speed = self.speed_head(global_feat).squeeze(-1)

        time_bins = int(x.shape[-1])
        time_axis = torch.linspace(0.0, 1.0, time_bins, device=x.device, dtype=trajectory_time.dtype).view(1, 1, -1)
        sigma = float(self._heatmap_sigma_t) / float(max(1, time_bins - 1))
        sigma = max(sigma, 1e-3)
        vis_prob = torch.sigmoid(visibility_logits).unsqueeze(-1)
        dist = (time_axis - trajectory_time.unsqueeze(-1)) / sigma
        heatmap_prob = vis_prob * torch.exp(-0.5 * dist.square())
        heatmap_logits = torch.logit(heatmap_prob.clamp(1e-4, 1.0 - 1e-4))

        return {
            "heatmap_logits": heatmap_logits,
            "visibility_logits": visibility_logits,
            "trajectory_time": trajectory_time,
            "objectness_logits": objectness_logits,
            "direction_logits": direction_logits,
            "speed": speed,
            "line_endpoints": line_endpoints,
        }


# Backward-compatible name used by the existing scripts/tests.
SingleVehiclePeakNet = VehicleTraceNet


def heatmap_to_time_prediction(heatmap_logits: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    logits = heatmap_logits.to(torch.float32)
    if logits.ndim != 3:
        raise ValueError("heatmap_logits must have shape [batch, channel, time]")
    temp = float(max(1e-3, temperature))
    probs = torch.softmax(logits / temp, dim=-1)
    bins = torch.linspace(0.0, 1.0, int(logits.shape[-1]), device=logits.device, dtype=probs.dtype)
    return torch.sum(probs * bins.view(1, 1, -1), dim=-1)


def build_single_vehicle_heatmap_target(
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


def build_single_vehicle_line_target(time: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    time = time.to(torch.float32)
    visibility = visibility.to(torch.float32)
    n_channels = int(time.shape[0])
    if n_channels <= 0:
        raise ValueError("time must have at least one channel")
    device = time.device
    mask = visibility > 0.5
    if int(mask.sum().item()) < 2:
        if int(mask.sum().item()) == 1:
            value = float(time[mask][0].item())
            return torch.tensor([value, value], dtype=torch.float32, device=device)
        return torch.zeros((2,), dtype=torch.float32, device=device)
    ch = torch.arange(n_channels, dtype=torch.float32, device=device)
    x = ch[mask]
    y = time[mask]
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    denom = torch.sum((x - x_mean) ** 2)
    if float(denom.item()) <= 1e-9:
        value = float(torch.mean(y).item())
        return torch.tensor([value, value], dtype=torch.float32, device=device)
    slope = torch.sum((x - x_mean) * (y - y_mean)) / denom
    intercept = y_mean - slope * x_mean
    t0 = torch.clamp(intercept, 0.0, 1.0)
    t1 = torch.clamp(intercept + slope * float(max(0, n_channels - 1)), 0.0, 1.0)
    return torch.stack([t0, t1]).to(torch.float32)


def build_single_vehicle_trajectory_target(time: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
    time_np = time.to(torch.float32).cpu().numpy()
    vis_np = visibility.to(torch.float32).cpu().numpy()
    n_channels = int(time_np.shape[0])
    if n_channels <= 0:
        raise ValueError("time must have at least one channel")
    ch = np.arange(n_channels, dtype=np.float32)
    mask = vis_np > 0.5
    if int(mask.sum()) >= 2:
        slope, intercept = np.polyfit(ch[mask].astype(np.float64), time_np[mask].astype(np.float64), deg=1)
        traj = np.clip(intercept + slope * ch, 0.0, 1.0).astype(np.float32)
        traj[mask] = np.clip(time_np[mask], 0.0, 1.0)
        return torch.from_numpy(traj)
    if int(mask.sum()) == 1:
        value = float(np.clip(time_np[mask][0], 0.0, 1.0))
        return torch.full((n_channels,), value, dtype=torch.float32)
    return torch.zeros((n_channels,), dtype=torch.float32)


def single_vehicle_heatmap_loss(outputs: dict[str, torch.Tensor], target_heatmap: torch.Tensor) -> torch.Tensor:
    pred = outputs["heatmap_logits"]
    tgt = target_heatmap.to(pred.device, dtype=pred.dtype)
    return F.binary_cross_entropy_with_logits(pred, tgt, reduction="mean")


def score_single_vehicle_window(
    outputs: dict[str, torch.Tensor],
    *,
    speed_norm_kmh: float = 150.0,
) -> dict[str, float]:
    obj_prob = torch.sigmoid(outputs["objectness_logits"].detach())
    heatmap_prob = torch.sigmoid(outputs["heatmap_logits"].detach())
    vis_prob = torch.sigmoid(outputs.get("visibility_logits", outputs["objectness_logits"].detach().unsqueeze(-1))).detach()
    if obj_prob.ndim == 0:
        obj_prob = obj_prob.unsqueeze(0)
    heatmap_peak = float(torch.max(heatmap_prob).item())
    heatmap_mean = float(torch.mean(heatmap_prob).item())
    heatmap_contrast = max(0.0, heatmap_peak - heatmap_mean)
    visibility_coverage = float(torch.mean((vis_prob > 0.5).to(torch.float32)).item())
    confidence = (
        0.50 * float(obj_prob.item())
        + 0.25 * min(1.0, heatmap_contrast / max(1e-6, heatmap_peak + 1e-3))
        + 0.25 * visibility_coverage
    )
    direction_prob = torch.softmax(outputs["direction_logits"].detach(), dim=-1)
    speed_kmh = float(torch.sigmoid(outputs["speed"].detach()).item() * float(speed_norm_kmh))
    return {
        "confidence": float(confidence),
        "objectness": float(obj_prob.item()),
        "heatmap_peak": float(heatmap_peak),
        "heatmap_mean": float(heatmap_mean),
        "heatmap_contrast": float(heatmap_contrast),
        "visibility_coverage": float(visibility_coverage),
        "direction_forward_prob": float(direction_prob[..., 0].item()),
        "direction_reverse_prob": float(direction_prob[..., 1].item()),
        "speed_kmh": float(speed_kmh),
    }


def estimate_direction_index_from_outputs(outputs: dict[str, torch.Tensor]) -> int:
    """Estimate forward(0)/reverse(1) from trajectory geometry when available."""
    if "trajectory_time" in outputs:
        traj = outputs["trajectory_time"].detach().to(torch.float32)
        if traj.ndim == 1 and traj.numel() >= 2:
            ch = torch.arange(traj.numel(), dtype=torch.float32, device=traj.device)
            mask = torch.isfinite(traj)
            if int(mask.sum().item()) >= 2:
                x = ch[mask]
                y = traj[mask]
                x_mean = torch.mean(x)
                y_mean = torch.mean(y)
                denom = torch.sum((x - x_mean) ** 2)
                if float(denom.item()) > 1e-9:
                    slope = torch.sum((x - x_mean) * (y - y_mean)) / denom
                    if float(torch.abs(slope).item()) > 1e-5:
                        return 0 if float(slope.item()) >= 0.0 else 1
        elif traj.ndim == 2 and traj.shape[-1] >= 2:
            start = torch.mean(traj[..., : max(1, min(4, traj.shape[-1]))])
            end = torch.mean(traj[..., -max(1, min(4, traj.shape[-1])) :])
            if torch.isfinite(start) and torch.isfinite(end) and abs(float(end.item() - start.item())) > 1e-5:
                return 0 if float(end.item()) >= float(start.item()) else 1
    if "line_endpoints" in outputs:
        line = torch.sigmoid(outputs["line_endpoints"].detach().to(torch.float32))
        if line.ndim >= 2 and line.shape[-1] == 2:
            return 0 if float(torch.mean(line[..., 1] - line[..., 0]).item()) >= 0.0 else 1
    if "direction_logits" in outputs:
        return int(outputs["direction_logits"].detach().argmax(dim=-1).item())
    return 0


def _rasterize_time_prior(
    trajectory_time: torch.Tensor,
    visibility_logits: torch.Tensor,
    *,
    time_bins: int,
    sigma_t: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    traj = trajectory_time.to(torch.float32).clamp(0.0, 1.0)
    vis = torch.sigmoid(visibility_logits.to(torch.float32))
    time_axis = torch.linspace(0.0, 1.0, int(time_bins), device=traj.device, dtype=traj.dtype).view(1, 1, -1)
    sigma = max(float(sigma_t) / float(max(1, time_bins - 1)), 1e-3)
    dist = (time_axis - traj.unsqueeze(-1)) / sigma
    prior_prob = vis.unsqueeze(-1) * torch.exp(-0.5 * dist.square())
    prior_prob = prior_prob.clamp(1e-4, 1.0 - 1e-4)
    prior_logits = torch.logit(prior_prob)
    return prior_logits, traj


def single_vehicle_supervised_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_heatmap: torch.Tensor,
    target_competitor_heatmap: Optional[torch.Tensor] = None,
    target_line: Optional[torch.Tensor] = None,
    target_trajectory: Optional[torch.Tensor] = None,
    target_visibility: torch.Tensor,
    target_time: torch.Tensor,
    target_objectness: Optional[torch.Tensor] = None,
    target_direction: Optional[torch.Tensor] = None,
    target_speed: Optional[torch.Tensor] = None,
    heatmap_weight: float = 1.0,
    visibility_weight: float = 0.5,
    time_weight: float = 1.0,
    time_consistency_weight: float = 0.0,
    competitor_weight: float = 0.0,
    objectness_weight: float = 0.25,
    direction_weight: float = 0.2,
    speed_weight: float = 0.2,
    time_temperature: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_heatmap = outputs["heatmap_logits"]
    heatmap_tgt = target_heatmap.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
    visibility_tgt = target_visibility.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
    time_tgt = target_time.to(pred_heatmap.device, dtype=pred_heatmap.dtype)

    loss_heatmap = F.binary_cross_entropy_with_logits(pred_heatmap, heatmap_tgt, reduction="mean")
    if "visibility_logits" in outputs:
        loss_visibility = F.binary_cross_entropy_with_logits(outputs["visibility_logits"], visibility_tgt, reduction="mean")
    else:
        visibility_logits = torch.logsumexp(pred_heatmap, dim=-1)
        loss_visibility = F.binary_cross_entropy_with_logits(visibility_logits, visibility_tgt, reduction="mean")

    if "trajectory_time" in outputs:
        pred_time = outputs["trajectory_time"].to(pred_heatmap.device, dtype=pred_heatmap.dtype)
    else:
        pred_time = heatmap_to_time_prediction(pred_heatmap, temperature=float(time_temperature))

    visible_mask = visibility_tgt > 0.5
    if bool(visible_mask.any()):
        loss_time = F.smooth_l1_loss(pred_time[visible_mask], time_tgt[visible_mask], reduction="mean")
    else:
        loss_time = pred_time.sum() * 0.0

    if float(time_consistency_weight) > 0.0:
        pair_mask = (visibility_tgt[:, 1:] > 0.5) & (visibility_tgt[:, :-1] > 0.5)
        if bool(pair_mask.any()):
            pred_delta = pred_time[:, 1:] - pred_time[:, :-1]
            tgt_delta = time_tgt[:, 1:] - time_tgt[:, :-1]
            loss_time_consistency = F.smooth_l1_loss(pred_delta[pair_mask], tgt_delta[pair_mask], reduction="mean")
        else:
            loss_time_consistency = pred_time.sum() * 0.0
    else:
        loss_time_consistency = pred_time.sum() * 0.0

    if target_competitor_heatmap is not None and float(competitor_weight) > 0.0:
        comp_tgt = target_competitor_heatmap.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        comp_mask = comp_tgt > 0.05
        if bool(comp_mask.any()):
            loss_competitor = F.binary_cross_entropy_with_logits(
                pred_heatmap[comp_mask],
                torch.zeros_like(pred_heatmap[comp_mask]),
                reduction="mean",
            )
        else:
            loss_competitor = pred_heatmap.sum() * 0.0
    else:
        loss_competitor = pred_heatmap.sum() * 0.0

    if target_line is not None and "line_endpoints" in outputs:
        line_tgt = target_line.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        line_pred = torch.sigmoid(outputs["line_endpoints"])
        loss_line = F.smooth_l1_loss(line_pred, line_tgt, reduction="mean")
    else:
        loss_line = pred_heatmap.sum() * 0.0

    if target_trajectory is not None and "trajectory_time" in outputs:
        traj_tgt = target_trajectory.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        traj_pred = outputs["trajectory_time"].to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        loss_trajectory = F.smooth_l1_loss(traj_pred[visible_mask], time_tgt[visible_mask], reduction="mean") if bool(visible_mask.any()) else pred_heatmap.sum() * 0.0
        traj_delta = traj_pred[:, 1:] - traj_pred[:, :-1]
        traj_tgt_delta = traj_tgt[:, 1:] - traj_tgt[:, :-1]
        loss_trajectory_smooth = F.smooth_l1_loss(traj_delta, traj_tgt_delta, reduction="mean")
    else:
        loss_trajectory = pred_heatmap.sum() * 0.0
        loss_trajectory_smooth = pred_heatmap.sum() * 0.0

    total = (
        float(heatmap_weight) * loss_heatmap
        + float(visibility_weight) * loss_visibility
        + float(time_weight) * loss_time
        + float(time_consistency_weight) * loss_time_consistency
        + float(competitor_weight) * loss_competitor
        + 0.55 * loss_line
        + 1.25 * loss_trajectory
        + 0.35 * loss_trajectory_smooth
    )
    metrics: dict[str, torch.Tensor] = {
        "loss_heatmap": loss_heatmap.detach(),
        "loss_visibility": loss_visibility.detach(),
        "loss_time": loss_time.detach(),
        "loss_time_consistency": loss_time_consistency.detach(),
        "loss_competitor": loss_competitor.detach(),
        "loss_line": loss_line.detach(),
        "loss_trajectory": loss_trajectory.detach(),
        "loss_trajectory_smooth": loss_trajectory_smooth.detach(),
    }

    if target_objectness is not None:
        obj_tgt = target_objectness.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        loss_objectness = F.binary_cross_entropy_with_logits(outputs["objectness_logits"], obj_tgt, reduction="mean")
        total = total + float(objectness_weight) * loss_objectness
        metrics["loss_objectness"] = loss_objectness.detach()
    if target_direction is not None:
        dir_tgt = target_direction.to(pred_heatmap.device, dtype=torch.long)
        loss_direction = F.cross_entropy(outputs["direction_logits"], dir_tgt, reduction="mean")
        total = total + float(direction_weight) * loss_direction
        metrics["loss_direction"] = loss_direction.detach()
    if target_speed is not None:
        spd_tgt = target_speed.to(pred_heatmap.device, dtype=pred_heatmap.dtype)
        loss_speed = F.smooth_l1_loss(outputs["speed"], spd_tgt, reduction="mean")
        total = total + float(speed_weight) * loss_speed
        metrics["loss_speed"] = loss_speed.detach()

    metrics["loss_total"] = total.detach()
    return total, metrics


def predict_single_vehicle_track(
    model: VehicleTraceNet,
    data_window: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[InferenceConfig | dict[str, Any]] = None,
    device: Optional[str] = None,
    return_prior: bool = False,
) -> list[Track] | tuple[list[Track], np.ndarray]:
    cfg = _as_inference_config(config)
    raw = np.asarray(data_window, dtype=np.float32)
    x = prepare_window_input(raw, int(cfg.time_downsample))
    resolved_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(resolved_device))
        obj_prob = float(torch.sigmoid(outputs["objectness_logits"][0]).item())
        heatmap_prob_ds = torch.sigmoid(outputs["heatmap_logits"][0]).detach()
        vis_prob = torch.sigmoid(outputs.get("visibility_logits", outputs["objectness_logits"].new_zeros((1, raw.shape[0])))[0]).detach()
        traj_pred = outputs.get("trajectory_time")
        traj_hint = traj_pred[0].detach().cpu().numpy() if traj_pred is not None else None
        line_pred = torch.sigmoid(outputs["line_endpoints"][0]).detach().cpu().numpy() if "line_endpoints" in outputs else None
        heatmap_peak = float(heatmap_prob_ds.max().item())
        heatmap_mean = float(heatmap_prob_ds.mean().item())
        heatmap_contrast = max(0.0, heatmap_peak - heatmap_mean)
        visibility_coverage = float((vis_prob > 0.5).to(torch.float32).mean().item())
        confidence = 0.45 * obj_prob + 0.25 * min(1.0, heatmap_contrast / max(1e-6, heatmap_peak + 1e-3)) + 0.30 * visibility_coverage
        effective_prior_weight = float(cfg.prior_weight)
        if confidence < max(float(cfg.low_confidence_objectness_threshold), 0.5 * float(cfg.objectness_threshold)):
            effective_prior_weight = float(cfg.low_confidence_prior_weight)
        if heatmap_peak >= float(cfg.low_confidence_heatmap_threshold):
            effective_prior_weight = max(float(effective_prior_weight), 1.15)
        prior = F.interpolate(
            heatmap_prob_ds.unsqueeze(0).unsqueeze(0),
            size=raw.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        prior_time_hint = None
        if traj_hint is not None:
            prior_time_hint = np.asarray(traj_hint, dtype=np.float32)
        elif line_pred is not None:
            prior_time_hint = np.linspace(float(line_pred[0]), float(line_pred[1]), int(raw.shape[0]), dtype=np.float32)

    direction_norm = str(direction).strip().lower()
    if direction_norm in {"auto", "both", "dual"}:
        predicted_dir = estimate_direction_index_from_outputs(outputs)
        predicted_direction = "forward" if predicted_dir == 0 else "reverse"
        direction_candidates = [predicted_direction, "reverse" if predicted_direction == "forward" else "forward"]
    else:
        direction_candidates = [direction_norm]

    def _decode(vmin: float, vmax: float, direction_name: str) -> list[Track]:
        return extract_single_vehicle_track(
            raw,
            float(fs),
            float(dx_m),
            str(direction_name),
            float(vmin),
            float(vmax),
            config=cfg.single_vehicle_tracker,
            prior_heatmap=prior,
            prior_weight=float(effective_prior_weight),
            prior_time_hint=prior_time_hint,
        )

    def _decode_raw(vmin: float, vmax: float, direction_name: str) -> list[Track]:
        return extract_single_vehicle_track(
            raw,
            float(fs),
            float(dx_m),
            str(direction_name),
            float(vmin),
            float(vmax),
            config=cfg.single_vehicle_tracker,
            prior_heatmap=None,
            prior_weight=0.0,
            prior_time_hint=None,
        )

    def _score(tracks: list[Track]) -> tuple[float, int]:
        if not tracks:
            return (float("-inf"), 0)
        track = tracks[0]
        return (
            float(track.total_score) + 2.0 * float(track.mean_speed_kmh if np.isfinite(track.mean_speed_kmh) else 0.0) / max(1.0, float(vmax_kmh)),
            int(len(track.points)),
        )

    candidates: list[tuple[tuple[float, int], list[Track]]] = []
    for direction_name in direction_candidates:
        tracks = _decode(float(vmin_kmh), float(vmax_kmh), direction_name)
        if not tracks:
            slack = max(30.0, float(cfg.single_vehicle_tracker.kalman_speed_gate_kmh))
            wide_vmin = 1.0
            wide_vmax = max(180.0, float(max(vmin_kmh, vmax_kmh)) + slack)
            if wide_vmax - wide_vmin >= 20.0:
                tracks = _decode(wide_vmin, wide_vmax, direction_name)
        if not tracks:
            tracks = _decode_raw(1.0, 180.0, direction_name)
        candidates.append((_score(tracks), tracks))

    candidates.sort(key=lambda item: item[0], reverse=True)
    tracks = candidates[0][1] if candidates else []
    if return_prior:
        return tracks, prior
    return tracks


def save_checkpoint(
    path: str | Path,
    model: VehicleTraceNet,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "vehicle_trace",
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    checkpoint_path = Path(path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, str(tmp_path))
    tmp_path.replace(checkpoint_path)


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[VehicleTraceNet, dict[str, Any]]:
    resolved_device = device or _auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = VehicleTraceNet(model_config).to(resolved_device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.has_trained_line_head = not any(str(key).startswith("line_head") for key in missing)
    model.has_trained_trajectory_head = not any(str(key).startswith("trajectory_head") for key in missing)
    if missing:
        print(f"VehicleTrace checkpoint loaded with newly initialized keys: {missing}", flush=True)
    if unexpected:
        print(f"VehicleTrace checkpoint ignored unexpected keys: {unexpected}", flush=True)
    model.eval()
    return model, checkpoint


__all__ = [
    "InferenceConfig",
    "ModelConfig",
    "SingleVehiclePeakNet",
    "SingleVehicleTrackerConfig",
    "VehicleTraceNet",
    "estimate_direction_index_from_outputs",
    "build_single_vehicle_heatmap_target",
    "build_single_vehicle_line_target",
    "build_single_vehicle_trajectory_target",
    "heatmap_to_time_prediction",
    "load_checkpoint_model",
    "prepare_window_input",
    "predict_single_vehicle_track",
    "save_checkpoint",
    "score_single_vehicle_window",
    "single_vehicle_heatmap_loss",
    "single_vehicle_supervised_loss",
]
