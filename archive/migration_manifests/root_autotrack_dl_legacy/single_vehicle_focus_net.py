"""Single-vehicle focus network.

This is a fresh single-track design:

- the network predicts a target mask prior and a competitor suppression mask
- it also predicts per-channel visibility, coarse trajectory time, direction,
  speed, and objectness
- the final decoder is still the classic graph/Kalman/Hungarian stack, but it
  now receives a learned single-vehicle prior instead of rediscovering the
  whole path from raw peaks

The intent is to replace the older slot-style routes for the one-vehicle
identification task.
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
from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.single_vehicle_net import (
    build_single_vehicle_heatmap_target,
    build_single_vehicle_line_target,
    build_single_vehicle_trajectory_target,
    prepare_window_input,
)
from autotrack.dl.trajectory_energy_model import ConvBlock, UpBlock, _smooth_track_with_kalman


@dataclass
class FocusModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    hidden_dim: int = 64
    pooled_channels: int = 16
    dropout: float = 0.08
    heatmap_sigma_t: float = 2.0


@dataclass
class FocusInferenceConfig:
    time_downsample: int = 10
    min_visible_channels: int = 4
    objectness_threshold: float = 0.35
    visibility_threshold: float = 0.5
    prior_weight: float = 1.0
    competitor_weight: float = 0.9
    single_vehicle_tracker: SingleVehicleTrackerConfig = field(default_factory=SingleVehicleTrackerConfig)


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


def _as_inference_config(config: Optional[FocusInferenceConfig | dict[str, Any]]) -> FocusInferenceConfig:
    if config is None:
        return FocusInferenceConfig()
    if isinstance(config, FocusInferenceConfig):
        return config
    if isinstance(config, dict):
        defaults = FocusInferenceConfig()
        tracker_cfg = config.get("single_vehicle_tracker", None)
        if isinstance(tracker_cfg, SingleVehicleTrackerConfig):
            tracker = tracker_cfg
        elif isinstance(tracker_cfg, dict):
            tracker = SingleVehicleTrackerConfig(**tracker_cfg)
        else:
            tracker = defaults.single_vehicle_tracker
        return FocusInferenceConfig(
            time_downsample=int(config.get("time_downsample", defaults.time_downsample)),
            min_visible_channels=int(config.get("min_visible_channels", defaults.min_visible_channels)),
            objectness_threshold=float(config.get("objectness_threshold", defaults.objectness_threshold)),
            visibility_threshold=float(config.get("visibility_threshold", defaults.visibility_threshold)),
            prior_weight=float(config.get("prior_weight", defaults.prior_weight)),
            competitor_weight=float(config.get("competitor_weight", defaults.competitor_weight)),
            single_vehicle_tracker=tracker,
        )
    raise TypeError("config must be FocusInferenceConfig / dict / None")


class SingleVehicleFocusNet(nn.Module):
    def __init__(self, config: Optional[FocusModelConfig] = None):
        super().__init__()
        self.config = config or FocusModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        pooled = int(max(8, c.pooled_channels))
        self.enc1 = ConvBlock(int(c.in_channels), 32)
        self.enc2 = ConvBlock(32, 48, stride=(1, 2))
        self.enc3 = ConvBlock(48, hidden, stride=(2, 2))
        self.bottleneck = ConvBlock(hidden, hidden)
        self.up2 = UpBlock(hidden, 48, 48)
        self.up1 = UpBlock(48, 32, 32)
        feature_channels = 32
        self.feature_head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1, bias=False),
            nn.GroupNorm(max(1, min(8, feature_channels // 4)), feature_channels),
            nn.GELU(),
        )
        self.target_mask_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.competitor_mask_head = nn.Conv2d(feature_channels, 1, kernel_size=1)
        channel_latent_dim = feature_channels * 2
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
        self.objectness_head = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.speed_head = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.line_head = nn.Sequential(
            nn.Linear(feature_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self._heatmap_sigma_t = float(max(1e-3, c.heatmap_sigma_t))

    def _channel_latent(self, feat: torch.Tensor, n_channels: int) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(feat, output_size=(n_channels, 1)).squeeze(-1)
        mx = F.adaptive_max_pool2d(feat, output_size=(n_channels, 1)).squeeze(-1)
        return torch.cat([avg, mx], dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, in_channels, channels, time]")
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        feat = self.bottleneck(enc3)
        feat = self.up2(feat, enc2)
        feat = self.up1(feat, enc1)
        feat = self.feature_head(feat)

        target_mask_logits = self.target_mask_head(feat).squeeze(1)
        competitor_mask_logits = self.competitor_mask_head(feat).squeeze(1)

        batch, _, n_channels, _ = feat.shape
        channel_latent = self._channel_latent(feat, int(n_channels))
        channel_latent = self.channel_proj(channel_latent).transpose(1, 2).contiguous()
        global_feat = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)

        visibility_logits = self.visibility_head(channel_latent).squeeze(-1)
        trajectory_time = torch.sigmoid(self.trajectory_head(channel_latent).squeeze(-1))
        objectness_logits = self.objectness_head(global_feat).squeeze(-1)
        direction_logits = self.direction_head(global_feat)
        speed = self.speed_head(global_feat).squeeze(-1)
        line_endpoints = torch.sigmoid(self.line_head(global_feat))

        time_bins = int(x.shape[-1])
        time_axis = torch.linspace(0.0, 1.0, time_bins, device=x.device, dtype=trajectory_time.dtype).view(1, 1, -1)
        sigma = float(self._heatmap_sigma_t) / float(max(1, time_bins - 1))
        sigma = max(sigma, 1e-3)
        vis_prob = torch.sigmoid(visibility_logits).unsqueeze(-1)
        dist = (time_axis - trajectory_time.unsqueeze(-1)) / sigma
        target_prior_prob = vis_prob * torch.exp(-0.5 * dist.square())
        target_prior_logits = torch.logit(target_prior_prob.clamp(1e-4, 1.0 - 1e-4))

        return {
            "target_mask_logits": target_mask_logits,
            "competitor_mask_logits": competitor_mask_logits,
            "target_prior_logits": target_prior_logits,
            "visibility_logits": visibility_logits,
            "trajectory_time": trajectory_time,
            "objectness_logits": objectness_logits,
            "direction_logits": direction_logits,
            "speed": speed,
            "line_endpoints": line_endpoints,
        }


def build_focus_targets(
    time: torch.Tensor,
    visibility: torch.Tensor,
    *,
    n_channels: int,
    time_bins: int,
    sigma_ch: float = 0.8,
    sigma_t: float = 2.0,
    competitor_time: Optional[torch.Tensor] = None,
    competitor_visibility: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_mask = build_single_vehicle_heatmap_target(
        time,
        visibility,
        n_channels=int(n_channels),
        time_bins=int(time_bins),
        sigma_ch=float(sigma_ch),
        sigma_t=float(sigma_t),
    )
    competitor_mask = torch.zeros_like(target_mask)
    if competitor_time is not None and competitor_visibility is not None:
        if bool(torch.as_tensor(competitor_visibility).max().item() > 0.0):
            competitor_mask = build_single_vehicle_heatmap_target(
                competitor_time,
                competitor_visibility,
                n_channels=int(n_channels),
                time_bins=int(time_bins),
                sigma_ch=float(sigma_ch),
                sigma_t=float(sigma_t),
            )
    line = build_single_vehicle_line_target(time, visibility)
    trajectory = build_single_vehicle_trajectory_target(time, visibility)
    return target_mask, competitor_mask, line, trajectory


def single_vehicle_focus_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_mask: torch.Tensor,
    target_visibility: torch.Tensor,
    target_time: torch.Tensor,
    target_objectness: Optional[torch.Tensor] = None,
    target_direction: Optional[torch.Tensor] = None,
    target_speed: Optional[torch.Tensor] = None,
    competitor_mask: Optional[torch.Tensor] = None,
    target_line: Optional[torch.Tensor] = None,
    target_trajectory: Optional[torch.Tensor] = None,
    target_weight: float = 1.0,
    competitor_weight: float = 0.9,
    visibility_weight: float = 0.5,
    time_weight: float = 1.0,
    objectness_weight: float = 0.25,
    direction_weight: float = 0.2,
    speed_weight: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_target = outputs["target_mask_logits"]
    pred_competitor = outputs["competitor_mask_logits"]
    tgt_mask = target_mask.to(pred_target.device, dtype=pred_target.dtype)
    vis_tgt = target_visibility.to(pred_target.device, dtype=pred_target.dtype)
    time_tgt = target_time.to(pred_target.device, dtype=pred_target.dtype)

    loss_target_bce = F.binary_cross_entropy_with_logits(pred_target, tgt_mask, reduction="mean")
    target_prob = torch.sigmoid(pred_target)
    dims = tuple(range(1, target_prob.ndim))
    inter = torch.sum(target_prob * tgt_mask, dim=dims)
    denom = torch.sum(target_prob, dim=dims) + torch.sum(tgt_mask, dim=dims)
    loss_target_dice = torch.mean(1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6))

    if competitor_mask is not None:
        comp_tgt = competitor_mask.to(pred_target.device, dtype=pred_target.dtype)
        loss_competitor = F.binary_cross_entropy_with_logits(pred_competitor, comp_tgt, reduction="mean")
    else:
        loss_competitor = pred_competitor.sum() * 0.0

    loss_visibility = F.binary_cross_entropy_with_logits(outputs["visibility_logits"], vis_tgt, reduction="mean")
    pred_time = outputs["trajectory_time"].to(pred_target.device, dtype=pred_target.dtype)
    visible = vis_tgt > 0.5
    if bool(visible.any()):
        loss_time = F.smooth_l1_loss(pred_time[visible], time_tgt[visible], reduction="mean")
    else:
        loss_time = pred_time.sum() * 0.0

    loss_line = pred_target.sum() * 0.0
    if target_line is not None:
        line_tgt = target_line.to(pred_target.device, dtype=pred_target.dtype)
        loss_line = F.smooth_l1_loss(outputs["line_endpoints"], line_tgt, reduction="mean")

    loss_trajectory = pred_target.sum() * 0.0
    if target_trajectory is not None:
        traj_tgt = target_trajectory.to(pred_target.device, dtype=pred_target.dtype)
        loss_trajectory = F.smooth_l1_loss(pred_time[visible], traj_tgt[visible], reduction="mean") if bool(visible.any()) else pred_target.sum() * 0.0

    total = (
        float(target_weight) * (loss_target_bce + loss_target_dice)
        + float(competitor_weight) * loss_competitor
        + float(visibility_weight) * loss_visibility
        + float(time_weight) * loss_time
        + 0.55 * loss_line
        + 1.25 * loss_trajectory
    )

    if target_objectness is not None:
        obj_tgt = target_objectness.to(pred_target.device, dtype=pred_target.dtype)
        loss_objectness = F.binary_cross_entropy_with_logits(outputs["objectness_logits"], obj_tgt, reduction="mean")
        total = total + float(objectness_weight) * loss_objectness
    else:
        loss_objectness = pred_target.sum() * 0.0

    if target_direction is not None:
        dir_tgt = target_direction.to(pred_target.device, dtype=torch.long)
        loss_direction = F.cross_entropy(outputs["direction_logits"], dir_tgt, reduction="mean")
        total = total + float(direction_weight) * loss_direction
    else:
        loss_direction = pred_target.sum() * 0.0

    if target_speed is not None:
        spd_tgt = target_speed.to(pred_target.device, dtype=pred_target.dtype)
        loss_speed = F.smooth_l1_loss(outputs["speed"], spd_tgt, reduction="mean")
        total = total + float(speed_weight) * loss_speed
    else:
        loss_speed = pred_target.sum() * 0.0

    metrics = {
        "loss_target_bce": loss_target_bce.detach(),
        "loss_target_dice": loss_target_dice.detach(),
        "loss_competitor": loss_competitor.detach(),
        "loss_visibility": loss_visibility.detach(),
        "loss_time": loss_time.detach(),
        "loss_line": loss_line.detach(),
        "loss_trajectory": loss_trajectory.detach(),
        "loss_objectness": loss_objectness.detach(),
        "loss_direction": loss_direction.detach(),
        "loss_speed": loss_speed.detach(),
        "loss_total": total.detach(),
    }
    return total, metrics


def predict_single_vehicle_focus_track(
    model: SingleVehicleFocusNet,
    data_window: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[FocusInferenceConfig | dict[str, Any]] = None,
    device: Optional[str] = None,
) -> list[Track]:
    cfg = _as_inference_config(config)
    raw = np.asarray(data_window, dtype=np.float32)
    x = prepare_window_input(raw, int(cfg.time_downsample))
    resolved_device = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(resolved_device))
        target_prior = torch.sigmoid(outputs["target_mask_logits"][0]).detach()
        competitor_prior = torch.sigmoid(outputs["competitor_mask_logits"][0]).detach()
        target_prior = torch.clamp(target_prior - float(cfg.competitor_weight) * competitor_prior, min=0.0)
        prior = F.interpolate(
            target_prior.unsqueeze(0).unsqueeze(0),
            size=raw.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        prior_time_hint = outputs["trajectory_time"][0].detach().cpu().numpy()
        prior_weight = float(cfg.prior_weight)
        obj_prob = float(torch.sigmoid(outputs["objectness_logits"][0]).item())
        if obj_prob < float(cfg.objectness_threshold):
            prior_weight *= 0.5

    direction_norm = str(direction).strip().lower()
    if direction_norm in {"auto", "both", "dual"}:
        direction_name = "forward" if int(torch.argmax(outputs["direction_logits"][0]).item()) == 0 else "reverse"
    else:
        direction_name = direction_norm
    tracks = extract_single_vehicle_track(
        raw,
        float(fs),
        float(dx_m),
        str(direction_name),
        float(vmin_kmh),
        float(vmax_kmh),
        config=cfg.single_vehicle_tracker,
        prior_heatmap=prior,
        prior_weight=float(prior_weight),
        prior_time_hint=prior_time_hint,
    )
    if not tracks:
        tracks = extract_single_vehicle_track(
            raw,
            float(fs),
            float(dx_m),
            str(direction_name),
            1.0,
            max(180.0, float(vmax_kmh)),
            config=cfg.single_vehicle_tracker,
            prior_heatmap=None,
            prior_weight=0.0,
            prior_time_hint=None,
        )
    if cfg.single_vehicle_tracker.kalman_fill_missing and tracks:
        tracks = [_smooth_track_with_kalman(tracks[0], n_time_bins=int(raw.shape[1]))]
    return tracks


def save_checkpoint(
    path: str | Path,
    model: SingleVehicleFocusNet,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: FocusModelConfig,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "single_vehicle_focus",
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


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[SingleVehicleFocusNet, dict[str, Any]]:
    resolved_device = device or _auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = FocusModelConfig(**dict(checkpoint.get("model_config", {})))
    model = SingleVehicleFocusNet(model_config).to(resolved_device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"SingleVehicleFocusNet checkpoint loaded with newly initialized keys: {missing}", flush=True)
    if unexpected:
        print(f"SingleVehicleFocusNet checkpoint ignored unexpected keys: {unexpected}", flush=True)
    model.eval()
    return model, checkpoint
