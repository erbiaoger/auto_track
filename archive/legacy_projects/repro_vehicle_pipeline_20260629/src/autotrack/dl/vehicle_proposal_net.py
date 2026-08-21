"""Dense multi-vehicle proposal network.

This is the new full-segment front-end for the track reconstruction pipeline.
It predicts a dense vehicle-likelihood heatmap over the whole window, plus a
coarse objectness score for window ranking. Trajectory extraction is left to the
graph/Kalman/Hungarian post-processing stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from autotrack.dl.single_vehicle_net import _auto_torch_device
from autotrack.dl.trajectory_energy_model import ConvBlock, UpBlock


@dataclass
class ProposalModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    hidden_dim: int = 64
    dropout: float = 0.1


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
) -> torch.Tensor:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = _robust_scale(arr_ds)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    return torch.from_numpy(raw[None, :, :].astype(np.float32, copy=False))


class VehicleProposalNet(nn.Module):
    def __init__(self, config: Optional[ProposalModelConfig] = None):
        super().__init__()
        self.config = config or ProposalModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.enc1 = ConvBlock(int(c.in_channels), 32)
        self.enc2 = ConvBlock(32, 64, stride=(1, 2))
        self.enc3 = ConvBlock(64, hidden, stride=(2, 2))
        self.bottleneck = ConvBlock(hidden, hidden)
        self.up2 = UpBlock(hidden, 64, 64)
        self.up1 = UpBlock(64, 32, 32)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(32, max(1, hidden // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(1, hidden // 2), 1, kernel_size=1),
        )
        self.objectness_head = nn.Sequential(
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        feat = self.bottleneck(enc3)
        feat = self.up2(feat, enc2)
        feat = self.up1(feat, enc1)
        heatmap_logits = self.heatmap_head(feat).squeeze(1)
        pooled = F.adaptive_avg_pool2d(feat, output_size=(1, 1)).flatten(1)
        return {
            "heatmap_logits": heatmap_logits,
            "objectness_logits": self.objectness_head(pooled).squeeze(-1),
        }


def build_multi_vehicle_heatmap_target(gt_masks: torch.Tensor) -> torch.Tensor:
    masks = gt_masks.to(torch.float32)
    if masks.ndim != 3:
        raise ValueError("gt_masks must have shape [n_vehicle, n_channel, n_time]")
    if masks.shape[0] == 0:
        return torch.zeros((masks.shape[1], masks.shape[2]), dtype=torch.float32, device=masks.device)
    return torch.amax(masks, dim=0).clamp(0.0, 1.0)


def build_multi_vehicle_objectness_target(gt_masks: torch.Tensor) -> torch.Tensor:
    return torch.tensor(1.0 if int(gt_masks.shape[0]) > 0 else 0.0, dtype=torch.float32, device=gt_masks.device)


def proposal_heatmap_loss(
    outputs: dict[str, torch.Tensor],
    target_heatmap: torch.Tensor,
    *,
    dice_weight: float = 0.25,
    positive_weight: float | None = None,
) -> torch.Tensor:
    pred = outputs["heatmap_logits"]
    tgt = target_heatmap.to(pred.device, dtype=pred.dtype)
    bce = F.binary_cross_entropy_with_logits(pred, tgt, reduction="none")
    if positive_weight is None or float(positive_weight) <= 0.0:
        dims = tuple(range(1, tgt.ndim)) if tgt.ndim > 1 else ()
        pos_ratio = tgt.mean(dim=dims, keepdim=True).clamp(1e-4, 0.5) if dims else tgt.mean().clamp(1e-4, 0.5)
        pos_weight = ((1.0 - pos_ratio) / pos_ratio).clamp(4.0, 32.0)
    else:
        pos_weight = torch.full_like(tgt, float(max(1e-6, positive_weight)))
    weight = torch.where(tgt > 0.5, pos_weight, torch.ones_like(tgt))
    bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
    prob = torch.sigmoid(pred)
    dims = tuple(range(1, prob.ndim))
    inter = torch.sum(prob * tgt, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(tgt, dim=dims)
    dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
    return bce + float(max(0.0, dice_weight)) * torch.mean(dice)


def vehicle_proposal_loss(
    outputs: dict[str, torch.Tensor],
    *,
    target_heatmap: torch.Tensor,
    target_objectness: Optional[torch.Tensor] = None,
    heatmap_weight: float = 1.0,
    dice_weight: float = 0.25,
    objectness_weight: float = 0.2,
    heatmap_positive_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss_heatmap = proposal_heatmap_loss(
        outputs,
        target_heatmap,
        dice_weight=float(dice_weight),
        positive_weight=heatmap_positive_weight,
    )
    total = float(heatmap_weight) * loss_heatmap
    metrics: dict[str, torch.Tensor] = {"loss_heatmap": loss_heatmap.detach()}
    if target_objectness is not None:
        obj_tgt = target_objectness.to(outputs["objectness_logits"].device, dtype=outputs["objectness_logits"].dtype)
        loss_objectness = F.binary_cross_entropy_with_logits(outputs["objectness_logits"], obj_tgt, reduction="mean")
        total = total + float(objectness_weight) * loss_objectness
        metrics["loss_objectness"] = loss_objectness.detach()
    metrics["loss_total"] = total.detach()
    return total, metrics


def score_vehicle_proposal_window(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    obj_prob = torch.sigmoid(outputs["objectness_logits"].detach())
    heatmap_prob = torch.sigmoid(outputs["heatmap_logits"].detach())
    if obj_prob.ndim == 0:
        obj_prob = obj_prob.unsqueeze(0)
    heatmap_peak = float(heatmap_prob.max().item())
    heatmap_mean = float(heatmap_prob.mean().item())
    heatmap_contrast = max(0.0, heatmap_peak - heatmap_mean)
    confidence = 0.7 * float(obj_prob.item()) + 0.3 * min(1.0, heatmap_contrast / max(1e-6, heatmap_peak + 1e-3))
    return {
        "confidence": float(confidence),
        "objectness": float(obj_prob.item()),
        "heatmap_peak": float(heatmap_peak),
        "heatmap_mean": float(heatmap_mean),
        "heatmap_contrast": float(heatmap_contrast),
    }


def save_checkpoint(
    path: str | Path,
    model: VehicleProposalNet,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ProposalModelConfig,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "vehicle_proposal",
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


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[VehicleProposalNet, dict[str, Any]]:
    resolved_device = device or _auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ProposalModelConfig(**dict(checkpoint.get("model_config", {})))
    model = VehicleProposalNet(model_config).to(resolved_device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        print(f"VehicleProposalNet checkpoint loaded with newly initialized keys: {missing}", flush=True)
    if unexpected:
        print(f"VehicleProposalNet checkpoint ignored unexpected keys: {unexpected}", flush=True)
    model.eval()
    return model, checkpoint
