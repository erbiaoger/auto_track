"""PeakLineNet semantic trajectory-line segmentation model.

Purpose:
    Convert a sparse channel-time peak-point image into a continuous semantic
    trajectory-line probability map. This model predicts "there is a vehicle
    trajectory here" and intentionally does not assign vehicle IDs.

Example:
    uv run python -m autotrack.dl.train_peak_line \
        --data-dir datasets/peak_line/train \
        --out-dir models/peak_line_cuda \
        --device cuda \
        --amp on

Outputs:
    line_logits [B, 1, C, T_down]
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from autotrack.dl.trajectory_set_model import WindowDatasetConfig, auto_torch_device


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 8
    dropout: float = 0.05


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: tuple[int, int] = (1, 1), dropout: float = 0.0):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, *, dropout: float = 0.0):
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class PeakLineNet(nn.Module):
    """Lightweight U-Net for sparse peak image to semantic line map."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        base = int(c.base_channels)
        dropout = float(c.dropout)
        self.enc1 = ConvBlock(int(c.in_channels), base, dropout=dropout)
        self.enc2 = ConvBlock(base, base * 2, stride=(1, 2), dropout=dropout)
        self.enc3 = ConvBlock(base * 2, base * 4, stride=(2, 2), dropout=dropout)
        self.enc4 = ConvBlock(base * 4, base * 8, stride=(2, 2), dropout=dropout)
        self.mid = ConvBlock(base * 8, base * 8, dropout=dropout)
        self.up3 = UpBlock(base * 8, base * 4, base * 4, dropout=dropout)
        self.up2 = UpBlock(base * 4, base * 2, base * 2, dropout=dropout)
        self.up1 = UpBlock(base * 2, base, base, dropout=dropout)
        self.head = nn.Conv2d(base, int(c.out_channels), kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        mid = self.mid(e4)
        d3 = self.up3(mid, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        logits = self.head(d1)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"line_logits": logits}


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob, dim=dims) + torch.sum(target, dim=dims)
    return torch.mean(1.0 - (2.0 * inter + float(eps)) / (denom + float(eps)))


def focal_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * target + (1.0 - prob) * (1.0 - target)
    return torch.mean(torch.pow(1.0 - pt, float(gamma)) * bce)


def peak_line_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    pos_weight: float = 20.0,
    dice_weight: float = 1.0,
    focal_weight: float = 0.25,
    focal_gamma: float = 2.0,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["line_logits"]
    target = targets["line_mask"].to(dtype=logits.dtype)
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    loss_dice = dice_loss_from_logits(logits, target)
    loss_focal = focal_loss_from_logits(logits, target, gamma=float(focal_gamma))
    loss = loss_bce + float(dice_weight) * loss_dice + float(focal_weight) * loss_focal
    metrics = peak_line_metrics(outputs, targets, threshold=float(threshold))
    metrics.update(
        {
            "loss": float(loss.detach().cpu().item()),
            "loss_bce": float(loss_bce.detach().cpu().item()),
            "loss_dice": float(loss_dice.detach().cpu().item()),
            "loss_focal": float(loss_focal.detach().cpu().item()),
        }
    )
    return loss, metrics


def peak_line_metrics(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], *, threshold: float = 0.5) -> dict[str, float]:
    logits = outputs["line_logits"]
    target = targets["line_mask"].to(dtype=torch.float32) >= 0.5
    pred = torch.sigmoid(logits) >= float(threshold)
    tp = torch.logical_and(pred, target).sum().to(torch.float32)
    fp = torch.logical_and(pred, torch.logical_not(target)).sum().to(torch.float32)
    fn = torch.logical_and(torch.logical_not(pred), target).sum().to(torch.float32)
    eps = torch.tensor(1e-6, device=logits.device, dtype=torch.float32)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    prob = torch.sigmoid(logits)
    return {
        "pixel_precision": float(precision.detach().cpu().item()),
        "pixel_recall": float(recall.detach().cpu().item()),
        "pixel_f1": float(f1.detach().cpu().item()),
        "pixel_iou": float(iou.detach().cpu().item()),
        "target_positive_frac": float(target.to(torch.float32).mean().detach().cpu().item()),
        "pred_positive_frac": float(pred.to(torch.float32).mean().detach().cpu().item()),
        "mean_probability": float(prob.mean().detach().cpu().item()),
        "max_probability": float(prob.max().detach().cpu().item()),
    }


def save_checkpoint(
    path: str | Path,
    model: PeakLineNet,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    dataset_config: WindowDatasetConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_family": "peak_line",
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


def load_checkpoint_model(checkpoint_path: str | Path, device: Optional[str] = None) -> tuple[PeakLineNet, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = PeakLineNet(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, checkpoint
