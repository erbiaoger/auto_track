from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        temporal_dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = (1, temporal_dilation)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=(1, temporal_dilation),
            bias=False,
        )
        self.norm1 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=padding,
            dilation=(1, temporal_dilation),
            bias=False,
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activation = nn.SiLU(inplace=True)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.activation(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return self.activation(hidden + residual)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([inputs, skip], dim=1))


class MultiModalResUNet(nn.Module):
    """Small CPU-capable U-Net for 50-station DAS feature images."""

    def __init__(
        self,
        *,
        in_channels: int = 5,
        base_channels: int = 12,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.stem = ResidualBlock(in_channels, base_channels)
        self.encoder1 = ResidualBlock(base_channels, base_channels * 2, stride=2)
        self.encoder2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2)
        self.context = nn.Sequential(
            ResidualBlock(base_channels * 4, base_channels * 4, temporal_dilation=2),
            ResidualBlock(base_channels * 4, base_channels * 4, temporal_dilation=4),
            ResidualBlock(base_channels * 4, base_channels * 4, temporal_dilation=8),
            ResidualBlock(base_channels * 4, base_channels * 4, temporal_dilation=16),
        )
        self.decoder1 = DecoderBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder2 = DecoderBlock(base_channels * 2, base_channels, base_channels)
        self.centerline_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.slowness_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.crossing_head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.embedding_head = nn.Conv2d(base_channels, embedding_dim, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 4:
            raise ValueError(f"expected [batch, feature, station, time], got {inputs.shape}")
        skip0 = self.stem(inputs)
        skip1 = self.encoder1(skip0)
        hidden = self.context(self.encoder2(skip1))
        hidden = self.decoder1(hidden, skip1)
        feature_map = self.decoder2(hidden, skip0)
        embedding = F.normalize(self.embedding_head(feature_map), p=2, dim=1, eps=1e-6)
        return {
            "centerline_logits": self.centerline_head(feature_map),
            "slowness": torch.sigmoid(self.slowness_head(feature_map)),
            "crossing_logits": self.crossing_head(feature_map),
            "embedding": embedding,
            "feature_map": feature_map,
        }
