from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hybrid_vehicle_tracker.types import HoughSeed


@dataclass
class HoughOutput:
    logits: torch.Tensor
    raw_scores: torch.Tensor
    slopes: torch.Tensor
    intercepts: torch.Tensor
    valid_support: torch.Tensor


class PhysicalHoughHead(nn.Module):
    """Differentiable line pooling using physical station coordinates."""

    def __init__(
        self,
        feature_channels: int,
        *,
        slope_min: float = 0.04,
        slope_max: float = 0.06,
        slope_bins: int = 17,
        intercept_step_s: float = 0.5,
        motion_direction: int = 1,
    ) -> None:
        super().__init__()
        if int(motion_direction) not in (-1, 1):
            raise ValueError("motion_direction must be +1 or -1")
        self.feature_projection = nn.Conv2d(feature_channels, 1, kernel_size=1)
        self.motion_direction = int(motion_direction)
        # Store positive magnitudes in the checkpoint for compatibility with
        # the first project version.  The signed values are materialized in
        # forward(), so an old checkpoint can be used with the corrected
        # DAY11 direction without rewriting its state dict.
        self.register_buffer("slopes", torch.linspace(slope_min, slope_max, slope_bins))
        self.intercept_step_s = float(intercept_step_s)
        self.score_mlp = nn.Sequential(
            nn.Linear(6, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def _intercepts(
        self, positions_m: torch.Tensor, duration_s: float, device: torch.device
    ) -> torch.Tensor:
        span = float(positions_m[-1] - positions_m[0])
        signed = self.motion_direction * self.slopes
        # Include every intercept for which a line with any allowed slope can
        # intersect the [0, duration] window.  The old formula only worked for
        # positive slopes and silently removed all reverse-direction DAY11
        # lines.
        low_shift = float(torch.min(-signed * span).item())
        high_shift = float(torch.max(-signed * span).item())
        lower = min(0.0, low_shift)
        upper = max(float(duration_s), float(duration_s) + high_shift)
        count = max(2, int(np.ceil((upper - lower) / self.intercept_step_s)) + 1)
        return torch.linspace(lower, upper, count, device=device)

    def forward(
        self,
        feature_map: torch.Tensor,
        positions_m: torch.Tensor,
        *,
        duration_s: float,
        feature_rate_hz: float,
        evidence_map: torch.Tensor | None = None,
    ) -> HoughOutput:
        if evidence_map is None:
            evidence = torch.sigmoid(self.feature_projection(feature_map))
        else:
            evidence = evidence_map
            if evidence.ndim == 3:
                evidence = evidence[:, None]
        # Physical station locations are around 83,500 m and overflow float16.
        # Keep the complete Hough accumulation in float32 under AMP.
        evidence = evidence.float()
        if evidence.shape[-2] != positions_m.numel():
            raise ValueError("physical position count does not match station dimension")

        device = evidence.device
        dtype = torch.float32
        positions = positions_m.to(device=device, dtype=torch.float32)
        positions = positions - positions[0]
        slopes = self.motion_direction * self.slopes.to(device=device, dtype=torch.float32)
        intercepts = self._intercepts(positions, duration_s, device).to(dtype=dtype)
        slope_grid, intercept_grid = torch.meshgrid(slopes, intercepts, indexing="ij")
        flat_slopes = slope_grid.reshape(-1)
        flat_intercepts = intercept_grid.reshape(-1)
        sample_times = flat_intercepts[:, None] + flat_slopes[:, None] * positions[None, :]
        max_grid_time = max(duration_s - 1.0 / feature_rate_hz, 1.0 / feature_rate_hz)
        time_coordinate = 2.0 * sample_times / max_grid_time - 1.0
        station_coordinate = torch.linspace(-1.0, 1.0, positions.numel(), device=device, dtype=dtype)
        station_coordinate = station_coordinate[None, :].expand_as(time_coordinate)
        grid = torch.stack([time_coordinate, station_coordinate], dim=-1)
        grid = grid[None].expand(evidence.shape[0], -1, -1, -1)

        valid = (sample_times >= 0.0) & (sample_times <= max_grid_time)
        sampled = F.grid_sample(
            evidence,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, 0]
        valid_float = valid.to(dtype=dtype)
        support = valid_float.sum(dim=-1).clamp_min(1.0)
        pooled_mean = (sampled * valid_float[None]).sum(dim=-1) / support[None]
        masked = sampled.masked_fill(~valid[None], -1.0)
        pooled_max = masked.max(dim=-1).values.clamp_min(0.0)
        masked_zero = sampled * valid_float[None]
        second_moment = (masked_zero.square().sum(dim=-1) / support[None]).clamp_min(0.0)
        # The epsilon prevents an infinite sqrt gradient on perfectly flat lines.
        pooled_std = (second_moment - pooled_mean.square()).clamp_min(1e-8).sqrt()
        active_fraction = (
            ((sampled >= 0.45) & valid[None]).to(dtype=dtype).sum(dim=-1) / support[None]
        )
        top_count = max(1, int(np.ceil(0.25 * positions.numel())))
        top_values = torch.topk(masked, k=min(top_count, masked.shape[-1]), dim=-1).values
        top_values = top_values.clamp_min(0.0)
        top_mean = top_values.mean(dim=-1)
        coverage = support / float(positions.numel())
        raw_scores = (
            0.40 * pooled_mean + 0.30 * top_mean + 0.30 * active_fraction
        ) * torch.sqrt(coverage[None])
        descriptors = torch.stack(
            [
                pooled_mean,
                pooled_max,
                pooled_std,
                active_fraction,
                top_mean,
                coverage[None].expand_as(pooled_mean),
            ],
            dim=-1,
        )
        logits = self.score_mlp(descriptors).squeeze(-1)
        shape = (evidence.shape[0], slopes.numel(), intercepts.numel())
        return HoughOutput(
            logits=logits.reshape(shape),
            raw_scores=raw_scores.reshape(shape),
            slopes=slopes,
            intercepts=intercepts,
            valid_support=support.reshape(slopes.numel(), intercepts.numel()),
        )

    @staticmethod
    def topk_seeds(
        output: HoughOutput,
        *,
        top_k: int,
        learned: bool,
        min_support: int = 5,
    ) -> list[HoughSeed]:
        if learned:
            learned_scores = torch.sigmoid(output.logits[0])
            score_map = (0.55 * learned_scores + 0.45 * output.raw_scores[0]).detach().cpu().numpy()
        else:
            score_map = output.raw_scores[0].detach().cpu().numpy()
        support = output.valid_support.detach().cpu().numpy()
        slopes = output.slopes.detach().cpu().numpy()
        intercepts = output.intercepts.detach().cpu().numpy()
        flat_order = np.argsort(score_map.ravel())[::-1]
        selected: list[tuple[int, int]] = []
        seeds: list[HoughSeed] = []
        for flat_index in flat_order:
            slope_index, intercept_index = np.unravel_index(flat_index, score_map.shape)
            if support[slope_index, intercept_index] < min_support:
                continue
            if any(
                abs(slope_index - old_slope) <= 1
                and abs(intercept_index - old_intercept) <= 2
                for old_slope, old_intercept in selected
            ):
                continue
            selected.append((slope_index, intercept_index))
            seeds.append(
                HoughSeed(
                    slope_s_per_m=float(slopes[slope_index]),
                    intercept_s=float(intercepts[intercept_index]),
                    score=float(score_map[slope_index, intercept_index]),
                    support=int(round(support[slope_index, intercept_index])),
                    source="deep_hough" if learned else "physical_hough",
                )
            )
            if len(seeds) >= top_k:
                break
        return seeds
