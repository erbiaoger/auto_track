from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from hybrid_vehicle_tracker.config import ModelConfig
from hybrid_vehicle_tracker.models.edge_gnn import EdgeAssociationGNN
from hybrid_vehicle_tracker.models.physical_hough import PhysicalHoughHead
from hybrid_vehicle_tracker.models.resunet import MultiModalResUNet


class HybridPerceptionModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.perception = MultiModalResUNet(
            base_channels=config.base_channels,
            embedding_dim=config.embedding_dim,
        )
        self.hough = PhysicalHoughHead(
            config.base_channels,
            slope_bins=config.hough_slopes,
            intercept_step_s=config.hough_intercept_step_s,
            motion_direction=config.motion_direction,
        )
        self.edge_gnn = EdgeAssociationGNN(
            node_features=9 + config.embedding_dim,
            edge_features=9,
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.perception(inputs)

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> None:
        checkpoint = {
            "state_dict": self.state_dict(),
            "model_config": vars(self.config),
            "metadata": metadata,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, Path(path))

    def load_checkpoint(self, path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.load_state_dict(state_dict, strict=True)
        return dict(checkpoint.get("metadata", {}))
