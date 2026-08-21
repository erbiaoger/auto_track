from __future__ import annotations

import torch
from torch import nn


class _MLP(nn.Sequential):
    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, out_features),
        )


class EdgeAssociationGNN(nn.Module):
    """Two-hop message-passing network for physics-gated peak graphs."""

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        *,
        hidden_dim: int = 64,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()
        self.node_encoder = _MLP(node_features, hidden_dim, hidden_dim)
        self.edge_encoder = _MLP(edge_features, hidden_dim, hidden_dim)
        self.edge_updates = nn.ModuleList(
            [_MLP(hidden_dim * 3, hidden_dim, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.node_updates = nn.ModuleList(
            [_MLP(hidden_dim * 2, hidden_dim, hidden_dim) for _ in range(message_passing_steps)]
        )
        self.edge_classifier = _MLP(hidden_dim * 3, hidden_dim, 1)
        self.merge_classifier = _MLP(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes = self.node_encoder(node_features)
        if edge_index.numel() == 0:
            empty = edge_features.new_empty((0,))
            return empty, self.merge_classifier(nodes).squeeze(-1)
        edges = self.edge_encoder(edge_features)
        source, target = edge_index[0], edge_index[1]
        for edge_update, node_update in zip(self.edge_updates, self.node_updates):
            edges = edges + edge_update(torch.cat([nodes[source], nodes[target], edges], dim=-1))
            aggregated = torch.zeros_like(nodes)
            aggregated.index_add_(0, source, edges)
            aggregated.index_add_(0, target, edges)
            degree = torch.zeros(nodes.shape[0], device=nodes.device, dtype=nodes.dtype)
            ones = torch.ones(edges.shape[0], device=nodes.device, dtype=nodes.dtype)
            degree.index_add_(0, source, ones)
            degree.index_add_(0, target, ones)
            aggregated = aggregated / degree.clamp_min(1.0)[:, None]
            nodes = nodes + node_update(torch.cat([nodes, aggregated], dim=-1))
        edge_logits = self.edge_classifier(
            torch.cat([nodes[source], nodes[target], edges], dim=-1)
        ).squeeze(-1)
        merge_logits = self.merge_classifier(nodes).squeeze(-1)
        return edge_logits, merge_logits
