from __future__ import annotations

import torch

from hybrid_vehicle_tracker.models.edge_gnn import EdgeAssociationGNN
from hybrid_vehicle_tracker.models.physical_hough import PhysicalHoughHead
from hybrid_vehicle_tracker.models.resunet import MultiModalResUNet


def test_resunet_output_shapes():
    model = MultiModalResUNet(base_channels=4, embedding_dim=8)
    inputs = torch.randn(2, 5, 10, 80)
    outputs = model(inputs)
    assert outputs["centerline_logits"].shape == (2, 1, 10, 80)
    assert outputs["slowness"].shape == (2, 1, 10, 80)
    assert outputs["embedding"].shape == (2, 8, 10, 80)


def test_physical_hough_finds_known_line_and_is_differentiable():
    stations = 10
    rate = 10.0
    duration = 60.0
    positions = torch.arange(stations, dtype=torch.float32) * 100.0
    evidence = torch.zeros(1, 1, stations, int(duration * rate), requires_grad=True)
    slope = 0.05
    intercept = 5.0
    with torch.no_grad():
        for station, position in enumerate(positions):
            index = int(round((intercept + slope * float(position)) * rate))
            evidence[0, 0, station, index] = 1.0
    head = PhysicalHoughHead(4, slope_bins=9, intercept_step_s=0.5)
    output = head(
        torch.zeros(1, 4, stations, int(duration * rate)),
        positions,
        duration_s=duration,
        feature_rate_hz=rate,
        evidence_map=evidence,
    )
    seeds = head.topk_seeds(output, top_k=1, learned=False, min_support=5)
    assert abs(seeds[0].slope_s_per_m - slope) <= 0.003
    assert abs(seeds[0].intercept_s - intercept) <= 0.75
    output.raw_scores.sum().backward()
    assert evidence.grad is not None


def test_edge_gnn_scores_edges_and_nodes():
    model = EdgeAssociationGNN(node_features=12, edge_features=9, hidden_dim=16)
    nodes = torch.randn(5, 12)
    edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    attributes = torch.randn(4, 9)
    edge_logits, merge_logits = model(nodes, edges, attributes)
    assert edge_logits.shape == (4,)
    assert merge_logits.shape == (5,)
