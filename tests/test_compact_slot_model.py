from __future__ import annotations

import torch

from autotrack.dl.compact_slot_model import ModelConfig, TrackSlotPredictor, track_slot_detection_metrics, track_slot_set_loss


def _make_targets(batch: int = 2, channels: int = 50) -> dict[str, torch.Tensor]:
    time = torch.zeros((batch, 2, channels), dtype=torch.float32)
    visibility = torch.zeros((batch, 2, channels), dtype=torch.float32)
    direction = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    speed = torch.tensor([[0.55, 0.62], [0.58, 0.65]], dtype=torch.float32)
    gt_valid = torch.tensor([[True, True], [True, False]], dtype=torch.bool)
    for b in range(batch):
        for g in range(2):
            for ch in range(6, 18):
                visibility[b, g, ch] = 1.0
                time[b, g, ch] = (0.2 * g + 0.01 * ch + 0.05 * b) % 1.0
    return {
        "time": time,
        "visibility": visibility,
        "direction": direction,
        "speed": speed,
        "gt_valid": gt_valid,
    }


def test_compact_slot_forward_and_loss() -> None:
    torch.manual_seed(0)
    model = TrackSlotPredictor(ModelConfig(max_tracks=6, hidden_dim=64, temporal_channels=24))
    x = torch.randn(2, 1, 50, 256)
    targets = _make_targets()
    outputs = model(x, targets=targets)
    assert outputs["objectness_logits"].shape == (2, 6)
    assert outputs["time"].shape == (2, 6, 50)
    loss, metrics = track_slot_set_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert metrics["gt"] == 3.0
    det = track_slot_detection_metrics(outputs, targets)
    assert "track_f1" in det
