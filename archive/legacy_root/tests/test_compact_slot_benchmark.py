from __future__ import annotations

from pathlib import Path

import torch

from autotrack.dl.build_multi_vehicle_benchmark import main as build_multi_vehicle_benchmark
from autotrack.dl.compact_slot_model import ModelConfig, TrackSlotPredictor, track_slot_detection_metrics, track_slot_set_loss
from autotrack.dl.multi_vehicle_benchmark_dataset import MultiVehicleBenchmarkDataset, stack_benchmark_batch


def test_compact_slot_benchmark_dataset_and_forward(tmp_path: Path) -> None:
    out_file = tmp_path / "multi.pt"
    rc = build_multi_vehicle_benchmark(
        [
            "--out-file",
            str(out_file),
            "--samples",
            "2",
            "--vehicles-min",
            "6",
            "--vehicles-max",
            "8",
            "--noise-std",
            "0",
            "--artifact-dropout-ratio",
            "0",
            "--artifact-decoy-ratio",
            "0",
            "--artifact-competing-ratio",
            "0",
            "--overwrite",
        ]
    )
    assert rc == 0

    ds = MultiVehicleBenchmarkDataset(out_file)
    x, targets = stack_benchmark_batch([ds[0], ds[1]])
    assert tuple(x.shape)[1:] == tuple(ds[0][0].shape)
    assert targets["time"].ndim == 3

    model = TrackSlotPredictor(ModelConfig(n_channels=int(x.shape[2]), in_channels=int(x.shape[1]), max_tracks=16, hidden_dim=64, num_heads=4))
    outputs = model(x)
    loss, metrics = track_slot_set_loss(outputs, targets)
    assert torch.isfinite(loss)
    assert metrics["gt"] > 0
    det = track_slot_detection_metrics(outputs, targets)
    assert "track_f1" in det
