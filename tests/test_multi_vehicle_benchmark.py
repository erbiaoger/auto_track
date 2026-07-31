from __future__ import annotations

from pathlib import Path

import torch

from autotrack.dl.build_multi_vehicle_benchmark import main as build_multi_vehicle_benchmark
from autotrack.dl.trajectory_energy_model import extract_vehicle_tracks_from_energy


def test_build_multi_vehicle_benchmark_and_decode(tmp_path: Path) -> None:
    out_file = tmp_path / "multi.pt"
    rc = build_multi_vehicle_benchmark(
        [
            "--out-file",
            str(out_file),
            "--samples",
            "4",
            "--vehicles-min",
            "8",
            "--vehicles-max",
            "10",
            "--noise-std",
            "0",
            "--artifact-dropout-ratio",
            "0",
            "--artifact-decoy-ratio",
            "0",
            "--artifact-competing-ratio",
            "0",
        ]
    )
    assert rc == 0
    payload = torch.load(str(out_file), map_location="cpu", weights_only=False)
    assert payload["format"] == "multi_vehicle_benchmark_v1"
    assert payload["length"] == 4
    sample = payload["samples"][0]
    assert tuple(sample["x"].shape)[1] == 50
    assert tuple(sample["energy"].shape) == tuple(sample["x"].shape)[1:]
    assert int(sample["target"]["gt_masks"].shape[0]) >= 8
    tracks = extract_vehicle_tracks_from_energy(
        sample["energy"].to(torch.float32).numpy(),
        fs=1000.0,
        dx_m=100.0,
        direction="forward",
        vmin_kmh=70.0,
        vmax_kmh=90.0,
        config={"time_downsample": 1, "min_visible_channels": 3, "peak_prominence": 0.01, "peak_min_height": 0.01},
    )
    assert len(tracks) >= 2
