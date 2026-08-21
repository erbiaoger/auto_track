from __future__ import annotations

from pathlib import Path

import torch

from autotrack.dl.build_single_vehicle_benchmark_realistic import main as build_realistic_benchmark


def test_build_single_vehicle_benchmark_realistic(tmp_path: Path) -> None:
    out_file = tmp_path / "bench.pt"
    rc = build_realistic_benchmark(
        [
            "--out-file",
            str(out_file),
            "--samples",
            "4",
            "--window-seconds",
            "120",
            "--dx-m",
            "100",
            "--speed-min-kmh",
            "70",
            "--speed-max-kmh",
            "90",
            "--artifact-dropout-ratio",
            "0.25",
            "--artifact-decoy-ratio",
            "0.25",
            "--artifact-competing-ratio",
            "0.25",
        ]
    )
    assert rc == 0
    payload = torch.load(str(out_file), map_location="cpu", weights_only=False)
    assert payload["format"] == "single_vehicle_benchmark_v1"
    assert payload["length"] == 4
    assert payload["meta"]["window_seconds"] == 120.0
    assert payload["meta"]["dx_m"] == 100.0
    sample = payload["samples"][0]
    assert tuple(sample["x"].shape)[1] == 50
    assert tuple(sample["target"]["time"].shape)[1] == 50
    assert float(sample["target"]["speed"][0].item()) > 0.0
