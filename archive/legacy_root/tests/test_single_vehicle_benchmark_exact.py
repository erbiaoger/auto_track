from __future__ import annotations

from pathlib import Path

import torch

from autotrack.dl.build_single_vehicle_benchmark_exact import main as build_exact_benchmark


def test_build_single_vehicle_benchmark_exact(tmp_path: Path) -> None:
    out_file = tmp_path / "exact.pt"
    rc = build_exact_benchmark(
        [
            "--out-file",
            str(out_file),
            "--samples",
            "4",
            "--window-seconds",
            "60",
            "--speed-min-kmh",
            "70",
            "--speed-max-kmh",
            "90",
        ]
    )
    assert rc == 0
    payload = torch.load(str(out_file), map_location="cpu", weights_only=False)
    assert payload["format"] == "single_vehicle_benchmark_v1"
    assert payload["length"] == 4
    assert payload["meta"]["window_seconds"] == 60.0
    assert payload["meta"]["artifact_dropout_ratio"] == 0.0
    assert payload["meta"]["artifact_decoy_ratio"] == 0.0
    assert payload["meta"]["artifact_competing_ratio"] == 0.0
    sample = payload["samples"][0]
    assert tuple(sample["x"].shape) == (1, 50, 6000)
    assert tuple(sample["target"]["time"].shape) == (1, 50)
    assert int(sample["target"]["direction"][0].item()) in {0, 1}
