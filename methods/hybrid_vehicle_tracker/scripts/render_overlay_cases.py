#!/usr/bin/env python3
"""Render several reference-style Gauss-window overlay cases from one run."""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import _plot_overlay
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/day11_120s.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("runs/day11_threshold_090_real/modal/raw_threshold.npy"))
    parser.add_argument("--pre", type=Path, default=Path("runs/day11_threshold_090_real/modal/pre_probability.npy"))
    parser.add_argument("--gauss", type=Path, default=Path("runs/day11_threshold_090_real/modal/gauss_threshold.npy"))
    parser.add_argument("--mapping", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day11_threshold_090_real/tracks/overlay_cases"))
    args = parser.parse_args()

    config = load_tracker_config(args.config)
    config.data.raw_path = str(args.raw)
    config.data.pre_path = str(args.pre)
    config.data.gauss_path = str(args.gauss)
    if args.mapping is not None:
        config.data.mapping_path = str(args.mapping)
    config.data.start_s = 0.0
    config.data.duration_s = 120.0
    config.runtime.device = "cuda"

    tracker = HybridVehicleTracker(config)
    batch = tracker.predict_from_paths()
    if tracker.last_artifacts is None:
        raise RuntimeError("inference completed without artifacts")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        "case_full_000_120s.png": None,
        "case_000_030s.png": (0.0, 30.0),
        "case_030_060s.png": (30.0, 60.0),
        "case_060_090s.png": (60.0, 90.0),
        "case_090_120s.png": (90.0, 120.0),
    }
    for filename, time_window in cases.items():
        _plot_overlay(
            batch,
            tracker.last_artifacts,
            args.output_dir / filename,
            time_window=time_window,
        )
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
