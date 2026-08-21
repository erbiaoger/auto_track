#!/usr/bin/env python3
"""Generate several complete 0--120 s overlays at different thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import _plot_overlay
from convert_threshold_and_predict import convert_and_assemble
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.8, 0.9])
    parser.add_argument("--output-root", type=Path, default=Path("runs/full_window_cases"))
    parser.add_argument("--mapping", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json"))
    parser.add_argument("--sig-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_raw/11"))
    parser.add_argument("--pred-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_pre/new11"))
    parser.add_argument("--converter", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/0002pre2guass.py"))
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    for threshold in args.thresholds:
        tag = f"threshold_{threshold:.2f}".replace(".", "")
        run_dir = args.output_root / tag
        modal_dir = run_dir / "modal"
        gauss_dir = run_dir / "gauss"
        tracks_dir = run_dir / "tracks"
        raw_path, pre_path, gauss_path, manifest = convert_and_assemble(
            converter_path=args.converter,
            sig_dir=args.sig_dir,
            pred_dir=args.pred_dir,
            gauss_dir=gauss_dir,
            mapping_path=args.mapping,
            cache_dir=modal_dir,
            threshold=threshold,
            duration_s=120.0,
            save_mode="grid",
        )
        config = load_tracker_config("configs/day11_120s.yaml")
        config.data.raw_path = str(raw_path)
        config.data.pre_path = str(pre_path)
        config.data.gauss_path = str(gauss_path)
        config.data.mapping_path = str(args.mapping)
        config.data.start_s = 0.0
        config.data.duration_s = 120.0
        config.runtime.device = "cuda"
        tracker = HybridVehicleTracker(config)
        batch = tracker.predict_from_paths()
        if tracker.last_artifacts is None:
            raise RuntimeError("inference completed without artifacts")
        tracks_dir.mkdir(parents=True, exist_ok=True)
        _plot_overlay(batch, tracker.last_artifacts, tracks_dir / "tracks_overlay_0_120s.png")
        (tracks_dir / "summary.json").write_text(
            json.dumps(
                {
                    "threshold": threshold,
                    "track_count": len(batch.tracks),
                    "observation_count": len(batch.observations),
                    "diagnostics": batch.diagnostics,
                    "manifest": manifest,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"threshold={threshold:.2f}: {tracks_dir / 'tracks_overlay_0_120s.png'} ({len(batch.tracks)} tracks)")


if __name__ == "__main__":
    main()
