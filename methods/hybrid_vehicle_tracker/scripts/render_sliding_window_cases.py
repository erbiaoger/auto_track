#!/usr/bin/env python3
"""Run fixed-threshold 120 s windows at several sliding start times."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert_threshold_and_predict import convert_and_assemble
from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import _plot_overlay
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/v6_peakset_long_exact/hybrid_final.pt"))
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--stride-s", type=float, default=60.0)
    parser.add_argument("--start-s", nargs="*", type=float, default=None)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=Path("runs/day11_threshold_090_sliding"))
    parser.add_argument("--mapping", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json"))
    parser.add_argument("--sig-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_raw/11"))
    parser.add_argument("--pred-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_pre/new11"))
    parser.add_argument("--converter", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/0002pre2guass.py"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    starts = args.start_s if args.start_s else [i * args.stride_s for i in range(args.count)]
    for start_s in starts:
        tag = f"window_{int(start_s):04d}_{int(start_s + args.duration_s):04d}"
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
            threshold=args.threshold,
            duration_s=args.duration_s,
            start_s=start_s,
            save_mode="grid",
        )
        config = load_tracker_config("configs/day11_120s.yaml")
        config.data.raw_path = str(raw_path)
        config.data.pre_path = str(pre_path)
        config.data.gauss_path = str(gauss_path)
        config.data.mapping_path = str(args.mapping)
        config.data.start_s = 0.0
        config.data.duration_s = args.duration_s
        config.runtime.device = args.device
        config.model.checkpoint = str(args.checkpoint)
        tracker = HybridVehicleTracker(config)
        batch = tracker.predict_from_paths()
        if tracker.last_artifacts is None:
            raise RuntimeError("inference completed without artifacts")
        tracks_dir.mkdir(parents=True, exist_ok=True)
        _plot_overlay(
            batch,
            tracker.last_artifacts,
            tracks_dir / "tracks_overlay.png",
            time_offset_s=start_s,
        )
        summary = {
            "threshold": args.threshold,
            "window_start_s": start_s,
            "window_end_s": start_s + args.duration_s,
            "track_count": len(batch.tracks),
            "observation_count": len(batch.observations),
            "diagnostics": batch.diagnostics,
            "conversion": manifest,
        }
        (tracks_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"threshold={args.threshold:.2f}, window={start_s:.0f}–{start_s + args.duration_s:.0f}s: "
            f"{tracks_dir / 'tracks_overlay.png'} ({len(batch.tracks)} tracks)"
        )


if __name__ == "__main__":
    main()
