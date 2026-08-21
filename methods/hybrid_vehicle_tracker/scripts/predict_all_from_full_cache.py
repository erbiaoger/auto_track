#!/usr/bin/env python3
"""GPU-predict every sliding window from one full-day threshold cache."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import _plot_overlay
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def _write_window_outputs(batch, artifacts, tracks_dir: Path, *, start_s: float, elapsed_s: float) -> None:
    tracks_dir.mkdir(parents=True, exist_ok=True)
    with (tracks_dir / "tracks.jsonl").open("w", encoding="utf-8") as handle:
        for track in batch.tracks:
            handle.write(json.dumps(track.to_dict(), ensure_ascii=False) + "\n")
    with (tracks_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for item in batch.observations:
            row = vars(item).copy()
            row.pop("embedding", None)
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (tracks_dir / "track_points.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "track_id", "channel_index", "station_id", "position_m", "time_s",
            "observed", "observation_id", "residual_s", "ambiguous",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for track in batch.tracks:
            for point in track.points:
                writer.writerow({"track_id": track.track_id, **vars(point)})
    _plot_overlay(
        batch,
        artifacts,
        tracks_dir / "tracks_overlay.png",
        time_offset_s=start_s,
    )
    summary = {
        "window_start_s": start_s,
        "window_end_s": start_s + batch.duration_s,
        "track_count": len(batch.tracks),
        "observation_count": len(batch.observations),
        "elapsed_s": elapsed_s,
        "diagnostics": batch.diagnostics,
    }
    (tracks_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/day11_threshold_090_full_day_cache"))
    parser.add_argument("--config", type=Path, default=Path("configs/day11_120s_v9.yaml"))
    parser.add_argument("--mapping", type=Path, default=Path(
        "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json"
    ))
    parser.add_argument("--output-root", type=Path, default=Path("runs/day11_threshold_090_all_v9_gpu"))
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--stride-s", type=float, default=60.0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = json.loads((args.cache_dir / "full_day_manifest.json").read_text(encoding="utf-8"))
    raw = np.load(manifest["raw_path"], mmap_mode="r")
    pre = np.load(manifest["pre_path"], mmap_mode="r")
    gauss = np.load(manifest["gauss_path"], mmap_mode="r")
    duration_total_s = float(manifest["duration_s"])
    starts = list(np.arange(0.0, duration_total_s - args.duration_s + 1e-6, args.stride_s))

    config = load_tracker_config(args.config)
    config.data.raw_path = str(manifest["raw_path"])
    config.data.pre_path = str(manifest["pre_path"])
    config.data.gauss_path = str(manifest["gauss_path"])
    config.data.mapping_path = str(args.mapping)
    config.data.sample_rate_hz = float(manifest["sample_rate_hz"])
    config.data.duration_s = float(args.duration_s)
    config.runtime.device = args.device
    tracker = HybridVehicleTracker(config)
    if args.device == "cuda":
        torch.cuda.synchronize()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_start = time.perf_counter()
    for index, start_s in enumerate(starts):
        if args.device == "cuda":
            torch.cuda.synchronize()
        window_start = time.perf_counter()
        batch = tracker.predict(
            raw,
            pre,
            gauss,
            args.mapping,
            start_s=float(start_s),
            duration_s=float(args.duration_s),
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - window_start
        tag = f"window_{int(round(start_s)):04d}_{int(round(start_s + args.duration_s)):04d}"
        if tracker.last_artifacts is None:
            raise RuntimeError("prediction completed without inference artifacts")
        _write_window_outputs(
            batch,
            tracker.last_artifacts,
            args.output_root / tag / "tracks",
            start_s=float(start_s),
            elapsed_s=elapsed_s,
        )
        row = {
            "index": index,
            "window_start_s": float(start_s),
            "window_end_s": float(start_s + args.duration_s),
            "elapsed_s": elapsed_s,
            "track_count": len(batch.tracks),
            "observation_count": len(batch.observations),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    total_elapsed_s = time.perf_counter() - total_start
    report = {
        "device": args.device,
        "checkpoint": config.model.checkpoint,
        "cache_manifest": str(args.cache_dir / "full_day_manifest.json"),
        "threshold": manifest["threshold"],
        "duration_total_s": duration_total_s,
        "window_duration_s": args.duration_s,
        "stride_s": args.stride_s,
        "window_count": len(rows),
        "total_elapsed_s": total_elapsed_s,
        "mean_window_elapsed_s": float(np.mean([row["elapsed_s"] for row in rows])),
        "min_window_elapsed_s": float(np.min([row["elapsed_s"] for row in rows])),
        "max_window_elapsed_s": float(np.max([row["elapsed_s"] for row in rows])),
        "total_track_count": int(sum(int(row["track_count"]) for row in rows)),
        "windows": rows,
    }
    (args.output_root / "prediction_timing.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "device", "window_count", "total_elapsed_s", "mean_window_elapsed_s",
        "total_track_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
