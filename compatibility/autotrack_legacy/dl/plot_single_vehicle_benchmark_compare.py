"""Plot a single-vehicle benchmark sample with target, competitor, and prediction overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.single_vehicle_net import InferenceConfig, SingleVehicleTrackerConfig, load_checkpoint_model, predict_single_vehicle_track


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a single-vehicle benchmark sample with overlays.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="single_vehicle_benchmark_v1 .pt file")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path for prediction overlay.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output PNG path.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index inside the benchmark.")
    parser.add_argument("--device", default="cpu", help="Torch device for model inference.")
    parser.add_argument("--dpi", type=int, default=170, help="PNG DPI.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    samples = list(payload.get("samples", []))
    if not samples:
        raise ValueError(f"benchmark has no samples: {args.benchmark_file}")
    idx = int(max(0, min(len(samples) - 1, int(args.sample_index))))
    sample = samples[idx]
    target = sample["target"]

    raw_window = target["raw_window"].to(torch.float32).cpu().numpy()
    gt_time = target["time"][0].to(torch.float32).cpu().numpy()
    gt_vis = target["visibility"][0].to(torch.float32).cpu().numpy()
    comp_time = target.get("artifact_competing_time")
    comp_vis = target.get("artifact_competing_visibility")
    if comp_time is not None:
        comp_time = comp_time.to(torch.float32).cpu().numpy()
    if comp_vis is not None:
        comp_vis = comp_vis.to(torch.float32).cpu().numpy()
    gt_dir = int(target["direction"][0].item())
    gt_speed = float(target["speed"][0].item()) * float(payload.get("meta", {}).get("speed_norm_kmh", 150.0))
    fs = float(payload.get("meta", {}).get("fs", 1000.0))
    dx_m = float(payload.get("meta", {}).get("dx_m", 100.0))
    speed_min = max(1.0, gt_speed - max(15.0, 0.2 * gt_speed))
    speed_max = gt_speed + max(15.0, 0.2 * gt_speed)

    model, _ = load_checkpoint_model(Path(args.model).expanduser(), device=str(args.device))
    pred_cfg = InferenceConfig(
        time_downsample=int(payload.get("meta", {}).get("time_downsample", 10)),
        min_visible_channels=int(payload.get("meta", {}).get("min_visible_channels", 3)),
        objectness_threshold=0.35,
        peak_threshold=0.25,
        prior_weight=1.0,
        single_vehicle_tracker=SingleVehicleTrackerConfig(
            candidate_prominence=0.22,
            candidate_min_distance=180,
            candidate_max_peaks_per_channel=32,
            max_skip_channels=8,
            min_track_channels=8,
            min_track_score=8.0,
            kalman_bridge_gap_channels=12,
            kalman_fill_missing=True,
            kalman_gate_seconds=0.35,
            kalman_speed_gate_kmh=30.0,
        ),
    )
    tracks = predict_single_vehicle_track(
        model,
        raw_window,
        fs,
        dx_m,
        "auto",
        speed_min,
        speed_max,
        pred_cfg,
        device=str(args.device),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    window_samples = raw_window.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.4), constrained_layout=True, dpi=int(args.dpi))
    ax.imshow(raw_window, aspect="auto", origin="lower", cmap="magma")
    ax.plot(
        [float(gt_time[ch]) * (window_samples - 1) for ch in range(len(gt_time)) if gt_vis[ch] > 0.5],
        [ch for ch in range(len(gt_vis)) if gt_vis[ch] > 0.5],
        color="lime",
        linewidth=2.4,
        linestyle="--",
        label="GT target",
    )
    if comp_time is not None and comp_vis is not None:
        ax.plot(
            [float(comp_time[ch]) * (window_samples - 1) for ch in range(len(comp_time)) if comp_vis[ch] > 0.5],
            [ch for ch in range(len(comp_vis)) if comp_vis[ch] > 0.5],
            color="orange",
            linewidth=2.4,
            linestyle=":",
            label="competitor",
        )
    if tracks:
        tr = tracks[0]
        ax.plot([p.t_idx for p in tr.points], [p.ch_idx for p in tr.points], color="cyan", linewidth=2.2, label="prediction")
    ax.set_title(f"sample {idx} | competitor_dir={int(target.get('artifact_competing_direction', torch.tensor([-1.0]))[0].item())}")
    ax.set_xlabel("time [sample idx]")
    ax.set_ylabel("channel")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.75)
    ax.set_xlim(0, window_samples - 1)
    ax.set_ylim(-0.5, raw_window.shape[0] - 0.5)
    args.out_file = Path(args.out_file).expanduser()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(args.out_file))
    plt.close(fig)

    meta = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "out_file": str(args.out_file),
        "sample_index": int(idx),
    }
    args.out_file.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(args.out_file), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
