"""Run the trajectory-set predictor on a benchmark sample and render overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from autotrack.dl.trajectory_set_model import InferenceConfig, load_checkpoint_model, predict_tracks_from_window


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict from a multi_vehicle_benchmark_v1 file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Benchmark .pt file")
    parser.add_argument("--model", required=True, type=Path, help="Trajectory-set checkpoint")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--sample-index", type=int, default=0, help="Benchmark sample index")
    parser.add_argument("--max-tracks", type=int, default=16, help="Maximum number of tracks to render")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI")
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _draw_gt(ax: Any, target: dict[str, torch.Tensor], *, window_seconds: float) -> None:
    colors = ["#00ffff", "#00ff66", "#ffff00", "#ff9900", "#ff4444", "#ffffff"]
    if "gt_valid" in target:
        gt_valid = target["gt_valid"].to(torch.bool)
    else:
        gt_valid = torch.ones((int(target["time"].shape[0]),), dtype=torch.bool)
    time = target["time"].to(torch.float32)
    for idx, gt_idx in enumerate(torch.where(gt_valid)[0].tolist()):
        vis = torch.where(target["visibility"][gt_idx] > 0.5)[0].tolist()
        if not vis:
            continue
        pts = [(float(time[gt_idx, ch].item()) * float(window_seconds), int(ch)) for ch in vis]
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=12,
            c="#b8b8b8",
            alpha=0.8,
            marker="o",
            linewidths=0.0,
            zorder=20,
        )
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=colors[idx % len(colors)], linewidth=1.0, alpha=0.75)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    sample = payload["samples"][int(args.sample_index)]
    meta = dict(payload.get("meta", {}))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    time_downsample = int(meta.get("time_downsample", 10))
    x = sample["x"].to(torch.float32)
    if x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

    checkpoint = torch.load(str(Path(args.model).expanduser()), map_location=device, weights_only=False)
    model, _ = load_checkpoint_model(Path(args.model).expanduser(), device=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()

    cfg = InferenceConfig(
        time_downsample=int(time_downsample),
        objectness_threshold=0.2,
        visibility_threshold=0.45,
        min_visible_channels=3,
        refine_radius_samples=120,
        max_tracks=int(args.max_tracks),
        dedup_tolerance_samples=180,
        clip_ratio=float(meta.get("clip_ratio", 1.35)),
    )
    fs_eff = float(fs) / float(max(1, int(time_downsample)))
    window_seconds = float(meta.get("window_seconds", float(max(1, x.shape[1] - 1)) / float(max(1e-9, fs_eff))))
    tracks = predict_tracks_from_window(
        model,
        x.cpu().numpy(),
        fs=float(fs),
        x_axis_m=np.arange(x.shape[0], dtype=np.float32) * float(dx_m),
        config=cfg,
        device=device,
    )

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), dpi=int(args.plot_dpi), constrained_layout=True)
    ax.imshow(
        x.cpu().numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0.0, float(window_seconds), -0.5, float(x.shape[0]) - 0.5),
    )
    colors = ["cyan", "lime", "yellow", "orange", "red", "white"]
    for idx, tr in enumerate(tracks):
        pts = sorted(tr.points, key=lambda p: float(p.time_s))
        ax.plot([float(p.time_s) for p in pts], [int(p.ch_idx) for p in pts], color=colors[idx % len(colors)], linewidth=2.0)
    _draw_gt(ax, sample["target"], window_seconds=float(meta.get("window_seconds", window_seconds)))
    ax.set_title(f"trajectory-set benchmark sample {int(args.sample_index)} | tracks={len(tracks)}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("channel")
    overlay = out_dir / "overlay.png"
    fig.savefig(str(overlay))
    plt.close(fig)

    summary = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "device": device,
        "sample_index": int(args.sample_index),
        "max_tracks": int(args.max_tracks),
        "tracks": int(len(tracks)),
        "overlay": str(overlay),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
