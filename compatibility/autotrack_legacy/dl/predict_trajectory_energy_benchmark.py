"""Run the trajectory-energy model on a benchmark file and render overlays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from autotrack.dl.trajectory_energy_model import TrajectoryEnergyNet, extract_vehicle_tracks_from_energy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict from a single_vehicle_benchmark_v1 file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Benchmark .pt file")
    parser.add_argument("--model", required=True, type=Path, help="Trajectory-energy checkpoint")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--sample-index", type=int, default=0, help="Benchmark sample index")
    parser.add_argument("--max-tracks", type=int, default=16, help="Maximum number of tracks to render")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI")
    parser.add_argument("--decoder-profile", default="strict", choices=["auto", "balanced", "strict", "recall"], help="Decoder profile used by the trajectory-energy extractor.")
    parser.add_argument("--scene-active-ratio-threshold", type=float, default=0.033, help="Active-ratio threshold used when decoder-profile=auto.")
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    sample = payload["samples"][int(args.sample_index)]
    meta = dict(payload.get("meta", {}))
    time_downsample = int(meta.get("time_downsample", 10))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    x = sample["x"].to(torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")

    checkpoint = torch.load(str(Path(args.model).expanduser()), map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = TrajectoryEnergyNet().to(device)
    if model_config:
        from autotrack.dl.trajectory_energy_model import ModelConfig

        model = TrajectoryEnergyNet(ModelConfig(**model_config)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(device))
    energy = torch.sigmoid(outputs["energy_logits"][0]).detach().cpu().numpy()
    tracks = extract_vehicle_tracks_from_energy(
        energy,
        fs=float(fs),
        dx_m=float(dx_m),
        direction="both",
        vmin_kmh=70.0,
        vmax_kmh=90.0,
        config={
            "time_downsample": int(time_downsample),
            "decoder_profile": str(args.decoder_profile),
            "scene_active_ratio_threshold": float(args.scene_active_ratio_threshold),
            "min_track_channels": 4,
            "min_track_score": 1.0,
            "max_skip_channels": 8,
            "peak_prominence": 0.02,
            "peak_min_height": 0.04,
            "peak_distance_samples": 8,
            "dedup_time_tol": 800,
            "dedup_channel_overlap": 3,
            "dedup_overlap_ratio": 0.65,
        },
    )
    tracks = tracks[: max(0, int(args.max_tracks))]

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), dpi=int(args.plot_dpi), constrained_layout=True)
    ax.imshow(energy, aspect="auto", origin="lower", cmap="magma")
    colors = ["cyan", "lime", "yellow", "orange", "red", "white"]
    for idx, tr in enumerate(tracks):
        pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
        ax.plot([float(p.t_idx) for p in pts], [int(p.ch_idx) for p in pts], color=colors[idx % len(colors)], linewidth=2.0)
    ax.set_title(f"trajectory-energy benchmark sample {int(args.sample_index)} | tracks={len(tracks)}")
    ax.set_xlabel("time bin")
    ax.set_ylabel("channel")
    fig.savefig(str(out_dir / "overlay.png"))
    plt.close(fig)

    summary = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "device": device,
        "sample_index": int(args.sample_index),
        "max_tracks": int(args.max_tracks),
        "tracks": int(len(tracks)),
        "overlay": str(out_dir / "overlay.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
