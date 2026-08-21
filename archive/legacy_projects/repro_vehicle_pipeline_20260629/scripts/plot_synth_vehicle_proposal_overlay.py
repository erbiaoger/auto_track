from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from autotrack.dl.multi_vehicle_pipeline import MultiVehiclePipelineConfig, _proposal_peak_boxes, extract_multi_vehicle_tracks
from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.vehicle_proposal_net import load_checkpoint_model as load_proposal_checkpoint_model, prepare_window_input


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render proposal and postprocessed overlays for a synthetic traffic sample.")
    parser.add_argument("--proposal-model", required=True, type=Path, help="Trained vehicle proposal checkpoint.")
    parser.add_argument(
        "--refine-model",
        type=Path,
        default=Path("models/vehicle_trace_exact_train_checkpoint_best.pt"),
        help="Single-vehicle refinement checkpoint.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--sample-index", type=int, default=0, help="Synthetic sample index.")
    parser.add_argument("--plot-dpi", type=int, default=170, help="Figure DPI.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample factor.")
    parser.add_argument("--vehicles-min", type=int, default=8, help="Minimum vehicles in the synthetic scene.")
    parser.add_argument("--vehicles-max", type=int, default=22, help="Maximum vehicles in the synthetic scene.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument(
        "--background-pt",
        type=Path,
        default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/xi_gauss_50_120s_large/test/shard_000000.pt"),
        help="Optional real shard .pt used as background texture.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-refine", action="store_true", help="Disable refinement and only render proposal boxes.")
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
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = OnlineSyntheticTrajectoryDataset(
        length=max(1, int(args.sample_index) + 1),
        n_channels=int(args.n_channels),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=0.0,
        slow_speed_min_kmh=float(args.speed_min_kmh),
        slow_speed_max_kmh=float(args.speed_max_kmh),
        fast_speed_min_kmh=float(args.speed_min_kmh),
        fast_speed_max_kmh=float(args.speed_max_kmh),
        noise_std=0.12,
        amp_min=0.8,
        amp_max=2.0,
        sigma_min_s=0.03,
        sigma_max_s=0.08,
        primary_ratio=0.5,
        min_visible_channels=3,
        speed_norm_kmh=150.0,
        clip_ratio=1.35,
        input_mode="raw",
        seed=int(args.seed),
        scene_mode="realistic_traffic",
        vehicle_count_profile="mixed_density",
        speed_variation_ratio=0.08,
        same_direction_cluster_ratio=0.35,
        crossing_ratio=0.35,
        parallel_close_ratio=0.20,
        multi_gap_dropout_ratio=0.55,
        return_raw_window=True,
        background_pt=args.background_pt if args.background_pt is not None and Path(args.background_pt).expanduser().is_file() else None,
        artifact_dropout_ratio=0.35,
        artifact_decoy_ratio=0.25,
        artifact_competing_ratio=0.45,
    )

    raw_x, target = dataset[int(args.sample_index)]
    raw = target["raw_window"].to(torch.float32).cpu().numpy()
    proposal_model, _ = load_proposal_checkpoint_model(Path(args.proposal_model).expanduser(), device=device)
    proposal_model.eval()
    x = prepare_window_input(raw, time_downsample=int(args.time_downsample), clip_ratio=1.35).unsqueeze(0).to(device)
    with torch.inference_mode():
        outputs = proposal_model(x)
    heat = torch.sigmoid(outputs["heatmap_logits"][0]).detach().cpu().numpy()
    boxes = _proposal_peak_boxes(heat, min_score=0.5, pad_channels=3, pad_time=20, max_peaks=12)

    tracks = extract_multi_vehicle_tracks(
        raw,
        fs=float(args.fs),
        dx_m=float(args.dx_m),
        direction="both",
        vmin_kmh=float(args.speed_min_kmh),
        vmax_kmh=float(args.speed_max_kmh),
        config=MultiVehiclePipelineConfig(
            candidate_limit=32,
            candidate_min_score=2.5,
            dedup_tolerance_samples=180,
            dedup_min_overlap_channels=3,
            dedup_min_overlap_ratio=0.45,
            crop_channel_margin=4,
            crop_time_margin_s=4.0,
            refine_with_model=not bool(args.no_refine),
            min_model_confidence=0.10,
            proposal_model_path=str(Path(args.proposal_model).expanduser()),
            proposal_prior_weight=1.25,
            proposal_time_downsample=int(args.time_downsample),
            iterative_extraction=not bool(args.no_refine),
            max_iterations=6 if bool(args.no_refine) else 10,
        ),
        model_path=str(Path(args.refine_model).expanduser()) if (args.refine_model is not None and not bool(args.no_refine)) else None,
        device=device,
    )

    raw_ds = raw[:, :: int(args.time_downsample)]
    window_seconds = float(raw.shape[1]) / float(args.fs)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=int(args.plot_dpi), constrained_layout=True)
    colors = ["cyan", "lime", "yellow", "orange", "red", "white"]

    ax = axes[0]
    ax.imshow(raw_ds, aspect="auto", origin="lower", cmap="magma", extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5))
    ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        alpha=0.52,
        extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5),
        vmin=0.0,
        vmax=1.0,
    )
    for score, (ch0, ch1, t0, t1) in boxes:
        x0 = float(t0) * window_seconds / float(heat.shape[1])
        x1 = float(t1) * window_seconds / float(heat.shape[1])
        rect = Rectangle((x0, ch0 - 0.5), x1 - x0, ch1 - ch0, fill=False, edgecolor="white", linewidth=1.4, alpha=0.95)
        ax.add_patch(rect)
    ax.set_title(f"Proposal overlay | boxes={len(boxes)}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("channel")

    ax = axes[1]
    ax.imshow(raw_ds, aspect="auto", origin="lower", cmap="magma", extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5))
    for j, tr in enumerate(tracks):
        pts = sorted(tr.points, key=lambda p: float(p.time_s))
        ax.plot([float(p.time_s) for p in pts], [int(p.ch_idx) for p in pts], color=colors[j % len(colors)], linewidth=2.2)
    ax.set_title(f"Postprocessed overlay | tracks={len(tracks)}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("channel")

    overlay = out_dir / "synthetic_vehicle_overlay.png"
    fig.savefig(str(overlay))
    plt.close(fig)

    summary = {
        "proposal_model": str(Path(args.proposal_model).expanduser()),
        "refine_model": str(Path(args.refine_model).expanduser()) if args.refine_model is not None else None,
        "device": device,
        "sample_index": int(args.sample_index),
        "proposal_boxes": int(len(boxes)),
        "post_tracks": int(len(tracks)),
        "overlay": str(overlay),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
