from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotrack.dl.shard_vehicle_pipeline import (  # noqa: E402
    ShardRunConfig,
    build_summary,
    extract_tracks_from_shard_bundle,
    load_shard_bundle,
    write_tracks_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the vehicle recognition pipeline on a shard .pt file.")
    parser.add_argument("--shard", required=True, type=Path, help="Input shard .pt file.")
    parser.add_argument("--meta", type=Path, default=None, help="Optional meta.json path. Defaults to sibling meta.json.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for CSV, JSON, and plots.")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "vehicle_trace_exact_train_checkpoint_best.pt", help="Single-vehicle checkpoint.")
    parser.add_argument("--proposal-model", type=Path, default=ROOT / "models" / "vehicle_proposal_rich_v4_checkpoint_best.pt", help="Dense proposal checkpoint.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--direction", default="auto", choices=["forward", "reverse", "both", "auto"], help="Vehicle motion direction.")
    parser.add_argument("--vmin-kmh", type=float, default=40.0, help="Minimum plausible speed.")
    parser.add_argument("--vmax-kmh", type=float, default=140.0, help="Maximum plausible speed.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of windows to process. 0 means all.")
    parser.add_argument("--candidate-limit", type=int, default=48, help="Maximum candidate tracks to refine per window.")
    parser.add_argument("--candidate-min-score", type=float, default=8.0, help="Minimum candidate score before refinement.")
    parser.add_argument("--dedup-tolerance-samples", type=int, default=180, help="Track deduplication tolerance in samples.")
    parser.add_argument("--dedup-min-overlap-channels", type=int, default=2, help="Minimum shared channels for deduplication.")
    parser.add_argument("--dedup-min-overlap-ratio", type=float, default=0.45, help="Minimum overlap ratio for deduplication.")
    parser.add_argument("--crop-channel-margin", type=int, default=4, help="Channels to add around each candidate crop.")
    parser.add_argument("--crop-time-margin-seconds", type=float, default=4.0, help="Time margin around each candidate crop.")
    parser.add_argument("--no-refine", action="store_true", help="Disable single-vehicle refinement.")
    parser.add_argument("--min-model-confidence", type=float, default=0.18, help="Minimum model confidence before refinement.")
    parser.add_argument("--model-confidence-weight", type=float, default=0.7, help="Weight of model confidence in final score.")
    parser.add_argument("--graph-confidence-weight", type=float, default=0.3, help="Weight of graph candidate score in final score.")
    parser.add_argument("--proposal-prior-weight", type=float, default=1.25, help="Weight of proposal prior before graph extraction.")
    parser.add_argument("--proposal-time-downsample", type=int, default=10, help="Time downsample factor used by the proposal model.")
    parser.add_argument("--no-iterative-extraction", action="store_true", help="Disable residual-suppressed iterative extraction.")
    parser.add_argument("--max-iterations", type=int, default=12, help="Maximum number of extraction passes.")
    parser.add_argument("--plot", action="store_true", help="Render an overview plot of the merged tracks.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Plot DPI.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_shard_bundle(args.shard, args.meta)
    run_cfg = ShardRunConfig(
        model_path=args.model,
        proposal_model_path=args.proposal_model,
        device=_resolve_device(args.device),
        direction=str(args.direction),
        vmin_kmh=float(args.vmin_kmh),
        vmax_kmh=float(args.vmax_kmh),
        max_samples=int(args.max_samples),
        candidate_limit=int(args.candidate_limit),
        candidate_min_score=float(args.candidate_min_score),
        dedup_tolerance_samples=int(args.dedup_tolerance_samples),
        dedup_min_overlap_channels=int(args.dedup_min_overlap_channels),
        dedup_min_overlap_ratio=float(args.dedup_min_overlap_ratio),
        crop_channel_margin=int(args.crop_channel_margin),
        crop_time_margin_s=float(args.crop_time_margin_seconds),
        refine_with_model=not bool(args.no_refine),
        min_model_confidence=float(args.min_model_confidence),
        model_confidence_weight=float(args.model_confidence_weight),
        graph_confidence_weight=float(args.graph_confidence_weight),
        proposal_prior_weight=float(args.proposal_prior_weight),
        proposal_time_downsample=int(args.proposal_time_downsample),
        iterative_extraction=not bool(args.no_iterative_extraction),
        max_iterations=int(args.max_iterations),
    )

    result = extract_tracks_from_shard_bundle(bundle, run_cfg)
    csv_path = write_tracks_csv(out_dir / "tracks.csv", result.tracks)
    summary = build_summary(bundle, result, run_cfg)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if args.plot:
        fig, ax = plt.subplots(figsize=(16, 6), dpi=int(args.plot_dpi), constrained_layout=True)
        colors = ["cyan", "lime", "yellow", "orange", "red", "white"]
        for idx, tr in enumerate(result.tracks):
            pts = sorted(tr.points, key=lambda p: float(p.time_s))
            ax.plot(
                [float(p.time_s) for p in pts],
                [int(p.ch_idx) for p in pts],
                linewidth=2.0,
                color=colors[idx % len(colors)],
                alpha=0.9,
            )
        ax.set_title(f"Merged vehicle tracks | tracks={len(result.tracks)}")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("channel")
        plot_path = out_dir / "tracks_overview.png"
        fig.savefig(plot_path)
        print(plot_path)

    print(csv_path)
    print(summary_path)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
