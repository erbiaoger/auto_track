"""Build the validated real-vehicle mixed benchmark and finetune the single-vehicle model on it."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_realistic import main as build_realistic_benchmark_main
from autotrack.dl.build_single_vehicle_benchmark_from_track_slot import main as build_trackslot_benchmark_main
from autotrack.dl.merge_single_vehicle_benchmarks import main as merge_benchmarks_main
from autotrack.dl.train_single_vehicle import main as train_single_vehicle_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the real-vehicle single-track model from mixed benchmarks.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and generated benchmarks.")
    parser.add_argument(
        "--realbg-npy",
        type=Path,
        default=Path("datasets/03gauss_large.npy"),
        help="Real background .npy used for the real-background benchmark slice.",
    )
    parser.add_argument(
        "--trackslot-dir",
        type=Path,
        default=Path("datasets/track_slot_v3_120s_realistic/train"),
        help="TrackSlot shard directory used to build one-vehicle samples.",
    )
    parser.add_argument("--clean-samples", type=int, default=128, help="Number of clean synthetic one-vehicle samples.")
    parser.add_argument("--realbg-samples", type=int, default=16, help="Number of real-background one-vehicle samples.")
    parser.add_argument("--trackslot-samples", type=int, default=256, help="Number of TrackSlot-derived one-vehicle samples.")
    parser.add_argument("--clean-weight", type=float, default=1.0, help="Sampler weight for clean synthetic samples.")
    parser.add_argument("--realbg-weight", type=float, default=2.0, help="Sampler weight for real-background samples.")
    parser.add_argument("--trackslot-weight", type=float, default=2.0, help="Sampler weight for TrackSlot-derived samples.")
    parser.add_argument("--device", default="auto", help="Torch device for finetuning.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Hidden dimension for the single-vehicle model.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--log-every", type=int, default=20, help="Training log cadence.")
    parser.add_argument("--keep-benchmarks", action="store_true", help="Keep generated benchmark .pt files instead of cleaning them up.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_bench = out_dir / "clean_benchmark.pt"
    realbg_bench = out_dir / "realbg_benchmark.pt"
    trackslot_bench = out_dir / "trackslot_benchmark.pt"
    mixed_bench = out_dir / "mixed_benchmark.pt"

    clean_argv = [
        "--out-file",
        str(clean_bench),
        "--samples",
        str(int(args.clean_samples)),
        "--seed",
        "7",
        "--n-channels",
        "50",
        "--fs",
        "1000",
        "--dx-m",
        "100",
        "--window-seconds",
        "120",
        "--time-downsample",
        "10",
        "--speed-min-kmh",
        "70",
        "--speed-max-kmh",
        "90",
        "--noise-std",
        "0.0",
        "--amp-min",
        "6",
        "--amp-max",
        "6",
        "--sigma-min-s",
        "0.25",
        "--sigma-max-s",
        "0.25",
        "--min-visible-channels",
        "3",
        "--raw-window-dtype",
        "float32",
        "--background-scale",
        "1.0",
        "--artifact-dropout-ratio",
        "0.0",
        "--artifact-decoy-ratio",
        "0.0",
        "--artifact-competing-ratio",
        "0.0",
    ]
    if build_realistic_benchmark_main(clean_argv) != 0:
        return 1

    realbg_argv = [
        "--out-file",
        str(realbg_bench),
        "--samples",
        str(int(args.realbg_samples)),
        "--seed",
        "11",
        "--n-channels",
        "50",
        "--fs",
        "1000",
        "--dx-m",
        "100",
        "--window-seconds",
        "120",
        "--time-downsample",
        "10",
        "--speed-min-kmh",
        "70",
        "--speed-max-kmh",
        "90",
        "--noise-std",
        "0.0",
        "--amp-min",
        "6",
        "--amp-max",
        "6",
        "--sigma-min-s",
        "0.25",
        "--sigma-max-s",
        "0.25",
        "--min-visible-channels",
        "3",
        "--background-npy",
        str(Path(args.realbg_npy).expanduser()),
        "--background-layout",
        "time_channel",
        "--background-channel-start",
        "0",
        "--background-scale",
        "1.0",
        "--artifact-dropout-ratio",
        "0.40",
        "--artifact-decoy-ratio",
        "0.50",
        "--artifact-competing-ratio",
        "0.30",
        "--artifact-competing-opposite-direction-ratio",
        "0.25",
        "--raw-window-dtype",
        "float16",
    ]
    if build_realistic_benchmark_main(realbg_argv) != 0:
        return 1

    trackslot_argv = [
        "--data-dir",
        str(Path(args.trackslot_dir).expanduser()),
        "--out-file",
        str(trackslot_bench),
        "--max-samples",
        str(int(args.trackslot_samples)),
        "--slot-policy",
        "most_visible",
        "--raw-window-dtype",
        "float16",
    ]
    if build_trackslot_benchmark_main(trackslot_argv) != 0:
        return 1

    merge_argv = [
        "--out-file",
        str(mixed_bench),
        "--in-file",
        str(clean_bench),
        str(realbg_bench),
        str(trackslot_bench),
        "--input-weight",
        str(float(args.clean_weight)),
        str(float(args.realbg_weight)),
        str(float(args.trackslot_weight)),
    ]
    if merge_benchmarks_main(merge_argv) != 0:
        return 1

    train_argv = [
        "--out-dir",
        str(out_dir),
        "--benchmark-file",
        str(mixed_bench),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--lr",
        str(float(args.lr)),
        "--weight-decay",
        str(float(args.weight_decay)),
        "--log-every",
        str(int(args.log_every)),
    ]
    if train_single_vehicle_main(train_argv) != 0:
        return 1

    if not bool(args.keep_benchmarks):
        for path in (clean_bench, realbg_bench, trackslot_bench, mixed_bench):
            try:
                path.unlink(missing_ok=True)
                path.with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
