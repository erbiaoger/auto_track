"""Train the redesigned single-vehicle model on the near-perfect synthetic benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_exact import main as build_exact_benchmark_main
from autotrack.dl.train_single_vehicle import main as train_single_vehicle_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-vehicle model on the exact synthetic benchmark.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and benchmark.")
    parser.add_argument("--benchmark-file", type=Path, default=None, help="Optional existing exact benchmark file to train on.")
    parser.add_argument("--samples", type=int, default=64, help="Number of synthetic benchmark windows.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden dimension.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Benchmark window duration.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--log-every", type=int, default=0, help="Training log cadence.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Checkpoint cadence.")
    parser.add_argument("--keep-benchmark", action="store_true", help="Keep the generated exact benchmark file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_file = out_dir / "exact_benchmark.pt"

    if args.benchmark_file is None:
        build_argv = [
            "--out-file",
            str(bench_file),
            "--samples",
            str(int(args.samples)),
            "--window-seconds",
            str(float(args.window_seconds)),
            "--speed-min-kmh",
            str(float(args.speed_min_kmh)),
            "--speed-max-kmh",
            str(float(args.speed_max_kmh)),
        ]
        if build_exact_benchmark_main(build_argv) != 0:
            return 1
    else:
        bench_file = Path(args.benchmark_file).expanduser()

    train_argv = [
        "--out-dir",
        str(out_dir / "train"),
        "--benchmark-file",
        str(bench_file),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--log-every",
        str(int(args.log_every)),
        "--checkpoint-every",
        str(int(args.checkpoint_every)),
    ]
    if train_single_vehicle_main(train_argv) != 0:
        return 1

    if args.benchmark_file is None and not bool(args.keep_benchmark):
        try:
            bench_file.unlink(missing_ok=True)
            bench_file.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
