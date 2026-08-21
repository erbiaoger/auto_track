"""End-to-end exact synthetic workflow: train, evaluate, and plot one vehicle."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.validate_single_vehicle_exact import main as validate_exact_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the exact synthetic single-vehicle workflow.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for exact train/eval artifacts.")
    parser.add_argument("--samples", type=int, default=64, help="Number of exact benchmark samples.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden dimension.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Benchmark window duration.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--keep-benchmark", action="store_true", help="Keep the generated exact benchmark file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_argv = [
        "--out-dir",
        str(Path(args.out_dir).expanduser()),
        "--samples",
        str(int(args.samples)),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--device",
        str(args.device),
        "--window-seconds",
        str(float(args.window_seconds)),
        "--speed-min-kmh",
        str(float(args.speed_min_kmh)),
        "--speed-max-kmh",
        str(float(args.speed_max_kmh)),
    ]
    if bool(args.keep_benchmark):
        validate_argv.append("--keep-benchmark")
    return int(validate_exact_main(validate_argv))


if __name__ == "__main__":
    raise SystemExit(main())
