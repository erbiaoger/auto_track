"""Validate the exact synthetic single-vehicle workflow end to end."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_exact import main as build_exact_benchmark_main
from autotrack.dl.evaluate_single_vehicle import main as evaluate_single_vehicle_main
from autotrack.dl.plot_single_vehicle_benchmark_compare import main as plot_single_vehicle_benchmark_compare_main
from autotrack.dl.train_single_vehicle_exact import main as train_single_vehicle_exact_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the exact synthetic single-vehicle workflow.")
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
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_file = out_dir / "exact_benchmark.pt"

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

    train_argv = [
        "--out-dir",
        str(out_dir),
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
        "0",
        "--checkpoint-every",
        "1",
    ]
    if train_single_vehicle_exact_main(train_argv) != 0:
        return 1

    eval_argv = [
        "--model",
        str(out_dir / "train" / "checkpoint_best.pt"),
        "--out-dir",
        str(out_dir / "eval"),
        "--benchmark-file",
        str(bench_file),
        "--device",
        str(args.device),
        "--samples",
        str(int(args.samples)),
        "--batch-size",
        "1",
    ]
    if evaluate_single_vehicle_main(eval_argv) != 0:
        return 1

    plot_argv = [
        "--benchmark-file",
        str(bench_file),
        "--model",
        str(out_dir / "train" / "checkpoint_best.pt"),
        "--out-file",
        str(out_dir / "compare.png"),
        "--sample-index",
        "0",
        "--device",
        str(args.device),
    ]
    if plot_single_vehicle_benchmark_compare_main(plot_argv) != 0:
        return 1

    if not bool(args.keep_benchmark):
        try:
            bench_file.unlink(missing_ok=True)
            bench_file.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
