"""End-to-end clean real-background workflow: finetune, evaluate, and plot one vehicle."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.evaluate_single_vehicle import main as evaluate_single_vehicle_main
from autotrack.dl.plot_single_vehicle_benchmark_compare import main as plot_single_vehicle_benchmark_compare_main
from autotrack.dl.train_single_vehicle_realbg_exact import main as train_single_vehicle_realbg_exact_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the clean real-background single-vehicle workflow.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for train/eval artifacts.")
    parser.add_argument(
        "--realbg-npy",
        type=Path,
        default=Path("datasets/03gauss_large.npy"),
        help="Real background .npy used for the benchmark.",
    )
    parser.add_argument("--samples", type=int, default=64, help="Number of benchmark windows.")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Model hidden dimension.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Benchmark window duration.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--keep-benchmark", action="store_true", help="Keep the generated benchmark file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_file = out_dir / "realbg_clean_benchmark.pt"

    train_argv = [
        "--out-dir",
        str(out_dir),
        "--realbg-npy",
        str(Path(args.realbg_npy).expanduser()),
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
        train_argv.append("--keep-benchmark")
    if train_single_vehicle_realbg_exact_main(train_argv) != 0:
        return 1

    eval_argv = [
        "--model",
        str(out_dir / "checkpoint_best.pt"),
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
        str(out_dir / "checkpoint_best.pt"),
        "--out-file",
        str(out_dir / "compare.png"),
        "--sample-index",
        "0",
        "--device",
        str(args.device),
    ]
    if plot_single_vehicle_benchmark_compare_main(plot_argv) != 0:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
