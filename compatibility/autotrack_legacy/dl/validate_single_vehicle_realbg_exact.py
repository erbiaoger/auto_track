"""Validate the redesigned single-vehicle model on real-background clean windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_realistic import main as build_realistic_benchmark_main
from autotrack.dl.evaluate_single_vehicle import main as evaluate_single_vehicle_main
from autotrack.dl.plot_single_vehicle_benchmark_compare import main as plot_single_vehicle_benchmark_compare_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a single-vehicle checkpoint on real-background clean windows.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path to validate.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for benchmark/eval artifacts.")
    parser.add_argument(
        "--realbg-npy",
        type=Path,
        default=Path("datasets/03gauss_large.npy"),
        help="Real background .npy used for the benchmark.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Number of benchmark samples.")
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
    bench_file = out_dir / "realbg_exact_benchmark.pt"

    build_argv = [
        "--out-file",
        str(bench_file),
        "--samples",
        str(int(args.samples)),
        "--seed",
        "101",
        "--n-channels",
        "50",
        "--fs",
        "1000",
        "--dx-m",
        "100",
        "--window-seconds",
        str(float(args.window_seconds)),
        "--time-downsample",
        "10",
        "--speed-min-kmh",
        str(float(args.speed_min_kmh)),
        "--speed-max-kmh",
        str(float(args.speed_max_kmh)),
        "--noise-std",
        "0.0",
        "--amp-min",
        "6.0",
        "--amp-max",
        "6.0",
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
        "0.0",
        "--artifact-decoy-ratio",
        "0.0",
        "--artifact-competing-ratio",
        "0.0",
        "--artifact-competing-opposite-direction-ratio",
        "0.0",
        "--raw-window-dtype",
        "float32",
    ]
    if build_realistic_benchmark_main(build_argv) != 0:
        return 1

    eval_argv = [
        "--model",
        str(Path(args.model).expanduser()),
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
        str(Path(args.model).expanduser()),
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
