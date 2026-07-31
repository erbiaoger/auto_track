"""Build a near-perfect single-vehicle benchmark for synthetic exactness checks.

This preset intentionally strips away the distractions that belong to the
robustness benchmark. It is meant to answer one question only:

Can the redesigned one-vehicle pipeline recover a clean trajectory almost
perfectly when the scene really contains one clear vehicle?
"""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_realistic import main as build_realistic_benchmark_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a near-perfect single-vehicle benchmark.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output .pt benchmark file.")
    parser.add_argument("--samples", type=int, default=64, help="Number of benchmark windows.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels per sample.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization denominator.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Layout of the real background .npy.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="First channel index to keep from the real background.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to the real background window.")
    parser.add_argument("--raw-window-dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Storage dtype for raw_window in the benchmark file.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    argv_forward = [
        "--out-file",
        str(args.out_file),
        "--samples",
        str(int(args.samples)),
        "--seed",
        str(int(args.seed)),
        "--n-channels",
        str(int(args.n_channels)),
        "--fs",
        str(float(args.fs)),
        "--dx-m",
        str(float(args.dx_m)),
        "--window-seconds",
        str(float(args.window_seconds)),
        "--time-downsample",
        str(int(args.time_downsample)),
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
        str(int(args.min_visible_channels)),
        "--speed-norm-kmh",
        str(float(args.speed_norm_kmh)),
        "--clip-ratio",
        str(float(args.clip_ratio)),
        "--artifact-dropout-ratio",
        "0.0",
        "--artifact-decoy-ratio",
        "0.0",
        "--artifact-competing-ratio",
        "0.0",
        "--artifact-competing-opposite-direction-ratio",
        "0.0",
        "--raw-window-dtype",
        str(args.raw_window_dtype),
    ]
    if args.background_npy is not None:
        argv_forward.extend(["--background-npy", str(args.background_npy)])
        argv_forward.extend(["--background-layout", str(args.background_layout)])
        argv_forward.extend(["--background-channel-start", str(int(args.background_channel_start))])
        argv_forward.extend(["--background-scale", str(float(args.background_scale))])
    return int(build_realistic_benchmark_main(argv_forward))


if __name__ == "__main__":
    raise SystemExit(main())
