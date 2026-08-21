"""Build a fixed benchmark from labeled real data and finetune on it in one step."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_from_labels import main as build_benchmark_main
from autotrack.dl.train_single_vehicle import main as train_single_vehicle_main


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Build a single-vehicle benchmark from manual labels and finetune a tracker on it."
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for benchmark and checkpoints.")
    parser.add_argument("--source-npy", required=True, type=Path, help="Real DAS .npy file.")
    parser.add_argument("--labels-json", required=True, type=Path, help="manual_labels.json or compatible project JSON.")
    parser.add_argument(
        "--benchmark-out-file",
        type=Path,
        default=None,
        help="Optional explicit benchmark .pt path. Defaults to <out-dir>/labels_benchmark.pt.",
    )
    parser.add_argument(
        "--keep-benchmark",
        action="store_true",
        help="Keep the intermediate benchmark file even if it was auto-generated.",
    )
    parser.add_argument("--array-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Input array layout.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Benchmark window length in seconds.")
    parser.add_argument("--margin-seconds", type=float, default=5.0, help="Extra margin around each labeled track.")
    parser.add_argument("--windows-per-track", type=int, default=3, help="Number of windows generated per labeled track.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum benchmark windows; 0 means all tracks.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to extracted raw window.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Fallback sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by labels.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


def main(argv: list[str] | None = None) -> int:
    args, remaining = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_file = Path(args.benchmark_out_file).expanduser() if args.benchmark_out_file is not None else out_dir / "labels_benchmark.pt"

    build_argv = [
        "--out-file",
        str(benchmark_file),
        "--source-npy",
        str(Path(args.source_npy).expanduser()),
        "--labels-json",
        str(Path(args.labels_json).expanduser()),
        "--array-layout",
        str(args.array_layout),
        "--channel-start",
        str(int(args.channel_start)),
        "--channel-count",
        str(int(args.channel_count)),
        "--window-seconds",
        str(float(args.window_seconds)),
        "--margin-seconds",
        str(float(args.margin_seconds)),
        "--windows-per-track",
        str(int(args.windows_per_track)),
        "--max-samples",
        str(int(args.max_samples)),
        "--background-scale",
        str(float(args.background_scale)),
        "--fs",
        str(float(args.fs)),
        "--dx-m",
        str(float(args.dx_m)),
        "--time-downsample",
        str(int(args.time_downsample)),
        "--speed-norm-kmh",
        str(float(args.speed_norm_kmh)),
        "--clip-ratio",
        str(float(args.clip_ratio)),
    ]
    if build_benchmark_main(build_argv) != 0:
        return 1

    train_argv = [
        "--out-dir",
        str(out_dir),
        "--benchmark-file",
        str(benchmark_file),
        *remaining,
    ]
    result = train_single_vehicle_main(train_argv)

    if result == 0 and not bool(args.keep_benchmark) and args.benchmark_out_file is None:
        try:
            benchmark_file.unlink(missing_ok=True)
            benchmark_file.with_suffix(".json").unlink(missing_ok=True)
        except OSError:
            pass
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
