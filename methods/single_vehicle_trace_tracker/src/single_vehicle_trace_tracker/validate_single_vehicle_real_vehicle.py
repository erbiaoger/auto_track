"""Validate a trained real-vehicle single-track checkpoint on the mixed benchmark and raw smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.build_single_vehicle_benchmark_realistic import main as build_realistic_benchmark_main
from autotrack.dl.build_single_vehicle_benchmark_from_track_slot import main as build_trackslot_benchmark_main
from autotrack.dl.evaluate_single_vehicle import main as evaluate_single_vehicle_main
from autotrack.dl.merge_single_vehicle_benchmarks import main as merge_benchmarks_main
from autotrack.dl.predict_single_vehicle_real_vehicle import main as predict_real_vehicle_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the real-vehicle single-track checkpoint.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path to validate.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for validation outputs.")
    parser.add_argument(
        "--realbg-npy",
        type=Path,
        default=Path("datasets/03gauss_large.npy"),
        help="Real background .npy used for the validation benchmark.",
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
    parser.add_argument("--device", default="auto", help="Torch device for evaluation and smoke inference.")
    parser.add_argument("--eval-samples", type=int, default=32, help="Sample count for benchmark evaluation.")
    parser.add_argument("--smoke-windows", type=int, default=24, help="Number of windows for the raw smoke check.")
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

    eval_argv = [
        "--model",
        str(Path(args.model).expanduser()),
        "--out-dir",
        str(out_dir / "eval"),
        "--benchmark-file",
        str(mixed_bench),
        "--device",
        str(args.device),
        "--samples",
        str(int(args.eval_samples)),
    ]
    if evaluate_single_vehicle_main(eval_argv) != 0:
        return 1

    smoke_argv = [
        "--model",
        str(Path(args.model).expanduser()),
        "--input",
        str(Path(args.realbg_npy).expanduser()),
        "--out-dir",
        str(out_dir / "smoke"),
        "--device",
        str(args.device),
        "--max-windows",
        str(int(args.smoke_windows)),
    ]
    if predict_real_vehicle_main(smoke_argv) != 0:
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
