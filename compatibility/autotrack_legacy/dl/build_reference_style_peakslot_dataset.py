"""Build a reference-style TrackSlot -> PeakSlot+prior dataset for inspection.

This is the style that matches the sample you pointed to:

- multi-vehicle windows, not a single-vehicle benchmark
- TrackSlot heatmap + GT polylines
- PeakSlot candidate tables after conversion
- waveform-line U-Net prior appended as the second input channel

The script is a thin orchestrator around the existing generators. It keeps the
data flow explicit so it is hard to confuse with the single-vehicle benchmark,
and it can also render the same diagnostic figures as the reference sample
directory the user pointed to.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reference-style TrackSlot/PeakSlot dataset.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak-slot dataset directory.")
    parser.add_argument("--track-out-dir", type=Path, default=None, help="Optional intermediate track-slot directory.")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("datasets/profiles/xi_gauss_50_realbg/realism_profile.json"),
        help="realism_profile.json forwarded to track-slot generation.",
    )
    parser.add_argument("--profile-strength", type=float, default=1.0, help="Profile blend factor forwarded to track-slot generation.")
    parser.add_argument("--unet-checkpoint", required=True, type=Path, help="waveform_line_task U-Net checkpoint.")
    parser.add_argument(
        "--waveform-task-dir",
        type=Path,
        default=Path("/csim2/zhangzhiyu/MyProjects/waveform_line_task"),
        help="Path containing waveform_line_task model/ and render.py modules.",
    )
    parser.add_argument("--num-samples", type=int, default=4, help="Number of samples to generate.")
    parser.add_argument("--shard-size", type=int, default=4, help="Samples per shard.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window length in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--n-ch", type=int, default=50, help="Number of channels.")
    parser.add_argument("--vehicles-min", type=int, default=10, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=18, help="Maximum vehicles per window.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--interaction-ratio", type=float, default=0.5, help="Fraction of interaction-heavy samples.")
    parser.add_argument("--isolated-noise-ratio", type=float, default=0.8, help="Fraction of samples with isolated noise peaks.")
    parser.add_argument("--random-dead-channel-ratio", type=float, default=0.5, help="Fraction of samples with extra dead channels.")
    parser.add_argument("--zero-background-ratio", type=float, default=0.4, help="Fraction of samples with missing channel-time blocks.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Peak candidate minimum distance.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Peak candidate minimum height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Peak candidate prominence.")
    parser.add_argument("--prior-threshold", type=float, default=0.0, help="Set prior values below this probability to 0.")
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Multiplier applied to the U-Net prior.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device for U-Net inference.")
    parser.add_argument("--batch-size", type=int, default=16, help="U-Net inference batch size.")
    parser.add_argument("--image-size", type=int, default=512, help="Rendered image size for U-Net input.")
    parser.add_argument("--waveform-line-width", type=int, default=1, help="Rendered waveform trace width.")
    parser.add_argument("--wiggle-fraction", type=float, default=0.28, help="Rendered trace wiggle fraction.")
    parser.add_argument("--robust-percentile", type=float, default=99.5, help="Rendered trace robust scaling percentile.")
    parser.add_argument("--plot-out-dir", type=Path, default=None, help="Optional directory for diagnostic plot images.")
    parser.add_argument("--plot-sample-indices", default="0,1,2,3", help="Comma-separated sample indices to plot.")
    parser.add_argument("--plot-peaks", action="store_true", help="Draw peak candidates on the target panel.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directories.")
    return parser.parse_args(argv)


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _python_module_cmd(module: str) -> list[str]:
    if shutil.which("uv") is not None:
        return ["uv", "run", "python", "-m", module]
    return [sys.executable, "-m", module]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    track_out_dir = Path(args.track_out_dir).expanduser() if args.track_out_dir is not None else out_dir.with_name(f"{out_dir.name}_track")
    plot_out_dir = Path(args.plot_out_dir).expanduser() if args.plot_out_dir is not None else out_dir.with_name(f"{out_dir.name}_plots")

    generate_cmd = [
        *_python_module_cmd("autotrack.dl.generate_track_slot_dataset"),
        "--out-dir",
        str(track_out_dir),
        "--num-samples",
        str(int(args.num_samples)),
        "--shard-size",
        str(int(args.shard_size)),
        "--window-seconds",
        str(float(args.window_seconds)),
        "--time-downsample",
        str(int(args.time_downsample)),
        "--n-ch",
        str(int(args.n_ch)),
        "--vehicles-min",
        str(int(args.vehicles_min)),
        "--vehicles-max",
        str(int(args.vehicles_max)),
        "--speed-min-kmh",
        str(float(args.speed_min_kmh)),
        "--speed-max-kmh",
        str(float(args.speed_max_kmh)),
        "--interaction-ratio",
        str(float(args.interaction_ratio)),
        "--isolated-noise-ratio",
        str(float(args.isolated_noise_ratio)),
        "--random-dead-channel-ratio",
        str(float(args.random_dead_channel_ratio)),
        "--zero-background-ratio",
        str(float(args.zero_background_ratio)),
        "--overwrite",
    ]
    if args.profile is not None:
        generate_cmd.extend(["--profile", str(Path(args.profile).expanduser())])
        generate_cmd.extend(["--profile-strength", str(float(args.profile_strength))])
    _run(generate_cmd)

    prior_cmd = [
        *_python_module_cmd("autotrack.dl.build_peak_slot_unetprior_from_track"),
        "--track-dir",
        str(track_out_dir),
        "--out-dir",
        str(out_dir),
        "--unet-checkpoint",
        str(Path(args.unet_checkpoint).expanduser()),
        "--waveform-task-dir",
        str(Path(args.waveform_task_dir).expanduser()),
        "--peak-candidates-per-channel",
        str(int(args.peak_candidates_per_channel)),
        "--peak-min-distance-s",
        str(float(args.peak_min_distance_s)),
        "--peak-min-height",
        str(float(args.peak_min_height)),
        "--peak-prominence",
        str(float(args.peak_prominence)),
        "--prior-threshold",
        str(float(args.prior_threshold)),
        "--prior-scale",
        str(float(args.prior_scale)),
        "--device",
        str(args.device),
        "--batch-size",
        str(int(args.batch_size)),
        "--image-size",
        str(int(args.image_size)),
        "--waveform-line-width",
        str(int(args.waveform_line_width)),
        "--wiggle-fraction",
        str(float(args.wiggle_fraction)),
        "--robust-percentile",
        str(float(args.robust_percentile)),
        "--overwrite",
    ]
    _run(prior_cmd)

    if bool(args.plot_sample_indices):
        plot_cmd = [
            *_python_module_cmd("autotrack.dl.plot_peak_slot_unetprior_samples"),
            "--data-dir",
            str(out_dir),
            "--out-dir",
            str(plot_out_dir),
            "--sample-indices",
            str(args.plot_sample_indices),
        ]
        if bool(args.plot_peaks):
            plot_cmd.append("--plot-peaks")
        _run(plot_cmd)

    print(f"built reference-style dataset: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
