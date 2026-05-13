"""Segment a real DAS `.npy` array into PeakSlotNet tensor shards.

Purpose:
    Convert one real DAS heatmap stored as a NumPy array into 120 s (or other
    length) `peak_slot` shards that can be read directly by
    `autotrack.dl.predict_peak_slot_dataset`. This is intended for unlabeled
    real data inspection: the output contains input tensors and peak candidates,
    while all GT tensors are empty placeholders.

Expected input:
    By default the source array layout is `[time, channel]`, for example
    `gauss_section.npy` with shape `[200001, 51]`. The script slices channels,
    cuts overlapping time windows, normalizes each window with the same helper
    used by the model inference path, and detects fixed-count peak candidates
    per channel.

Example:
    uv run python -m autotrack.dl.segment_real_npy_to_peak_slot \
        --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/saved_arrays/gauss_section.npy \
        --out-dir datasets/peak_slot/xi_gauss_50_120s_stride60 \
        --window-seconds 120 \
        --stride-seconds 60 \
        --fs 1000 \
        --channel-start 0 \
        --channel-count 50 \
        --overwrite

Outputs:
    The output directory contains `meta.json` and `shard_*.pt`. Each shard stores:
    x, peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, visibility,
    direction, speed, and gt_valid. For unlabeled real data, `gt_valid` has shape
    `[sample, 0]`, and `has_ground_truth=false` is recorded in metadata.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_tensor
from autotrack.dl.trajectory_set_model import prepare_window_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut a real DAS .npy array into PeakSlotNet shards.")
    parser.add_argument("--input", required=True, type=Path, help="Input NumPy array path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak_slot dataset directory.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Output segment length in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=60.0, help="Sliding-window stride in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample factor for model input.")
    parser.add_argument("--channel-start", type=int, default=0, help="First source channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of source channels to keep.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Robust normalized clipping ratio.")
    parser.add_argument("--input-mode", choices=["raw", "raw_abs"], default="raw", help="Input feature mode for prepare_window_input.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization metadata value.")
    parser.add_argument("--x-dtype", choices=["float32", "float16"], default="float32", help="Stored dtype for x tensor.")
    parser.add_argument("--shard-size", type=int, default=256, help="Samples per shard.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates K per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum distance between peaks on one channel.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Minimum normalized peak height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum normalized peak prominence.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="Metadata value for GT matching tolerance.")
    parser.add_argument("--overwrite", action="store_true", help="Replace a non-empty output directory.")
    return parser.parse_args()


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _window_starts(n_time: int, window_samples: int, stride_samples: int) -> list[int]:
    if window_samples <= 0:
        raise ValueError("window_samples must be > 0")
    if stride_samples <= 0:
        raise ValueError("stride_samples must be > 0")
    if n_time < window_samples:
        raise ValueError(f"Input has {n_time} samples, shorter than one window of {window_samples} samples.")
    max_start = int(n_time - window_samples)
    starts = list(range(0, max_start + 1, int(stride_samples)))
    if not starts:
        starts = [0]
    if starts[-1] != max_start:
        starts.append(max_start)
    return sorted(set(int(s) for s in starts))


def _extract_window(
    source: np.ndarray,
    *,
    layout: str,
    start: int,
    window_samples: int,
    channel_start: int,
    channel_count: int,
) -> np.ndarray:
    channel_end = int(channel_start + channel_count)
    if layout == "time_channel":
        if source.ndim != 2:
            raise ValueError("time_channel input must be a 2-D array")
        if channel_start < 0 or channel_end > int(source.shape[1]):
            raise ValueError(f"Requested channel slice [{channel_start}, {channel_end}) outside source shape {source.shape}")
        window = source[int(start) : int(start + window_samples), int(channel_start) : int(channel_end)].T
    else:
        if source.ndim != 2:
            raise ValueError("channel_time input must be a 2-D array")
        if channel_start < 0 or channel_end > int(source.shape[0]):
            raise ValueError(f"Requested channel slice [{channel_start}, {channel_end}) outside source shape {source.shape}")
        window = source[int(channel_start) : int(channel_end), int(start) : int(start + window_samples)]
    return np.nan_to_num(np.array(window, dtype=np.float32, copy=True), copy=False)


def _peak_stats(peak_valid: torch.Tensor) -> dict[str, int | float]:
    per_channel = peak_valid.to(torch.int64).sum(dim=1)
    return {
        "valid_peak_total": int(per_channel.sum().item()),
        "valid_peak_min_per_channel": int(per_channel.min().item()) if per_channel.numel() else 0,
        "valid_peak_median_per_channel": float(torch.median(per_channel.to(torch.float32)).item()) if per_channel.numel() else 0.0,
        "valid_peak_max_per_channel": int(per_channel.max().item()) if per_channel.numel() else 0,
    }


def _save_shard(
    out_path: Path,
    *,
    xs: list[torch.Tensor],
    peak_times: list[torch.Tensor],
    peak_amps: list[torch.Tensor],
    peak_valids: list[torch.Tensor],
    peak_indices: list[torch.Tensor],
    n_channels: int,
) -> int:
    sample_count = len(xs)
    if sample_count <= 0:
        return 0
    payload = {
        "x": torch.stack(xs, dim=0).contiguous(),
        "peak_time": torch.stack(peak_times, dim=0).contiguous(),
        "peak_amp": torch.stack(peak_amps, dim=0).contiguous(),
        "peak_valid": torch.stack(peak_valids, dim=0).contiguous(),
        "peak_index": torch.stack(peak_indices, dim=0).contiguous(),
        "gt_peak_index": torch.empty((sample_count, 0, int(n_channels)), dtype=torch.long),
        "visibility": torch.empty((sample_count, 0, int(n_channels)), dtype=torch.float32),
        "direction": torch.empty((sample_count, 0), dtype=torch.long),
        "speed": torch.empty((sample_count, 0), dtype=torch.float32),
        "gt_valid": torch.empty((sample_count, 0), dtype=torch.bool),
    }
    torch.save(payload, str(out_path))
    return sample_count


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fs = float(args.fs)
    window_samples = int(round(float(args.window_seconds) * fs))
    stride_samples = int(round(float(args.stride_seconds) * fs))
    time_downsample = int(max(1, args.time_downsample))
    channel_count = int(args.channel_count)
    if channel_count <= 0:
        raise ValueError("--channel-count must be > 0")

    source = np.load(str(input_path), mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D input array, got shape {source.shape}")
    n_time = int(source.shape[0] if args.array_layout == "time_channel" else source.shape[1])
    starts = _window_starts(n_time, window_samples, stride_samples)
    _prepare_out_dir(out_dir, bool(args.overwrite))

    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(args.peak_candidates_per_channel),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=float(args.peak_match_tolerance_s),
    )
    shard_size = int(max(1, args.shard_size))
    x_dtype = torch.float16 if str(args.x_dtype) == "float16" else torch.float32

    shards: list[str] = []
    per_window_peak_stats: list[dict[str, Any]] = []
    current_xs: list[torch.Tensor] = []
    current_peak_times: list[torch.Tensor] = []
    current_peak_amps: list[torch.Tensor] = []
    current_peak_valids: list[torch.Tensor] = []
    current_peak_indices: list[torch.Tensor] = []
    total_samples = 0
    t0 = time.perf_counter()

    for sample_idx, start in enumerate(starts):
        data_window = _extract_window(
            source,
            layout=str(args.array_layout),
            start=int(start),
            window_samples=window_samples,
            channel_start=int(args.channel_start),
            channel_count=channel_count,
        )
        x = prepare_window_input(
            data_window,
            time_downsample=time_downsample,
            clip_ratio=float(args.clip_ratio),
            input_mode=str(args.input_mode),
        ).to(dtype=x_dtype)
        peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates_from_tensor(
            x[0].to(torch.float32),
            fs=fs,
            time_downsample=time_downsample,
            window_samples=window_samples,
            config=peak_cfg,
        )
        stat = _peak_stats(peak_valid)
        stat.update(
            {
                "sample_index": int(sample_idx),
                "window_start_sample": int(start),
                "window_start_seconds": float(start) / fs,
            }
        )
        per_window_peak_stats.append(stat)
        current_xs.append(x)
        current_peak_times.append(peak_time)
        current_peak_amps.append(peak_amp)
        current_peak_valids.append(peak_valid)
        current_peak_indices.append(peak_index)

        if len(current_xs) >= shard_size:
            shard_name = f"shard_{len(shards):06d}.pt"
            total_samples += _save_shard(
                out_dir / shard_name,
                xs=current_xs,
                peak_times=current_peak_times,
                peak_amps=current_peak_amps,
                peak_valids=current_peak_valids,
                peak_indices=current_peak_indices,
                n_channels=channel_count,
            )
            shards.append(shard_name)
            current_xs = []
            current_peak_times = []
            current_peak_amps = []
            current_peak_valids = []
            current_peak_indices = []
            print(f"wrote {shard_name}: total_samples={total_samples}", flush=True)

    if current_xs:
        shard_name = f"shard_{len(shards):06d}.pt"
        total_samples += _save_shard(
            out_dir / shard_name,
            xs=current_xs,
            peak_times=current_peak_times,
            peak_amps=current_peak_amps,
            peak_valids=current_peak_valids,
            peak_indices=current_peak_indices,
            n_channels=channel_count,
        )
        shards.append(shard_name)
        print(f"wrote {shard_name}: total_samples={total_samples}", flush=True)

    totals = [int(item["valid_peak_total"]) for item in per_window_peak_stats]
    meta = {
        "format": "peak_slot_shards_v1",
        "mode": "peak_slot_real_npy_segments",
        "created_at_unix": time.time(),
        "source_file": str(input_path),
        "source_shape_time_channel": list(source.shape) if args.array_layout == "time_channel" else [int(source.shape[1]), int(source.shape[0])],
        "source_array": input_path.stem,
        "layout_note": f"source layout is {args.array_layout}; shard x is [sample, feature, channel, downsampled_time]",
        "num_samples": int(total_samples),
        "shard_size": int(shard_size),
        "shards": shards,
        "n_channels": int(channel_count),
        "original_n_channels": int(source.shape[1] if args.array_layout == "time_channel" else source.shape[0]),
        "channel_slice": [int(args.channel_start), int(args.channel_start + channel_count)],
        "in_channels": 2 if str(args.input_mode) == "raw_abs" else 1,
        "input_mode": str(args.input_mode),
        "fs": fs,
        "dx_m": float(args.dx_m),
        "window_seconds": float(args.window_seconds),
        "stride_seconds": float(args.stride_seconds),
        "window_samples": int(window_samples),
        "stride_samples": int(stride_samples),
        "window_start_samples": [int(s) for s in starts],
        "window_start_seconds": [float(s) / fs for s in starts],
        "time_downsample": int(time_downsample),
        "downsampled_time": int((window_samples + time_downsample - 1) // time_downsample),
        "downsampled_time_samples": int((window_samples + time_downsample - 1) // time_downsample),
        "speed_norm_kmh": float(args.speed_norm_kmh),
        "clip_ratio": float(args.clip_ratio),
        "x_dtype": str(args.x_dtype),
        "max_gt": 0,
        "has_ground_truth": False,
        "note": "No manual trajectory labels are included; gt_* tensors are empty placeholders for prediction/inspection only.",
        "peak_candidates_per_channel": int(args.peak_candidates_per_channel),
        "peak_none_index": int(args.peak_candidates_per_channel),
        "peak_detection": {
            "candidates_per_channel": int(args.peak_candidates_per_channel),
            "min_distance_s": float(args.peak_min_distance_s),
            "min_height": float(args.peak_min_height),
            "prominence": float(args.peak_prominence),
            "match_tolerance_s": float(args.peak_match_tolerance_s),
        },
        "peak_stats": {
            "valid_peak_total": int(sum(totals)),
            "valid_peak_min_per_window": int(min(totals)) if totals else 0,
            "valid_peak_median_per_window": float(np.median(np.asarray(totals, dtype=np.float32))) if totals else 0.0,
            "valid_peak_max_per_window": int(max(totals)) if totals else 0,
            "per_window": per_window_peak_stats,
        },
        "elapsed_seconds": float(time.perf_counter() - t0),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: samples={total_samples}, shards={len(shards)}, starts={meta['window_start_samples']}, "
        f"out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
