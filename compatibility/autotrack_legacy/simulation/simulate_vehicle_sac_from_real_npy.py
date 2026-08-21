"""Generate GUI-loadable SAC data by sampling real `.npy` background windows.

用途：
    直接从真实 DAS `.npy` 中抽取一个窗口作为背景，然后叠加可控的模拟车辆轨迹，
    以得到“背景分布最接近真实数据”的训练/检查样本。相比纯参数模拟，这条路
    更适合缩小训练域和真实域的差距。

用法：
    PYTHONPATH=. uv run python autotrack/simulation/simulate_vehicle_sac_from_real_npy.py \
        --input /Volumes/SanDisk2T4/MyProjects/BaFang/xi/00gauss_large.npy \
        --out-dir datasets/gui_demo_peakslot/realbg_mix_120s \
        --window-start-seconds 1092 \
        --window-seconds 120 \
        --channel-count 50 \
        --primary-count 18 \
        --secondary-count 4 \
        --fixed-amp 6.0 \
        --speed-range-kmh 70 86 \
        --device mps

输出：
    - CH*.sac
    - tracks.json
    - vehicles.csv
    - sim_config.json
    - preview.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulate_vehicle_sac import (
    build_vehicle_table,
    parse_args,
    save_preview,
    validate_config,
    write_sac_files,
    write_tracks_json,
    write_vehicle_csv,
)
from autotrack.simulation.simulate_vehicle_sac_torch import _auto_device, overlay_vehicle_pulses_torch


def _parse_extra_args(argv: list[str]) -> tuple[list[str], argparse.Namespace]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="", help="Torch device for overlay generation: cuda, mps, cpu, or auto.")
    parser.add_argument("--input", required=True, type=Path, help="Real background .npy path.")
    parser.add_argument("--array-layout", choices=["time_channel", "channel_time"], default="time_channel", help="Input array layout.")
    parser.add_argument("--window-start-seconds", type=float, default=-1.0, help="Window start in seconds; <0 means random.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Multiply sampled real background by this factor before overlay.")
    parser.add_argument("--background-clip-min", type=float, default=0.0, help="Minimum clip value applied to sampled background.")
    parser.add_argument("--background-clip-max", type=float, default=1.0, help="Maximum clip value applied to sampled background.")
    parser.add_argument("--seed-offset", type=int, default=900000, help="Offset used for random background window sampling.")
    known, remaining = parser.parse_known_args(argv)
    return remaining, known


def _load_background_window(
    input_path: Path,
    *,
    layout: str,
    fs: float,
    duration_s: float,
    channel_start: int,
    channel_count: int,
    window_start_s: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.load(str(input_path), mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D .npy array, got shape {arr.shape}")
    n_time = int(arr.shape[0] if layout == "time_channel" else arr.shape[1])
    n_channels_all = int(arr.shape[1] if layout == "time_channel" else arr.shape[0])
    end_ch = int(channel_start + channel_count)
    if channel_start < 0 or end_ch > n_channels_all:
        raise ValueError(f"Channel slice [{channel_start}, {end_ch}) outside source shape {arr.shape}")
    window_samples = int(round(float(duration_s) * float(fs)))
    if window_samples <= 0 or window_samples > n_time:
        raise ValueError(f"Invalid window length {window_samples} for input length {n_time}")
    max_start = int(n_time - window_samples)
    if float(window_start_s) >= 0.0:
        start = int(round(float(window_start_s) * float(fs)))
        start = max(0, min(max_start, start))
    else:
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    if layout == "time_channel":
        window = np.array(arr[start : start + window_samples, channel_start:end_ch], dtype=np.float32, copy=True).T
    else:
        window = np.array(arr[channel_start:end_ch, start : start + window_samples], dtype=np.float32, copy=True)
    window = np.nan_to_num(window, copy=False)
    meta = {
        "input_file": str(input_path),
        "window_start_seconds": float(start) / float(fs),
        "window_end_seconds": float(start + window_samples) / float(fs),
        "window_samples": int(window_samples),
        "channel_start": int(channel_start),
        "channel_count": int(channel_count),
    }
    return window, meta


def _write_config_json(cfg: Any, rows: list[dict[str, Any]], *, background_meta: dict[str, Any], extra: argparse.Namespace) -> None:
    n_primary = int(sum(1 for r in rows if r["direction"] == "primary"))
    n_secondary = int(len(rows) - n_primary)
    payload = {
        **asdict(cfg),
        "out_dir": str(cfg.out_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": cfg.n_samples,
        "n_primary": n_primary,
        "n_secondary": n_secondary,
        "background_source": background_meta,
        "background_scale": float(extra.background_scale),
        "background_clip_min": float(extra.background_clip_min),
        "background_clip_max": float(extra.background_clip_max),
    }
    out_json = cfg.out_dir / "sim_config.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    remaining, extra = _parse_extra_args(raw_argv)
    raw_device = str(extra.device).strip()
    device_name = _auto_device() if raw_device in {"", "auto"} else raw_device

    old_argv = sys.argv
    sys.argv = [old_argv[0], *remaining]
    try:
        cfg = parse_args()
    finally:
        sys.argv = old_argv

    validate_config(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    background, background_meta = _load_background_window(
        Path(extra.input).expanduser(),
        layout=str(extra.array_layout),
        fs=float(cfg.fs),
        duration_s=float(cfg.duration_s),
        channel_start=int(extra.channel_start),
        channel_count=int(extra.channel_count),
        window_start_s=float(extra.window_start_seconds),
        seed=int(cfg.seed) + int(extra.seed_offset),
    )
    if int(background.shape[0]) != int(cfg.n_ch):
        raise ValueError(
            f"Sampled background channel count is {background.shape[0]}, but cfg.n_ch={cfg.n_ch}. "
            "Set --n-ch to match --channel-count."
        )
    background = np.clip(
        background * float(extra.background_scale),
        float(extra.background_clip_min),
        float(extra.background_clip_max),
    ).astype(np.float32, copy=False)

    device = torch.device(device_name)
    print(f"Using torch device for real-background mixing: {device}", flush=True)
    t0 = time.perf_counter()
    data = torch.from_numpy(background).to(device)
    rng = np.random.default_rng(int(cfg.seed))
    rows = build_vehicle_table(cfg, rng)
    tracks = overlay_vehicle_pulses_torch(data, rows, cfg, rng)
    matrix_elapsed = time.perf_counter() - t0
    data_np = data.detach().to("cpu").numpy().astype(np.float32, copy=False)

    t_write = time.perf_counter()
    write_sac_files(data_np, cfg)
    write_vehicle_csv(rows, cfg)
    write_tracks_json(tracks, cfg)
    save_preview(data_np, cfg)
    _write_config_json(cfg, rows, background_meta=background_meta, extra=extra)
    write_elapsed = time.perf_counter() - t_write

    n_primary = sum(1 for r in rows if r["direction"] == "primary")
    n_secondary = len(rows) - n_primary
    print(f"Done. Output: {cfg.out_dir}", flush=True)
    print(f"SAC files: {cfg.n_ch}, samples/channel: {cfg.n_samples}, fs: {cfg.fs} Hz", flush=True)
    print(f"Vehicles: {len(rows)} (primary={n_primary}, secondary={n_secondary})", flush=True)
    print(
        f"Background window: [{background_meta['window_start_seconds']:.1f}, {background_meta['window_end_seconds']:.1f}] s from {background_meta['input_file']}",
        flush=True,
    )
    print(f"Matrix generation: {matrix_elapsed:.2f}s, write/preview: {write_elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
