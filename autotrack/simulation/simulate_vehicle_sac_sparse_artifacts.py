"""Generate SAC simulation with isolated Gaussian artifacts and many missing channels.

用途：
    为 GUI / Deep Learning 提取流程生成更贴近当前真实背景假设的 SAC 数据：
    背景不是连续小噪声，而是由大量孤立高斯峰、整道缺失、局部缺块组成。
    车辆轨迹仍然沿用项目现有模拟器的真值输出格式，因此可直接导入 GUI，
    也可用 `tracks.json` 做对照检查。

用法：
    PYTHONPATH=. uv run python autotrack/simulation/simulate_vehicle_sac_sparse_artifacts.py \
        --out-dir datasets/gui_demo_peakslot/artifact_sparse_120s \
        --seed 404 \
        --primary-count 18 \
        --secondary-count 4 \
        --duration-s 120 \
        --fs 1000 \
        --n-ch 50 \
        --dx-m 100 \
        --fixed-amp 6.0 \
        --speed-range-kmh 70 86 \
        --isolated-peak-count 320 \
        --isolated-peak-amp-range 1.0 7.0 \
        --isolated-peak-sigma-t-range 0.05 0.22 \
        --isolated-peak-sigma-ch-range 0.03 0.30 \
        --dead-channel-count 10 \
        --drop-block-count 70 \
        --drop-block-channel-width-range 1 4 \
        --drop-block-duration-range-s 0.6 4.0 \
        --device mps

输出：
    - CH*.sac：每通道一个 SAC 文件，可直接导入 GUI。
    - tracks.json：车辆真值轨迹。
    - vehicles.csv：车辆参数。
    - sim_config.json：仿真参数与伪迹统计。
    - preview.png：预览图。

说明：
    该脚本默认不加入连续白噪声；如果需要可用 `--continuous-noise-std`
    额外叠加，但默认建议保持 0，以便突出“孤立高斯峰 + 缺道”背景。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from simulate_vehicle_sac import (
    _json_float,
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
    parser.add_argument("--device", default="", help="Torch device for matrix generation: cuda, mps, cpu, or auto.")
    parser.add_argument("--continuous-noise-std", type=float, default=0.0, help="Optional continuous background noise std; default 0.")
    parser.add_argument("--isolated-peak-count", type=int, default=320, help="Number of isolated Gaussian artifact peaks.")
    parser.add_argument("--isolated-peak-amp-range", nargs=2, type=float, default=[1.0, 7.0], metavar=("MIN", "MAX"), help="Amplitude range of isolated artifact peaks.")
    parser.add_argument("--isolated-peak-sigma-t-range", nargs=2, type=float, default=[0.05, 0.22], metavar=("MIN", "MAX"), help="Time sigma range (s) of isolated artifact peaks.")
    parser.add_argument("--isolated-peak-sigma-ch-range", nargs=2, type=float, default=[0.03, 0.30], metavar=("MIN", "MAX"), help="Channel sigma range of isolated artifact peaks.")
    parser.add_argument("--dead-channel-count", type=int, default=10, help="Number of fully missing channels.")
    parser.add_argument("--drop-block-count", type=int, default=70, help="Number of channel-time missing blocks.")
    parser.add_argument("--drop-block-channel-width-range", nargs=2, type=int, default=[1, 4], metavar=("MIN", "MAX"), help="Channel width range of missing blocks.")
    parser.add_argument("--drop-block-duration-range-s", nargs=2, type=float, default=[0.6, 4.0], metavar=("MIN", "MAX"), help="Duration range (s) of missing blocks.")
    parser.add_argument("--artifact-seed-offset", type=int, default=100000, help="Seed offset used for isolated artifacts and missing blocks.")
    known, remaining = parser.parse_known_args(argv)
    return remaining, known


def _validate_extra_args(extra: argparse.Namespace, n_ch: int, duration_s: float) -> None:
    amp_min, amp_max = float(extra.isolated_peak_amp_range[0]), float(extra.isolated_peak_amp_range[1])
    sigma_t_min, sigma_t_max = float(extra.isolated_peak_sigma_t_range[0]), float(extra.isolated_peak_sigma_t_range[1])
    sigma_ch_min, sigma_ch_max = float(extra.isolated_peak_sigma_ch_range[0]), float(extra.isolated_peak_sigma_ch_range[1])
    width_min, width_max = int(extra.drop_block_channel_width_range[0]), int(extra.drop_block_channel_width_range[1])
    dur_min, dur_max = float(extra.drop_block_duration_range_s[0]), float(extra.drop_block_duration_range_s[1])
    if float(extra.continuous_noise_std) < 0.0:
        raise ValueError("--continuous-noise-std must be >= 0.")
    if int(extra.isolated_peak_count) < 0:
        raise ValueError("--isolated-peak-count must be >= 0.")
    if amp_min <= 0.0 or amp_max <= 0.0 or amp_min > amp_max:
        raise ValueError("--isolated-peak-amp-range must satisfy 0 < min <= max.")
    if sigma_t_min <= 0.0 or sigma_t_max <= 0.0 or sigma_t_min > sigma_t_max:
        raise ValueError("--isolated-peak-sigma-t-range must satisfy 0 < min <= max.")
    if sigma_ch_min <= 0.0 or sigma_ch_max <= 0.0 or sigma_ch_min > sigma_ch_max:
        raise ValueError("--isolated-peak-sigma-ch-range must satisfy 0 < min <= max.")
    if int(extra.dead_channel_count) < 0 or int(extra.dead_channel_count) > int(n_ch):
        raise ValueError("--dead-channel-count must be in [0, n_ch].")
    if int(extra.drop_block_count) < 0:
        raise ValueError("--drop-block-count must be >= 0.")
    if width_min <= 0 or width_max <= 0 or width_min > width_max:
        raise ValueError("--drop-block-channel-width-range must satisfy 0 < min <= max.")
    if width_max > int(n_ch):
        raise ValueError("--drop-block-channel-width-range max must be <= n_ch.")
    if dur_min <= 0.0 or dur_max <= 0.0 or dur_min > dur_max or dur_max > float(duration_s):
        raise ValueError("--drop-block-duration-range-s must satisfy 0 < min <= max <= duration_s.")


def _sample_isolated_artifacts(
    data: torch.Tensor,
    *,
    cfg: Any,
    extra: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n_ch = int(cfg.n_ch)
    n_samples = int(cfg.n_samples)
    t_axis = torch.arange(n_samples, device=data.device, dtype=torch.float32) / float(cfg.fs)
    ch_axis = torch.arange(n_ch, device=data.device, dtype=torch.float32)
    amp_min, amp_max = float(extra.isolated_peak_amp_range[0]), float(extra.isolated_peak_amp_range[1])
    sigma_t_min, sigma_t_max = float(extra.isolated_peak_sigma_t_range[0]), float(extra.isolated_peak_sigma_t_range[1])
    sigma_ch_min, sigma_ch_max = float(extra.isolated_peak_sigma_ch_range[0]), float(extra.isolated_peak_sigma_ch_range[1])

    for artifact_id in range(int(extra.isolated_peak_count)):
        center_time = float(rng.uniform(0.0, float(cfg.duration_s)))
        center_ch = float(rng.uniform(0.0, max(0.0, float(n_ch - 1))))
        amp = float(rng.uniform(amp_min, amp_max))
        sigma_t = float(rng.uniform(sigma_t_min, sigma_t_max))
        sigma_ch = float(rng.uniform(sigma_ch_min, sigma_ch_max))
        t_profile = torch.exp(-0.5 * ((t_axis - center_time) / max(1e-6, sigma_t)) ** 2)
        ch_profile = torch.exp(-0.5 * ((ch_axis - center_ch) / max(1e-6, sigma_ch)) ** 2)
        blob = amp * ch_profile[:, None] * t_profile[None, :]
        data += blob.to(dtype=data.dtype)
        records.append(
            {
                "artifact_id": int(artifact_id),
                "type": "isolated_gaussian_peak",
                "center_time_s": float(center_time),
                "center_channel": float(center_ch),
                "amp": float(amp),
                "sigma_t_s": float(sigma_t),
                "sigma_ch": float(sigma_ch),
            }
        )
    return records


def _apply_dead_channels(
    data: torch.Tensor,
    *,
    extra: argparse.Namespace,
    rng: np.random.Generator,
) -> list[int]:
    count = int(extra.dead_channel_count)
    if count <= 0:
        return []
    order = rng.permutation(int(data.shape[0])).tolist()
    dead_channels = sorted(int(ch) for ch in order[:count])
    if dead_channels:
        data[dead_channels, :] = 0.0
    return dead_channels


def _apply_drop_blocks(
    data: torch.Tensor,
    *,
    cfg: Any,
    extra: argparse.Namespace,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    n_ch = int(cfg.n_ch)
    n_samples = int(cfg.n_samples)
    width_min, width_max = int(extra.drop_block_channel_width_range[0]), int(extra.drop_block_channel_width_range[1])
    dur_min, dur_max = float(extra.drop_block_duration_range_s[0]), float(extra.drop_block_duration_range_s[1])
    for block_id in range(int(extra.drop_block_count)):
        width = int(rng.integers(width_min, width_max + 1))
        start_ch = int(rng.integers(0, max(1, n_ch - width + 1)))
        duration_s = float(rng.uniform(dur_min, dur_max))
        duration_samples = int(max(1, round(duration_s * float(cfg.fs))))
        start_t = int(rng.integers(0, max(1, n_samples - duration_samples + 1)))
        end_ch = start_ch + width
        end_t = min(n_samples, start_t + duration_samples)
        data[start_ch:end_ch, start_t:end_t] = 0.0
        records.append(
            {
                "drop_block_id": int(block_id),
                "channel_start": int(start_ch),
                "channel_end_exclusive": int(end_ch),
                "time_start_s": float(start_t) / float(cfg.fs),
                "time_end_s": float(end_t) / float(cfg.fs),
                "duration_s": float(end_t - start_t) / float(cfg.fs),
            }
        )
    return records


def _write_config_json(cfg: Any, rows: list[dict[str, Any]], *, extra: argparse.Namespace, artifact_summary: dict[str, Any]) -> None:
    n_primary = int(sum(1 for r in rows if r["direction"] == "primary"))
    n_secondary = int(len(rows) - n_primary)
    payload = {
        **asdict(cfg),
        "out_dir": str(cfg.out_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": cfg.n_samples,
        "n_primary": n_primary,
        "n_secondary": n_secondary,
        "continuous_noise_std": float(extra.continuous_noise_std),
        "isolated_peak_count": int(extra.isolated_peak_count),
        "isolated_peak_amp_range": [float(extra.isolated_peak_amp_range[0]), float(extra.isolated_peak_amp_range[1])],
        "isolated_peak_sigma_t_range": [float(extra.isolated_peak_sigma_t_range[0]), float(extra.isolated_peak_sigma_t_range[1])],
        "isolated_peak_sigma_ch_range": [float(extra.isolated_peak_sigma_ch_range[0]), float(extra.isolated_peak_sigma_ch_range[1])],
        "dead_channel_count": int(extra.dead_channel_count),
        "drop_block_count": int(extra.drop_block_count),
        "drop_block_channel_width_range": [int(extra.drop_block_channel_width_range[0]), int(extra.drop_block_channel_width_range[1])],
        "drop_block_duration_range_s": [float(extra.drop_block_duration_range_s[0]), float(extra.drop_block_duration_range_s[1])],
        "artifact_summary": artifact_summary,
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
    _validate_extra_args(extra, n_ch=int(cfg.n_ch), duration_s=float(cfg.duration_s))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_name)
    print(f"Using torch device for sparse-artifact generation: {device}", flush=True)
    t0 = time.perf_counter()

    gen_device = device if device.type != "mps" else torch.device("cpu")
    generator = torch.Generator(device=gen_device)
    generator.manual_seed(int(cfg.seed))
    if float(extra.continuous_noise_std) > 0.0:
        data = torch.normal(
            mean=0.0,
            std=float(extra.continuous_noise_std),
            size=(int(cfg.n_ch), int(cfg.n_samples)),
            generator=generator,
            device=gen_device,
            dtype=torch.float32,
        )
    else:
        data = torch.zeros((int(cfg.n_ch), int(cfg.n_samples)), device=gen_device, dtype=torch.float32)
    if gen_device != device:
        data = data.to(device)

    rng = np.random.default_rng(int(cfg.seed))
    rows = build_vehicle_table(cfg, rng)
    tracks = overlay_vehicle_pulses_torch(data, rows, cfg, rng)

    artifact_rng = np.random.default_rng(int(cfg.seed) + int(extra.artifact_seed_offset))
    isolated_records = _sample_isolated_artifacts(data, cfg=cfg, extra=extra, rng=artifact_rng)
    dead_channels = _apply_dead_channels(data, extra=extra, rng=artifact_rng)
    drop_block_records = _apply_drop_blocks(data, cfg=cfg, extra=extra, rng=artifact_rng)

    matrix_elapsed = time.perf_counter() - t0
    data_np = data.detach().to("cpu").numpy().astype(np.float32, copy=False)

    t_write = time.perf_counter()
    write_sac_files(data_np, cfg)
    write_vehicle_csv(rows, cfg)
    write_tracks_json(tracks, cfg)
    save_preview(data_np, cfg)
    artifact_summary = {
        "dead_channels": dead_channels,
        "dead_channel_count": int(len(dead_channels)),
        "isolated_gaussian_peak_count": int(len(isolated_records)),
        "drop_block_count": int(len(drop_block_records)),
        "drop_blocks_preview": drop_block_records[:10],
        "isolated_peaks_preview": isolated_records[:10],
    }
    _write_config_json(cfg, rows, extra=extra, artifact_summary=artifact_summary)
    write_elapsed = time.perf_counter() - t_write

    n_primary = sum(1 for r in rows if r["direction"] == "primary")
    n_secondary = len(rows) - n_primary
    print(f"Done. Output: {cfg.out_dir}", flush=True)
    print(f"SAC files: {cfg.n_ch}, samples/channel: {cfg.n_samples}, fs: {cfg.fs} Hz", flush=True)
    print(f"Vehicles: {len(rows)} (primary={n_primary}, secondary={n_secondary})", flush=True)
    print(
        f"Artifacts: isolated_gaussian_peaks={len(isolated_records)}, dead_channels={len(dead_channels)}, drop_blocks={len(drop_block_records)}",
        flush=True,
    )
    print(f"Matrix generation: {matrix_elapsed:.2f}s, write/preview: {write_elapsed:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
