"""Generate synthetic DAS SAC data with torch acceleration.

用途：
    这是 `simulate_vehicle_sac.py` 的独立加速版本，不修改原文件。它复用原
    模拟器的 CLI 参数、车辆表生成、SAC/CSV/JSON 写出逻辑，只把白噪声生成
    和车辆高斯脉冲叠加放到 PyTorch 上执行。可用 `--device cuda` 在 NVIDIA
    显卡上生成数据，也可用 `--device mps` 或 `--device cpu`。

用例：
    uv run python simulate_vehicle_sac_torch.py \
        --out-dir datasets/train/sim_0001 \
        --seed 1 \
        --primary-count 300 \
        --secondary-count 60 \
        --noise-std 0.3 \
        --fixed-amp 6.0 \
        --duration-s 3600 \
        --fs 1000 \
        --n-ch 50 \
        --dx-m 100 \
        --speed-range-kmh 60 110 \
        --speed-jitter-kmh-range -3 3 \
        --accel-count 20 \
        --decel-count 20 \
        --stop-go-count 10 \
        --device cuda

输出：
    与 `simulate_vehicle_sac.py` 一致：CH*.sac、vehicles.csv、tracks.json、
    sim_config.json、preview.png。

注意：
    GPU 只能加速数据矩阵生成；写 SAC 文件仍然是 CPU/磁盘 IO，生成大规模
    1 小时数据时这部分仍会占用明显时间。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from simulate_vehicle_sac import (
    _json_float,
    _model_direction,
    _travel_time_with_motion,
    build_vehicle_table,
    parse_args,
    save_preview,
    write_config_json,
    write_sac_files,
    write_tracks_json,
    write_vehicle_csv,
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_device_arg(argv: list[str]) -> tuple[list[str], str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="", help="Torch device for matrix generation: cuda, mps, cpu, or auto.")
    known, remaining = parser.parse_known_args(argv)
    raw_device = str(known.device).strip()
    device = _auto_device() if raw_device in {"", "auto"} else raw_device
    return remaining, device


def overlay_vehicle_pulses_torch(
    data: torch.Tensor,
    rows: list[dict],
    cfg,
    rng: np.random.Generator,
) -> list[dict]:
    n_samples = int(data.shape[1])
    device = data.device
    jitter_min_kmh, jitter_max_kmh = cfg.speed_jitter_kmh_range
    jitter_channel_count = min(cfg.n_ch, cfg.speed_jitter_channel_count)
    apply_jitter = ((jitter_min_kmh != 0.0) or (jitter_max_kmh != 0.0)) and (jitter_channel_count > 0)
    tracks: list[dict] = []

    for row in rows:
        points: list[dict] = []
        t_entry = float(row["t_entry"])
        amp = float(row["amp"])
        sigma_t = float(row["sigma_t"])
        direction = str(row["direction"])
        base_speed_kmh = float(row["speed_kmh"])
        jitter_channel_set: set[int] = set()
        if apply_jitter:
            chosen = rng.choice(cfg.n_ch, size=jitter_channel_count, replace=False)
            jitter_channel_set = {int(v) for v in np.asarray(chosen, dtype=int).tolist()}
            row["jitter_channels"] = ";".join(str(ch) for ch in sorted(jitter_channel_set))
        else:
            row["jitter_channels"] = ""

        half_width = int(np.ceil(4.0 * sigma_t * cfg.fs))
        if half_width <= 0:
            continue

        for ch in range(cfg.n_ch):
            if direction == "primary":
                dist_m = ch * cfg.dx_m
            else:
                dist_m = (cfg.n_ch - 1 - ch) * cfg.dx_m

            t_center_base = t_entry + _travel_time_with_motion(row, dist_m)
            if apply_jitter and (ch in jitter_channel_set):
                jitter_kmh = float(rng.uniform(jitter_min_kmh, jitter_max_kmh))
                speed_scale = max(1e-6, (base_speed_kmh + jitter_kmh) / base_speed_kmh)
                t_center = t_entry + (t_center_base - t_entry) / speed_scale
            else:
                t_center = t_center_base

            if t_center < 0.0 or t_center >= cfg.duration_s:
                continue

            center_idx = int(round(t_center * cfg.fs))
            center_idx = int(max(0, min(n_samples - 1, center_idx)))
            points.append(
                {
                    "ch_idx": int(ch),
                    "offset_m": float(ch * cfg.dx_m),
                    "time_s": float(t_center),
                    "t_idx": int(center_idx),
                    "amp": float(amp),
                    "sigma_t": float(sigma_t),
                }
            )

            left = max(0, center_idx - half_width)
            right = min(n_samples - 1, center_idx + half_width)
            if left > right:
                continue
            idx = torch.arange(left, right + 1, device=device, dtype=torch.float32)
            dt = (idx / float(cfg.fs)) - float(t_center)
            pulse = float(amp) * torch.exp(-0.5 * (dt / float(sigma_t)) ** 2)
            data[int(ch), left : right + 1] += pulse.to(dtype=data.dtype)

        tracks.append(
            {
                "track_id": int(row["id"]),
                "direction": _model_direction(str(direction)),
                "sim_direction": str(direction),
                "t_entry": float(row["t_entry"]),
                "speed_kmh": float(row["speed_kmh"]),
                "speed_mps": float(row["speed_mps"]),
                "amp": float(row["amp"]),
                "sigma_t": float(row["sigma_t"]),
                "jitter_channels": str(row.get("jitter_channels", "")),
                "motion_type": str(row.get("motion_type", "constant")),
                "event_start_dist_m": _json_float(row.get("event_start_dist_m", np.nan)),
                "event_duration_s": _json_float(row.get("event_duration_s", np.nan)),
                "event_accel_mps2": _json_float(row.get("event_accel_mps2", np.nan)),
                "stop_duration_s": _json_float(row.get("stop_duration_s", np.nan)),
                "stop_brake_mps2": _json_float(row.get("stop_brake_mps2", np.nan)),
                "restart_accel_mps2": _json_float(row.get("restart_accel_mps2", np.nan)),
                "points": points,
            }
        )
    return tracks


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    remaining, device_name = _parse_device_arg(raw_argv)
    old_argv = sys.argv
    sys.argv = [old_argv[0], *remaining]
    try:
        cfg = parse_args()
    finally:
        sys.argv = old_argv

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)
    print(f"Using torch device for data generation: {device}")
    t0 = time.perf_counter()

    rng = np.random.default_rng(cfg.seed)
    if cfg.add_noise:
        gen_device = device if device.type != "mps" else torch.device("cpu")
        generator = torch.Generator(device=gen_device)
        generator.manual_seed(int(cfg.seed))
        data = torch.normal(
            mean=0.0,
            std=float(cfg.noise_std),
            size=(int(cfg.n_ch), int(cfg.n_samples)),
            generator=generator,
            device=gen_device,
            dtype=torch.float32,
        )
        if gen_device != device:
            data = data.to(device)
    else:
        data = torch.zeros((int(cfg.n_ch), int(cfg.n_samples)), device=device, dtype=torch.float32)

    rows = build_vehicle_table(cfg, rng)
    tracks = overlay_vehicle_pulses_torch(data, rows, cfg, rng)
    matrix_elapsed = time.perf_counter() - t0
    data_np = data.detach().to("cpu").numpy().astype(np.float32, copy=False)

    t_write = time.perf_counter()
    write_sac_files(data_np, cfg)
    write_vehicle_csv(rows, cfg)
    write_tracks_json(tracks, cfg)
    write_config_json(cfg, rows)
    save_preview(data_np, cfg)
    write_elapsed = time.perf_counter() - t_write

    n_primary = sum(1 for r in rows if r["direction"] == "primary")
    n_secondary = len(rows) - n_primary
    speed_kmh_vals = np.array([r["speed_kmh"] for r in rows], dtype=float)
    print(f"Done. Output: {cfg.out_dir}")
    print(f"SAC files: {cfg.n_ch}, samples/channel: {cfg.n_samples}, fs: {cfg.fs} Hz")
    print(f"Vehicles: {len(rows)} (primary={n_primary}, secondary={n_secondary})")
    print(f"Matrix generation: {matrix_elapsed:.2f}s, write/preview: {write_elapsed:.2f}s")
    if speed_kmh_vals.size > 0:
        print(
            "Speed km/h: "
            f"min={speed_kmh_vals.min():.2f}, max={speed_kmh_vals.max():.2f}, "
            f"mean={speed_kmh_vals.mean():.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
