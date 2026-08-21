"""Generate synthetic DAS SAC data and exact vehicle trajectory labels.

用途：
    生成一组模拟 DAS 车辆振动 SAC 文件，同时输出车辆参数表、配置文件、
    预览图，以及用于深度学习训练的 `tracks.json` 真值轨迹文件。

用例：
    uv run python KF/auto_track/simulate_vehicle_sac.py \
        --out-dir KF/auto_track/data \
        --primary-count 200 \
        --secondary-count 20 \
        --duration-s 3600 \
        --fs 1000

输出：
    - CH*.sac：每个通道一个 SAC 文件。
    - vehicles.csv：每辆车的模拟参数。
    - tracks.json：每辆车在每个通道上的真实轨迹坐标点。
    - sim_config.json：本次模拟配置。
    - preview.png：前若干秒时空图预览。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace, UTCDateTime
from obspy.core.util.attribdict import AttribDict

plt.rcParams["font.family"] = "Times New Roman"


@dataclass
class SimulationConfig:
    out_dir: Path
    seed: int = 42
    noise_std: float = 1.0
    n_veh: int = 48
    dir_ratio: float = 0.7
    primary_count: int | None = None
    secondary_count: int | None = None
    add_noise: bool = True
    fixed_amp: float = 1.0
    amp_range: tuple[float, float] = (3.0, 6.0)
    sigma_range: tuple[float, float] = (0.06, 0.18)
    n_ch: int = 50
    dx_m: float = 100.0
    fs: float = 1000.0
    duration_s: float = 3600.0
    speed_range_kmh: tuple[float, float] = (60.0, 120.0)
    speed_jitter_kmh_range: tuple[float, float] = (0.0, 0.0)
    speed_jitter_channel_count: int = 3
    accel_count: int = 0
    decel_count: int = 0
    stop_go_count: int = 0
    accel_mps2: float = 0.8
    decel_mps2: float = 0.8
    stop_brake_mps2: float = 1.2
    restart_accel_mps2: float = 0.8
    accel_duration_range_s: tuple[float, float] = (4.0, 10.0)
    decel_duration_range_s: tuple[float, float] = (4.0, 10.0)
    stop_duration_range_s: tuple[float, float] = (2.0, 8.0)
    event_start_ratio_range: tuple[float, float] = (0.1, 0.7)
    preview_seconds: float = 600.0

    @property
    def n_samples(self) -> int:
        return int(round(self.duration_s * self.fs))


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="Generate DAS synthetic 1-hour SAC data with multi-vehicle Gaussian trajectories."
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for SAC files and metadata.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--noise-std", type=float, default=1.0, help="White noise std.")
    parser.add_argument(
        "--n-veh",
        "--vehicle-count",
        dest="n_veh",
        type=int,
        default=48,
        help="Number of vehicles in one hour.",
    )
    parser.add_argument(
        "--dir-ratio",
        type=float,
        default=0.7,
        help="Primary direction ratio in [0, 1]. Secondary ratio is (1-ratio).",
    )
    parser.add_argument("--primary-count", type=int, default=None, help="Exact number of primary-direction vehicles.")
    parser.add_argument(
        "--secondary-count",
        type=int,
        default=None,
        help="Exact number of secondary-direction vehicles.",
    )
    parser.add_argument("--no-noise", action="store_true", help="Disable background white noise.")
    parser.add_argument(
        "--fixed-amp",
        type=float,
        default=1.0,
        help="Fixed Gaussian pulse amplitude for all vehicles (default: 1.0).",
    )
    parser.add_argument(
        "--amp-range",
        nargs=2,
        type=float,
        default=[3.0, 6.0],
        metavar=("MIN", "MAX"),
        help="Amplitude multipliers of noise_std.",
    )
    parser.add_argument(
        "--sigma-range",
        nargs=2,
        type=float,
        default=[0.06, 0.18],
        metavar=("MIN", "MAX"),
        help="Gaussian sigma range in seconds.",
    )
    parser.add_argument("--n-ch", type=int, default=50, help="Number of channels.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--duration-s", type=float, default=3600.0, help="Duration in seconds.")
    parser.add_argument(
        "--speed-range-kmh",
        nargs=2,
        type=float,
        default=[60.0, 120.0],
        metavar=("MIN", "MAX"),
        help="Per-vehicle random speed range in km/h.",
    )
    parser.add_argument(
        "--speed-jitter-kmh-range",
        nargs=2,
        type=float,
        default=[0.0, 0.0],
        metavar=("MIN", "MAX"),
        help=(
            "Per-channel random speed jitter in km/h (sampled per vehicle per channel). "
            "Example: -3 3 means each channel speed is base_speed +/- up to 3 km/h."
        ),
    )
    parser.add_argument(
        "--speed-jitter-channel-count",
        type=int,
        default=3,
        help="Number of channels per vehicle that apply speed jitter; others keep base speed.",
    )
    parser.add_argument("--accel-count", type=int, default=0, help="Number of vehicles with acceleration events.")
    parser.add_argument("--decel-count", type=int, default=0, help="Number of vehicles with deceleration events.")
    parser.add_argument("--stop-go-count", type=int, default=0, help="Number of vehicles with stop-and-go events.")
    parser.add_argument("--accel-mps2", type=float, default=0.8, help="Acceleration for accel events (m/s^2).")
    parser.add_argument(
        "--decel-mps2",
        type=float,
        default=0.8,
        help="Deceleration magnitude for decel events (m/s^2).",
    )
    parser.add_argument(
        "--stop-brake-mps2",
        type=float,
        default=1.2,
        help="Braking deceleration magnitude for stop-go events (m/s^2).",
    )
    parser.add_argument(
        "--restart-accel-mps2",
        type=float,
        default=0.8,
        help="Restart acceleration for stop-go events (m/s^2).",
    )
    parser.add_argument(
        "--accel-duration-range-s",
        nargs=2,
        type=float,
        default=[4.0, 10.0],
        metavar=("MIN", "MAX"),
        help="Duration range (s) of acceleration events.",
    )
    parser.add_argument(
        "--decel-duration-range-s",
        nargs=2,
        type=float,
        default=[4.0, 10.0],
        metavar=("MIN", "MAX"),
        help="Duration range (s) of deceleration events.",
    )
    parser.add_argument(
        "--stop-duration-range-s",
        nargs=2,
        type=float,
        default=[2.0, 8.0],
        metavar=("MIN", "MAX"),
        help="Stop duration range (s) for stop-go events.",
    )
    parser.add_argument(
        "--event-start-ratio-range",
        nargs=2,
        type=float,
        default=[0.1, 0.7],
        metavar=("MIN", "MAX"),
        help="Event start position ratio in travel distance (0~1).",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=600.0,
        help="Preview figure time span in seconds.",
    )
    args = parser.parse_args()
    n_veh = int(args.n_veh)
    primary_count = args.primary_count
    secondary_count = args.secondary_count
    if primary_count is not None and secondary_count is not None:
        n_veh = int(primary_count + secondary_count)
    elif primary_count is not None:
        secondary_count = int(n_veh - primary_count)
    elif secondary_count is not None:
        primary_count = int(n_veh - secondary_count)

    cfg = SimulationConfig(
        out_dir=args.out_dir,
        seed=args.seed,
        noise_std=args.noise_std,
        n_veh=n_veh,
        dir_ratio=args.dir_ratio,
        primary_count=primary_count,
        secondary_count=secondary_count,
        add_noise=(not args.no_noise),
        fixed_amp=args.fixed_amp,
        amp_range=(float(args.amp_range[0]), float(args.amp_range[1])),
        sigma_range=(float(args.sigma_range[0]), float(args.sigma_range[1])),
        n_ch=args.n_ch,
        dx_m=args.dx_m,
        fs=args.fs,
        duration_s=args.duration_s,
        speed_range_kmh=(float(args.speed_range_kmh[0]), float(args.speed_range_kmh[1])),
        speed_jitter_kmh_range=(
            float(args.speed_jitter_kmh_range[0]),
            float(args.speed_jitter_kmh_range[1]),
        ),
        speed_jitter_channel_count=int(args.speed_jitter_channel_count),
        accel_count=args.accel_count,
        decel_count=args.decel_count,
        stop_go_count=args.stop_go_count,
        accel_mps2=args.accel_mps2,
        decel_mps2=args.decel_mps2,
        stop_brake_mps2=args.stop_brake_mps2,
        restart_accel_mps2=args.restart_accel_mps2,
        accel_duration_range_s=(float(args.accel_duration_range_s[0]), float(args.accel_duration_range_s[1])),
        decel_duration_range_s=(float(args.decel_duration_range_s[0]), float(args.decel_duration_range_s[1])),
        stop_duration_range_s=(float(args.stop_duration_range_s[0]), float(args.stop_duration_range_s[1])),
        event_start_ratio_range=(
            float(args.event_start_ratio_range[0]),
            float(args.event_start_ratio_range[1]),
        ),
        preview_seconds=args.preview_seconds,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: SimulationConfig) -> None:
    min_motion_speed_mps = 0.5
    if cfg.primary_count is None or cfg.secondary_count is None:
        if not (0.0 <= cfg.dir_ratio <= 1.0):
            raise ValueError("--dir-ratio must be in [0, 1].")
    if cfg.n_veh < 0:
        raise ValueError("--n-veh / --vehicle-count must be >= 0.")
    if cfg.fixed_amp <= 0:
        raise ValueError("--fixed-amp must be > 0.")
    if cfg.add_noise and cfg.noise_std <= 0:
        raise ValueError("--noise-std must be > 0 when noise is enabled.")
    if (not cfg.add_noise) and cfg.noise_std < 0:
        raise ValueError("--noise-std must be >= 0 when noise is disabled.")
    if cfg.primary_count is not None and cfg.primary_count < 0:
        raise ValueError("--primary-count must be >= 0.")
    if cfg.secondary_count is not None and cfg.secondary_count < 0:
        raise ValueError("--secondary-count must be >= 0.")
    if cfg.primary_count is not None and cfg.secondary_count is not None:
        if cfg.primary_count + cfg.secondary_count != cfg.n_veh:
            raise ValueError("primary_count + secondary_count must equal total vehicle count.")
    if cfg.fs <= 0:
        raise ValueError("--fs must be > 0.")
    if cfg.duration_s <= 0:
        raise ValueError("--duration-s must be > 0.")
    if cfg.n_ch <= 0:
        raise ValueError("--n-ch must be > 0.")
    if cfg.dx_m <= 0:
        raise ValueError("--dx-m must be > 0.")
    if cfg.preview_seconds <= 0:
        raise ValueError("--preview-seconds must be > 0.")
    if cfg.accel_count < 0 or cfg.decel_count < 0 or cfg.stop_go_count < 0:
        raise ValueError("--accel-count / --decel-count / --stop-go-count must be >= 0.")
    if cfg.accel_count + cfg.decel_count + cfg.stop_go_count > cfg.n_veh:
        raise ValueError("Sum of --accel-count/--decel-count/--stop-go-count must be <= vehicle count.")
    if cfg.accel_mps2 <= 0:
        raise ValueError("--accel-mps2 must be > 0.")
    if cfg.decel_mps2 <= 0:
        raise ValueError("--decel-mps2 must be > 0.")
    if cfg.stop_brake_mps2 <= 0:
        raise ValueError("--stop-brake-mps2 must be > 0.")
    if cfg.restart_accel_mps2 <= 0:
        raise ValueError("--restart-accel-mps2 must be > 0.")
    amp_min, amp_max = cfg.amp_range
    sig_min, sig_max = cfg.sigma_range
    v_min, v_max = cfg.speed_range_kmh
    jitter_min_kmh, jitter_max_kmh = cfg.speed_jitter_kmh_range
    accel_t_min, accel_t_max = cfg.accel_duration_range_s
    decel_t_min, decel_t_max = cfg.decel_duration_range_s
    stop_t_min, stop_t_max = cfg.stop_duration_range_s
    event_ratio_min, event_ratio_max = cfg.event_start_ratio_range
    if amp_min <= 0 or amp_max <= 0 or amp_min >= amp_max:
        raise ValueError("--amp-range must satisfy 0 < min < max.")
    if sig_min <= 0 or sig_max <= 0 or sig_min >= sig_max:
        raise ValueError("--sigma-range must satisfy 0 < min < max.")
    if v_min <= 0 or v_max <= 0 or v_min >= v_max:
        raise ValueError("--speed-range-kmh must satisfy 0 < min < max.")
    if jitter_min_kmh > jitter_max_kmh:
        raise ValueError("--speed-jitter-kmh-range must satisfy min <= max.")
    if v_min + jitter_min_kmh <= 0:
        raise ValueError(
            "--speed-jitter-kmh-range min is too negative. "
            "Ensure speed_range_min + jitter_min > 0."
        )
    if cfg.speed_jitter_channel_count < 0:
        raise ValueError("--speed-jitter-channel-count must be >= 0.")
    if cfg.speed_jitter_channel_count > cfg.n_ch:
        raise ValueError("--speed-jitter-channel-count must be <= --n-ch.")
    if accel_t_min <= 0 or accel_t_max <= 0 or accel_t_min >= accel_t_max:
        raise ValueError("--accel-duration-range-s must satisfy 0 < min < max.")
    if decel_t_min <= 0 or decel_t_max <= 0 or decel_t_min >= decel_t_max:
        raise ValueError("--decel-duration-range-s must satisfy 0 < min < max.")
    if stop_t_min <= 0 or stop_t_max <= 0 or stop_t_min >= stop_t_max:
        raise ValueError("--stop-duration-range-s must satisfy 0 < min < max.")
    if event_ratio_min < 0 or event_ratio_max > 1 or event_ratio_min >= event_ratio_max:
        raise ValueError("--event-start-ratio-range must satisfy 0 <= min < max <= 1.")
    if v_min / 3.6 <= min_motion_speed_mps:
        raise ValueError("--speed-range-kmh min is too low for motion events; increase minimum speed.")


def sample_directions(cfg: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    n_veh = cfg.n_veh
    if cfg.primary_count is not None and cfg.secondary_count is not None:
        n_primary = int(cfg.primary_count)
        n_secondary = int(cfg.secondary_count)
    else:
        n_primary = int(round(n_veh * cfg.dir_ratio))
        n_primary = max(0, min(n_veh, n_primary))
        n_secondary = n_veh - n_primary
    if n_primary + n_secondary != n_veh:
        raise ValueError("Direction counts do not match total vehicle count.")
    directions = np.array(["primary"] * n_primary + ["secondary"] * n_secondary, dtype=object)
    rng.shuffle(directions)
    return directions


def _sample_event_start_dist_m(cfg: SimulationConfig, rng: np.random.Generator) -> float:
    max_dist = max(0.0, (cfg.n_ch - 1) * cfg.dx_m)
    start_ratio = rng.uniform(cfg.event_start_ratio_range[0], cfg.event_start_ratio_range[1])
    return float(start_ratio * max_dist)


def _pick_event_indices(cfg: SimulationConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(cfg.n_veh)
    rng.shuffle(idx)
    accel_idx = idx[: cfg.accel_count]
    decel_idx = idx[cfg.accel_count : cfg.accel_count + cfg.decel_count]
    stop_idx = idx[cfg.accel_count + cfg.decel_count : cfg.accel_count + cfg.decel_count + cfg.stop_go_count]
    return accel_idx, decel_idx, stop_idx


def _add_motion_profiles(rows: list[dict], cfg: SimulationConfig, rng: np.random.Generator) -> None:
    min_motion_speed_mps = 0.5
    for row in rows:
        row["motion_type"] = "constant"
        row["event_start_dist_m"] = np.nan
        row["event_duration_s"] = np.nan
        row["event_accel_mps2"] = np.nan
        row["stop_duration_s"] = np.nan
        row["stop_brake_mps2"] = np.nan
        row["restart_accel_mps2"] = np.nan

    if not rows:
        return

    accel_idx, decel_idx, stop_idx = _pick_event_indices(cfg, rng)
    for idx in accel_idx:
        row = rows[int(idx)]
        row["motion_type"] = "accel"
        row["event_start_dist_m"] = _sample_event_start_dist_m(cfg, rng)
        row["event_duration_s"] = float(rng.uniform(cfg.accel_duration_range_s[0], cfg.accel_duration_range_s[1]))
        row["event_accel_mps2"] = float(cfg.accel_mps2)

    for idx in decel_idx:
        row = rows[int(idx)]
        speed_mps = float(row["speed_mps"])
        raw_duration = float(rng.uniform(cfg.decel_duration_range_s[0], cfg.decel_duration_range_s[1]))
        max_safe_duration = max(1e-3, (speed_mps - min_motion_speed_mps) / cfg.decel_mps2)
        duration = min(raw_duration, max_safe_duration)
        row["motion_type"] = "decel"
        row["event_start_dist_m"] = _sample_event_start_dist_m(cfg, rng)
        row["event_duration_s"] = float(duration)
        row["event_accel_mps2"] = float(-cfg.decel_mps2)

    for idx in stop_idx:
        row = rows[int(idx)]
        row["motion_type"] = "stop_go"
        row["event_start_dist_m"] = _sample_event_start_dist_m(cfg, rng)
        row["stop_duration_s"] = float(rng.uniform(cfg.stop_duration_range_s[0], cfg.stop_duration_range_s[1]))
        row["stop_brake_mps2"] = float(cfg.stop_brake_mps2)
        row["restart_accel_mps2"] = float(cfg.restart_accel_mps2)


def _travel_time_with_motion(row: dict, x_m: float) -> float:
    min_motion_speed_mps = 0.5
    speed_mps = float(row["speed_mps"])
    motion_type = str(row.get("motion_type", "constant"))

    if motion_type == "constant":
        return x_m / speed_mps

    event_start_dist_m = float(row.get("event_start_dist_m", np.nan))
    if not np.isfinite(event_start_dist_m) or x_m <= event_start_dist_m:
        return x_m / speed_mps

    t_before_event = event_start_dist_m / speed_mps
    x_rel = x_m - event_start_dist_m

    if motion_type in {"accel", "decel"}:
        event_duration_s = float(row.get("event_duration_s", np.nan))
        event_accel_mps2 = float(row.get("event_accel_mps2", np.nan))
        if not np.isfinite(event_duration_s) or not np.isfinite(event_accel_mps2):
            return x_m / speed_mps
        if event_duration_s <= 0 or abs(event_accel_mps2) < 1e-12:
            return x_m / speed_mps

        v_end = max(speed_mps + event_accel_mps2 * event_duration_s, min_motion_speed_mps)
        x_event = speed_mps * event_duration_s + 0.5 * event_accel_mps2 * (event_duration_s**2)
        x_event = max(0.0, float(x_event))

        if x_rel <= x_event:
            disc = speed_mps * speed_mps + 2.0 * event_accel_mps2 * x_rel
            disc = max(0.0, float(disc))
            tau = (-speed_mps + np.sqrt(disc)) / event_accel_mps2
            tau = float(np.clip(tau, 0.0, event_duration_s))
            return t_before_event + tau

        return t_before_event + event_duration_s + (x_rel - x_event) / v_end

    if motion_type == "stop_go":
        stop_duration_s = float(row.get("stop_duration_s", np.nan))
        stop_brake_mps2 = float(row.get("stop_brake_mps2", np.nan))
        restart_accel_mps2 = float(row.get("restart_accel_mps2", np.nan))
        if not np.isfinite(stop_duration_s) or not np.isfinite(stop_brake_mps2) or not np.isfinite(restart_accel_mps2):
            return x_m / speed_mps
        if stop_duration_s < 0 or stop_brake_mps2 <= 0 or restart_accel_mps2 <= 0:
            return x_m / speed_mps

        t_brake = speed_mps / stop_brake_mps2
        x_brake = (speed_mps * speed_mps) / (2.0 * stop_brake_mps2)
        t_restart = speed_mps / restart_accel_mps2
        x_restart = (speed_mps * speed_mps) / (2.0 * restart_accel_mps2)

        if x_rel <= x_brake:
            disc = speed_mps * speed_mps - 2.0 * stop_brake_mps2 * x_rel
            disc = max(0.0, float(disc))
            tau = (speed_mps - np.sqrt(disc)) / stop_brake_mps2
            tau = float(np.clip(tau, 0.0, t_brake))
            return t_before_event + tau

        x_after_brake = x_rel - x_brake
        if x_after_brake <= x_restart:
            tau = np.sqrt(max(0.0, 2.0 * x_after_brake / restart_accel_mps2))
            tau = float(np.clip(tau, 0.0, t_restart))
            return t_before_event + t_brake + stop_duration_s + tau

        return t_before_event + t_brake + stop_duration_s + t_restart + (x_after_brake - x_restart) / speed_mps

    return x_m / speed_mps


def build_vehicle_table(cfg: SimulationConfig, rng: np.random.Generator) -> list[dict]:
    directions = sample_directions(cfg, rng)
    rows: list[dict] = []
    for vid in range(cfg.n_veh):
        speed_kmh = rng.uniform(cfg.speed_range_kmh[0], cfg.speed_range_kmh[1])
        speed_mps = speed_kmh / 3.6
        amp = float(cfg.fixed_amp)
        sigma_t = rng.uniform(cfg.sigma_range[0], cfg.sigma_range[1])
        t_entry = rng.uniform(0.0, cfg.duration_s)
        rows.append(
            {
                "id": vid,
                "direction": str(directions[vid]),
                "t_entry": float(t_entry),
                "speed_kmh": float(speed_kmh),
                "speed_mps": float(speed_mps),
                "amp": float(amp),
                "sigma_t": float(sigma_t),
            }
        )
    _add_motion_profiles(rows, cfg, rng)
    return rows


def _model_direction(sim_direction: str) -> str:
    if sim_direction == "primary":
        return "forward"
    if sim_direction == "secondary":
        return "reverse"
    return str(sim_direction)


def overlay_vehicle_pulses(
    data: np.ndarray, rows: list[dict], cfg: SimulationConfig, rng: np.random.Generator
) -> list[dict]:
    n_samples = data.shape[1]
    jitter_min_kmh, jitter_max_kmh = cfg.speed_jitter_kmh_range
    jitter_channel_count = min(cfg.n_ch, cfg.speed_jitter_channel_count)
    apply_jitter = ((jitter_min_kmh != 0.0) or (jitter_max_kmh != 0.0)) and (jitter_channel_count > 0)
    tracks: list[dict] = []
    for row in rows:
        points: list[dict] = []
        t_entry = row["t_entry"]
        amp = row["amp"]
        sigma_t = row["sigma_t"]
        direction = row["direction"]
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

            idx = np.arange(left, right + 1)
            dt = (idx / cfg.fs) - t_center
            pulse = amp * np.exp(-0.5 * (dt / sigma_t) ** 2)
            data[ch, left : right + 1] += pulse.astype(np.float32, copy=False)
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


def _json_float(value: object) -> float | None:
    try:
        val = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(val):
        return None
    return val


def write_sac_files(data: np.ndarray, cfg: SimulationConfig) -> None:
    starttime = UTCDateTime(1970, 1, 1)
    delta = 1.0 / cfg.fs
    for ch in range(cfg.n_ch):
        tr = Trace(data=data[ch].astype(np.float32, copy=False))
        tr.stats.network = "SIM"
        tr.stats.station = f"CH{ch:03d}"
        tr.stats.channel = "DAS"
        tr.stats.starttime = starttime
        tr.stats.delta = delta
        tr.stats.distance = float(ch * cfg.dx_m)
        tr.stats.sac = AttribDict({"dist": float(ch * cfg.dx_m / 1000.0), "b": 0.0})
        out_path = cfg.out_dir / f"CH{ch:03d}.sac"
        tr.write(str(out_path), format="SAC")


def write_vehicle_csv(rows: list[dict], cfg: SimulationConfig) -> None:
    out_csv = cfg.out_dir / "vehicles.csv"
    fieldnames = [
        "id",
        "direction",
        "t_entry",
        "speed_kmh",
        "speed_mps",
        "amp",
        "sigma_t",
        "jitter_channels",
        "motion_type",
        "event_start_dist_m",
        "event_duration_s",
        "event_accel_mps2",
        "stop_duration_s",
        "stop_brake_mps2",
        "restart_accel_mps2",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row: dict[str, object] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, float) and not np.isfinite(value):
                    clean_row[key] = ""
                else:
                    clean_row[key] = value
            writer.writerow(clean_row)


def write_config_json(cfg: SimulationConfig, rows: list[dict]) -> None:
    n_primary = int(sum(1 for r in rows if r["direction"] == "primary"))
    n_secondary = int(len(rows) - n_primary)
    payload = {
        **asdict(cfg),
        "out_dir": str(cfg.out_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": cfg.n_samples,
        "n_primary": n_primary,
        "n_secondary": n_secondary,
    }
    out_json = cfg.out_dir / "sim_config.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_tracks_json(tracks: list[dict], cfg: SimulationConfig) -> None:
    payload = {
        "schema_version": 1,
        "source": "simulate_vehicle_sac.py",
        "fs": float(cfg.fs),
        "dx_m": float(cfg.dx_m),
        "n_ch": int(cfg.n_ch),
        "duration_s": float(cfg.duration_s),
        "n_samples": int(cfg.n_samples),
        "coordinate_system": {
            "ch_idx": "0-based channel index",
            "offset_m": "channel offset in meters",
            "time_s": "absolute time from SAC start in seconds",
            "t_idx": "sample index at fs",
        },
        "tracks": tracks,
    }
    out_json = cfg.out_dir / "tracks.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_preview(data: np.ndarray, cfg: SimulationConfig) -> None:
    preview_samples = min(cfg.n_samples, int(round(cfg.preview_seconds * cfg.fs)))
    if preview_samples <= 0:
        return

    preview = data[:, :preview_samples]
    max_cols = 3000
    stride = max(1, preview.shape[1] // max_cols)
    preview_ds = preview[:, ::stride]

    p99 = np.percentile(np.abs(preview_ds), 99.0)
    clip = float(max(p99, 1e-6))
    shown = np.clip(preview_ds, -clip, clip)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        shown,
        aspect="auto",
        origin="lower",
        cmap="seismic",
        extent=[0.0, preview_samples / cfg.fs, 0.0, (cfg.n_ch - 1) * cfg.dx_m / 1000.0],
        vmin=-clip,
        vmax=clip,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (km)")
    ax.set_title(f"Simulated DAS Preview (first {preview_samples / cfg.fs:.1f} s)")
    fig.colorbar(im, ax=ax, label="Amplitude")
    fig.tight_layout()
    fig.savefig(cfg.out_dir / "preview.png", dpi=150)
    plt.close(fig)



def main() -> None:
    cfg = parse_args()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    if cfg.add_noise:
        data = rng.normal(loc=0.0, scale=cfg.noise_std, size=(cfg.n_ch, cfg.n_samples)).astype(np.float32)
    else:
        data = np.zeros((cfg.n_ch, cfg.n_samples), dtype=np.float32)

    rows = build_vehicle_table(cfg, rng)
    tracks = overlay_vehicle_pulses(data, rows, cfg, rng)

    write_sac_files(data, cfg)
    write_vehicle_csv(rows, cfg)
    write_tracks_json(tracks, cfg)
    write_config_json(cfg, rows)
    save_preview(data, cfg)

    n_primary = sum(1 for r in rows if r["direction"] == "primary")
    n_secondary = len(rows) - n_primary
    speed_kmh_vals = np.array([r["speed_kmh"] for r in rows], dtype=float)
    motion_summary = {
        "constant": sum(1 for r in rows if r.get("motion_type") == "constant"),
        "accel": sum(1 for r in rows if r.get("motion_type") == "accel"),
        "decel": sum(1 for r in rows if r.get("motion_type") == "decel"),
        "stop_go": sum(1 for r in rows if r.get("motion_type") == "stop_go"),
    }
    print(f"Done. Output: {cfg.out_dir}")
    print(f"SAC files: {cfg.n_ch}, samples/channel: {cfg.n_samples}, fs: {cfg.fs} Hz")
    print(f"Vehicles: {len(rows)} (primary={n_primary}, secondary={n_secondary})")
    print(f"Noise: {'enabled' if cfg.add_noise else 'disabled'} (noise_std={cfg.noise_std})")
    print(
        "Speed jitter: "
        f"range_kmh=({cfg.speed_jitter_kmh_range[0]}, {cfg.speed_jitter_kmh_range[1]}), "
        f"channels_per_vehicle={cfg.speed_jitter_channel_count}"
    )
    print(f"Gaussian pulse amp (fixed): {cfg.fixed_amp}")
    print(
        "Motion types: "
        f"constant={motion_summary['constant']}, "
        f"accel={motion_summary['accel']}, "
        f"decel={motion_summary['decel']}, "
        f"stop_go={motion_summary['stop_go']}"
    )
    if speed_kmh_vals.size > 0:
        print(
            "Speed km/h: "
            f"min={speed_kmh_vals.min():.2f}, max={speed_kmh_vals.max():.2f}, "
            f"mean={speed_kmh_vals.mean():.2f}"
        )
    else:
        print("Speed km/h: N/A (no vehicles)")


if __name__ == "__main__":
    main()
