from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


@dataclass
class TrackPoint:
    ch_idx: int
    t_idx: int
    time_s: float
    offset_m: float
    amp: float
    score: float


@dataclass
class Track:
    track_id: int
    direction: str
    points: list[TrackPoint]
    total_score: float
    mean_speed_kmh: float


@dataclass
class ExtractorConfig:
    sigma_seconds: tuple[float, ...] = (0.06, 0.10, 0.14, 0.18)
    use_template_enhancement: bool = False
    enhance_decimate: int = 2
    prominence: float = 0.4
    min_peak_distance: int = 500
    max_skip_channels: int = 4
    lambda_speed: float = 2.0
    lambda_skip: float = 0.55
    speed_change_tolerance_kmh: float = 18.0
    speed_penalty_power: float = 0.85
    speed_penalty_cap: float = 2.5
    min_track_channels: int = 12
    min_track_score: float = 10.0
    edge_relax_enabled: bool = True
    edge_min_track_channels: int = 4
    edge_time_margin_seconds: float = 15.0
    edge_min_score_scale: float = 0.2
    nms_time_radius: int = 180
    nms_channel_radius: int = 1
    max_tracks: int = 256
    max_peaks_per_channel: int = 400


def _as_config(config: Optional[ExtractorConfig | dict]) -> ExtractorConfig:
    if config is None:
        return ExtractorConfig()
    if isinstance(config, ExtractorConfig):
        return config
    if isinstance(config, dict):
        return ExtractorConfig(**config)
    raise TypeError("config must be ExtractorConfig / dict / None")


def _enhance_with_gaussian_templates(
    abs_data: np.ndarray,
    fs: float,
    sigma_seconds: Sequence[float],
) -> np.ndarray:
    enhanced = np.zeros_like(abs_data, dtype=np.float32)
    for sigma_s in sigma_seconds:
        sigma_samples = max(1.0, float(sigma_s) * float(fs))
        response = gaussian_filter1d(abs_data, sigma=sigma_samples, axis=1, mode="nearest")
        np.maximum(enhanced, response.astype(np.float32, copy=False), out=enhanced)
    return enhanced


def _build_nodes(
    data: np.ndarray,
    fs: float,
    config: ExtractorConfig,
) -> list[dict[str, np.ndarray]]:
    abs_data = np.abs(data).astype(np.float32, copy=False)
    decim = int(max(1, config.enhance_decimate))
    if decim > 1:
        abs_work = abs_data[:, ::decim]
        fs_work = float(fs) / float(decim)
        peak_distance = int(max(1, round(float(config.min_peak_distance) / float(decim))))
    else:
        abs_work = abs_data
        fs_work = float(fs)
        peak_distance = int(max(1, config.min_peak_distance))

    if bool(config.use_template_enhancement) and len(config.sigma_seconds) > 0:
        enhanced = _enhance_with_gaussian_templates(abs_work, fs_work, config.sigma_seconds)
    else:
        # When template enhancement is disabled, peaks are detected directly on |signal|.
        enhanced = abs_work

    channel_nodes: list[dict[str, np.ndarray]] = []
    for ch in range(data.shape[0]):
        peaks, props = find_peaks(
            enhanced[ch],
            prominence=config.prominence,
            distance=peak_distance,
        )
        if peaks.size == 0:
            channel_nodes.append(
                {
                    "t": np.empty((0,), dtype=np.int32),
                    "amp": np.empty((0,), dtype=np.float32),
                    "score": np.empty((0,), dtype=np.float32),
                }
            )
            continue

        if decim > 1:
            peaks = np.minimum(
                peaks.astype(np.int64, copy=False) * int(decim),
                int(data.shape[1] - 1),
            ).astype(np.int32, copy=False)

        prominences = props.get("prominences", np.zeros(peaks.shape[0], dtype=np.float32)).astype(np.float32)
        amps = abs_data[ch, peaks].astype(np.float32, copy=False)
        amp_ref = float(np.median(amps) + 1e-6)
        amp_norm = amps / amp_ref
        node_score = prominences + 0.2 * amp_norm

        if peaks.size > config.max_peaks_per_channel:
            idx_top = np.argsort(node_score)[-config.max_peaks_per_channel :]
            peaks = peaks[idx_top]
            amps = amps[idx_top]
            node_score = node_score[idx_top]

        order = np.argsort(peaks)
        channel_nodes.append(
            {
                "t": peaks[order].astype(np.int32, copy=False),
                "amp": amps[order].astype(np.float32, copy=False),
                "score": node_score[order].astype(np.float32, copy=False),
            }
        )
    return channel_nodes


def _dt_bounds(direction: str, delta_x_m: float, vmin_mps: float, vmax_mps: float) -> tuple[float, float]:
    if direction == "forward":
        return delta_x_m / vmax_mps, delta_x_m / vmin_mps
    if direction == "reverse":
        return -delta_x_m / vmin_mps, -delta_x_m / vmax_mps
    raise ValueError("direction must be either forward or reverse")


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    ts = np.array([p.time_s for p in points], dtype=np.float64)
    chs = np.array([p.ch_idx for p in points], dtype=np.float64)
    dt = np.diff(ts)
    dch = np.diff(chs)
    valid = np.abs(dt) > 1e-9
    if not np.any(valid):
        return float("nan")
    speed_mps = np.abs(dch[valid]) * dx_m / np.abs(dt[valid])
    return float(3.6 * np.mean(speed_mps))


def _extract_best_track(
    nodes: list[dict[str, np.ndarray]],
    fs: float,
    dx_m: float,
    direction: str,
    vmin_mps: float,
    vmax_mps: float,
    n_samples: int,
    config: ExtractorConfig,
    track_id: int,
) -> Optional[Track]:
    n_ch = len(nodes)
    dp_score: list[np.ndarray] = []
    dp_len: list[np.ndarray] = []
    dp_speed: list[np.ndarray] = []
    dp_tmin: list[np.ndarray] = []
    dp_tmax: list[np.ndarray] = []
    dp_prev_ch: list[np.ndarray] = []
    dp_prev_idx: list[np.ndarray] = []

    speed_scale = max(1e-6, float(vmax_mps - vmin_mps))
    speed_tol_mps = max(0.0, float(config.speed_change_tolerance_kmh) / 3.6)
    speed_pow = float(max(0.5, config.speed_penalty_power))
    speed_pen_cap = float(max(0.0, config.speed_penalty_cap))

    for ch in range(n_ch):
        t_curr = nodes[ch]["t"]
        s_curr = nodes[ch]["score"]
        n_curr = t_curr.size
        if n_curr == 0:
            dp_score.append(np.empty((0,), dtype=np.float32))
            dp_len.append(np.empty((0,), dtype=np.int32))
            dp_speed.append(np.empty((0,), dtype=np.float32))
            dp_tmin.append(np.empty((0,), dtype=np.int32))
            dp_tmax.append(np.empty((0,), dtype=np.int32))
            dp_prev_ch.append(np.empty((0,), dtype=np.int16))
            dp_prev_idx.append(np.empty((0,), dtype=np.int32))
            continue

        score_arr = s_curr.astype(np.float32, copy=True)
        len_arr = np.ones((n_curr,), dtype=np.int32)
        speed_arr = np.full((n_curr,), np.nan, dtype=np.float32)
        tmin_arr = t_curr.astype(np.int32, copy=True)
        tmax_arr = t_curr.astype(np.int32, copy=True)
        prev_ch_arr = np.full((n_curr,), -1, dtype=np.int16)
        prev_idx_arr = np.full((n_curr,), -1, dtype=np.int32)

        for dch in range(1, config.max_skip_channels + 1):
            pch = ch - dch
            if pch < 0:
                continue
            t_prev = nodes[pch]["t"]
            if t_prev.size == 0:
                continue
            prev_score = dp_score[pch]
            prev_len = dp_len[pch]
            prev_speed = dp_speed[pch]
            prev_tmin = dp_tmin[pch]
            prev_tmax = dp_tmax[pch]
            delta_x = float(dch * dx_m)
            dt_low, dt_high = _dt_bounds(direction, delta_x, vmin_mps, vmax_mps)
            skip_penalty = float(config.lambda_skip * max(0, dch - 1))

            for j in range(n_curr):
                dt = (float(t_curr[j]) - t_prev.astype(np.float64)) / float(fs)
                valid = (dt >= dt_low) & (dt <= dt_high)
                if not np.any(valid):
                    continue
                idx = np.where(valid)[0]
                dt_sel = dt[idx]
                speed_sel = delta_x / np.maximum(np.abs(dt_sel), 1e-9)
                prev_speed_sel = prev_speed[idx].astype(np.float64)
                prev_speed_sel = np.where(np.isfinite(prev_speed_sel), prev_speed_sel, speed_sel)
                speed_delta = np.abs(speed_sel - prev_speed_sel)
                speed_excess = np.maximum(0.0, speed_delta - speed_tol_mps)
                speed_pen = config.lambda_speed * np.power(speed_excess / speed_scale, speed_pow)
                if speed_pen_cap > 0:
                    speed_pen = np.minimum(speed_pen, speed_pen_cap)
                cand = prev_score[idx].astype(np.float64) + float(s_curr[j]) - speed_pen - skip_penalty
                tie = cand + 1e-4 * prev_len[idx].astype(np.float64)
                k = int(np.argmax(tie))
                cand_score = float(cand[k])
                cand_len = int(prev_len[idx[k]]) + 1

                if cand_score > float(score_arr[j]) + 1e-12 or (
                    abs(cand_score - float(score_arr[j])) <= 1e-12 and cand_len > int(len_arr[j])
                ):
                    prev_idx = int(idx[k])
                    score_arr[j] = np.float32(cand_score)
                    len_arr[j] = np.int32(cand_len)
                    prev_ch_arr[j] = np.int16(pch)
                    prev_idx_arr[j] = np.int32(prev_idx)
                    speed_arr[j] = np.float32(0.7 * prev_speed_sel[k] + 0.3 * speed_sel[k])
                    tmin_arr[j] = np.int32(min(int(prev_tmin[prev_idx]), int(t_curr[j])))
                    tmax_arr[j] = np.int32(max(int(prev_tmax[prev_idx]), int(t_curr[j])))

        dp_score.append(score_arr)
        dp_len.append(len_arr)
        dp_speed.append(speed_arr)
        dp_tmin.append(tmin_arr)
        dp_tmax.append(tmax_arr)
        dp_prev_ch.append(prev_ch_arr)
        dp_prev_idx.append(prev_idx_arr)

    # Allow a looser acceptance rule near the window boundaries, where a valid
    # track may be truncated by the analysis window and therefore look shorter.
    edge_margin_samples = int(max(0, round(float(config.edge_time_margin_seconds) * float(fs))))
    edge_margin_samples = int(min(max(0, n_samples - 1), edge_margin_samples))
    relaxed_len = int(max(2, min(config.min_track_channels, config.edge_min_track_channels)))
    relaxed_score = float(config.min_track_score * max(0.0, float(config.edge_min_score_scale)))

    # Among all per-node DP optima, choose one global best endpoint that also
    # satisfies the track validity thresholds.
    best = None
    best_value = -np.inf
    for ch in range(n_ch):
        if dp_score[ch].size == 0:
            continue
        # Standard acceptance rule for complete in-window tracks.
        strict_valid = (dp_len[ch] >= config.min_track_channels) & (dp_score[ch] >= config.min_track_score)
        if config.edge_relax_enabled and relaxed_len < config.min_track_channels:
            # Relax the thresholds only for paths that touch the start or end of
            # the current window, where partial tracks are expected.
            near_start = dp_tmin[ch] <= edge_margin_samples
            near_end = dp_tmax[ch] >= (n_samples - 1 - edge_margin_samples)
            edge_touch = near_start | near_end
            relaxed_valid = edge_touch & (dp_len[ch] >= relaxed_len) & (dp_score[ch] >= relaxed_score)
            valid = strict_valid | relaxed_valid
        else:
            valid = strict_valid
        if not np.any(valid):
            continue
        idxs = np.where(valid)[0]
        # Break near-ties by preferring the longer path.
        values = dp_score[ch][idxs].astype(np.float64) + 1e-3 * dp_len[ch][idxs].astype(np.float64)
        k = int(np.argmax(values))
        val = float(values[k])
        if val > best_value:
            best_value = val
            best = (ch, int(idxs[k]))

    if best is None:
        return None

    # Recover the full track by following parent pointers backward from the
    # chosen endpoint, then reverse to restore chronological order.
    path: list[tuple[int, int]] = []
    ch, idx = best
    while ch >= 0 and idx >= 0:
        path.append((ch, idx))
        next_ch = int(dp_prev_ch[ch][idx]) if dp_prev_ch[ch].size > 0 else -1
        next_idx = int(dp_prev_idx[ch][idx]) if dp_prev_idx[ch].size > 0 else -1
        ch, idx = next_ch, next_idx
    path.reverse()

    points: list[TrackPoint] = []
    for pch, pidx in path:
        t_idx = int(nodes[pch]["t"][pidx])
        amp = float(nodes[pch]["amp"][pidx])
        score = float(nodes[pch]["score"][pidx])
        points.append(
            TrackPoint(
                ch_idx=pch,
                t_idx=t_idx,
                time_s=float(t_idx) / float(fs),
                offset_m=float(pch) * float(dx_m),
                amp=amp,
                score=score,
            )
        )

    mean_speed = _mean_speed_kmh(points, dx_m)
    total_score = float(dp_score[best[0]][best[1]])
    return Track(
        track_id=track_id,
        direction=direction,
        points=points,
        total_score=total_score,
        mean_speed_kmh=mean_speed,
    )


def _suppress_nodes(nodes: list[dict[str, np.ndarray]], track: Track, config: ExtractorConfig) -> None:
    n_ch = len(nodes)
    t_radius = int(max(1, config.nms_time_radius))
    ch_radius = int(max(0, config.nms_channel_radius))

    for p in track.points:
        for ch in range(max(0, p.ch_idx - ch_radius), min(n_ch, p.ch_idx + ch_radius + 1)):
            t_arr = nodes[ch]["t"]
            if t_arr.size == 0:
                continue
            keep = np.abs(t_arr - int(p.t_idx)) > t_radius
            nodes[ch]["t"] = nodes[ch]["t"][keep]
            nodes[ch]["amp"] = nodes[ch]["amp"][keep]
            nodes[ch]["score"] = nodes[ch]["score"][keep]


def extract_all(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[ExtractorConfig | dict] = None,
) -> list[Track]:
    cfg = _as_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array with shape [n_channel, n_sample]")
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if dx_m <= 0:
        raise ValueError("dx_m must be > 0")
    if vmin_kmh <= 0 or vmax_kmh <= 0:
        raise ValueError("speed range must be > 0")
    if vmin_kmh > vmax_kmh:
        raise ValueError("vmin_kmh cannot be greater than vmax_kmh")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be either forward or reverse")

    vmin_mps = float(vmin_kmh) / 3.6
    vmax_mps = float(vmax_kmh) / 3.6

    base_nodes = _build_nodes(arr, fs, cfg)
    nodes = [
        {
            "t": n["t"].copy(),
            "amp": n["amp"].copy(),
            "score": n["score"].copy(),
        }
        for n in base_nodes
    ]

    tracks: list[Track] = []
    for tid in range(int(max(1, cfg.max_tracks))):
        best = _extract_best_track(
            nodes=nodes,
            fs=fs,
            dx_m=dx_m,
            direction=direction,
            vmin_mps=vmin_mps,
            vmax_mps=vmax_mps,
            n_samples=arr.shape[1],
            config=cfg,
            track_id=tid,
        )
        if best is None:
            break
        tracks.append(best)
        _suppress_nodes(nodes, best, cfg)

    return tracks
