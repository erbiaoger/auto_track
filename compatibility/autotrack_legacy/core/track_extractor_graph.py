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
    lambda_prediction: float = 1.25
    lambda_skip: float = 0.55
    speed_change_tolerance_kmh: float = 18.0
    speed_penalty_power: float = 0.85
    speed_penalty_cap: float = 2.5
    prediction_tolerance_ratio: float = 0.20
    prediction_tolerance_min_seconds: float = 0.08
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
    k_best_per_node: int = 3


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


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float, x_axis_m: Optional[np.ndarray] = None) -> float:
    if len(points) < 2:
        return float("nan")
    ts = np.array([p.time_s for p in points], dtype=np.float64)
    chs = np.array([p.ch_idx for p in points], dtype=np.float64)
    dt = np.diff(ts)
    dch = np.diff(chs)
    valid = np.abs(dt) > 1e-9
    if not np.any(valid):
        return float("nan")
    if x_axis_m is None:
        distance_m = np.abs(dch[valid]) * dx_m
    else:
        indices = chs.astype(np.int64)
        distance_m = np.abs(np.diff(np.asarray(x_axis_m, dtype=np.float64)[indices]))[valid]
    speed_mps = distance_m / np.abs(dt[valid])
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
    prior_time_hint: Optional[np.ndarray] = None,
    prior_channel_weight: float = 0.0,
    x_axis_m: Optional[np.ndarray] = None,
) -> Optional[Track]:
    n_ch = len(nodes)
    k_best = int(max(1, config.k_best_per_node))
    dp_score: list[np.ndarray] = []
    dp_len: list[np.ndarray] = []
    dp_speed: list[np.ndarray] = []
    dp_tmin: list[np.ndarray] = []
    dp_tmax: list[np.ndarray] = []
    dp_prev_ch: list[np.ndarray] = []
    dp_prev_idx: list[np.ndarray] = []
    dp_prev_slot: list[np.ndarray] = []

    speed_scale = max(1e-6, float(vmax_mps - vmin_mps))
    speed_tol_mps = max(0.0, float(config.speed_change_tolerance_kmh) / 3.6)
    speed_pow = float(max(0.5, config.speed_penalty_power))
    speed_pen_cap = float(max(0.0, config.speed_penalty_cap))
    pred_lambda = float(max(0.0, config.lambda_prediction))
    pred_tol_ratio = float(max(0.0, config.prediction_tolerance_ratio))
    pred_tol_min_s = float(max(0.0, config.prediction_tolerance_min_seconds))
    prior_time_arr = None
    if prior_time_hint is not None:
        prior_time_arr = np.asarray(prior_time_hint, dtype=np.float64)
        if prior_time_arr.ndim != 1 or prior_time_arr.shape[0] != n_ch:
            raise ValueError("prior_time_hint must have shape [n_channels]")

    for ch in range(n_ch):
        t_curr = nodes[ch]["t"]
        s_curr = nodes[ch]["score"]
        n_curr = t_curr.size
        if n_curr == 0:
            empty_shape = (0, k_best)
            dp_score.append(np.empty(empty_shape, dtype=np.float32))
            dp_len.append(np.empty(empty_shape, dtype=np.int32))
            dp_speed.append(np.empty(empty_shape, dtype=np.float32))
            dp_tmin.append(np.empty(empty_shape, dtype=np.int32))
            dp_tmax.append(np.empty(empty_shape, dtype=np.int32))
            dp_prev_ch.append(np.empty(empty_shape, dtype=np.int16))
            dp_prev_idx.append(np.empty(empty_shape, dtype=np.int32))
            dp_prev_slot.append(np.empty(empty_shape, dtype=np.int8))
            continue

        score_arr = np.full((n_curr, k_best), -np.inf, dtype=np.float32)
        len_arr = np.zeros((n_curr, k_best), dtype=np.int32)
        speed_arr = np.full((n_curr, k_best), np.nan, dtype=np.float32)
        tmin_arr = np.tile(t_curr.astype(np.int32, copy=False)[:, None], (1, k_best))
        tmax_arr = np.tile(t_curr.astype(np.int32, copy=False)[:, None], (1, k_best))
        prev_ch_arr = np.full((n_curr, k_best), -1, dtype=np.int16)
        prev_idx_arr = np.full((n_curr, k_best), -1, dtype=np.int32)
        prev_slot_arr = np.full((n_curr, k_best), -1, dtype=np.int8)
        score_arr[:, 0] = s_curr.astype(np.float32, copy=False)
        len_arr[:, 0] = 1

        def _push_candidate(
            node_idx: int,
            cand_score: float,
            cand_len: int,
            cand_speed: float,
            cand_tmin: int,
            cand_tmax: int,
            cand_prev_ch: int,
            cand_prev_idx: int,
            cand_prev_slot: int,
        ) -> None:
            if not np.isfinite(cand_score):
                return
            ties = score_arr[node_idx].astype(np.float64) + 1e-4 * len_arr[node_idx].astype(np.float64)
            cand_tie = float(cand_score) + 1e-4 * float(cand_len)
            same_state = (
                (prev_ch_arr[node_idx] == cand_prev_ch)
                & (prev_idx_arr[node_idx] == cand_prev_idx)
                & (prev_slot_arr[node_idx] == cand_prev_slot)
            )
            if np.any(same_state):
                pos = int(np.argmax(same_state))
                if cand_tie <= float(ties[pos]) + 1e-12:
                    return
            worst = int(np.argmin(ties))
            if cand_tie <= float(ties[worst]) + 1e-12:
                return
            score_arr[node_idx, worst] = np.float32(cand_score)
            len_arr[node_idx, worst] = np.int32(cand_len)
            speed_arr[node_idx, worst] = np.float32(cand_speed)
            tmin_arr[node_idx, worst] = np.int32(cand_tmin)
            tmax_arr[node_idx, worst] = np.int32(cand_tmax)
            prev_ch_arr[node_idx, worst] = np.int16(cand_prev_ch)
            prev_idx_arr[node_idx, worst] = np.int32(cand_prev_idx)
            prev_slot_arr[node_idx, worst] = np.int8(cand_prev_slot)

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
            if x_axis_m is None:
                delta_x = float(dch * dx_m)
            else:
                delta_x = abs(float(x_axis_m[ch]) - float(x_axis_m[pch]))
            dt_low, dt_high = _dt_bounds(direction, delta_x, vmin_mps, vmax_mps)
            skip_penalty = float(config.lambda_skip * max(0, dch - 1))

            for j in range(n_curr):
                dt = (float(t_curr[j]) - t_prev.astype(np.float64)) / float(fs)
                valid = (dt >= dt_low) & (dt <= dt_high)
                if not np.any(valid):
                    continue
                idx = np.where(valid)[0]
                for prev_idx in idx:
                    dt_curr = float(dt[prev_idx])
                    speed_curr = float(delta_x / max(abs(dt_curr), 1e-9))
                    for prev_slot in range(k_best):
                        prev_score_val = float(prev_score[prev_idx, prev_slot])
                        if not np.isfinite(prev_score_val):
                            continue
                        prev_speed_val = float(prev_speed[prev_idx, prev_slot])
                        speed_ref = prev_speed_val if np.isfinite(prev_speed_val) else speed_curr
                        speed_delta = abs(speed_curr - speed_ref)
                        speed_excess = max(0.0, speed_delta - speed_tol_mps)
                        speed_pen = float(config.lambda_speed) * float(np.power(speed_excess / speed_scale, speed_pow))
                        if speed_pen_cap > 0:
                            speed_pen = min(speed_pen, speed_pen_cap)

                        pred_pen = 0.0
                        if pred_lambda > 0.0 and np.isfinite(prev_speed_val) and prev_speed_val > 1e-9:
                            dt_pred = delta_x / prev_speed_val
                            dt_err = abs(dt_curr - dt_pred)
                            dt_tol = max(pred_tol_min_s, pred_tol_ratio * abs(dt_pred))
                            pred_pen = pred_lambda * max(0.0, dt_err - dt_tol) / max(1e-6, dt_tol)

                        prior_pen = 0.0
                        if prior_time_arr is not None and float(prior_channel_weight) > 0.0:
                            prior_time = float(prior_time_arr[ch])
                            if np.isfinite(prior_time):
                                prior_tol = max(pred_tol_min_s, 0.10)
                                abs_time = float(t_curr[j]) / float(fs)
                                prior_pen = float(prior_channel_weight) * max(
                                    0.0, abs(abs_time - prior_time) - prior_tol
                                ) / max(1e-6, prior_tol)

                        cand_score = prev_score_val + float(s_curr[j]) - speed_pen - pred_pen - prior_pen - skip_penalty
                        cand_len = int(prev_len[prev_idx, prev_slot]) + 1
                        cand_speed = 0.7 * speed_ref + 0.3 * speed_curr
                        cand_tmin = min(int(prev_tmin[prev_idx, prev_slot]), int(t_curr[j]))
                        cand_tmax = max(int(prev_tmax[prev_idx, prev_slot]), int(t_curr[j]))
                        _push_candidate(
                            node_idx=j,
                            cand_score=cand_score,
                            cand_len=cand_len,
                            cand_speed=cand_speed,
                            cand_tmin=cand_tmin,
                            cand_tmax=cand_tmax,
                            cand_prev_ch=pch,
                            cand_prev_idx=int(prev_idx),
                            cand_prev_slot=prev_slot,
                        )

        dp_score.append(score_arr)
        dp_len.append(len_arr)
        dp_speed.append(speed_arr)
        dp_tmin.append(tmin_arr)
        dp_tmax.append(tmax_arr)
        dp_prev_ch.append(prev_ch_arr)
        dp_prev_idx.append(prev_idx_arr)
        dp_prev_slot.append(prev_slot_arr)

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
        strict_valid = (dp_len[ch] >= config.min_track_channels) & np.isfinite(dp_score[ch]) & (dp_score[ch] >= config.min_track_score)
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
        idxs = np.argwhere(valid)
        # Break near-ties by preferring the longer path.
        values = dp_score[ch][valid].astype(np.float64) + 1e-3 * dp_len[ch][valid].astype(np.float64)
        k = int(np.argmax(values))
        val = float(values[k])
        if val > best_value:
            best_value = val
            best = (ch, int(idxs[k][0]), int(idxs[k][1]))

    if best is None:
        return None

    # Recover the full track by following parent pointers backward from the
    # chosen endpoint, then reverse to restore chronological order.
    path: list[tuple[int, int]] = []
    ch, idx, slot = best
    while ch >= 0 and idx >= 0 and slot >= 0:
        path.append((ch, idx))
        next_ch = int(dp_prev_ch[ch][idx, slot]) if dp_prev_ch[ch].size > 0 else -1
        next_idx = int(dp_prev_idx[ch][idx, slot]) if dp_prev_idx[ch].size > 0 else -1
        next_slot = int(dp_prev_slot[ch][idx, slot]) if dp_prev_slot[ch].size > 0 else -1
        ch, idx, slot = next_ch, next_idx, next_slot
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
                offset_m=(float(x_axis_m[pch]) if x_axis_m is not None else float(pch) * float(dx_m)),
                amp=amp,
                score=score,
            )
        )

    mean_speed = _mean_speed_kmh(points, dx_m, x_axis_m=x_axis_m)
    total_score = float(dp_score[best[0]][best[1], best[2]])
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
    x_axis_m: Optional[np.ndarray] = None,
) -> list[Track]:
    cfg = _as_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array with shape [n_channel, n_sample]")
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if dx_m <= 0:
        raise ValueError("dx_m must be > 0")
    axis = None
    if x_axis_m is not None:
        axis = np.asarray(x_axis_m, dtype=np.float64)
        if axis.ndim != 1 or axis.shape[0] != arr.shape[0]:
            raise ValueError("x_axis_m must have shape [n_channel]")
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis) <= 0):
            raise ValueError("x_axis_m must be finite and strictly increasing")
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
            x_axis_m=axis,
        )
        if best is None:
            break
        tracks.append(best)
        _suppress_nodes(nodes, best, cfg)

    return tracks


def extend_track(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    seed_points: list[TrackPoint],
    side: str,
    target_ch_idx: Optional[int] = None,
    target_t_idx: Optional[int] = None,
    config: Optional[ExtractorConfig | dict] = None,
) -> list[TrackPoint]:
    cfg = _as_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array with shape [n_channel, n_sample]")
    if fs <= 0 or dx_m <= 0:
        raise ValueError("fs and dx_m must be > 0")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be either forward or reverse")
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if not seed_points:
        return []

    points = sorted(seed_points, key=lambda p: p.ch_idx)
    nodes = _build_nodes(arr, fs, cfg)
    vmin_mps = float(vmin_kmh) / 3.6
    vmax_mps = float(vmax_kmh) / 3.6
    speed_scale = max(1e-6, float(vmax_mps - vmin_mps))
    speed_tol_mps = max(0.0, float(cfg.speed_change_tolerance_kmh) / 3.6)
    speed_pow = float(max(0.5, cfg.speed_penalty_power))
    speed_pen_cap = float(max(0.0, cfg.speed_penalty_cap))
    pred_lambda = float(max(0.0, cfg.lambda_prediction))
    pred_tol_ratio = float(max(0.0, cfg.prediction_tolerance_ratio))
    pred_tol_min_s = float(max(0.0, cfg.prediction_tolerance_min_seconds))

    if side == "right":
        anchor = points[-1]
        history = points[-min(3, len(points)) :]
        step_sign = 1
        start_ch = int(anchor.ch_idx) + 1
        stop_default = arr.shape[0] - 1
    else:
        anchor = points[0]
        history = points[: min(3, len(points))]
        step_sign = -1
        start_ch = int(anchor.ch_idx) - 1
        stop_default = 0

    if len(history) >= 2:
        if side == "right":
            ref_a, ref_b = history[-2], history[-1]
        else:
            ref_a, ref_b = history[1], history[0]
        dt_ref = abs(float(ref_b.time_s) - float(ref_a.time_s))
        dch_ref = abs(int(ref_b.ch_idx) - int(ref_a.ch_idx))
        speed_ref = float(dch_ref * dx_m / max(dt_ref, 1e-9)) if dch_ref > 0 else float("nan")
    else:
        speed_ref = float("nan")
    if not np.isfinite(speed_ref):
        speed_ref = 0.5 * (vmin_mps + vmax_mps)

    anchor0 = TrackPoint(
        ch_idx=int(anchor.ch_idx),
        t_idx=int(anchor.t_idx),
        time_s=float(anchor.time_s),
        offset_m=float(anchor.offset_m),
        amp=float(anchor.amp),
        score=float(anchor.score),
    )
    current = TrackPoint(
        ch_idx=int(anchor.ch_idx),
        t_idx=int(anchor.t_idx),
        time_s=float(anchor.time_s),
        offset_m=float(anchor.offset_m),
        amp=float(anchor.amp),
        score=float(anchor.score),
    )
    target = int(target_ch_idx) if target_ch_idx is not None else int(stop_default)
    target = max(0, min(target, int(arr.shape[0] - 1)))
    target_t_s = float(target_t_idx) / float(fs) if target_t_idx is not None else None
    dynamic_max_skip = int(max(int(cfg.max_skip_channels), min(16, max(6, abs(target - int(anchor.ch_idx))))))

    added: list[TrackPoint] = []
    while 0 <= start_ch < int(arr.shape[0]):
        if (step_sign > 0 and start_ch > target) or (step_sign < 0 and start_ch < target):
            break

        best_choice: Optional[tuple[int, int, float, float, float]] = None
        best_score = -np.inf
        best_next_ch = None

        for dch in range(1, dynamic_max_skip + 1):
            cand_ch = int(current.ch_idx) + step_sign * dch
            if cand_ch < 0 or cand_ch >= int(arr.shape[0]):
                continue
            if target_ch_idx is not None:
                if step_sign > 0 and cand_ch > target:
                    continue
                if step_sign < 0 and cand_ch < target:
                    continue

            delta_x = float(abs(cand_ch - int(current.ch_idx)) * dx_m)
            dt_low, dt_high = _dt_bounds(direction, delta_x, vmin_mps, vmax_mps)
            dt_lo = min(dt_low, dt_high)
            dt_hi = max(dt_low, dt_high)
            dt_slack = max(0.08, 0.35 * max(abs(dt_lo), abs(dt_hi)))
            pred_dt = float(delta_x / max(speed_ref, 1e-9))
            if direction == "reverse":
                pred_dt = -pred_dt
            pred_t = float(current.time_s) + pred_dt

            t_arr = nodes[cand_ch]["t"]
            if t_arr.size == 0:
                continue
            dt_arr = (t_arr.astype(np.float64) - float(current.t_idx)) / float(fs)
            valid = (dt_arr >= (dt_lo - dt_slack)) & (dt_arr <= (dt_hi + dt_slack))
            if not np.any(valid):
                continue

            valid_idx = np.where(valid)[0]
            for idx in valid_idx:
                dt_curr = float(dt_arr[idx])
                speed_curr = float(delta_x / max(abs(dt_curr), 1e-9))
                speed_delta = abs(speed_curr - speed_ref)
                speed_excess = max(0.0, speed_delta - speed_tol_mps)
                speed_pen = float(cfg.lambda_speed) * float(np.power(speed_excess / speed_scale, speed_pow))
                if speed_pen_cap > 0:
                    speed_pen = min(speed_pen, speed_pen_cap)

                pred_pen = 0.0
                dt_err = abs((float(t_arr[idx]) / float(fs)) - pred_t)
                dt_tol = max(pred_tol_min_s, pred_tol_ratio * max(abs(pred_dt), pred_tol_min_s))
                if pred_lambda > 0.0:
                    pred_pen = pred_lambda * max(0.0, dt_err - dt_tol) / max(1e-6, dt_tol)

                click_guide_pen = 0.0
                if target_t_s is not None and abs(target - int(anchor0.ch_idx)) >= 1:
                    frac = abs(cand_ch - int(anchor0.ch_idx)) / max(1.0, abs(target - int(anchor0.ch_idx)))
                    frac = float(np.clip(frac, 0.0, 1.0))
                    guide_t = float(anchor0.time_s) + frac * (float(target_t_s) - float(anchor0.time_s))
                    guide_err = abs((float(t_arr[idx]) / float(fs)) - guide_t)
                    guide_tol = max(0.12, 0.25 * abs(float(target_t_s) - float(anchor0.time_s)))
                    click_guide_pen = 1.75 * max(0.0, guide_err - guide_tol) / max(1e-6, guide_tol)

                skip_penalty = float(cfg.lambda_skip * max(0, dch - 1))
                cand_score = float(nodes[cand_ch]["score"][idx]) - speed_pen - pred_pen - click_guide_pen - skip_penalty
                if cand_score > best_score:
                    best_score = cand_score
                    best_choice = (
                        cand_ch,
                        int(idx),
                        speed_curr,
                        float(nodes[cand_ch]["amp"][idx]),
                        float(nodes[cand_ch]["score"][idx]),
                    )
                    best_next_ch = cand_ch

        if best_choice is None or best_next_ch is None:
            break

        cand_ch, node_idx, speed_ref, amp, node_score = best_choice
        t_idx = int(nodes[cand_ch]["t"][node_idx])
        current = TrackPoint(
            ch_idx=int(cand_ch),
            t_idx=t_idx,
            time_s=float(t_idx) / float(fs),
            offset_m=float(cand_ch) * float(dx_m),
            amp=float(amp),
            score=float(node_score),
        )
        added.append(current)
        start_ch = int(current.ch_idx) + step_sign

        if int(current.ch_idx) == target:
            break

    return added
