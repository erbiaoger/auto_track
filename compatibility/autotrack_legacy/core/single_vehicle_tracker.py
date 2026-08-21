"""Single-vehicle trajectory extraction utilities.

This module is the beginning of the new tracking path. It keeps the problem
explicitly one-vehicle-at-a-time: detect candidate peaks, decode one coherent
path with physical continuity, optionally bridge sparse gaps with Hungarian
assignment, and smooth the result with a constant-velocity Kalman filter.

The intent is to replace the current multi-slot PeakSlotNet workflow for the
single-vehicle task rather than patching it further.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, _build_nodes


@dataclass
class SingleVehicleTrackerConfig:
    candidate_prominence: float = 0.22
    candidate_min_distance: int = 180
    candidate_max_peaks_per_channel: int = 32
    max_skip_channels: int = 8
    min_track_channels: int = 8
    min_track_score: float = 8.0
    edge_relax_enabled: bool = True
    edge_min_track_channels: int = 4
    edge_time_margin_seconds: float = 8.0
    edge_min_score_scale: float = 0.5
    speed_change_tolerance_kmh: float = 18.0
    speed_penalty_power: float = 0.9
    speed_penalty_cap: float = 2.5
    lambda_speed: float = 2.0
    lambda_skip: float = 0.55
    kalman_process_var: float = 0.8
    kalman_meas_var: float = 0.18
    kalman_bridge_gap_channels: int = 12
    kalman_fill_missing: bool = True
    kalman_gate_seconds: float = 0.35
    kalman_speed_gate_kmh: float = 30.0
    direction: str = "forward"
    candidate_hypotheses: int = 4
    hypothesis_prior_weight: float = 4.0
    trajectory_fit_weight: float = 1.2
    trajectory_curvature_weight: float = 0.35


def _as_config(config: Optional[SingleVehicleTrackerConfig | dict[str, Any]]) -> SingleVehicleTrackerConfig:
    if config is None:
        return SingleVehicleTrackerConfig()
    if isinstance(config, SingleVehicleTrackerConfig):
        return config
    if isinstance(config, dict):
        defaults = SingleVehicleTrackerConfig()
        return SingleVehicleTrackerConfig(
            candidate_prominence=float(config.get("candidate_prominence", defaults.candidate_prominence)),
            candidate_min_distance=int(config.get("candidate_min_distance", defaults.candidate_min_distance)),
            candidate_max_peaks_per_channel=int(
                config.get("candidate_max_peaks_per_channel", defaults.candidate_max_peaks_per_channel)
            ),
            max_skip_channels=int(config.get("max_skip_channels", defaults.max_skip_channels)),
            min_track_channels=int(config.get("min_track_channels", defaults.min_track_channels)),
            min_track_score=float(config.get("min_track_score", defaults.min_track_score)),
            edge_relax_enabled=bool(config.get("edge_relax_enabled", defaults.edge_relax_enabled)),
            edge_min_track_channels=int(config.get("edge_min_track_channels", defaults.edge_min_track_channels)),
            edge_time_margin_seconds=float(config.get("edge_time_margin_seconds", defaults.edge_time_margin_seconds)),
            edge_min_score_scale=float(config.get("edge_min_score_scale", defaults.edge_min_score_scale)),
            speed_change_tolerance_kmh=float(config.get("speed_change_tolerance_kmh", defaults.speed_change_tolerance_kmh)),
            speed_penalty_power=float(config.get("speed_penalty_power", defaults.speed_penalty_power)),
            speed_penalty_cap=float(config.get("speed_penalty_cap", defaults.speed_penalty_cap)),
            lambda_speed=float(config.get("lambda_speed", defaults.lambda_speed)),
            lambda_skip=float(config.get("lambda_skip", defaults.lambda_skip)),
            kalman_process_var=float(config.get("kalman_process_var", defaults.kalman_process_var)),
            kalman_meas_var=float(config.get("kalman_meas_var", defaults.kalman_meas_var)),
            kalman_bridge_gap_channels=int(config.get("kalman_bridge_gap_channels", defaults.kalman_bridge_gap_channels)),
            kalman_fill_missing=bool(config.get("kalman_fill_missing", defaults.kalman_fill_missing)),
            kalman_gate_seconds=float(config.get("kalman_gate_seconds", defaults.kalman_gate_seconds)),
            kalman_speed_gate_kmh=float(config.get("kalman_speed_gate_kmh", defaults.kalman_speed_gate_kmh)),
            direction=str(config.get("direction", defaults.direction)),
            candidate_hypotheses=int(config.get("candidate_hypotheses", defaults.candidate_hypotheses)),
            hypothesis_prior_weight=float(config.get("hypothesis_prior_weight", defaults.hypothesis_prior_weight)),
            trajectory_fit_weight=float(config.get("trajectory_fit_weight", defaults.trajectory_fit_weight)),
            trajectory_curvature_weight=float(
                config.get("trajectory_curvature_weight", defaults.trajectory_curvature_weight)
            ),
        )
    raise TypeError("config must be SingleVehicleTrackerConfig / dict / None")


def _graph_config(cfg: SingleVehicleTrackerConfig) -> ExtractorConfig:
    return ExtractorConfig(
        prominence=float(cfg.candidate_prominence),
        min_peak_distance=int(max(1, cfg.candidate_min_distance)),
        max_peaks_per_channel=int(max(1, cfg.candidate_max_peaks_per_channel)),
        max_skip_channels=int(max(1, cfg.max_skip_channels)),
        edge_relax_enabled=bool(cfg.edge_relax_enabled),
        edge_min_track_channels=int(max(2, cfg.edge_min_track_channels)),
        edge_time_margin_seconds=float(cfg.edge_time_margin_seconds),
        edge_min_score_scale=float(cfg.edge_min_score_scale),
        min_track_channels=int(max(2, cfg.min_track_channels)),
        min_track_score=float(cfg.min_track_score),
    )


def _relaxed_graph_config(cfg: SingleVehicleTrackerConfig) -> ExtractorConfig:
    return ExtractorConfig(
        prominence=float(max(0.02, cfg.candidate_prominence * 0.55)),
        min_peak_distance=int(max(4, round(cfg.candidate_min_distance * 0.5))),
        max_peaks_per_channel=int(max(8, round(cfg.candidate_max_peaks_per_channel * 1.5))),
        max_skip_channels=int(max(cfg.max_skip_channels, 12)),
        edge_relax_enabled=True,
        edge_min_track_channels=int(max(2, round(cfg.edge_min_track_channels * 0.75))),
        edge_time_margin_seconds=float(cfg.edge_time_margin_seconds),
        edge_min_score_scale=float(cfg.edge_min_score_scale),
        min_track_channels=int(max(4, round(cfg.min_track_channels * 0.6))),
        min_track_score=float(max(2.0, cfg.min_track_score * 0.5)),
        lambda_speed=float(cfg.lambda_speed),
        lambda_skip=float(cfg.lambda_skip),
        speed_change_tolerance_kmh=float(cfg.speed_change_tolerance_kmh),
        speed_penalty_power=float(cfg.speed_penalty_power),
        speed_penalty_cap=float(cfg.speed_penalty_cap),
    )


def _dt_bounds(direction: str, delta_x_m: float, vmin_mps: float, vmax_mps: float) -> tuple[float, float]:
    if str(direction) == "forward":
        return delta_x_m / vmax_mps, delta_x_m / vmin_mps
    if str(direction) == "reverse":
        return -delta_x_m / vmin_mps, -delta_x_m / vmax_mps
    raise ValueError("direction must be either forward or reverse")


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: p.ch_idx)
    speeds: list[float] = []
    for left, right in zip(pts[:-1], pts[1:]):
        dch = abs(int(right.ch_idx) - int(left.ch_idx))
        dt = abs(float(right.time_s) - float(left.time_s))
        if dch > 0 and dt > 1e-9:
            speeds.append(3.6 * float(dch) * float(dx_m) / dt)
    return float(np.mean(speeds)) if speeds else float("nan")


def _recompute_track(track_id: int, direction: str, points: list[TrackPoint], dx_m: float) -> Track:
    ordered = sorted(points, key=lambda p: (int(p.ch_idx), int(p.t_idx)))
    return Track(
        track_id=int(track_id),
        direction=str(direction),
        points=ordered,
        total_score=float(sum(float(p.score) for p in ordered)),
        mean_speed_kmh=_mean_speed_kmh(ordered, float(dx_m)),
    )


def _track_prior_score(
    track: Track,
    *,
    prior_heatmap: Optional[np.ndarray],
    prior_time_hint: Optional[np.ndarray],
    fs: float,
    prior_weight: float,
    trajectory_fit_weight: float,
    trajectory_curvature_weight: float,
) -> float:
    score = float(track.total_score)
    prior_samples: list[float] = []
    if prior_heatmap is not None and float(prior_weight) > 0.0:
        prior = np.asarray(prior_heatmap, dtype=np.float32)
        for point in track.points:
            ch = int(point.ch_idx)
            t = int(point.t_idx)
            if 0 <= ch < prior.shape[0] and 0 <= t < prior.shape[1]:
                prior_samples.append(float(max(0.0, prior[ch, t])))
        if prior_samples:
            score += float(prior_weight) * float(np.mean(prior_samples)) * max(1.0, float(len(track.points)))
    if prior_time_hint is not None and float(prior_weight) > 0.0:
        hint = np.asarray(prior_time_hint, dtype=np.float64)
        time_errors: list[float] = []
        for point in track.points:
            ch = int(point.ch_idx)
            if 0 <= ch < hint.shape[0] and np.isfinite(hint[ch]):
                time_errors.append(abs(float(point.time_s) - float(hint[ch])))
        if time_errors:
            score -= float(prior_weight) * float(np.mean(time_errors)) * float(fs)
    if len(track.points) >= 3:
        ch = np.array([int(p.ch_idx) for p in track.points], dtype=np.float64)
        t = np.array([float(p.time_s) for p in track.points], dtype=np.float64)
        if np.ptp(ch) >= 2:
            slope, intercept = np.polyfit(ch, t, deg=1)
            resid = t - (slope * ch + intercept)
            fit_pen = float(np.mean(np.abs(resid)))
            if len(resid) >= 3:
                curv = np.diff(resid, n=2)
                curv_pen = float(np.mean(np.abs(curv)))
            else:
                curv_pen = 0.0
            score -= float(trajectory_fit_weight) * fit_pen * float(fs)
            score -= float(trajectory_curvature_weight) * curv_pen * float(fs)
    return score + 1e-3 * float(len(track.points))


def _extract_hypothesis_tracks(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    cfg: SingleVehicleTrackerConfig,
    *,
    prior_heatmap: Optional[np.ndarray],
    prior_time_hint: Optional[np.ndarray],
    prior_weight: float,
) -> list[Track]:
    from autotrack.core.track_extractor_graph import extract_all

    graph_cfg = replace(_graph_config(cfg), max_tracks=int(max(1, cfg.candidate_hypotheses)))
    work = np.asarray(data, dtype=np.float32)
    if prior_heatmap is not None:
        prior = np.asarray(prior_heatmap, dtype=np.float32)
        if prior.shape != work.shape:
            raise ValueError(f"prior_heatmap shape {tuple(prior.shape)} must match data shape {tuple(work.shape)}")
        work = np.abs(work) + float(max(0.0, prior_weight)) * np.maximum(prior, 0.0)
    tracks = extract_all(
        work,
        float(fs),
        float(dx_m),
        str(direction),
        float(vmin_kmh),
        float(vmax_kmh),
        config=graph_cfg,
    )
    if not tracks:
        return []
    ranked = sorted(
        tracks,
        key=lambda tr: _track_prior_score(
            tr,
            prior_heatmap=prior_heatmap,
            prior_time_hint=prior_time_hint,
            fs=float(fs),
            prior_weight=float(cfg.hypothesis_prior_weight),
            trajectory_fit_weight=float(cfg.trajectory_fit_weight),
            trajectory_curvature_weight=float(cfg.trajectory_curvature_weight),
        ),
        reverse=True,
    )
    return ranked


def merge_single_vehicle_track_fragments(
    tracks: list[Track],
    *,
    fs: float,
    dx_m: float,
    direction: str,
    merge_tolerance_samples: int = 180,
    max_gap_channels: int = 12,
    kalman_process_var: float = 0.8,
    kalman_meas_var: float = 0.18,
    kalman_fill_missing: bool = True,
) -> list[Track]:
    """Merge multiple window-level fragments of a single vehicle into one track.

    The input tracks are assumed to represent the same vehicle under overlapping
    sliding windows. The merge keeps the best time estimate per channel and then
    smooths the recovered channel-time line.
    """
    if not tracks:
        return []
    if len(tracks) == 1:
        track = tracks[0]
        smoothed = _kalman_smooth_channel_times(
            track.points,
            fs=float(fs),
            dx_m=float(dx_m),
            process_var=float(kalman_process_var),
            meas_var=float(kalman_meas_var),
            fill_missing=bool(kalman_fill_missing),
            bridge_gap_channels=int(max_gap_channels),
        )
        return [_recompute_track(int(track.track_id), str(direction), smoothed, float(dx_m))]

    grouped: dict[int, list[TrackPoint]] = {}
    for tr in tracks:
        for point in tr.points:
            grouped.setdefault(int(point.ch_idx), []).append(point)

    merged_points: list[TrackPoint] = []
    for ch in sorted(grouped):
        pts = sorted(grouped[ch], key=lambda p: (float(p.time_s), -float(p.score)))
        clusters: list[list[TrackPoint]] = []
        for point in pts:
            placed = False
            for cluster in clusters:
                if abs(float(cluster[-1].time_s) - float(point.time_s)) <= float(merge_tolerance_samples) / float(fs):
                    cluster.append(point)
                    placed = True
                    break
            if not placed:
                clusters.append([point])
        for cluster in clusters:
            scores = np.asarray([max(1e-6, float(p.score)) for p in cluster], dtype=np.float64)
            times = np.asarray([float(p.time_s) for p in cluster], dtype=np.float64)
            amps = np.asarray([float(p.amp) for p in cluster], dtype=np.float64)
            t_idx = int(round(float(np.average(times, weights=scores)) * float(fs)))
            merged_points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=int(t_idx),
                    time_s=float(np.average(times, weights=scores)),
                    offset_m=float(ch) * float(dx_m),
                    amp=float(np.max(amps)),
                    score=float(np.max(scores)),
                )
            )

    smoothed = _kalman_smooth_channel_times(
        merged_points,
        fs=float(fs),
        dx_m=float(dx_m),
        process_var=float(kalman_process_var),
        meas_var=float(kalman_meas_var),
        fill_missing=bool(kalman_fill_missing),
        bridge_gap_channels=int(max_gap_channels),
    )
    return [_recompute_track(0, str(direction), smoothed, float(dx_m))]


def _kalman_smooth_channel_times(
    points: list[TrackPoint],
    *,
    fs: float,
    dx_m: float,
    process_var: float,
    meas_var: float,
    fill_missing: bool,
    bridge_gap_channels: int,
) -> list[TrackPoint]:
    if len(points) < 2:
        return list(points)

    ordered = sorted(points, key=lambda p: int(p.ch_idx))
    ch_min = int(ordered[0].ch_idx)
    ch_max = int(ordered[-1].ch_idx)
    obs_map = {int(p.ch_idx): p for p in ordered}
    first_speed = abs(float(ordered[1].time_s) - float(ordered[0].time_s))
    if first_speed <= 1e-9:
        first_speed = 1.0

    # State = [time_s, slope_s_per_channel].
    x = np.array([float(ordered[0].time_s), (float(ordered[-1].time_s) - float(ordered[0].time_s)) / max(1, ch_max - ch_min)], dtype=np.float64)
    if not np.isfinite(x[1]) or abs(x[1]) <= 1e-9:
        x[1] = first_speed
    p = np.diag([float(meas_var), max(1.0, float(meas_var))]).astype(np.float64)
    f = lambda step: np.array([[1.0, float(step)], [0.0, 1.0]], dtype=np.float64)
    q_base = np.array([[0.25, 0.5], [0.5, 1.0]], dtype=np.float64) * float(process_var)
    h = np.array([[1.0, 0.0]], dtype=np.float64)
    r = np.array([[float(meas_var)]], dtype=np.float64)

    filtered_x: list[np.ndarray] = []
    filtered_p: list[np.ndarray] = []
    for ch in range(ch_min, ch_max + 1):
        ff = f(1.0)
        q = q_base.copy()
        x = ff @ x
        p = ff @ p @ ff.T + q
        obs = obs_map.get(ch)
        if obs is not None:
            z = np.array([[float(obs.time_s)]], dtype=np.float64)
            y = z - h @ x
            s = h @ p @ h.T + r
            k = p @ h.T @ np.linalg.inv(s)
            x = x + (k @ y).reshape(-1)
            p = (np.eye(2, dtype=np.float64) - k @ h) @ p
        filtered_x.append(x.copy())
        filtered_p.append(p.copy())

    smoothed_x = [arr.copy() for arr in filtered_x]
    smoothed_p = [arr.copy() for arr in filtered_p]
    for idx in range(len(smoothed_x) - 2, -1, -1):
        p_f = filtered_p[idx]
        p_pred = f(1.0) @ p_f @ f(1.0).T + q_base
        g = p_f @ f(1.0).T @ np.linalg.pinv(p_pred)
        smoothed_x[idx] = filtered_x[idx] + g @ (smoothed_x[idx + 1] - (f(1.0) @ filtered_x[idx]))
        smoothed_p[idx] = p_f + g @ (smoothed_p[idx + 1] - p_pred) @ g.T

    smoothed_points: list[TrackPoint] = []
    for offset, ch in enumerate(range(ch_min, ch_max + 1)):
        state = smoothed_x[offset]
        obs = obs_map.get(ch)
        if obs is None and not fill_missing:
            continue
        if obs is not None:
            t_idx = int(obs.t_idx)
            amp = float(obs.amp)
            score = float(obs.score)
        else:
            t_idx = int(round(float(state[0]) * float(fs)))
            amp = float(np.mean([p.amp for p in ordered]))
            score = float(np.mean([p.score for p in ordered]))
        smoothed_points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(t_idx),
                time_s=float(state[0]),
                offset_m=float(ch) * float(dx_m),
                amp=float(amp),
                score=float(score),
            )
        )
    # Avoid filling long holes blindly.
    if fill_missing and bridge_gap_channels > 0 and len(smoothed_points) >= 2:
        final: list[TrackPoint] = [smoothed_points[0]]
        for left, right in zip(smoothed_points[:-1], smoothed_points[1:]):
            gap = int(right.ch_idx) - int(left.ch_idx)
            if gap > 1 and gap <= int(bridge_gap_channels):
                dt = (float(right.time_s) - float(left.time_s)) / float(gap)
                for step in range(1, gap):
                    ch = int(left.ch_idx) + step
                    final.append(
                        TrackPoint(
                            ch_idx=int(ch),
                            t_idx=int(round(float(left.t_idx) + step * (float(right.t_idx) - float(left.t_idx)) / float(gap))),
                            time_s=float(left.time_s) + step * float(dt),
                            offset_m=float(ch) * float(dx_m),
                            amp=float(0.5 * (left.amp + right.amp)),
                            score=float(0.5 * (left.score + right.score)),
                        )
                    )
            final.append(right)
        smoothed_points = final
    return smoothed_points


def _match_gap_candidates_hungarian(
    candidates: list[list[tuple[int, float, float]]],
    predicted_times: dict[int, float],
    *,
    gate_seconds: float,
) -> list[tuple[int, int]]:
    """Match predicted channel times to gap candidates.

    `candidates` is a list of per-channel tuples `(peak_idx, time_s, score)`.
    Returns `(channel, peak_idx)` assignments.
    """

    flat: list[tuple[int, int, float, float]] = []
    for ch, items in enumerate(candidates):
        for peak_idx, time_s, score in items:
            flat.append((int(ch), int(peak_idx), float(time_s), float(score)))
    if not flat or not predicted_times:
        return []

    channels = sorted(predicted_times)
    cost = np.full((len(channels), len(flat)), 1e6, dtype=np.float64)
    for i, ch in enumerate(channels):
        pred = float(predicted_times[ch])
        for j, (cand_ch, peak_idx, time_s, score) in enumerate(flat):
            if cand_ch != ch:
                continue
            err = abs(time_s - pred)
            if err > float(gate_seconds):
                continue
            cost[i, j] = err - 0.01 * float(score)
    row_idx, col_idx = linear_sum_assignment(cost)
    out: list[tuple[int, int]] = []
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        if cost[r, c] < 1e5:
            out.append((int(channels[r]), int(flat[c][1])))
    return out


def extract_single_vehicle_track(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[SingleVehicleTrackerConfig | dict[str, Any]] = None,
    diagnostics: Optional[dict[str, Any]] = None,
    prior_heatmap: Optional[np.ndarray] = None,
    prior_weight: float = 1.0,
    prior_time_hint: Optional[np.ndarray] = None,
) -> list[Track]:
    """Extract one physically consistent vehicle track from a DAS window."""
    cfg = _as_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must have shape [n_channel, n_sample]")
    if fs <= 0 or dx_m <= 0:
        raise ValueError("fs and dx_m must be positive")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be forward or reverse")
    if vmin_kmh <= 0 or vmax_kmh <= 0 or vmin_kmh > vmax_kmh:
        raise ValueError("invalid speed window")

    graph_cfg = _graph_config(cfg)
    relaxed_graph_cfg = _relaxed_graph_config(cfg)
    work = arr
    prior = None
    if prior_heatmap is not None:
        prior = np.asarray(prior_heatmap, dtype=np.float32)
        if prior.shape != arr.shape:
            raise ValueError(f"prior_heatmap shape {tuple(prior.shape)} must match data shape {tuple(arr.shape)}")
        work = np.abs(arr) + float(max(0.0, prior_weight)) * np.maximum(prior, 0.0)
    nodes = _build_nodes(work, float(fs), graph_cfg)
    if prior is not None and float(prior_weight) > 0.0:
        prior_scale = float(prior_weight)
        for ch in range(int(arr.shape[0])):
            if nodes[ch]["t"].size == 0:
                continue
            prior_vals = np.clip(prior[ch, nodes[ch]["t"]], 0.0, None).astype(np.float32, copy=False)
            nodes[ch]["score"] = nodes[ch]["score"].astype(np.float32, copy=False) + prior_scale * prior_vals
    vmin_mps = float(vmin_kmh) / 3.6
    vmax_mps = float(vmax_kmh) / 3.6

    from autotrack.core.track_extractor_graph import _extract_best_track

    best: Optional[Track] = None
    if int(cfg.candidate_hypotheses) > 1:
        hypothesis_tracks = _extract_hypothesis_tracks(
            arr,
            float(fs),
            float(dx_m),
            str(direction),
            float(vmin_kmh),
            float(vmax_kmh),
            cfg,
            prior_heatmap=prior,
            prior_time_hint=prior_time_hint,
            prior_weight=float(prior_weight),
        )
        if hypothesis_tracks:
            best = hypothesis_tracks[0]

    if best is None:
        # This remains a single-track decoder: choose one coherent path and
        # suppress everything else. The extra hypotheses are only a ranking aid.
        best = _extract_best_track(
            nodes=[
                {
                    "t": n["t"].copy(),
                    "amp": n["amp"].copy(),
                    "score": n["score"].copy(),
                }
                for n in nodes
            ],
            fs=float(fs),
            dx_m=float(dx_m),
            direction=str(direction),
            vmin_mps=float(vmin_mps),
            vmax_mps=float(vmax_mps),
            n_samples=int(arr.shape[1]),
            config=graph_cfg,
            track_id=0,
            prior_time_hint=prior_time_hint,
            prior_channel_weight=float(prior_weight) * 1.5,
        )
    if best is None:
        best = _extract_best_track(
            nodes=[
                {
                    "t": n["t"].copy(),
                    "amp": n["amp"].copy(),
                    "score": n["score"].copy(),
                }
                for n in _build_nodes(work, float(fs), relaxed_graph_cfg)
            ],
            fs=float(fs),
            dx_m=float(dx_m),
            direction=str(direction),
            vmin_mps=float(vmin_mps),
            vmax_mps=float(vmax_mps),
            n_samples=int(arr.shape[1]),
            config=relaxed_graph_cfg,
            track_id=0,
            prior_time_hint=prior_time_hint,
            prior_channel_weight=float(prior_weight) * 1.5,
        )
    if best is None:
        if diagnostics is not None:
            diagnostics.update({"status": "no_track", "candidate_count": int(sum(int(n["t"].size) for n in nodes))})
        return []

    bridge_points = _kalman_smooth_channel_times(
        best.points,
        fs=float(fs),
        dx_m=float(dx_m),
        process_var=float(cfg.kalman_process_var),
        meas_var=float(cfg.kalman_meas_var),
        fill_missing=bool(cfg.kalman_fill_missing),
        bridge_gap_channels=int(cfg.kalman_bridge_gap_channels),
    )
    # Re-anchor missing channels against real candidates when the detector saw
    # them. This keeps the Kalman fill close to actual peaks instead of only
    # relying on linear interpolation.
    observed_channels = {int(p.ch_idx) for p in best.points}
    predicted_times: dict[int, float] = {}
    gap_candidates: list[list[tuple[int, float, float]]] = [[] for _ in range(int(arr.shape[0]))]
    for ch in range(int(arr.shape[0])):
        candidates = []
        for idx in range(int(nodes[ch]["t"].shape[0])):
            candidates.append((int(idx), float(nodes[ch]["t"][idx]) / float(fs), float(nodes[ch]["score"][idx])))
        gap_candidates[ch] = candidates
    for point in bridge_points:
        if int(point.ch_idx) not in observed_channels:
            predicted_times[int(point.ch_idx)] = float(point.time_s)
    assigned = _match_gap_candidates_hungarian(
        gap_candidates,
        predicted_times,
        gate_seconds=float(cfg.kalman_gate_seconds),
    )
    assigned_map = {(int(ch), int(peak_idx)) for ch, peak_idx in assigned}
    assigned_points: list[TrackPoint] = []
    for point in bridge_points:
        if int(point.ch_idx) in observed_channels:
            assigned_points.append(point)
            continue
        ch = int(point.ch_idx)
        match = None
        for peak_idx, time_s, score in gap_candidates[ch]:
            if (ch, int(peak_idx)) in assigned_map:
                match = (int(peak_idx), float(time_s), float(score))
                break
        if match is None:
            assigned_points.append(point)
            continue
        peak_idx, time_s, score = match
        assigned_points.append(
            TrackPoint(
                ch_idx=int(ch),
                t_idx=int(round(time_s * float(fs))),
                time_s=float(time_s),
                offset_m=float(ch) * float(dx_m),
                amp=float(score),
                score=float(score),
            )
        )
    track = _recompute_track(0, str(direction), assigned_points, float(dx_m))
    if diagnostics is not None:
        diagnostics.update(
            {
                "status": "ok",
                "candidate_count": int(sum(int(n["t"].size) for n in nodes)),
                "seed_point_count": int(len(best.points)),
                "final_point_count": int(len(track.points)),
                "mean_speed_kmh": float(track.mean_speed_kmh),
                "total_score": float(track.total_score),
            }
        )
    return [track]
