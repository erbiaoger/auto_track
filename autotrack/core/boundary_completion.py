"""Boundary-aware trajectory completion for channel-time windows.

This module repairs PeakSlotNet fragments after graph fusion. It treats the
current channel-time window as a rectangle and keeps/merges tracks that can be
explained as physically plausible lines crossing two window boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, _build_nodes, extend_track


@dataclass
class BoundaryCompletionConfig:
    boundary_completion_mode: str = "strict"
    boundary_max_gap_channels: int = 12
    boundary_fit_residual_s: float = 0.8
    boundary_margin_channels: int = 2
    boundary_margin_seconds: float = 3.0
    boundary_dt_slack_ratio: float = 0.35
    boundary_min_seed_channels: int = 4
    boundary_projected_completion_enabled: bool = True
    boundary_projected_min_span_channels: int = 18
    boundary_graph_prominence: float = 0.18
    boundary_graph_min_peak_distance: int = 120
    boundary_graph_max_skip_channels: int = 8
    boundary_bridge_search_radius_samples: int = 900
    boundary_nms_tolerance_samples: int = 180


@dataclass(frozen=True)
class _FitInfo:
    valid: bool
    slope_s_per_ch: float
    intercept_s: float
    residual_s: float
    speed_kmh: float
    edges: tuple[str, ...]
    edge_points: tuple[tuple[str, float, float], ...]
    supported_edges: tuple[str, ...]


def _as_config(config: Optional[BoundaryCompletionConfig | dict[str, Any]]) -> BoundaryCompletionConfig:
    if config is None:
        return BoundaryCompletionConfig()
    if isinstance(config, BoundaryCompletionConfig):
        return config
    if isinstance(config, dict):
        defaults = BoundaryCompletionConfig()
        return BoundaryCompletionConfig(
            boundary_completion_mode=str(config.get("boundary_completion_mode", defaults.boundary_completion_mode)),
            boundary_max_gap_channels=int(config.get("boundary_max_gap_channels", defaults.boundary_max_gap_channels)),
            boundary_fit_residual_s=float(config.get("boundary_fit_residual_s", defaults.boundary_fit_residual_s)),
            boundary_margin_channels=int(config.get("boundary_margin_channels", defaults.boundary_margin_channels)),
            boundary_margin_seconds=float(config.get("boundary_margin_seconds", defaults.boundary_margin_seconds)),
            boundary_dt_slack_ratio=float(config.get("boundary_dt_slack_ratio", defaults.boundary_dt_slack_ratio)),
            boundary_min_seed_channels=int(config.get("boundary_min_seed_channels", defaults.boundary_min_seed_channels)),
            boundary_projected_completion_enabled=_bool_value(
                config.get("boundary_projected_completion_enabled"),
                defaults.boundary_projected_completion_enabled,
            ),
            boundary_projected_min_span_channels=int(
                config.get("boundary_projected_min_span_channels", defaults.boundary_projected_min_span_channels)
            ),
            boundary_graph_prominence=float(
                config.get("boundary_graph_prominence", config.get("fusion_graph_prominence", defaults.boundary_graph_prominence))
            ),
            boundary_graph_min_peak_distance=int(
                config.get(
                    "boundary_graph_min_peak_distance",
                    config.get("fusion_graph_min_peak_distance", defaults.boundary_graph_min_peak_distance),
                )
            ),
            boundary_graph_max_skip_channels=int(
                config.get(
                    "boundary_graph_max_skip_channels",
                    config.get("fusion_graph_max_skip_channels", defaults.boundary_graph_max_skip_channels),
                )
            ),
            boundary_bridge_search_radius_samples=int(
                config.get(
                    "boundary_bridge_search_radius_samples",
                    config.get("fusion_bridge_search_radius_samples", defaults.boundary_bridge_search_radius_samples),
                )
            ),
            boundary_nms_tolerance_samples=int(
                config.get(
                    "boundary_nms_tolerance_samples",
                    config.get("fusion_nms_tolerance_samples", defaults.boundary_nms_tolerance_samples),
                )
            ),
        )
    raise TypeError("config must be BoundaryCompletionConfig / dict / None")


def _bool_value(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _graph_config(cfg: BoundaryCompletionConfig) -> ExtractorConfig:
    return ExtractorConfig(
        prominence=float(cfg.boundary_graph_prominence),
        min_peak_distance=int(max(1, cfg.boundary_graph_min_peak_distance)),
        max_skip_channels=int(max(1, cfg.boundary_graph_max_skip_channels)),
        edge_relax_enabled=True,
        edge_min_track_channels=2,
        min_track_channels=max(2, int(cfg.boundary_min_seed_channels)),
    )


def _opposite_direction(direction: str) -> str:
    return "reverse" if str(direction) == "forward" else "forward"


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: int(p.ch_idx))
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


def _merge_points(points: list[TrackPoint], added: list[TrackPoint], *, tolerance_samples: int) -> list[TrackPoint]:
    by_channel: dict[int, TrackPoint] = {int(p.ch_idx): p for p in points}
    tol = int(max(0, tolerance_samples))
    for point in added:
        ch = int(point.ch_idx)
        old = by_channel.get(ch)
        if old is None:
            by_channel[ch] = point
            continue
        if abs(int(old.t_idx) - int(point.t_idx)) <= tol and float(point.score) > float(old.score):
            by_channel[ch] = point
    return [by_channel[ch] for ch in sorted(by_channel)]


def _fit_track(
    track: Track,
    *,
    n_channels: int,
    duration_s: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    slack_ratio: float,
) -> _FitInfo:
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return _FitInfo(False, float("nan"), float("nan"), float("inf"), float("nan"), (), (), ())
    ch = np.asarray([float(p.ch_idx) for p in points], dtype=np.float64)
    ts = np.asarray([float(p.time_s) for p in points], dtype=np.float64)
    if np.unique(ch).size < 2:
        return _FitInfo(False, float("nan"), float("nan"), float("inf"), float("nan"), (), (), ())
    slope, intercept = np.polyfit(ch, ts, deg=1)
    residual = float(np.median(np.abs(ts - (float(slope) * ch + float(intercept)))))
    if str(direction) == "forward" and slope <= 0.0:
        return _FitInfo(False, float(slope), float(intercept), residual, float("nan"), (), (), ())
    if str(direction) == "reverse" and slope >= 0.0:
        return _FitInfo(False, float(slope), float(intercept), residual, float("nan"), (), (), ())
    speed_kmh = 3.6 * float(dx_m) / max(1e-9, abs(float(slope)))
    if not (float(vmin_kmh) * (1.0 - slack_ratio) <= speed_kmh <= float(vmax_kmh) * (1.0 + slack_ratio)):
        return _FitInfo(False, float(slope), float(intercept), residual, speed_kmh, (), (), ())
    edge_points = _line_rectangle_edge_hits(float(slope), float(intercept), int(n_channels), float(duration_s))
    edges = tuple(edge for edge, _, _ in edge_points)
    return _FitInfo(
        bool(len(edges) >= 2),
        float(slope),
        float(intercept),
        residual,
        speed_kmh,
        tuple(edges),
        tuple(edge_points),
        (),
    )


def _line_rectangle_edges(slope: float, intercept: float, n_channels: int, duration_s: float) -> list[str]:
    return [edge for edge, _, _ in _line_rectangle_edge_hits(slope, intercept, n_channels, duration_s)]


def _line_rectangle_edge_hits(slope: float, intercept: float, n_channels: int, duration_s: float) -> list[tuple[str, float, float]]:
    max_ch = float(max(0, int(n_channels) - 1))
    max_t = float(max(0.0, duration_s))
    eps = 1e-6
    hits: list[tuple[str, float, float]] = []
    for edge, ch in (("left", 0.0), ("right", max_ch)):
        t = float(slope) * ch + float(intercept)
        if -eps <= t <= max_t + eps:
            hits.append((edge, ch, float(np.clip(t, 0.0, max_t))))
    if abs(float(slope)) > eps:
        for edge, t in (("top", 0.0), ("bottom", max_t)):
            ch = (float(t) - float(intercept)) / float(slope)
            if -eps <= ch <= max_ch + eps:
                hits.append((edge, float(np.clip(ch, 0.0, max_ch)), t))

    unique: list[tuple[str, float, float]] = []
    coords: list[tuple[float, float]] = []
    for edge, ch, t in hits:
        if any(abs(ch - old_ch) <= 1e-4 and abs(t - old_t) <= 1e-4 for old_ch, old_t in coords):
            continue
        coords.append((ch, t))
        unique.append((edge, ch, t))
    return unique


def _supported_fit_edges(
    track: Track,
    fit: _FitInfo,
    *,
    n_channels: int,
    duration_s: float,
    cfg: BoundaryCompletionConfig,
) -> tuple[str, ...]:
    if not fit.valid or fit.residual_s > float(cfg.boundary_fit_residual_s):
        return ()
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if not points:
        return ()
    min_ch = min(int(p.ch_idx) for p in points)
    max_ch = max(int(p.ch_idx) for p in points)
    min_t = min(float(p.time_s) for p in points)
    max_t = max(float(p.time_s) for p in points)
    ch_margin = int(max(0, cfg.boundary_margin_channels))
    t_margin = float(max(0.0, cfg.boundary_margin_seconds))
    supported: list[str] = []
    for edge, _, _ in fit.edge_points:
        if edge == "left" and min_ch <= ch_margin:
            supported.append(edge)
        elif edge == "right" and max_ch >= int(n_channels) - 1 - ch_margin:
            supported.append(edge)
        elif edge == "top" and min_t <= t_margin:
            supported.append(edge)
        elif edge == "bottom" and max_t >= float(duration_s) - t_margin:
            supported.append(edge)
    return tuple(dict.fromkeys(supported))


def _with_supported_edges(fit: _FitInfo, supported_edges: tuple[str, ...]) -> _FitInfo:
    return _FitInfo(
        valid=fit.valid,
        slope_s_per_ch=fit.slope_s_per_ch,
        intercept_s=fit.intercept_s,
        residual_s=fit.residual_s,
        speed_kmh=fit.speed_kmh,
        edges=fit.edges,
        edge_points=fit.edge_points,
        supported_edges=tuple(supported_edges),
    )


def _edge_supported(track: Track, fit: _FitInfo, *, n_channels: int, duration_s: float, cfg: BoundaryCompletionConfig) -> bool:
    supported_edges = _supported_fit_edges(track, fit, n_channels=n_channels, duration_s=duration_s, cfg=cfg)
    return len(supported_edges) >= 2


def _projected_completion_supported(track: Track, fit: _FitInfo, *, n_channels: int, cfg: BoundaryCompletionConfig) -> bool:
    if not bool(cfg.boundary_projected_completion_enabled):
        return False
    if not fit.valid or fit.residual_s > float(cfg.boundary_fit_residual_s):
        return False
    if len(fit.edges) < 2 or len(fit.supported_edges) < 1:
        return False
    if len(track.points) < int(cfg.boundary_min_seed_channels):
        return False
    channels = [int(p.ch_idx) for p in track.points]
    if not channels:
        return False
    span = int(max(channels) - min(channels) + 1)
    required_span = min(
        int(max(1, n_channels)),
        int(max(cfg.boundary_min_seed_channels, cfg.boundary_projected_min_span_channels)),
    )
    return span >= required_span


def _track_complete(
    track: Track,
    *,
    n_channels: int,
    duration_s: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    cfg: BoundaryCompletionConfig,
) -> tuple[bool, _FitInfo]:
    fit = _fit_track(
        track,
        n_channels=n_channels,
        duration_s=duration_s,
        dx_m=dx_m,
        direction=direction,
        vmin_kmh=vmin_kmh,
        vmax_kmh=vmax_kmh,
        slack_ratio=float(cfg.boundary_dt_slack_ratio),
    )
    supported_edges = _supported_fit_edges(track, fit, n_channels=n_channels, duration_s=duration_s, cfg=cfg)
    fit = _with_supported_edges(fit, supported_edges)
    complete = len(supported_edges) >= 2 or _projected_completion_supported(track, fit, n_channels=n_channels, cfg=cfg)
    return bool(complete), fit


def _entry_exit_edges(fit: _FitInfo) -> tuple[str, str]:
    supported = set(fit.supported_edges)
    hits = [hit for hit in fit.edge_points if hit[0] in supported]
    if len(hits) < 2:
        hits = list(fit.edge_points)
    if len(hits) < 2:
        return "", ""
    ordered = sorted(hits, key=lambda item: float(item[2]))
    return str(ordered[0][0]), str(ordered[-1][0])


def _edge_missing_channels(track: Track, edge: str, *, n_channels: int) -> int:
    if not track.points:
        return 0
    pts = sorted(track.points, key=lambda p: int(p.ch_idx))
    if edge == "left":
        return max(0, int(pts[0].ch_idx))
    if edge == "right":
        return max(0, int(n_channels) - 1 - int(pts[-1].ch_idx))
    return 0


def _track_diag_row(track: Track, before: int, after: int, complete: bool, status: str, fit: _FitInfo, n_channels: int) -> dict[str, Any]:
    entry_edge, exit_edge = _entry_exit_edges(fit)
    if complete and len(fit.supported_edges) >= 2:
        completion_reason = "supported_two_boundaries"
    elif complete:
        completion_reason = "projected_boundary_completion"
    else:
        completion_reason = "missing_supported_boundary"
    return {
        "track_id": int(track.track_id),
        "before_points": int(before),
        "after_points": int(after),
        "added_points": max(0, int(after) - int(before)),
        "complete": bool(complete),
        "status": str(status),
        "fit_residual_s": float(fit.residual_s),
        "fit_speed_kmh": float(fit.speed_kmh),
        "boundary_edges": list(fit.edges),
        "supported_boundary_edges": list(fit.supported_edges),
        "entry_edge": entry_edge,
        "exit_edge": exit_edge,
        "missing_to_entry_channels": int(_edge_missing_channels(track, entry_edge, n_channels=n_channels)),
        "missing_to_exit_channels": int(_edge_missing_channels(track, exit_edge, n_channels=n_channels)),
        "completion_reason": completion_reason,
    }


def _speed_ok(
    left: TrackPoint,
    right: TrackPoint,
    *,
    direction: str,
    dx_m: float,
    vmin_kmh: float,
    vmax_kmh: float,
    slack_ratio: float,
) -> bool:
    dch = int(right.ch_idx) - int(left.ch_idx)
    if dch <= 0:
        return False
    dt = float(right.time_s) - float(left.time_s)
    if str(direction) == "forward" and dt <= 0.0:
        return False
    if str(direction) == "reverse" and dt >= 0.0:
        return False
    speed_kmh = 3.6 * abs(float(dch) * float(dx_m)) / max(1e-9, abs(dt))
    slack = float(max(0.0, slack_ratio))
    return float(vmin_kmh) * (1.0 - slack) <= speed_kmh <= float(vmax_kmh) * (1.0 + slack)


def _link_cost(
    left: Track,
    right: Track,
    *,
    data_shape: tuple[int, int],
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    cfg: BoundaryCompletionConfig,
) -> Optional[float]:
    if str(left.direction) != str(direction) or str(right.direction) != str(direction):
        return None
    lpts = sorted(left.points, key=lambda p: int(p.ch_idx))
    rpts = sorted(right.points, key=lambda p: int(p.ch_idx))
    if not lpts or not rpts:
        return None
    if {int(p.ch_idx) for p in lpts} & {int(p.ch_idx) for p in rpts}:
        return None
    l_end = lpts[-1]
    r_start = rpts[0]
    gap = int(r_start.ch_idx) - int(l_end.ch_idx)
    if gap < 1 or gap > int(max(1, cfg.boundary_max_gap_channels)):
        return None
    if not _speed_ok(
        l_end,
        r_start,
        direction=direction,
        dx_m=dx_m,
        vmin_kmh=vmin_kmh,
        vmax_kmh=vmax_kmh,
        slack_ratio=float(cfg.boundary_dt_slack_ratio),
    ):
        return None
    merged = _recompute_track(int(left.track_id), str(direction), list(left.points) + list(right.points), float(dx_m))
    complete, fit = _track_complete(
        merged,
        n_channels=int(data_shape[0]),
        duration_s=float(data_shape[1]) / float(fs),
        dx_m=dx_m,
        direction=direction,
        vmin_kmh=vmin_kmh,
        vmax_kmh=vmax_kmh,
        cfg=cfg,
    )
    if not fit.valid or fit.residual_s > float(cfg.boundary_fit_residual_s):
        return None
    complete_bonus = -2.0 if complete else 0.0
    return float(gap) + float(fit.residual_s) + complete_bonus


def _link_fragments(
    tracks: list[Track],
    *,
    data_shape: tuple[int, int],
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    cfg: BoundaryCompletionConfig,
) -> tuple[list[Track], int]:
    merged = [_recompute_track(i, tr.direction, list(tr.points), float(dx_m)) for i, tr in enumerate(tracks)]
    link_count = 0
    while True:
        best: Optional[tuple[int, int]] = None
        best_cost = float("inf")
        for i, left in enumerate(merged):
            for j, right in enumerate(merged):
                if i == j:
                    continue
                cost = _link_cost(
                    left,
                    right,
                    data_shape=data_shape,
                    fs=fs,
                    dx_m=dx_m,
                    direction=direction,
                    vmin_kmh=vmin_kmh,
                    vmax_kmh=vmax_kmh,
                    cfg=cfg,
                )
                if cost is not None and cost < best_cost:
                    best = (i, j)
                    best_cost = float(cost)
        if best is None:
            break
        i, j = best
        stitched = _recompute_track(int(merged[i].track_id), str(direction), list(merged[i].points) + list(merged[j].points), float(dx_m))
        merged = [tr for k, tr in enumerate(merged) if k not in {i, j}] + [stitched]
        link_count += 1
    return [_recompute_track(i, tr.direction, list(tr.points), float(dx_m)) for i, tr in enumerate(merged)], int(link_count)


def _bridge_internal_gaps(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    track: Track,
    *,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    nodes: list[dict[str, np.ndarray]],
    cfg: BoundaryCompletionConfig,
) -> list[TrackPoint]:
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return []
    added: list[TrackPoint] = []
    radius = int(max(1, cfg.boundary_bridge_search_radius_samples))
    max_gap = int(max(1, cfg.boundary_max_gap_channels))
    for left, right in zip(points[:-1], points[1:]):
        gap = int(right.ch_idx) - int(left.ch_idx)
        if gap <= 1 or gap > max_gap:
            continue
        if not _speed_ok(
            left,
            right,
            direction=direction,
            dx_m=dx_m,
            vmin_kmh=vmin_kmh,
            vmax_kmh=vmax_kmh,
            slack_ratio=float(cfg.boundary_dt_slack_ratio),
        ):
            continue
        for ch in range(int(left.ch_idx) + 1, int(right.ch_idx)):
            if ch < 0 or ch >= int(data.shape[0]) or nodes[ch]["t"].size == 0:
                continue
            frac = float(ch - int(left.ch_idx)) / float(gap)
            guide_t = int(round(float(left.t_idx) + frac * (float(right.t_idx) - float(left.t_idx))))
            t_arr = nodes[ch]["t"].astype(np.int64, copy=False)
            candidates = np.where(np.abs(t_arr - int(guide_t)) <= radius)[0]
            if candidates.size == 0:
                continue
            scores = nodes[ch]["score"][candidates].astype(np.float64) - 0.002 * np.abs(t_arr[candidates] - int(guide_t))
            best_pos = int(candidates[int(np.argmax(scores))])
            t_idx = int(nodes[ch]["t"][best_pos])
            candidate = TrackPoint(
                ch_idx=int(ch),
                t_idx=t_idx,
                time_s=float(t_idx) / float(fs),
                offset_m=float(ch) * float(dx_m),
                amp=float(nodes[ch]["amp"][best_pos]),
                score=float(nodes[ch]["score"][best_pos]),
            )
            if _speed_ok(
                left,
                candidate,
                direction=direction,
                dx_m=dx_m,
                vmin_kmh=vmin_kmh,
                vmax_kmh=vmax_kmh,
                slack_ratio=float(cfg.boundary_dt_slack_ratio),
            ) and _speed_ok(
                candidate,
                right,
                direction=direction,
                dx_m=dx_m,
                vmin_kmh=vmin_kmh,
                vmax_kmh=vmax_kmh,
                slack_ratio=float(cfg.boundary_dt_slack_ratio),
            ):
                added.append(candidate)
    return added


def _extend_channel_edges(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    track: Track,
    *,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    graph_cfg: ExtractorConfig,
    cfg: BoundaryCompletionConfig,
) -> list[TrackPoint]:
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if len(points) < int(cfg.boundary_min_seed_channels):
        return []
    added: list[TrackPoint] = []
    min_ch = int(points[0].ch_idx)
    max_ch = int(points[-1].ch_idx)
    if 0 < min_ch <= int(cfg.boundary_max_gap_channels):
        added.extend(
            extend_track(
                data,
                fs,
                dx_m,
                _opposite_direction(direction),
                vmin_kmh,
                vmax_kmh,
                points,
                side="left",
                target_ch_idx=0,
                config=graph_cfg,
            )
        )
    right_gap = int(data.shape[0]) - 1 - max_ch
    if 0 < right_gap <= int(cfg.boundary_max_gap_channels):
        added.extend(
            extend_track(
                data,
                fs,
                dx_m,
                direction,
                vmin_kmh,
                vmax_kmh,
                points,
                side="right",
                target_ch_idx=int(data.shape[0]) - 1,
                config=graph_cfg,
            )
        )
    return added


def _should_reject_repair(track: Track, complete: bool, fit: _FitInfo, cfg: BoundaryCompletionConfig) -> bool:
    if complete:
        return False
    if len(track.points) >= int(cfg.boundary_min_seed_channels):
        return False
    return not fit.valid or fit.residual_s > float(cfg.boundary_fit_residual_s)


def complete_tracks_to_boundaries(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    tracks: list[Track],
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[BoundaryCompletionConfig | dict[str, Any]] = None,
    diagnostics: Optional[dict[str, Any]] = None,
) -> list[Track]:
    """Repair/keep tracks using channel-time boundary plausibility."""
    cfg = _as_config(config)
    validation_only = False
    if isinstance(config, dict):
        raw_validation = str(config.get("boundary_validation_only", "false")).strip().lower()
        validation_only = raw_validation in {"1", "true", "yes", "on"}
    mode = str(cfg.boundary_completion_mode).strip().lower()
    if mode not in {"off", "repair", "strict"}:
        mode = "repair"
    if mode == "off":
        if diagnostics is not None:
            diagnostics.update(
                {
                    "boundary_completion_enabled": False,
                    "boundary_input_track_count": int(len(tracks)),
                    "boundary_output_track_count": int(len(tracks)),
                    "boundary_completed_count": 0,
                    "boundary_linked_fragment_count": 0,
                    "boundary_extended_point_count": 0,
                    "boundary_rejected_fragment_count": 0,
                    "boundary_incomplete_after_count": 0,
                }
            )
        return list(tracks)

    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array with shape [n_channel, n_sample]")
    if fs <= 0.0 or dx_m <= 0.0:
        raise ValueError("fs and dx_m must be > 0")

    graph_cfg = _graph_config(cfg)
    linked, linked_count = _link_fragments(
        list(tracks),
        data_shape=(int(arr.shape[0]), int(arr.shape[1])),
        fs=float(fs),
        dx_m=float(dx_m),
        direction=str(direction),
        vmin_kmh=float(vmin_kmh),
        vmax_kmh=float(vmax_kmh),
        cfg=cfg,
    )

    nodes = [] if validation_only else _build_nodes(arr, float(fs), graph_cfg)
    completed_count = 0
    extended_count = 0
    rejected_count = 0
    incomplete_count = 0
    out: list[Track] = []
    rows: list[dict[str, Any]] = []
    duration_s = float(arr.shape[1]) / float(fs)

    for track in linked:
        before = int(len(track.points))
        added: list[TrackPoint] = []
        if not validation_only:
            added.extend(
                _bridge_internal_gaps(
                    arr,
                    float(fs),
                    float(dx_m),
                    track,
                    direction=str(direction),
                    vmin_kmh=float(vmin_kmh),
                    vmax_kmh=float(vmax_kmh),
                    nodes=nodes,
                    cfg=cfg,
                )
            )
            added.extend(
                _extend_channel_edges(
                    arr,
                    float(fs),
                    float(dx_m),
                    track,
                    direction=str(direction),
                    vmin_kmh=float(vmin_kmh),
                    vmax_kmh=float(vmax_kmh),
                    graph_cfg=graph_cfg,
                    cfg=cfg,
                )
            )
        merged_points = _merge_points(track.points, added, tolerance_samples=int(cfg.boundary_nms_tolerance_samples))
        repaired = _recompute_track(int(track.track_id), str(track.direction), merged_points, float(dx_m))
        after = int(len(repaired.points))
        complete, fit = _track_complete(
            repaired,
            n_channels=int(arr.shape[0]),
            duration_s=duration_s,
            dx_m=float(dx_m),
            direction=str(direction),
            vmin_kmh=float(vmin_kmh),
            vmax_kmh=float(vmax_kmh),
            cfg=cfg,
        )
        if complete:
            completed_count += 1

        reject = bool(mode == "strict" and not complete) or (mode == "repair" and _should_reject_repair(repaired, complete, fit, cfg))
        if reject:
            rejected_count += 1
            rows.append(_track_diag_row(repaired, before, after, complete, "rejected", fit, int(arr.shape[0])))
            continue

        if not complete:
            incomplete_count += 1
        extended_count += max(0, after - before)
        out.append(repaired)
        rows.append(_track_diag_row(repaired, before, after, complete, "completed" if complete else "kept_incomplete", fit, int(arr.shape[0])))

    result = [_recompute_track(i, tr.direction, list(tr.points), float(dx_m)) for i, tr in enumerate(out)]
    if diagnostics is not None:
        diagnostics.update(
            {
                "boundary_completion_enabled": True,
                "boundary_completion_mode": str(mode),
                "boundary_validation_only": bool(validation_only),
                "boundary_input_track_count": int(len(tracks)),
                "boundary_output_track_count": int(len(result)),
                "boundary_completed_count": int(completed_count),
                "boundary_linked_fragment_count": int(linked_count),
                "boundary_extended_point_count": int(extended_count),
                "boundary_rejected_fragment_count": int(rejected_count),
                "boundary_incomplete_after_count": int(incomplete_count),
                "boundary_tracks": rows,
            }
        )
    return result
