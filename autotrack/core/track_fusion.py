"""Trajectory fusion helpers for extending PeakSlotNet tracks with graph search.

Purpose:
    Use high-confidence PeakSlotNet tracks as seeds and extend them with the
    classic peak graph search. This module does not discover new tracks from
    scratch; it only fills endpoints and small internal channel gaps for tracks
    that PeakSlotNet already detected.

Example:
    from autotrack.core.track_fusion import extend_peakslot_tracks_with_graph

    fused = extend_peakslot_tracks_with_graph(
        data=window,
        fs=1000.0,
        dx_m=100.0,
        tracks=peakslot_tracks,
        direction="forward",
        vmin_kmh=60.0,
        vmax_kmh=100.0,
        config={"fusion_mode": "graph_extend"},
    )

Outputs:
    A list of `Track` objects with the same track ids and directions as the
    input tracks, optionally containing additional graph-supported points.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from autotrack.core.track_extractor_graph import ExtractorConfig, Track, TrackPoint, _build_nodes, extend_track


@dataclass
class TrackFusionConfig:
    fusion_mode: str = "off"
    fusion_min_seed_channels: int = 4
    fusion_extend_left: bool = True
    fusion_extend_right: bool = True
    fusion_graph_prominence: float = 0.18
    fusion_graph_min_peak_distance: int = 120
    fusion_graph_max_skip_channels: int = 8
    fusion_min_added_channels: int = 1
    fusion_nms_tolerance_samples: int = 180
    fusion_bridge_search_radius_samples: int = 900


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


def _as_fusion_config(config: Optional[TrackFusionConfig | dict[str, Any]]) -> TrackFusionConfig:
    if config is None:
        return TrackFusionConfig()
    if isinstance(config, TrackFusionConfig):
        return config
    if isinstance(config, dict):
        defaults = TrackFusionConfig()
        return TrackFusionConfig(
            fusion_mode=str(config.get("fusion_mode", defaults.fusion_mode)),
            fusion_min_seed_channels=int(config.get("fusion_min_seed_channels", defaults.fusion_min_seed_channels)),
            fusion_extend_left=_bool_value(config.get("fusion_extend_left"), defaults.fusion_extend_left),
            fusion_extend_right=_bool_value(config.get("fusion_extend_right"), defaults.fusion_extend_right),
            fusion_graph_prominence=float(config.get("fusion_graph_prominence", defaults.fusion_graph_prominence)),
            fusion_graph_min_peak_distance=int(
                config.get("fusion_graph_min_peak_distance", defaults.fusion_graph_min_peak_distance)
            ),
            fusion_graph_max_skip_channels=int(
                config.get("fusion_graph_max_skip_channels", defaults.fusion_graph_max_skip_channels)
            ),
            fusion_min_added_channels=int(config.get("fusion_min_added_channels", defaults.fusion_min_added_channels)),
            fusion_nms_tolerance_samples=int(
                config.get("fusion_nms_tolerance_samples", defaults.fusion_nms_tolerance_samples)
            ),
            fusion_bridge_search_radius_samples=int(
                config.get("fusion_bridge_search_radius_samples", defaults.fusion_bridge_search_radius_samples)
            ),
        )
    raise TypeError("config must be TrackFusionConfig / dict / None")


def _graph_config(cfg: TrackFusionConfig) -> ExtractorConfig:
    return ExtractorConfig(
        prominence=float(cfg.fusion_graph_prominence),
        min_peak_distance=int(max(1, cfg.fusion_graph_min_peak_distance)),
        max_skip_channels=int(max(1, cfg.fusion_graph_max_skip_channels)),
        edge_relax_enabled=True,
        edge_min_track_channels=2,
        min_track_channels=max(2, int(cfg.fusion_min_seed_channels)),
    )


def _opposite_direction(direction: str) -> str:
    return "reverse" if str(direction) == "forward" else "forward"


def _mean_speed_kmh(points: list[TrackPoint], dx_m: float) -> float:
    if len(points) < 2:
        return float("nan")
    pts = sorted(points, key=lambda p: p.ch_idx)
    speeds: list[float] = []
    for left, right in zip(pts[:-1], pts[1:]):
        dch = abs(int(right.ch_idx) - int(left.ch_idx))
        dt = abs(float(right.time_s) - float(left.time_s))
        if dch <= 0 or dt <= 1e-9:
            continue
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


def _merge_points(
    original_points: list[TrackPoint],
    added_points: list[TrackPoint],
    *,
    tolerance_samples: int,
) -> list[TrackPoint]:
    by_channel: dict[int, tuple[TrackPoint, bool]] = {int(p.ch_idx): (p, True) for p in original_points}
    tol = int(max(0, tolerance_samples))
    for point in added_points:
        ch = int(point.ch_idx)
        current = by_channel.get(ch)
        if current is None:
            by_channel[ch] = (point, False)
            continue
        existing, is_original = current
        if is_original and abs(int(existing.t_idx) - int(point.t_idx)) > tol:
            continue
        if float(point.score) > float(existing.score):
            by_channel[ch] = (point, False)
    return [value[0] for value in by_channel.values()]


def _speed_ok(
    left: TrackPoint,
    right: TrackPoint,
    *,
    direction: str,
    dx_m: float,
    vmin_kmh: float,
    vmax_kmh: float,
    slack_ratio: float = 0.35,
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
    return float(vmin_kmh) * (1.0 - slack_ratio) <= speed_kmh <= float(vmax_kmh) * (1.0 + slack_ratio)


def _bridge_internal_gaps(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    track: Track,
    *,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    graph_cfg: ExtractorConfig,
    fusion_cfg: TrackFusionConfig,
) -> list[TrackPoint]:
    points = sorted(track.points, key=lambda p: int(p.ch_idx))
    if len(points) < 2:
        return []
    nodes = _build_nodes(data, float(fs), graph_cfg)
    radius = int(max(1, fusion_cfg.fusion_bridge_search_radius_samples))
    max_gap = int(max(1, fusion_cfg.fusion_graph_max_skip_channels))
    added: list[TrackPoint] = []

    for left, right in zip(points[:-1], points[1:]):
        gap = int(right.ch_idx) - int(left.ch_idx)
        if gap <= 1 or gap > max_gap:
            continue
        if not _speed_ok(left, right, direction=direction, dx_m=dx_m, vmin_kmh=vmin_kmh, vmax_kmh=vmax_kmh):
            continue
        for ch in range(int(left.ch_idx) + 1, int(right.ch_idx)):
            frac = float(ch - int(left.ch_idx)) / float(gap)
            guide_t = int(round(float(left.t_idx) + frac * (float(right.t_idx) - float(left.t_idx))))
            if ch < 0 or ch >= len(nodes) or nodes[ch]["t"].size == 0:
                continue
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
            if _speed_ok(left, candidate, direction=direction, dx_m=dx_m, vmin_kmh=vmin_kmh, vmax_kmh=vmax_kmh) and _speed_ok(
                candidate, right, direction=direction, dx_m=dx_m, vmin_kmh=vmin_kmh, vmax_kmh=vmax_kmh
            ):
                added.append(candidate)
    return added


def extend_peakslot_tracks_with_graph(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    tracks: list[Track],
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[TrackFusionConfig | dict[str, Any]] = None,
    diagnostics: Optional[dict[str, Any]] = None,
) -> list[Track]:
    """Extend existing PeakSlotNet tracks with graph-search endpoint and gap points."""
    cfg = _as_fusion_config(config)
    if str(cfg.fusion_mode).lower() != "graph_extend":
        if diagnostics is not None:
            diagnostics.update(
                {
                    "fusion_enabled": False,
                    "fusion_input_track_count": int(len(tracks)),
                    "fusion_output_track_count": int(len(tracks)),
                    "fusion_added_point_count": 0,
                    "fusion_tracks": [],
                }
            )
        return list(tracks)

    arr = np.asarray(data, dtype=np.float32)
    graph_cfg = _graph_config(cfg)
    fused: list[Track] = []
    diag_rows: list[dict[str, Any]] = []
    total_added = 0
    for track in tracks:
        before_count = int(len(track.points))
        if before_count < int(cfg.fusion_min_seed_channels):
            fused.append(track)
            diag_rows.append(
                {
                    "track_id": int(track.track_id),
                    "before_points": before_count,
                    "after_points": before_count,
                    "added_points": 0,
                    "status": "skipped_short_seed",
                }
            )
            continue

        seed = sorted(track.points, key=lambda p: int(p.ch_idx))
        added: list[TrackPoint] = []
        if bool(cfg.fusion_extend_left):
            added.extend(
                extend_track(
                    arr,
                    fs,
                    dx_m,
                    _opposite_direction(direction),
                    vmin_kmh,
                    vmax_kmh,
                    seed,
                    side="left",
                    config=graph_cfg,
                )
            )
        if bool(cfg.fusion_extend_right):
            added.extend(
                extend_track(arr, fs, dx_m, direction, vmin_kmh, vmax_kmh, seed, side="right", config=graph_cfg)
            )
        added.extend(
            _bridge_internal_gaps(
                arr,
                fs,
                dx_m,
                track,
                direction=direction,
                vmin_kmh=vmin_kmh,
                vmax_kmh=vmax_kmh,
                graph_cfg=graph_cfg,
                fusion_cfg=cfg,
            )
        )

        merged = _merge_points(seed, added, tolerance_samples=int(cfg.fusion_nms_tolerance_samples))
        after_count = int(len(merged))
        added_count = max(0, after_count - before_count)
        total_added += added_count
        if added_count < int(cfg.fusion_min_added_channels):
            fused.append(track)
            diag_rows.append(
                {
                    "track_id": int(track.track_id),
                    "before_points": before_count,
                    "after_points": before_count,
                    "added_points": 0,
                    "status": "kept_original",
                }
            )
            continue
        fused_track = _recompute_track(int(track.track_id), str(track.direction), merged, float(dx_m))
        fused.append(fused_track)
        diag_rows.append(
            {
                "track_id": int(track.track_id),
                "before_points": before_count,
                "after_points": int(len(fused_track.points)),
                "added_points": added_count,
                "status": "fused",
            }
        )

    if diagnostics is not None:
        diagnostics.update(
            {
                "fusion_enabled": True,
                "fusion_input_track_count": int(len(tracks)),
                "fusion_output_track_count": int(len(fused)),
                "fusion_added_point_count": int(total_added),
                "fusion_tracks": diag_rows,
            }
        )
    return fused
