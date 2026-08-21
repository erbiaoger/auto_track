#!/usr/bin/env python3
"""Compare five recognizers using only the separated large-vehicle picks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEPARATION_SCRIPT = ROOT / "shared_data/2025-08-09/车辆三分离_大车_正向小车_反向小车.py"
sys.path[:0] = [
    str(ROOT / "common/src"),
    str(ROOT / "compatibility/autotrack_legacy"),
    str(ROOT / "methods/hybrid_vehicle_tracker/src"),
    str(ROOT / "methods/kalman_seed_tracker/src"),
    str(ROOT / "methods/hungarian_assignment_tracker/src"),
    str(ROOT / "methods/graph_search_tracker/src"),
    str(ROOT / "methods/peak_slot_tracker/src"),
    str(ROOT / "methods/vehicle_peak_set_tracker/src"),
    str(ROOT / "vehicle_replay_web/backend"),
]

# The maintained deep-learning sources still import their historical package
# name ``autotrack``.  Expose the checked-in compatibility package under that
# name for this offline comparison only; no source files are changed.
_legacy_root = ROOT / "compatibility/autotrack_legacy"
_legacy_package = types.ModuleType("autotrack")
_legacy_package.__path__ = [str(_legacy_root)]
sys.modules.setdefault("autotrack", _legacy_package)
_MODEL_CACHE = {}


def _section_vmax(section: np.ndarray) -> float:
    nonzero = section[section > 0]
    value = float(np.percentile(nonzero, 99.0)) if nonzero.size else 1.0
    return value if value > 0 else 1.0


def load_original():
    spec = importlib.util.spec_from_file_location("vehicle_three_way_compare", SEPARATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SEPARATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    default_dir = ROOT / "shared_data/2025-08-09/web_input/DAY02/source"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=default_dir / "raw_DAY02.npy")
    parser.add_argument("--pre", type=Path, default=default_dir / "pre_DAY02.npy")
    parser.add_argument("--prediction", type=Path, default=default_dir / "prediction_DAY02.npy")
    parser.add_argument("--gauss", type=Path, default=default_dir / "gauss_DAY02.npy")
    parser.add_argument("--mapping", type=Path, default=ROOT / "shared_data/2025-08-09/web_input/DAY02/mapping/raw_DAY02.mapping.json")
    parser.add_argument("--config", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/configs/day11_120s_v9.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/checkpoints/active/v9_synthetic_morphology_16384/hybrid_final.pt")
    parser.add_argument("--peak-slot-checkpoint", type=Path, default=ROOT / "methods/peak_slot_tracker/checkpoints/active/peak_slot_v4_120s_noisy_badch_cuda/checkpoint_best.pt")
    parser.add_argument("--peak-set-checkpoint", type=Path, default=ROOT / "methods/vehicle_peak_set_tracker/results/training/vehicle_peakset_run/peakguided_train_realshape_crossing_curriculum/checkpoint_best.pt")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--window-s", type=float, default=120.0)
    parser.add_argument("--stride-s", type=float, default=60.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=Path("/tmp/direct_large_method_comparison_DAY02_0_600s_waveform_style.png"))
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def sparse_plane(picks, start_s, duration_s, fs, n_channels):
    plane = np.zeros((int(round(duration_s * fs)), n_channels), dtype=np.float32)
    radius = max(1, int(round(0.08 * fs)))
    for item in picks:
        index = int(round((float(item["t"]) - start_s) * fs))
        channel = int(item["trace"])
        if not (0 <= channel < n_channels and 0 <= index < plane.shape[0]):
            continue
        amp = float(np.clip(item.get("prob", 1.0), 0.25, 1.0))
        left, right = max(0, index - radius), min(plane.shape[0], index + radius + 1)
        offsets = (np.arange(left, right, dtype=np.float32) - index) / radius
        plane[left:right, channel] = np.maximum(plane[left:right, channel], amp * np.exp(-0.5 * offsets * offsets))
    return plane


def mirror_geometry(geometry):
    from auto_track_common.types import Station, StationGeometry

    positions = np.asarray(geometry.positions_m, dtype=np.float64)
    max_position = float(positions[-1])
    stations = []
    for i in range(len(geometry)):
        source = geometry.stations[len(geometry) - 1 - i]
        stations.append(Station(i, str(source.station_id), max_position - positions[len(geometry) - 1 - i], source.sequence, source.location))
    return StationGeometry(tuple(stations))


def normalize_track(track, positions, mirror, index, prefix):
    points = []
    for point in list(getattr(track, "points", [])):
        channel = int(getattr(point, "channel_index", getattr(point, "ch_idx", 0)))
        if mirror:
            channel = len(positions) - 1 - channel
        if 0 <= channel < len(positions):
            points.append({"channel_index": channel, "position_m": float(positions[channel]), "time_s": float(getattr(point, "time_s", 0.0)), "observed": bool(getattr(point, "observed", True))})
    speed = getattr(track, "median_speed_kmh", getattr(track, "mean_speed_kmh", float("nan")))
    return SimpleNamespace(global_vehicle_id=f"{prefix}{index:04d}", points=sorted(points, key=lambda p: (p["time_s"], p["channel_index"])), median_speed_kmh=float(speed))


def run_method(method, plane, geometry, fs, args, mirror):
    work_plane = plane[:, ::-1] if mirror else plane
    work_geometry = mirror_geometry(geometry) if mirror else geometry
    positions = np.asarray(work_geometry.positions_m, dtype=np.float64)
    duration_s = float(plane.shape[0] / fs)
    if method == "kalman":
        import yaml
        from kalman_seed_tracker import KalmanVehicleTracker
        config = yaml.safe_load((ROOT / "methods/kalman_seed_tracker/configs/day11.yaml").read_text())
        tracks = KalmanVehicleTracker(config).predict_window(work_plane, sample_rate_hz=fs, geometry=work_geometry, duration_s=duration_s).tracks
    elif method == "hungarian":
        import yaml
        from hungarian_assignment_tracker import HungarianAssignmentTracker
        config = yaml.safe_load((ROOT / "methods/hungarian_assignment_tracker/configs/day11.yaml").read_text())
        tracks = HungarianAssignmentTracker(config).predict_window(work_plane, sample_rate_hz=fs, geometry=work_geometry, duration_s=duration_s).tracks
    elif method == "graph":
        from graph_search_tracker.auto_track_gpu import extract_all_gpu
        from graph_search_tracker.track_extractor_graph import ExtractorConfig
        # Match the graph-search worker's physical convention: the original
        # large-vehicle motion is reverse in station order, so reverse the
        # station axis and let the extractor search its forward direction.
        graph_axis = positions[-1] - positions[::-1]
        tracks = extract_all_gpu(work_plane.T[::-1], fs=fs, dx_m=float(np.median(np.diff(positions))), direction="forward", vmin_kmh=60.0, vmax_kmh=90.0, config=ExtractorConfig(prominence=0.4, min_peak_distance=int(round(0.5 * fs)), min_track_channels=12, edge_min_track_channels=4, max_tracks=256), x_axis_m=graph_axis)
        return [normalize_track(track, np.asarray(geometry.positions_m), mirror=True, index=i + 1, prefix=method[:1].upper()) for i, track in enumerate(tracks)]
    elif method == "peakslot":
        from peak_slot_tracker.peak_slot_model import InferenceConfig, load_checkpoint_model, predict_tracks_from_window
        cache_key = ("peakslot", str(args.peak_slot_checkpoint), args.device)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key], _ = load_checkpoint_model(args.peak_slot_checkpoint, device=args.device)
        model = _MODEL_CACHE[cache_key]
        config = InferenceConfig(time_downsample=10, objectness_threshold=0.45, peak_threshold=0.4, min_visible_channels=4, max_tracks=96, extra_candidate_slots=8, candidate_objectness_floor=0.20, viterbi_speed_min_kmh=60.0, viterbi_speed_max_kmh=90.0, decoder_mode="beam_global")
        tracks = predict_tracks_from_window(model, work_plane.T, fs, positions, config=config, device=args.device)
        tracks = [track for track in tracks if str(getattr(track, "direction", "reverse")) == "reverse"]
    elif method == "peakset":
        from vehicle_peak_set_tracker.vehicle_peak_set_transformer import PeakSetInferenceConfig, decode_peak_guided_vehicle_tracks, decode_vehicle_peak_tracks, load_checkpoint_model
        cache_key = ("peakset", str(args.peak_set_checkpoint), args.device)
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key], _ = load_checkpoint_model(args.peak_set_checkpoint, device=args.device)
        model = _MODEL_CACHE[cache_key]
        config = PeakSetInferenceConfig(objectness_threshold=0.25, complete_valid_threshold=0.35, anchor_threshold=0.45, min_visible_channels=5, max_tracks=24, dedup_tolerance_samples=30, speed_min_kmh=60.0, speed_max_kmh=90.0, graph_refine=False)
        decoded = decode_peak_guided_vehicle_tracks(model, work_plane.T, fs, positions, config=config, device=args.device) if type(model).__name__ == "PeakGuidedVehicleSetTransformer" else decode_vehicle_peak_tracks(model, work_plane.T, fs, positions, config=config, device=args.device)
        tracks = [item.track for item in decoded if str(getattr(item.track, "direction", "reverse")) == "reverse"]
    else:
        raise ValueError(method)
    return [normalize_track(track, np.asarray(geometry.positions_m), mirror, i + 1, method[:1].upper()) for i, track in enumerate(tracks)]


def _worker_track(track, index: int):
    from vehicle_replay_web.methods import WorkerPoint, WorkerTrack

    points = [
        WorkerPoint(
            channel_index=int(point["channel_index"]),
            station_id=str(point.get("station_id", f"CH{int(point['channel_index'])}")),
            position_m=float(point["position_m"]),
            time_s=float(point["time_s"]),
            observed=bool(point.get("observed", True)),
        )
        for point in track.points
    ]
    return WorkerTrack(
        track_id=str(getattr(track, "global_vehicle_id", f"local-{index}")),
        direction=str(getattr(track, "direction", "unknown")),
        points=points,
        median_speed_kmh=float(getattr(track, "median_speed_kmh", float("nan"))),
        confidence=float(getattr(track, "confidence", 1.0)) if getattr(track, "confidence", None) is not None else None,
        score=None,
        observed_count=sum(1 for point in points if point.observed),
        span_m=float(max((point.position_m for point in points), default=0.0) - min((point.position_m for point in points), default=0.0)),
        max_gap_m=float(max((right.position_m - left.position_m for left, right in zip(sorted(points, key=lambda item: item.position_m), sorted(points, key=lambda item: item.position_m)[1:])), default=0.0)),
        enters_window=False,
        exits_window=False,
        ambiguous_crossing=False,
    )


def _stitch_classic_method(method, large, geometry, fs, args, start_s, end_s):
    """Run one classic/deep method through the shared cross-window stitcher."""
    from vehicle_replay_web.methods import WorkerBatch
    from vehicle_replay_web.stitch import TrackStitcher

    stitcher = TrackStitcher(stride_s=float(args.stride_s), window_s=float(args.window_s))
    for window_start in np.arange(start_s, end_s - args.window_s + 1e-6, args.stride_s):
        window_picks = [item for item in large if window_start <= item["t"] <= window_start + args.window_s]
        plane = sparse_plane(window_picks, window_start, args.window_s, fs, len(geometry))
        local_tracks = run_method(method, plane, geometry, fs, args, mirror=False)
        worker_tracks = [_worker_track(track, index) for index, track in enumerate(local_tracks)]
        stitcher.update(WorkerBatch(worker_tracks, [], float(window_start), float(args.window_s), {}))
    return stitcher.finalize().tracks


def _consensus_points(points, *, project_to_line: bool) -> list[dict]:
    """Collapse repeated station arrivals and optionally draw their robust line.

    At a 2-second stride, one physical vehicle can be produced by dozens of
    nearly identical windows.  Median arrival time per station removes that
    oversampling; the final robust line fit keeps tiny independent-window
    timing shifts from becoming visible saw teeth in the comparison figure.
    """
    grouped = {}
    for point in points:
        grouped.setdefault(int(point["channel_index"]), []).append(point)
    rows = []
    for channel, group in grouped.items():
        times = np.asarray([float(point["time_s"]) for point in group], dtype=np.float64)
        time_s = float(np.median(times))
        chosen = min(group, key=lambda point: abs(float(point["time_s"]) - time_s))
        rows.append({**chosen, "time_s": time_s, "observed": any(bool(point.get("observed", True)) for point in group)})
    rows.sort(key=lambda point: int(point["channel_index"]))
    if len(rows) < 3 or not project_to_line:
        return rows

    position = np.asarray([float(point["position_m"]) for point in rows], dtype=np.float64)
    times = np.asarray([float(point["time_s"]) for point in rows], dtype=np.float64)
    keep = np.ones(len(rows), dtype=bool)
    for _ in range(2):
        if keep.sum() < 3:
            break
        line = np.polyfit(position[keep], times[keep], 1)
        residual = np.abs(times - np.polyval(line, position))
        mad = float(np.median(np.abs(residual[keep] - np.median(residual[keep]))))
        gate = max(0.35, min(1.2, 3.0 * 1.4826 * mad))
        updated = residual <= gate
        if np.array_equal(updated, keep):
            break
        keep = updated
    if keep.sum() < 3:
        return [row for row, accepted in zip(rows, keep) if accepted]
    line = np.polyfit(position[keep], times[keep], 1)
    return [{**row, "time_s": float(np.polyval(line, float(row["position_m"]))) } for row, accepted in zip(rows, keep) if accepted]


def _track_fit(track):
    points = _consensus_points(track.points, project_to_line=False)
    if len(points) < 3:
        return None
    position = np.asarray([float(point["position_m"]) for point in points], dtype=np.float64)
    times = np.asarray([float(point["time_s"]) for point in points], dtype=np.float64)
    if np.unique(position).size < 3:
        return None
    return points, np.polyfit(position, times, 1)


def _same_vehicle_fit(left_fit, right_fit) -> bool:
    left_points, left_line = left_fit
    right_points, right_line = right_fit
    left_by_channel = {int(point["channel_index"]): float(point["time_s"]) for point in left_points}
    right_by_channel = {int(point["channel_index"]): float(point["time_s"]) for point in right_points}
    common = sorted(set(left_by_channel).intersection(right_by_channel))
    if len(common) < 3:
        return False
    overlap = len(common) / max(1, min(len(left_by_channel), len(right_by_channel)))
    point_residual = float(np.median([abs(left_by_channel[channel] - right_by_channel[channel]) for channel in common]))
    positions = np.asarray([float(next(point["position_m"] for point in left_points if int(point["channel_index"]) == channel)) for channel in common], dtype=np.float64)
    line_residual = float(np.median(np.abs(np.polyval(left_line, positions) - np.polyval(right_line, positions))))
    left_speed = 3.6 / abs(float(left_line[0])) if abs(float(left_line[0])) > 1e-9 else float("inf")
    right_speed = 3.6 / abs(float(right_line[0])) if abs(float(right_line[0])) > 1e-9 else float("inf")
    # A strict same-station timing gate prevents neighbouring vehicles from
    # being merged even when they have almost parallel trajectories.
    return overlap >= 0.25 and point_residual <= 0.75 and line_residual <= 0.75 and abs(left_speed - right_speed) <= 8.0


def _same_vehicle(left, right) -> bool:
    left_fit = _track_fit(left)
    right_fit = _track_fit(right)
    return left_fit is not None and right_fit is not None and _same_vehicle_fit(left_fit, right_fit)


def dedup(groups):
    """Cluster repeated sliding-window outputs into consensus vehicle tracks."""
    tracks = [track for group in groups for track in group]
    fitted = [_track_fit(track) for track in tracks]
    parent = list(range(len(tracks)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # Compare only trajectories that are present during the same physical
    # time interval.  Intercept-at-zero is not a safe pre-filter here because
    # a tiny slope change is magnified by the long station geometry.
    candidates = sorted(
        (
            (index, min(float(point["time_s"]) for point in fitted[index][0]), max(float(point["time_s"]) for point in fitted[index][0]))
            for index, item in enumerate(fitted)
            if item is not None
        ),
        key=lambda item: item[1],
    )
    for offset, (left, left_start, left_end) in enumerate(candidates):
        left_fit = fitted[left]
        assert left_fit is not None
        for right, right_start, right_end in candidates[offset + 1:]:
            right_fit = fitted[right]
            assert right_fit is not None
            if right_start > left_end:
                break
            if min(left_end, right_end) - max(left_start, right_start) < 2.0:
                continue
            if _same_vehicle_fit(left_fit, right_fit):
                union(left, right)
    clusters = {}
    for index, track in enumerate(tracks):
        clusters.setdefault(find(index), []).append(track)
    result = []
    for members in clusters.values():
        best = max(members, key=lambda item: (len(item.points), np.isfinite(item.median_speed_kmh), item.median_speed_kmh))
        points = _consensus_points([point for item in members for point in item.points], project_to_line=True)
        result.append(SimpleNamespace(global_vehicle_id=best.global_vehicle_id, points=points, median_speed_kmh=float(best.median_speed_kmh)))
    return sorted(result, key=lambda item: (min((float(point["time_s"]) for point in item.points), default=float("inf")), item.global_vehicle_id))


def _same_or_contiguous_hybrid_track(left, right) -> bool:
    """Strictly join two Hybrid fragments from the same physical vehicle."""
    left_fit = _track_fit(left)
    right_fit = _track_fit(right)
    if left_fit is None or right_fit is None:
        return False
    left_points, left_line = left_fit
    right_points, right_line = right_fit
    if float(left_line[0] * right_line[0]) <= 0.0:
        return False
    left_speed = 3.6 / abs(float(left_line[0])) if abs(float(left_line[0])) > 1e-9 else float("inf")
    right_speed = 3.6 / abs(float(right_line[0])) if abs(float(right_line[0])) > 1e-9 else float("inf")
    if abs(left_speed - right_speed) > 5.0:
        return False
    left_position = np.asarray([float(point["position_m"]) for point in left_points], dtype=np.float64)
    right_position = np.asarray([float(point["position_m"]) for point in right_points], dtype=np.float64)
    low = max(float(left_position.min()), float(right_position.min()))
    high = min(float(left_position.max()), float(right_position.max()))
    if high >= low:
        probes = np.asarray([low, (low + high) / 2.0, high], dtype=np.float64)
        gate = 0.65
    elif float(left_position.max()) < float(right_position.min()):
        probes = np.asarray([(float(left_position.max()) + float(right_position.min())) / 2.0], dtype=np.float64)
        gate = 0.9
    else:
        probes = np.asarray([(float(left_position.min()) + float(right_position.max())) / 2.0], dtype=np.float64)
        gate = 0.9
    line_residual = float(np.median(np.abs(np.polyval(left_line, probes) - np.polyval(right_line, probes))))
    if line_residual > gate:
        return False
    left_by_channel = {int(point["channel_index"]): float(point["time_s"]) for point in left_points}
    right_by_channel = {int(point["channel_index"]): float(point["time_s"]) for point in right_points}
    common = sorted(set(left_by_channel).intersection(right_by_channel))
    if common:
        point_residual = float(np.median([abs(left_by_channel[channel] - right_by_channel[channel]) for channel in common]))
        if point_residual > 0.8:
            return False
    return True


def consolidate_hybrid_tracks(tracks):
    """Join fragment IDs, then render one consensus line per large vehicle."""
    parent = list(range(len(tracks)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(tracks)):
        for right in range(left + 1, len(tracks)):
            if _same_or_contiguous_hybrid_track(tracks[left], tracks[right]):
                union(left, right)
    clusters = {}
    for index, track in enumerate(tracks):
        clusters.setdefault(find(index), []).append(track)
    result = []
    for members in clusters.values():
        best = max(members, key=lambda item: (item.observed_count, item.span_m, item.confidence or 0.0))
        points = _consensus_points([point for item in members for point in item.points], project_to_line=True)
        if len(points) < 3:
            continue
        line = np.polyfit(
            np.asarray([float(point["position_m"]) for point in points], dtype=np.float64),
            np.asarray([float(point["time_s"]) for point in points], dtype=np.float64),
            1,
        )
        result.append(SimpleNamespace(
            global_vehicle_id=best.global_vehicle_id,
            points=points,
            median_speed_kmh=float(3.6 / abs(float(line[0]))) if abs(float(line[0])) > 1e-9 else float("nan"),
        ))
    return sorted(result, key=lambda item: (min(float(point["time_s"]) for point in item.points), item.global_vehicle_id))


def run_hybrid_large(original, raw, pre, picks_all, rms_min, rms_max, geometry, fs, args):
    """Reuse the original large-vehicle Hybrid path from the accepted figure."""
    from hybrid_vehicle_tracker.config import load_tracker_config
    from hybrid_vehicle_tracker.tracker import HybridVehicleTracker
    from vehicle_replay_web.methods import WorkerBatch, WorkerPoint, WorkerTrack
    from vehicle_replay_web.stitch import TrackStitcher

    config = load_tracker_config(args.config)
    config.runtime.device = args.device
    config.model.checkpoint = str(args.checkpoint)
    config.model.motion_direction = -1
    config.association.motion_direction = -1
    config.data.sample_rate_hz = fs
    config.data.duration_s = float(args.window_s)
    config.data.mapping_path = str(args.mapping)
    tracker = HybridVehicleTracker(config)
    stitcher = TrackStitcher(stride_s=float(args.stride_s), window_s=float(args.window_s))
    for window_start in np.arange(args.start_s, args.start_s + args.duration_s - args.window_s + 1e-6, args.stride_s):
        local_i0 = int(round(window_start * fs))
        local_i1 = local_i0 + int(round(args.window_s * fs))
        category_window = np.asarray(
            original.gauss_section(
                picks_all, "large", local_i0, local_i1 - local_i0, raw.shape[1], original.DT,
                original.GAUSS_WIDTH_S, rms_min, rms_max, original.AMP_MIN, original.AMP_MAX,
            ),
            dtype=np.float32,
        )
        raw_part = np.asarray(raw[local_i0:local_i1], dtype=np.float32)
        pre_part = np.asarray(pre[local_i0:local_i1], dtype=np.float32)
        mask = np.clip(category_window / max(float(original.AMP_MAX), 1e-6), 0.0, 1.0)
        raw_category = raw_part * mask
        non_event_pre = np.max(pre_part, axis=0, keepdims=True)
        pre_category = pre_part * mask + non_event_pre * (1.0 - mask)
        batch = tracker.predict(raw_category, pre_category, category_window, geometry, start_s=0.0, duration_s=float(args.window_s))
        worker_tracks = []
        for track in batch.tracks:
            points = [
                WorkerPoint(
                    channel_index=int(point.channel_index),
                    station_id=str(point.station_id),
                    position_m=float(point.position_m),
                    time_s=float(point.time_s),
                    observed=bool(point.observed),
                    observation_id=point.observation_id,
                    residual_s=point.residual_s,
                    ambiguous=bool(point.ambiguous),
                )
                for point in track.points
            ]
            worker_tracks.append(WorkerTrack(
                track_id=str(track.track_id), direction="unknown", points=points,
                median_speed_kmh=float(track.median_speed_kmh),
                confidence=float(track.confidence) if track.confidence is not None else None,
                score=float(track.score) if track.score is not None else None,
                observed_count=int(track.observed_count), span_m=float(track.span_m),
                max_gap_m=float(track.max_gap_m), enters_window=bool(track.enters_window),
                exits_window=bool(track.exits_window), ambiguous_crossing=bool(track.ambiguous_crossing),
        ))
        current = WorkerBatch(tracks=worker_tracks, observations=[], start_s=float(window_start), duration_s=float(args.window_s), diagnostics=batch.diagnostics)
        stitcher.update(current)
    final_update = stitcher.finalize()
    return final_update.tracks


def overlay(ax, tracks, prefix):
    color_map = plt.get_cmap("hsv")
    color_count = max(1, len(tracks))
    for index, track in enumerate(tracks):
        if len(track.points) < 2:
            continue
        color = matplotlib.colors.to_hex(color_map(index / color_count))
        times = np.asarray([float(point["time_s"]) for point in track.points])
        channels = np.asarray([int(point["channel_index"]) for point in track.points])
        ax.plot(times, channels, color=color, lw=1.35, alpha=0.9, zorder=8)
        ax.scatter(times, channels, s=7, color=color, edgecolors="white", linewidths=0.2, zorder=9)
        if index < 70:
            ax.text(times[-1] + 2, channels[-1], f"{prefix}{index + 1:03d}", color=color, fontsize=5.5, zorder=10)


def main():
    args = parse_args()
    original = load_original()
    raw = np.load(args.raw, mmap_mode="r")
    pre = np.load(args.pre, mmap_mode="r")
    prediction = np.load(args.prediction, mmap_mode="r")
    gauss = np.load(args.gauss, mmap_mode="r")
    fs = float(original.DT ** -1)
    end_s = min(float(raw.shape[0] / fs), args.start_s + args.duration_s)
    i0 = int(round(args.start_s * fs))
    i1 = int(round(end_s * fs))
    geometry = __import__("hybrid_vehicle_tracker.data.mapping", fromlist=["load_station_geometry"]).load_station_geometry(args.mapping)
    positions = np.asarray(geometry.positions_m, dtype=np.float64)
    print("[1/3] build picks; keep large vehicles only")
    picks_all = original.build_picks(raw, prediction, original.DT, thresh=original.PRED_THRESH, min_gap_s=original.MIN_GAP_S, energy_win_s=original.ENERGY_WIN_S, split_method=original.SPLIT_METHOD, split_pct=original.SPLIT_PCT, split_thr=original.SPLIT_THRESH)
    rms_min, rms_max = original.build_rms_norm_params(picks_all)
    picks = [item for item in picks_all if args.start_s <= item["t"] <= end_s]
    large = [item.copy() for item in picks if item["is_large"]]
    print(f"  picks={len(picks)}, large={len(large)}")
    large_plane = np.asarray(
        original.gauss_section(
            large,
            "large",
            i0,
            i1 - i0,
            raw.shape[1],
            original.DT,
            original.GAUSS_WIDTH_S,
            rms_min,
            rms_max,
            original.AMP_MIN,
            original.AMP_MAX,
        ),
        dtype=np.float32,
    )
    print("[2/3] run methods")
    method_labels = {"kalman": "Kalman", "graph": "Graph search", "hungarian": "Hungarian", "peakslot": "PeakSlotNet", "peakset": "Vehicle Peak-Set"}
    method_colors = {"kalman": "#0081a7", "graph": "#f77f00", "hungarian": "#7b2cbf", "peakslot": "#c1121f", "hybrid": "#d62728", "peakset": "#00a878"}
    method_tracks = {}
    for method in method_labels:
        method_tracks[method] = _stitch_classic_method(method, large, geometry, fs, args, args.start_s, end_s)
        print(f"  {method_labels[method]}: {len(method_tracks[method])}")
    hybrid_tracks = run_hybrid_large(original, raw, pre, picks_all, rms_min, rms_max, geometry, fs, args)
    method_tracks["hybrid"] = hybrid_tracks
    print(f"  Hybrid: {len(hybrid_tracks)}")
    print("[3/3] draw")
    print("[3/3] draw in the original waveform/gauss layout")
    t_win = np.arange(i0, i1, dtype=np.float32) / fs
    raw_win = np.asarray(raw[i0:i1], dtype=np.float32)
    gauss_win = np.asarray(gauss[i0:i1], dtype=np.float32)
    # Keep the raw waveform and the four requested method panels.  The
    # AGC/Gauss reference is an intermediate visual aid rather than a method,
    # and PeakSlotNet is intentionally excluded from this comparison figure.
    fig = plt.figure(figsize=(16, 18), facecolor="white")
    layout = gridspec.GridSpec(
        5,
        1,
        figure=fig,
        left=0.07,
        right=0.95,
        top=0.95,
        bottom=0.04,
        hspace=0.36,
        height_ratios=[1, 1.2, 1.2, 1.2, 1.2],
    )
    axes = [fig.add_subplot(layout[index]) for index in range(5)]
    for ax in axes:
        original.style_ax(ax)
    xlim = (float(t_win[0]), float(t_win[-1]))
    original.draw_wiggle(axes[0], t_win, raw_win, color="#000000", fill_color="#000000", scale=original.WIGGLE_SCALE, lw=original.WIGGLE_LW)
    axes[0].set_xlim(*xlim)
    axes[0].set_title("(1) Raw Signal", fontsize=9.5, fontweight="bold", loc="left", color="#222")
    axes[0].set_ylabel("Trace #", fontsize=8)
    axes[0].tick_params(labelbottom=False)
    method_items = [
        ("kalman", "Kalman"),
        ("graph", "Graph search"),
        ("hungarian", "Hungarian"),
        ("hybrid", "Hybrid"),
    ]
    for panel_index, (ax, (method, label)) in enumerate(zip(axes[1:], method_items), start=2):
        image = original.draw_imshow(ax, t_win, large_plane, original.CMAP_LARGE, alpha=original.IMSHOW_ALPHA, vmax=_section_vmax(large_plane))
        original.draw_ref_wiggle(ax, t_win, gauss_win, original.REF_WIGGLE_SCALE, original.REF_WIGGLE_ALPHA)
        ax.set_xlim(*xlim)
        ax.set_title(f"({panel_index}) Large Vehicles + {label} | N = {len(method_tracks[method])}", fontsize=9.5, fontweight="bold", loc="left", color=method_colors[method])
        ax.set_ylabel("Trace #", fontsize=8)
        ax.tick_params(labelbottom=False)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.012, aspect=25).ax.tick_params(labelsize=6)
        overlay(ax, method_tracks[method], method[:1].upper())
    axes[-1].tick_params(labelbottom=True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Large-vehicle recognition comparison | pick-only input | t = {args.start_s:.0f} - {end_s:.0f} s", fontsize=11, fontweight="bold", color="#111", y=0.975)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.write_text(json.dumps({"output": str(args.output), "start_s": args.start_s, "end_s": end_s, "input_policy": "classic/deep methods use large-vehicle pick points only; Hybrid uses original Raw+Pre+Gauss masked path", "small_vehicles": "excluded", "large_track_count_by_method": {key: len(value) for key, value in method_tracks.items()}, "methods": dict(method_items)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
