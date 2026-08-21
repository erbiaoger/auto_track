#!/usr/bin/env python3
"""Render the original three-way separation figure with deduplicated Hybrid tracks overlaid.

This is intentionally an offline check.  It does not start the web service or
use its rolling replay loop: the original separator classifies the full record,
the large-vehicle signal is passed to Hybrid in overlapping windows, and the
accepted tracks are matched and merged across windows before plotting.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SEPARATION_SCRIPT = ROOT / "shared_data/2025-08-09/车辆三分离_大车_正向小车_反向小车.py"
sys.path[:0] = [
    str(ROOT / "common/src"),
    str(ROOT / "methods/hybrid_vehicle_tracker/src"),
]


def load_original():
    spec = importlib.util.spec_from_file_location("vehicle_three_way_original", SEPARATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SEPARATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = ROOT / "shared_data/2025-08-09/web_input/DAY02/source"
    parser.add_argument("--raw", type=Path, default=default_dir / "raw_DAY02.npy")
    parser.add_argument("--pre", type=Path, default=default_dir / "pre_DAY02.npy")
    parser.add_argument("--prediction", type=Path, default=default_dir / "prediction_DAY02.npy")
    parser.add_argument("--gauss", type=Path, default=default_dir / "gauss_DAY02.npy")
    parser.add_argument("--mapping", type=Path, default=ROOT / "shared_data/2025-08-09/web_input/DAY02/mapping/raw_DAY02.mapping.json")
    parser.add_argument("--config", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/configs/day11_120s_v9.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/checkpoints/active/v9_synthetic_morphology_16384/hybrid_final.pt")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--window-s", type=float, default=120.0)
    parser.add_argument("--stride-s", type=float, default=60.0, help="recognition window stride; overlap enables cross-window matching")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/direct_large_recognition_DAY02_0_600s.png")
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _vmax(section: np.ndarray) -> float:
    nonzero = section[section > 0]
    value = float(np.percentile(nonzero, 99.0)) if nonzero.size else 1.0
    return value if value > 0 else 1.0


def _channel_from_point(point, positions: np.ndarray) -> int:
    if isinstance(point, dict):
        return int(point.get("channel_index", np.argmin(np.abs(positions - float(point["position_m"])))) )
    channel = getattr(point, "channel_index", None)
    if channel is not None:
        return int(channel)
    return int(np.argmin(np.abs(positions - float(point.position_m))))


def _overlay_tracks(
    ax,
    tracks: list[tuple[object, float, str]],
    positions: np.ndarray,
    *,
    label: bool,
    line_color: str | None = None,
    palette: tuple[str, ...] | None = None,
) -> None:
    colors = ("#00a878", "#7b2cbf", "#0081a7", "#f77f00", "#c1121f")
    for index, (track, offset_s, window_tag) in enumerate(tracks):
        points = list(getattr(track, "points", []))
        if len(points) < 2:
            continue
        time_values = np.asarray([
            float(point["time_s"]) if isinstance(point, dict) else float(point.time_s) + offset_s
            for point in points
        ])
        channel_values = np.asarray([_channel_from_point(point, positions) for point in points])
        color = line_color or (palette or colors)[index % len(palette or colors)]
        ax.plot(time_values, channel_values, color=color, lw=1.6, alpha=0.95, zorder=8)
        ax.scatter(time_values, channel_values, s=8, color=color, edgecolors="white", linewidths=0.25, zorder=9)
        if label:
            speed = float(getattr(track, "median_speed_kmh", float("nan")))
            vehicle_id = str(getattr(track, "global_vehicle_id", window_tag))
            ax.text(time_values[-1] + 3, channel_values[-1], f"{vehicle_id} {speed:.1f} km/h", color=color, fontsize=6.5, zorder=10)


def _peak_only_tracks(picks_window, original, positions: np.ndarray, direction: int, prefix: str) -> list[object]:
    """Build small-vehicle tracks from pick points only.

    No Raw/Pre/Gauss tensor is passed to a neural recognizer here.  The
    original separator's tau-p fitted lines provide the vehicle hypotheses,
    and only picks explicitly assigned to a fitted line are retained.
    """
    # Fit only the displayed interval.  Fitting the complete recording here
    # creates one candidate for every short-lived pick cluster, unlike the
    # original plotting program which fits the visible interval as a whole.
    small = [item.copy() for item in picks_window if not item["is_large"]]
    small, debug = original.split_directions(small, return_debug=True, verbose=False)
    tracks: list[object] = []
    next_id = 1
    for line_index, line in enumerate(debug["lines"]):
        if (1 if line["p"] > 0 else -1) != direction:
            continue
        rows = [
            item for item in small
            if item.get("on_line", False)
            and int(item.get("line_id", -1)) == line_index
            and int(item.get("direction", 0)) == direction
        ]
        traces = {int(item["trace"]) for item in rows}
        if len(traces) < int(original.MIN_TRACES):
            continue
        points = [
            {
                "channel_index": int(item["trace"]),
                "position_m": float(positions[int(item["trace"])]),
                "time_s": float(item["t"]),
                "observed": True,
            }
            for item in sorted(rows, key=lambda item: item["t"])
        ]
        # The separator's p is in seconds/trace.  Convert the fitted line to
        # seconds/metre with the actual station geometry before displaying a
        # small-vehicle speed; using 3.6/abs(p) would be only correct for a
        # one-metre station spacing.
        position_values = np.asarray([positions[int(item["trace"])] for item in rows], dtype=float)
        time_values = np.asarray([float(item["t"]) for item in rows], dtype=float)
        if np.unique(position_values).size >= 2:
            physical_slope = float(np.polyfit(position_values, time_values, 1)[0])
        else:
            physical_slope = float(line["p"])
        tracks.append(SimpleNamespace(
            global_vehicle_id=f"{prefix}{next_id:04d}",
            points=points,
            median_speed_kmh=float(3.6 / abs(physical_slope)) if physical_slope else float("nan"),
        ))
        next_id += 1
    print(f"  {prefix} peak-only fitted tracks: {len(tracks)}")
    return tracks


def main() -> None:
    args = parse_args()
    original = load_original()
    raw = np.load(args.raw, mmap_mode="r")
    pre = np.load(args.pre, mmap_mode="r")
    prediction = np.load(args.prediction, mmap_mode="r")
    gauss = np.load(args.gauss, mmap_mode="r")
    fs = float(original.DT ** -1)
    start_s = max(0.0, float(args.start_s))
    end_s = min(float(raw.shape[0] / fs), start_s + float(args.duration_s))
    i0, i1 = int(round(start_s * fs)), int(round(end_s * fs))
    t_win = np.arange(i0, i1, dtype=np.float32) / fs
    raw_win = np.asarray(raw[i0:i1], dtype=np.float32)
    agc_win = np.asarray(gauss[i0:i1], dtype=np.float32)

    print("[1/4] full-record original separation")
    picks_all = original.build_picks(
        raw,
        prediction,
        original.DT,
        thresh=original.PRED_THRESH,
        min_gap_s=original.MIN_GAP_S,
        energy_win_s=original.ENERGY_WIN_S,
        split_method=original.SPLIT_METHOD,
        split_pct=original.SPLIT_PCT,
        split_thr=original.SPLIT_THRESH,
    )
    rms_min, rms_max = original.build_rms_norm_params(picks_all)
    picks = [item for item in picks_all if start_s <= item["t"] <= end_s]
    picks_small = [item for item in picks if not item["is_large"]]
    picks_small, _ = original.split_directions(picks_small, return_debug=True, verbose=False)
    for item in picks:
        if item["is_large"]:
            item["direction"] = 0
            item["line_id"] = -1
            item["on_line"] = False
    gs_large = original.gauss_section(
        picks, "large", i0, i1 - i0, raw.shape[1], original.DT,
        original.GAUSS_WIDTH_S, rms_min, rms_max, original.AMP_MIN, original.AMP_MAX,
    )
    gs_fwd = original.gauss_section(
        picks, "fwd", i0, i1 - i0, raw.shape[1], original.DT,
        original.GAUSS_WIDTH_S, rms_min, rms_max, original.AMP_MIN, original.AMP_MAX,
    )
    gs_rev = original.gauss_section(
        picks, "rev", i0, i1 - i0, raw.shape[1], original.DT,
        original.GAUSS_WIDTH_S, rms_min, rms_max, original.AMP_MIN, original.AMP_MAX,
    )

    print(f"[2/4] Hybrid recognition on large/forward-small/reverse-small signals ({args.device})")
    from hybrid_vehicle_tracker.config import load_tracker_config
    from hybrid_vehicle_tracker.data.mapping import load_station_geometry
    from hybrid_vehicle_tracker.tracker import HybridVehicleTracker
    from vehicle_replay_web.methods import WorkerBatch, WorkerPoint, WorkerTrack
    from vehicle_replay_web.stitch import StreamingTrack, TrackStitcher

    geometry = load_station_geometry(args.mapping)
    positions = geometry.positions_m

    def as_worker_track(track) -> WorkerTrack:
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
        return WorkerTrack(
            track_id=str(track.track_id),
            direction="unknown",
            points=points,
            median_speed_kmh=float(track.median_speed_kmh),
            confidence=float(track.confidence) if track.confidence is not None else None,
            score=float(track.score) if track.score is not None else None,
            observed_count=int(track.observed_count),
            span_m=float(track.span_m),
            max_gap_m=float(track.max_gap_m),
            enters_window=bool(track.enters_window),
            exits_window=bool(track.exits_window),
            ambiguous_crossing=bool(track.ambiguous_crossing),
        )

    def run_category(category: str, direction: int, prefix: str, display_name: str) -> list[StreamingTrack]:
        config = load_tracker_config(args.config)
        config.runtime.device = args.device
        config.model.checkpoint = str(args.checkpoint)
        config.model.motion_direction = int(direction)
        config.association.motion_direction = int(direction)
        config.data.sample_rate_hz = fs
        config.data.duration_s = float(args.window_s)
        config.data.mapping_path = str(args.mapping)
        tracker = HybridVehicleTracker(config)
        stitcher = TrackStitcher(stride_s=float(args.stride_s), window_s=float(args.window_s))
        stitched: dict[str, StreamingTrack] = {}

        for window_start in np.arange(start_s, end_s - float(args.window_s) + 1e-6, float(args.stride_s)):
            local_i0 = int(round(window_start * fs))
            local_i1 = local_i0 + int(round(float(args.window_s) * fs))
            category_window = np.asarray(
                original.gauss_section(
                    picks_all, category, local_i0, local_i1 - local_i0, raw.shape[1], original.DT,
                    original.GAUSS_WIDTH_S, rms_min, rms_max, original.AMP_MIN, original.AMP_MAX,
                ),
                dtype=np.float32,
            )
            raw_part = np.asarray(raw[local_i0:local_i1], dtype=np.float32)
            pre_part = np.asarray(pre[local_i0:local_i1], dtype=np.float32)
            mask = np.clip(category_window / max(float(original.AMP_MAX), 1e-6), 0.0, 1.0)
            if category == "large":
                # Keep the original large-vehicle path unchanged.
                raw_category = raw_part * mask
                non_event_pre = np.max(pre_part, axis=0, keepdims=True)
                pre_category = pre_part * mask + non_event_pre * (1.0 - mask)
            else:
                # Small-vehicle category Gauss is sparse and should not be
                # used to hard-zero the real Raw/Pre waveforms.  Keep the
                # original modal evidence intact; category_window remains the
                # small forward/reverse constraint supplied to the tracker.
                raw_category = raw_part
                pre_category = pre_part
            batch = tracker.predict(raw_category, pre_category, category_window, geometry, start_s=0.0, duration_s=float(args.window_s))
            worker_batch = WorkerBatch(
                tracks=[as_worker_track(track) for track in batch.tracks],
                observations=[],
                start_s=float(window_start),
                duration_s=float(args.window_s),
                diagnostics=batch.diagnostics,
            )
            current_global = stitcher.update(worker_batch)
            for old_id in current_global.removed_track_ids:
                stitched.pop(old_id, None)
            for old_id, canonical_id in current_global.id_aliases.items():
                old_item = stitched.pop(old_id, None)
                if old_item is not None and canonical_id not in stitched:
                    stitched[canonical_id] = old_item
            for item in current_global.tracks:
                stitched[item.global_vehicle_id] = item
            print(f"  {display_name} {window_start:.0f}-{window_start + args.window_s:.0f}s: {len(batch.tracks)} tracks -> {len(stitched)} cumulative IDs")

        final_update = stitcher.finalize()
        for old_id in final_update.removed_track_ids:
            stitched.pop(old_id, None)
        for old_id, canonical_id in final_update.id_aliases.items():
            old_item = stitched.pop(old_id, None)
            if old_item is not None and canonical_id not in stitched:
                stitched[canonical_id] = old_item
        for item in final_update.tracks:
            stitched[item.global_vehicle_id] = item
        result = list(stitched.values())
        for item in result:
            item.global_vehicle_id = f"{prefix}{item.global_vehicle_id}"
        print(f"  {display_name} deduplicated total: {len(result)}")
        return result

    large_tracks = run_category("large", -1, "L", "large")
    # Small vehicles intentionally use pick points only.  This keeps the
    # large-vehicle Hybrid path unchanged while avoiding Raw/Pre leakage into
    # the small-vehicle recognition path.
    forward_tracks = _peak_only_tracks(picks, original, positions, 1, "F")
    reverse_tracks = _peak_only_tracks(picks, original, positions, -1, "R")
    recognized = {
        "large": [(track, 0.0, track.global_vehicle_id) for track in large_tracks],
        "fwd": [(track, 0.0, track.global_vehicle_id) for track in forward_tracks],
        "rev": [(track, 0.0, track.global_vehicle_id) for track in reverse_tracks],
    }

    total_tracks = sum(len(items) for items in recognized.values())
    print(f"[3/4] draw figure, deduplicated recognized tracks={total_tracks}")
    fig = plt.figure(figsize=original.FIGSIZE, facecolor="white")
    # The AGC/Gauss waveform is still used as a faint reference overlay in
    # recognition panels, but its standalone panel is intentionally omitted.
    gsp = gridspec.GridSpec(5, 1, figure=fig, left=0.07, right=0.95, top=0.95, bottom=0.05, hspace=0.38, height_ratios=[1, 1.2, 1.2, 1.2, 1.2])
    axes = [fig.add_subplot(gsp[index]) for index in range(5)]
    for ax in axes:
        original.style_ax(ax)
    xlim = (float(t_win[0]), float(t_win[-1]))
    original.draw_wiggle(axes[0], t_win, raw_win, color="#000000", fill_color="#000000", scale=original.WIGGLE_SCALE, lw=original.WIGGLE_LW)
    axes[0].set_xlim(*xlim); axes[0].set_title("(1) Raw Signal", fontsize=9.5, fontweight="bold", loc="left", color="#222"); axes[0].set_ylabel("Trace #", fontsize=8); axes[0].tick_params(labelbottom=False)

    sections = [(axes[1], gs_large, original.CMAP_LARGE, original.COL_LARGE, "(2) Large Vehicles + Recognition"), (axes[2], gs_fwd, original.CMAP_FWD, original.COL_FWD, "(3) Forward Small Vehicles + Recognition"), (axes[3], gs_rev, original.CMAP_REV, original.COL_REV, "(4) Reverse Small Vehicles + Recognition")]
    for ax, section, cmap, color, title in sections:
        image = original.draw_imshow(ax, t_win, section, cmap, alpha=original.IMSHOW_ALPHA, vmax=_vmax(section))
        original.draw_ref_wiggle(ax, t_win, agc_win, original.REF_WIGGLE_SCALE, original.REF_WIGGLE_ALPHA)
        ax.set_xlim(*xlim); ax.set_title(title, fontsize=9.5, fontweight="bold", loc="left", color=color); ax.set_ylabel("Trace #", fontsize=8); ax.tick_params(labelbottom=False)
        fig.colorbar(image, ax=ax, pad=0.01, fraction=0.012, aspect=25).ax.tick_params(labelsize=6)
    # Match the original deduplicated large-vehicle figure: each category
    # panel cycles through the same highly separated colors.  The combined
    # panel below keeps its fixed red/blue/orange category colors unchanged.
    track_palette = ("#00a878", "#7b2cbf", "#0081a7", "#f77f00", "#c1121f")
    _overlay_tracks(axes[1], recognized["large"], positions, label=True, palette=track_palette)
    _overlay_tracks(axes[2], recognized["fwd"], positions, label=True, palette=track_palette)
    _overlay_tracks(axes[3], recognized["rev"], positions, label=True, palette=track_palette)

    original.draw_imshow(axes[4], t_win, gs_large / (_vmax(gs_large) + 1e-12), original.CMAP_LARGE, alpha=0.86, vmax=1.0)
    original.draw_imshow(axes[4], t_win, gs_fwd / (_vmax(gs_fwd) + 1e-12), original.CMAP_FWD, alpha=0.45, vmax=1.0)
    original.draw_imshow(axes[4], t_win, gs_rev / (_vmax(gs_rev) + 1e-12), original.CMAP_REV, alpha=0.45, vmax=1.0)
    original.draw_ref_wiggle(axes[4], t_win, agc_win, original.REF_WIGGLE_SCALE * 0.8, original.REF_WIGGLE_ALPHA * 0.7)
    axes[4].set_xlim(*xlim); axes[4].set_title("(5) Combined + Hybrid Recognition (red / blue / orange)", fontsize=9.5, fontweight="bold", loc="left", color="#222"); axes[4].set_ylabel("Trace #", fontsize=8); axes[4].set_xlabel("Time (s)")
    _overlay_tracks(axes[4], recognized["large"], positions, label=False, line_color=original.COL_LARGE)
    _overlay_tracks(axes[4], recognized["fwd"], positions, label=False, line_color=original.COL_FWD)
    _overlay_tracks(axes[4], recognized["rev"], positions, label=False, line_color=original.COL_REV)
    axes[4].legend(handles=[plt.Line2D([], [], color=original.COL_LARGE, lw=2, label=f"Large recognized (N={len(recognized['large'])})"), plt.Line2D([], [], color=original.COL_FWD, lw=2, label=f"Forward small recognized (N={len(recognized['fwd'])})"), plt.Line2D([], [], color=original.COL_REV, lw=2, label=f"Reverse small recognized (N={len(recognized['rev'])})")], loc="upper right", fontsize=8, framealpha=0.85)
    fig.suptitle(f"Three-way vehicle recognition overlay | t = {start_s:.0f} - {end_s:.0f} s | deduplicated tracks = {total_tracks}", fontsize=11, fontweight="bold", color="#111", y=0.975)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "output": str(args.output),
                "start_s": start_s,
                "end_s": end_s,
                "device": args.device,
                "track_count": total_tracks,
                "track_count_by_category": {key: len(items) for key, items in recognized.items()},
                "large_pick_count": len([item for item in picks if item["is_large"]]),
                "small_forward_pick_count": len([item for item in picks if not item["is_large"] and item.get("direction") == 1]),
                "small_reverse_pick_count": len([item for item in picks if not item["is_large"] and item.get("direction") == -1]),
                "small_recognition_method": "pick_points_only_tau_p_fitted_lines",
                "large_recognition_method": "hybrid_large_masked_modal",
                "window_s": args.window_s,
                "stride_s": args.stride_s,
                "cross_window_deduplicated": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[4/4] saved: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
