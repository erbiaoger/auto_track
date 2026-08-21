#!/usr/bin/env python3
"""Render a clean raw-waveform / pick-point / final-track figure."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})


ROOT = Path(__file__).resolve().parents[1]
SEPARATION_SCRIPT = ROOT / "shared_data/2025-08-09/车辆三分离_大车_正向小车_反向小车.py"
sys.path[:0] = [
    str(ROOT / "common/src"),
    str(ROOT / "compatibility/autotrack_legacy"),
    str(ROOT / "methods/hybrid_vehicle_tracker/src"),
    str(ROOT / "methods/kalman_seed_tracker/src"),
    str(ROOT / "methods/hybrid_vehicle_tracker/src"),
    str(ROOT / "vehicle_replay_web/backend"),
]

_legacy_root = ROOT / "compatibility/autotrack_legacy"
_legacy_package = types.ModuleType("autotrack")
_legacy_package.__path__ = [str(_legacy_root)]
sys.modules.setdefault("autotrack", _legacy_package)


def load_original():
    spec = importlib.util.spec_from_file_location("vehicle_three_way_overlay", SEPARATION_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {SEPARATION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    source = ROOT / "shared_data/2025-08-09/web_input/DAY02/source"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=source / "raw_DAY02.npy")
    parser.add_argument("--prediction", type=Path, default=source / "prediction_DAY02.npy")
    parser.add_argument("--mapping", type=Path, default=ROOT / "shared_data/2025-08-09/web_input/DAY02/mapping/raw_DAY02.mapping.json")
    parser.add_argument("--config", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/configs/day11_120s_v9.yaml")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "methods/hybrid_vehicle_tracker/checkpoints/active/v9_synthetic_morphology_16384/hybrid_final.pt")
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--window-s", type=float, default=120.0)
    parser.add_argument("--stride-s", type=float, default=2.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/direct_large_final_raw_pick_track_overlay_DAY02_0_600s_stride2.png")
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def _draw_raw(ax, t: np.ndarray, raw: np.ndarray, original, *, color: str, fill_color: str, scale: float, lw: float) -> None:
    original.draw_wiggle(
        ax,
        t,
        raw,
        color=color,
        fill_color=fill_color,
        scale=scale,
        lw=lw,
    )


def _format_axis(ax, *, x_max: float, show_xlabel: bool = False) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylabel("Trace number", fontsize=13, fontweight="bold", labelpad=8, color="#222222")
    ax.grid(axis="x", color="#d9d9d9", lw=0.45, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=11, colors="#222222", direction="out", length=4, width=1.0)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontname("Times New Roman")
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    if show_xlabel:
        ax.set_xlabel("Time (s)", fontsize=13, fontweight="bold", labelpad=8, color="#222222")
    else:
        ax.tick_params(labelbottom=False)


def _scatter_picks(ax, picks: list[dict], *, alpha: float, size: float) -> None:
    groups = (
        ("large", "#e41a1c", "large pick"),
        ("forward", "#377eb8", "forward pick"),
        ("reverse", "#4daf4a", "reverse pick"),
    )
    for category, color, _label in groups:
        if category == "large":
            rows = [p for p in picks if bool(p.get("is_large"))]
        elif category == "forward":
            rows = [p for p in picks if not p.get("is_large") and int(p.get("direction", 0)) == 1]
        else:
            rows = [p for p in picks if not p.get("is_large") and int(p.get("direction", 0)) == -1]
        if not rows:
            continue
        ax.scatter(
            [float(p["t"]) for p in rows],
            [int(p["trace"]) for p in rows],
            s=size,
            color=color,
            alpha=alpha,
            linewidths=0.25,
            edgecolors="white",
            zorder=6,
            rasterized=True,
        )


def _draw_tracks(ax, tracks: list, *, label_limit: int = 0) -> None:
    # Keep the trajectory palette stable across figure revisions.  The
    # scientific layout may change, but a vehicle's display color should not.
    colors = plt.get_cmap("turbo")(np.linspace(0.06, 0.96, max(1, len(tracks))))
    for index, track in enumerate(sorted(tracks, key=lambda item: (item.last_seen_s, item.global_vehicle_id))):
        points = list(track.points)
        if len(points) < 2:
            continue
        points = sorted(points, key=lambda point: float(point.get("time_s", 0.0)))
        times = np.asarray([float(point["time_s"]) for point in points])
        channels = np.asarray([int(point.get("channel_index", -1)) for point in points])
        valid = np.isfinite(times) & np.isfinite(channels)
        times, channels = times[valid], channels[valid]
        if len(times) < 2:
            continue
        line_color = colors[index]
        halo = [patheffects.Stroke(linewidth=2.5, foreground="white", alpha=0.90), patheffects.Normal()]
        ax.plot(times, channels, color=line_color, lw=1.35, alpha=0.90, zorder=10, path_effects=halo)
        ax.scatter(times, channels, s=7, color=line_color, edgecolors="white", linewidths=0.25, zorder=11, rasterized=True)
        if label_limit and index < label_limit:
            speed = float(track.median_speed_kmh)
            label = f"{track.global_vehicle_id}  {speed:.1f} km/h" if np.isfinite(speed) else str(track.global_vehicle_id)
            end = int(np.argmax(times))
            edge_label = float(times[end]) > 560.0
            ax.text(
                float(times[end]) - 2.0 if edge_label else float(times[end]) + 2.0,
                float(channels[end]),
                label,
                fontsize=7.4,
                color=line_color,
                va="center",
                ha="right" if edge_label else "left",
                zorder=12,
                clip_on=True,
                path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white", alpha=0.95)],
            )


def _build_picks_memory_safe(raw, prediction, original) -> list[dict]:
    """Match the separator's picker without stacking full-record RMS arrays."""
    picks: list[dict] = []
    nt, nx = prediction.shape
    print("  计算局部 RMS（逐道低内存模式）...", flush=True)
    for channel in range(nx):
        rms = original.local_rms(raw[:, channel], original.DT, original.ENERGY_WIN_S)
        indices, probabilities = original.pick_peaks_1d(
            prediction[:, channel], original.DT, original.PRED_THRESH, original.MIN_GAP_S
        )
        for index, probability in zip(indices, probabilities):
            picks.append({
                "trace": channel,
                "idx": int(index),
                "t": float(index * original.DT),
                "prob": float(probability),
                "rms": float(rms[index]),
            })
        if channel == nx - 1 or (channel + 1) % 10 == 0:
            print(f"    processed traces: {channel + 1}/{nx}", flush=True)
    if not picks:
        return picks
    rms_values = np.asarray([item["rms"] for item in picks], dtype=np.float64)
    large = original.classify_picks(
        rms_values,
        original.SPLIT_METHOD,
        original.SPLIT_PCT,
        original.SPLIT_THRESH,
    )
    for item, is_large in zip(picks, large):
        item["is_large"] = bool(is_large)
    print(
        f"  共 {len(picks)} 个拾取点  →  大车 {int(np.sum(large))}   小车 {len(picks) - int(np.sum(large))}  (低内存模式)",
        flush=True,
    )
    return picks


def main() -> None:
    args = parse_args()
    original = load_original()
    raw_full = np.load(args.raw, mmap_mode="r")
    prediction = np.load(args.prediction, mmap_mode="r")
    fs = float(original.DT ** -1)
    start_s = float(args.start_s)
    end_s = min(float(raw_full.shape[0] / fs), start_s + float(args.duration_s))
    i0, i1 = int(round(start_s * fs)), int(round(end_s * fs))
    t = np.arange(i0, i1, dtype=np.float32) / fs
    raw = np.asarray(raw_full[i0:i1], dtype=np.float32)

    print("[1/4] build picks")
    picks_all = _build_picks_memory_safe(raw_full, prediction, original)
    picks = [dict(item) for item in picks_all if start_s <= float(item["t"]) <= end_s]
    small = [item for item in picks if not item["is_large"]]
    small, _ = original.split_directions(small, return_debug=True, verbose=False)
    # split_directions returns copied dictionaries; update the corresponding
    # displayed rows by their stable trace/time key.
    direction_lookup = {(int(item["trace"]), round(float(item["t"]), 6)): item for item in small}
    for item in picks:
        if item["is_large"]:
            item.update(direction=0, line_id=-1, on_line=False)
        else:
            assigned = direction_lookup.get((int(item["trace"]), round(float(item["t"]), 6)))
            if assigned is not None:
                item.update({key: assigned.get(key) for key in ("direction", "line_id", "on_line")})

    print(f"  picks={len(picks)}")
    print("[2/4] run final Hybrid stitching")
    # Reuse the maintained offline Hybrid adapter; it returns canonical,
    # smoothed tracks with absolute time coordinates.
    sys.path.insert(0, str(ROOT / "tools"))
    import compare_peak_only_methods_DAY02 as comparison

    from hybrid_vehicle_tracker.data.mapping import load_station_geometry

    geometry = load_station_geometry(args.mapping)
    rms_min, rms_max = original.build_rms_norm_params(picks_all)
    namespace = argparse.Namespace(
        device=args.device,
        config=args.config,
        checkpoint=args.checkpoint,
        mapping=args.mapping,
        start_s=start_s,
        duration_s=end_s - start_s,
        window_s=float(args.window_s),
        stride_s=float(args.stride_s),
    )
    import json
    cache_path = args.output.with_suffix(".tracks.json")
    if cache_path.exists():
        from types import SimpleNamespace
        tracks = [SimpleNamespace(**row) for row in json.loads(cache_path.read_text(encoding="utf-8"))]
        print(f"  loaded cached canonical tracks: {len(tracks)}")
    else:
        tracks = comparison.run_hybrid_large(
            original,
            raw_full,
            np.load(ROOT / "shared_data/2025-08-09/web_input/DAY02/source/pre_DAY02.npy", mmap_mode="r"),
            picks_all,
            rms_min,
            rms_max,
            geometry,
            fs,
            namespace,
        )
        cache_path.write_text(json.dumps([track.to_dict() for track in tracks], ensure_ascii=False), encoding="utf-8")
    print(f"  final canonical tracks={len(tracks)}")

    print("[3/4] draw polished figure")
    fig = plt.figure(figsize=(18, 17), facecolor="#ffffff")
    grid = fig.add_gridspec(3, 1, height_ratios=(1.0, 1.0, 1.12), hspace=0.18, left=0.075, right=0.985, top=0.98, bottom=0.075)
    axes = [fig.add_subplot(grid[i]) for i in range(3)]

    # A: original signal only.
    _draw_raw(ax=axes[0], t=t, raw=raw, original=original, color="#263238", fill_color="#90a4ae", scale=original.WIGGLE_SCALE, lw=0.48)
    _format_axis(axes[0], x_max=end_s, show_xlabel=False)
    axes[0].set_ylim(-0.6, raw.shape[1] - 0.4)
    axes[0].set_title("(a) Original signal", loc="left", fontsize=15, fontweight="bold", color="#222222", pad=10)

    # B: classification evidence.  Keep this panel deliberately clean so
    # the three pick classes can be inspected without track overlays.
    _format_axis(axes[1], x_max=end_s, show_xlabel=False)
    axes[1].set_ylim(-0.6, raw.shape[1] - 0.4)
    axes[1].set_facecolor("#fbfcfd")
    axes[1].set_title("(b) Classified pick points", loc="left", fontsize=15, fontweight="bold", color="#222222", pad=10)
    _scatter_picks(axes[1], picks, alpha=0.78, size=12)

    # C: final canonical tracks over a very faint signal reference.
    _draw_raw(ax=axes[2], t=t, raw=raw, original=original, color="#c2cdd2", fill_color="#f1f4f5", scale=original.WIGGLE_SCALE * 0.78, lw=0.35)
    _format_axis(axes[2], x_max=end_s, show_xlabel=True)
    axes[2].set_ylim(-0.6, raw.shape[1] - 0.4)
    axes[2].set_title(f"(c) Final stitched trajectories  (N = {len(tracks)})", loc="left", fontsize=15, fontweight="bold", color="#222222", pad=10)
    _scatter_picks(axes[2], picks, alpha=0.10, size=5)
    _draw_tracks(axes[2], tracks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    summary = {
        "output": str(args.output),
        "start_s": start_s,
        "end_s": end_s,
        "stride_s": args.stride_s,
        "window_s": args.window_s,
        "displayed_pick_count": len(picks),
        "final_canonical_track_count": len(tracks),
        "layout": "raw waveform / picks / final stitched tracks",
    }
    summary_path = args.summary or args.output.with_suffix(".json")
    summary_path.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4/4] saved: {args.output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
