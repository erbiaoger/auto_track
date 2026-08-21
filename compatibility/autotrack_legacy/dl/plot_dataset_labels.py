"""Plot tensor-shard data with ground-truth labels for manual inspection.

Purpose:
    This CLI reads generated TrackSlotNet or converted PeakSlotNet tensor
    shards and writes diagnostic figures that overlay ground-truth vehicle
    labels on the stored heatmap. Use it to verify whether labels are correct
    before training or prediction.

    For `track_slot_shards_v1`, GT points are drawn from:
        time [sample, gt, channel] and visibility [sample, gt, channel]

    For `peak_slot_shards_v1`, GT points are drawn from:
        gt_peak_index [sample, gt, channel] -> peak_time [sample, channel, K]
        and visibility [sample, gt, channel]

    This distinction is important when checking conversion quality. If the
    TrackSlotNet plot is correct but the PeakSlotNet plot is wrong, the problem
    is in peak-candidate conversion. If both are wrong, the source generated
    labels are wrong.

Example:
    uv run python -m autotrack.dl.plot_dataset_labels \
        --data-dir datasets/track_slot/train \
        --out-dir /tmp/track_slot_label_check \
        --sample-indices 0,6,12 \
        --plot-peaks

    uv run python -m autotrack.dl.plot_dataset_labels \
        --data-dir datasets/peak_slot/train \
        --out-dir /tmp/peak_slot_label_check \
        --sample-indices 6 \
        --plot-peaks

Arguments:
    --data-dir must contain `meta.json` and the `shard_*.pt` files listed in it.
    --out-dir receives PNG plots and `label_points.csv`.
    --sample-indices selects global sample indices, comma-separated.
    --start-sample and --plot-samples are used when --sample-indices is empty.
    --plot-peaks draws all valid PeakSlotNet peak candidates when present.
    --max-gt-tracks can limit the number of GT tracks drawn per sample.

Outputs:
    <out-dir>/plots/sample_000000.png, ...
        Heatmap plus GT label polylines. PeakSlot plots can also show candidate
        peaks as small blue points.
    <out-dir>/label_points.csv
        One row per plotted GT point with sample index, GT id, channel, time,
        direction, peak index, and label source.
    <out-dir>/summary.json
        Dataset format, selected samples, and output paths.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from autotrack.dl.trajectory_set_model import LABEL_TO_DIRECTION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot tensor shard heatmaps with ground-truth labels.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing meta.json and shard_*.pt.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for plots, CSV, and summary.json.")
    parser.add_argument("--sample-indices", default="", help="Comma-separated global sample indices to plot, e.g. 0,6,12.")
    parser.add_argument("--start-sample", type=int, default=0, help="First global sample index when --sample-indices is empty.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of consecutive samples to plot when --sample-indices is empty.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for output PNG figures.")
    parser.add_argument("--plot-peaks", action="store_true", help="Draw valid peak candidates when peak_time/peak_valid exist.")
    parser.add_argument("--max-gt-tracks", type=int, default=0, help="Maximum GT tracks drawn per sample; 0 draws all valid tracks.")
    parser.add_argument("--point-size", type=float, default=10.0, help="Scatter marker size for GT points.")
    parser.add_argument("--line-width", type=float, default=1.15, help="GT line width.")
    parser.add_argument("--line-alpha", type=float, default=0.75, help="GT line opacity.")
    parser.add_argument("--vmax-quantile", type=float, default=0.995, help="Absolute heatmap quantile used for grayscale clipping.")
    parser.add_argument("--plot-style", choices=["heatmap", "waveform"], default="heatmap", help="Background style: heatmap image or per-channel waveform traces.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _parse_sample_indices(text: str, *, start_sample: int, plot_samples: int) -> list[int]:
    raw = str(text).strip()
    if raw:
        out = []
        for item in raw.split(","):
            item = item.strip()
            if item:
                out.append(int(item))
        return sorted(set(out))
    start = int(max(0, start_sample))
    count = int(max(0, plot_samples))
    return list(range(start, start + count))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _iter_selected_samples(
    data_dir: Path,
    shards: Iterable[str],
    selected: set[int],
) -> Iterable[tuple[int, str, int, dict[str, torch.Tensor]]]:
    global_start = 0
    for shard_name in shards:
        shard_path = data_dir / shard_name
        payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        wanted = [idx for idx in sorted(selected) if global_start <= idx < global_start + n]
        for global_idx in wanted:
            local_idx = int(global_idx - global_start)
            sample = {
                key: value[local_idx] if torch.is_tensor(value) and int(value.shape[0]) == n else value
                for key, value in payload.items()
            }
            yield int(global_idx), str(shard_name), int(local_idx), sample
        global_start += n


def _heatmap_from_sample(sample: dict[str, torch.Tensor]) -> np.ndarray:
    x = sample["x"].detach().cpu().to(torch.float32)
    if x.ndim == 3:
        arr = x[0]
    elif x.ndim == 2:
        arr = x
    else:
        raise ValueError(f"Expected sample x with shape [in_channels,C,T] or [C,T], got {tuple(x.shape)}")
    return arr.numpy()


def _valid_gt_indices(sample: dict[str, torch.Tensor], max_gt_tracks: int) -> list[int]:
    valid = torch.where(sample["gt_valid"].to(torch.bool))[0].tolist()
    if int(max_gt_tracks) > 0:
        valid = valid[: int(max_gt_tracks)]
    return [int(item) for item in valid]


def _gt_points_from_track_slot(sample: dict[str, torch.Tensor], gt_idx: int, window_seconds: float) -> list[dict[str, Any]]:
    points = []
    visibility = sample["visibility"][gt_idx].to(torch.float32)
    times = sample["time"][gt_idx].to(torch.float32)
    for ch in torch.where(visibility > 0.5)[0].tolist():
        t_norm = float(times[int(ch)].item())
        points.append(
            {
                "channel": int(ch),
                "time_norm": t_norm,
                "time_s": t_norm * float(window_seconds),
                "peak_index": "",
                "label_source": "time",
            }
        )
    return points


def _gt_points_from_peak_slot(sample: dict[str, torch.Tensor], gt_idx: int, window_seconds: float) -> list[dict[str, Any]]:
    points = []
    visibility = sample["visibility"][gt_idx].to(torch.float32)
    gt_peak_index = sample["gt_peak_index"][gt_idx].to(torch.long)
    peak_time = sample["peak_time"].to(torch.float32)
    none_idx = int(peak_time.shape[-1])
    for ch in torch.where(visibility > 0.5)[0].tolist():
        peak_idx = int(gt_peak_index[int(ch)].item())
        if peak_idx >= none_idx:
            continue
        t_norm = float(peak_time[int(ch), peak_idx].item())
        points.append(
            {
                "channel": int(ch),
                "time_norm": t_norm,
                "time_s": t_norm * float(window_seconds),
                "peak_index": int(peak_idx),
                "label_source": "gt_peak_index",
            }
        )
    return points


def _gt_points(sample: dict[str, torch.Tensor], gt_idx: int, window_seconds: float) -> list[dict[str, Any]]:
    if "gt_peak_index" in sample and "peak_time" in sample:
        return _gt_points_from_peak_slot(sample, gt_idx, window_seconds)
    if "time" in sample:
        return _gt_points_from_track_slot(sample, gt_idx, window_seconds)
    raise ValueError("Sample does not contain TrackSlot time labels or PeakSlot gt_peak_index labels.")


def _direction(sample: dict[str, torch.Tensor], gt_idx: int) -> tuple[int, str]:
    label = int(sample["direction"][gt_idx].item()) if "direction" in sample else -1
    return label, LABEL_TO_DIRECTION.get(label, str(label))


def _monotonic_violations(points: list[dict[str, Any]], direction_label: int) -> int:
    if len(points) < 2:
        return 0
    ordered = sorted(points, key=lambda item: int(item["channel"]))
    times = np.array([float(item["time_norm"]) for item in ordered], dtype=np.float64)
    dt = np.diff(times)
    if int(direction_label) == 1:
        return int(np.sum(dt > 1e-5))
    return int(np.sum(dt < -1e-5))


def _write_label_rows(
    writer: csv.DictWriter,
    *,
    sample_index: int,
    shard_name: str,
    local_index: int,
    sample: dict[str, torch.Tensor],
    window_seconds: float,
    max_gt_tracks: int,
) -> tuple[int, int]:
    track_count = 0
    point_count = 0
    for plot_gt_id, gt_idx in enumerate(_valid_gt_indices(sample, int(max_gt_tracks))):
        direction_label, direction = _direction(sample, gt_idx)
        points = _gt_points(sample, gt_idx, window_seconds)
        violations = _monotonic_violations(points, direction_label)
        if not points:
            continue
        track_count += 1
        for point in points:
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "shard": str(shard_name),
                    "local_index": int(local_index),
                    "gt_track_id": int(plot_gt_id),
                    "source_gt_index": int(gt_idx),
                    "direction": str(direction),
                    "direction_label": int(direction_label),
                    "channel": int(point["channel"]),
                    "time_norm": f"{float(point['time_norm']):.8f}",
                    "time_s": f"{float(point['time_s']):.6f}",
                    "peak_index": point["peak_index"],
                    "label_source": str(point["label_source"]),
                    "track_monotonic_violations": int(violations),
                }
            )
            point_count += 1
    return track_count, point_count


def _plot_sample(
    out_path: Path,
    *,
    sample_index: int,
    sample: dict[str, torch.Tensor],
    window_seconds: float,
    dpi: int,
    plot_peaks: bool,
    max_gt_tracks: int,
    point_size: float,
    line_width: float,
    line_alpha: float,
    vmax_quantile: float,
    plot_style: str,
) -> tuple[int, int]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    arr = _heatmap_from_sample(sample)
    n_ch = int(arr.shape[0])
    finite = arr[np.isfinite(arr)]
    q = float(min(1.0, max(0.0, vmax_quantile)))
    vmax = max(float(np.quantile(np.abs(finite), q)), 1e-6) if finite.size else 1.0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    im = None
    style = str(plot_style).lower()
    x_axis = np.arange(n_ch, dtype=np.float64)
    x_label = "Channel index"
    if style == "waveform":
        dx_m = float(sample.get("_dx_m", 0.0))
        if dx_m > 0.0:
            x_axis = np.arange(n_ch, dtype=np.float64) * dx_m * 1e-3
            x_label = "Offset [km]"
        t_axis = np.linspace(0.0, float(window_seconds), int(arr.shape[1]), dtype=np.float64)
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        wiggle_amp = 0.27 * spacing
        scale = max(vmax, 1e-6)
        for ch in range(n_ch):
            ratio = np.clip(np.asarray(arr[ch], dtype=np.float64) / scale, -1.35, 1.35)
            ax.plot(x_axis[ch] + ratio * wiggle_amp, t_axis, color="0.45", linewidth=0.8, alpha=0.9)
    else:
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            vmin=-vmax,
            vmax=vmax,
            extent=(0.0, float(window_seconds), -0.5, float(n_ch) - 0.5),
            interpolation="nearest",
        )

    if bool(plot_peaks) and "peak_time" in sample and "peak_valid" in sample:
        peak_time = sample["peak_time"].to(torch.float32)
        peak_valid = sample["peak_valid"].to(torch.bool)
        peak_chs = []
        peak_ts = []
        for ch in range(int(peak_valid.shape[0])):
            valid = torch.where(peak_valid[ch])[0].tolist()
            peak_chs.extend([int(ch)] * len(valid))
            peak_ts.extend([float(peak_time[ch, int(idx)].item()) * float(window_seconds) for idx in valid])
        if peak_chs:
            if style == "waveform":
                peak_xs = [float(x_axis[int(ch)]) for ch in peak_chs]
                peak_ys = peak_ts
            else:
                peak_xs = peak_ts
                peak_ys = peak_chs
            ax.scatter(peak_xs, peak_ys, s=5, color="#1f77b4", alpha=0.35, linewidths=0, label="Peak candidates")

    cmap = plt.get_cmap("tab20")
    plotted_tracks = 0
    plotted_points = 0
    first_label = True
    for plot_gt_id, gt_idx in enumerate(_valid_gt_indices(sample, int(max_gt_tracks))):
        points = sorted(_gt_points(sample, gt_idx, window_seconds), key=lambda item: int(item["channel"]))
        if len(points) < 1:
            continue
        times = [float(point["time_s"]) for point in points]
        chs = [int(point["channel"]) for point in points]
        if style == "waveform":
            xs = [float(x_axis[int(ch)]) for ch in chs]
            ys = times
        else:
            xs = times
            ys = chs
        color = cmap(plot_gt_id % 20)
        if len(points) >= 2:
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=float(line_width),
                alpha=float(line_alpha),
                label="GT labels" if first_label else None,
            )
        ax.scatter(xs, ys, s=float(point_size), color=color, edgecolors="black", linewidths=0.2, alpha=0.95)
        first_label = False
        plotted_tracks += 1
        plotted_points += len(points)

    fmt = "peak_slot" if "gt_peak_index" in sample else "track_slot"
    ax.set_title(f"Dataset label overlay, sample {sample_index}  format={fmt}  GT={plotted_tracks}  points={plotted_points}")
    if style == "waveform":
        pad = 0.4 * (float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        ax.set_xlim(float(x_axis[0] - pad), float(x_axis[-1] + pad))
        ax.set_ylim(0.0, float(window_seconds))
        ax.invert_yaxis()
    else:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel index")
        ax.set_xlim(0.0, float(window_seconds))
        ax.set_ylim(-0.5, float(n_ch) - 0.5)
    if plotted_tracks > 0 or (plot_peaks and "peak_time" in sample):
        ax.legend(loc="upper right", frameon=True)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized heatmap amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return plotted_tracks, plotted_points


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {data_dir / 'meta.json'}")

    selected = _parse_sample_indices(
        str(args.sample_indices),
        start_sample=int(args.start_sample),
        plot_samples=int(args.plot_samples),
    )
    if not selected:
        raise ValueError("No samples selected. Use --sample-indices or --plot-samples > 0.")
    selected_set = set(int(item) for item in selected)
    window_seconds = float(meta.get("window_seconds", 1.0))

    csv_path = out_dir / "label_points.csv"
    fields = [
        "sample_index",
        "shard",
        "local_index",
        "gt_track_id",
        "source_gt_index",
        "direction",
        "direction_label",
        "channel",
        "time_norm",
        "time_s",
        "peak_index",
        "label_source",
        "track_monotonic_violations",
    ]
    found_samples = []
    total_tracks = 0
    total_points = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for sample_index, shard_name, local_index, sample in _iter_selected_samples(data_dir, shards, selected_set):
            sample["_dx_m"] = float(meta.get("dx_m", 0.0))
            track_count, point_count = _plot_sample(
                plots_dir / f"sample_{int(sample_index):06d}.png",
                sample_index=int(sample_index),
                sample=sample,
                window_seconds=window_seconds,
                dpi=int(args.plot_dpi),
                plot_peaks=bool(args.plot_peaks),
                max_gt_tracks=int(args.max_gt_tracks),
                point_size=float(args.point_size),
                line_width=float(args.line_width),
                line_alpha=float(args.line_alpha),
                vmax_quantile=float(args.vmax_quantile),
                plot_style=str(args.plot_style),
            )
            _write_label_rows(
                writer,
                sample_index=int(sample_index),
                shard_name=str(shard_name),
                local_index=int(local_index),
                sample=sample,
                window_seconds=window_seconds,
                max_gt_tracks=int(args.max_gt_tracks),
            )
            found_samples.append(int(sample_index))
            total_tracks += int(track_count)
            total_points += int(point_count)

    missing = sorted(selected_set - set(found_samples))
    summary = {
        "mode": "dataset_label_plot",
        "data_dir": str(data_dir),
        "format": str(meta.get("format", "unknown")),
        "selected_samples": selected,
        "plotted_samples": found_samples,
        "missing_samples": missing,
        "window_seconds": window_seconds,
        "total_plotted_tracks": int(total_tracks),
        "total_plotted_points": int(total_points),
        "outputs": {
            "plots_dir": str(plots_dir),
            "label_points_csv": str(csv_path),
            "summary_json": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: plotted_samples={len(found_samples)}, tracks={total_tracks}, points={total_points}, "
        f"missing={len(missing)}, out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
