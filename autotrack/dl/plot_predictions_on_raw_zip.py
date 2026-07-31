"""Overlay PeakSlot prediction CSV tracks on raw arrays stored in a zip.

Example:
    uv run python -m autotrack.dl.plot_predictions_on_raw_zip \
        --zip-path datasets/saved_arrays03.zip \
        --array-name signal_raw \
        --prediction-dir predicts/peak_slot_v4_120s_noisy_badch_cuda/prediction_large \
        --out-dir predicts/peak_slot_v4_120s_noisy_badch_cuda/prediction_large/raw_overlay_signal_raw \
        --plot-samples 16
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predicted tracks on raw zipped .npy arrays.")
    parser.add_argument("--zip-path", required=True, type=Path, help="Zip file containing time-channel .npy arrays.")
    parser.add_argument("--array-name", default="signal_raw", help="Array name in the zip, e.g. signal_raw or gauss_section.")
    parser.add_argument("--prediction-dir", required=True, type=Path, help="Directory containing predicted_tracks.csv and summary.json.")
    parser.add_argument("--data-dir", default=None, type=Path, help="Dataset meta.json directory. Defaults to summary.json data_dir.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for overlay PNGs and summary.json.")
    parser.add_argument("--sample-indices", default="", help="Comma-separated sample indices to plot.")
    parser.add_argument("--start-sample", type=int, default=0, help="First sample when --sample-indices is empty.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of samples when --sample-indices is empty.")
    parser.add_argument("--plot-style", choices=["waveform", "heatmap"], default="waveform", help="Background style.")
    parser.add_argument("--plot-downsample", type=int, default=0, help="Raw display downsample; 0 uses meta time_downsample.")
    parser.add_argument(
        "--normalize-traces",
        "--normalize-trace",
        action="store_true",
        help="Normalize each waveform trace independently before plotting.",
    )
    parser.add_argument("--direction-filter", choices=["all", "forward", "reverse"], default="all", help="Track direction filter.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum track objectness score to draw.")
    parser.add_argument("--max-tracks-per-sample", type=int, default=0, help="0 draws all tracks passing filters.")
    parser.add_argument("--dpi", type=int, default=160, help="Output PNG DPI.")
    parser.add_argument("--vmax-quantile", type=float, default=0.995, help="Robust amplitude quantile for display.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_data_dir(prediction_dir: Path, data_dir: Path | None) -> Path:
    if data_dir is not None:
        return data_dir
    summary_path = prediction_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.json not found and --data-dir was not provided: {summary_path}")
    raw = str(_load_json(summary_path).get("data_dir", "")).strip()
    if not raw:
        raise ValueError(f"summary.json does not contain data_dir: {summary_path}")
    path = Path(raw)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _parse_sample_indices(text: str, *, start_sample: int, plot_samples: int, sample_count: int) -> list[int]:
    if str(text).strip():
        selected = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    else:
        start = max(0, int(start_sample))
        selected = list(range(start, start + max(0, int(plot_samples))))
    return sorted({idx for idx in selected if 0 <= int(idx) < int(sample_count)})


def _zip_npy_name(zip_path: Path, array_name: str) -> str:
    wanted = str(array_name).strip().removesuffix(".npy")
    with zipfile.ZipFile(zip_path) as zf:
        names = [name for name in zf.namelist() if name.endswith(".npy")]
    for name in names:
        stem = Path(name).stem
        if stem == wanted or name == str(array_name):
            return name
    available = ", ".join(names)
    raise FileNotFoundError(f"Array {array_name!r} not found in {zip_path}. Available: {available}")


def _read_npy_header(stream: Any) -> tuple[tuple[int, ...], np.dtype, bool]:
    from numpy.lib import format

    version = format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, fortran_order, dtype = format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")
    return tuple(int(item) for item in shape), np.dtype(dtype), bool(fortran_order)


def _discard(stream: Any, n_bytes: int) -> None:
    remaining = int(n_bytes)
    chunk = 8 * 1024 * 1024
    while remaining > 0:
        data = stream.read(min(chunk, remaining))
        if not data:
            raise EOFError("Unexpected EOF while skipping .npy payload.")
        remaining -= len(data)


def _read_exact(stream: Any, n_bytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = int(n_bytes)
    chunk = 32 * 1024 * 1024
    while remaining > 0:
        data = stream.read(min(chunk, remaining))
        if not data:
            raise EOFError("Unexpected EOF while reading .npy payload.")
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def _load_time_channel_range(zip_path: Path, npy_name: str, *, row_start: int, row_end: int) -> tuple[np.ndarray, tuple[int, ...]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(npy_name) as stream:
            shape, dtype, fortran_order = _read_npy_header(stream)
            if fortran_order:
                raise ValueError("Fortran-order .npy arrays are not supported for streaming slice reads.")
            if len(shape) != 2:
                raise ValueError(f"Expected a time-channel array with shape [time, channel], got {shape}")
            n_time, n_channel = shape
            row_start = max(0, min(int(row_start), int(n_time)))
            row_end = max(row_start, min(int(row_end), int(n_time)))
            row_bytes = int(n_channel) * int(dtype.itemsize)
            _discard(stream, row_start * row_bytes)
            payload = _read_exact(stream, (row_end - row_start) * row_bytes)
    arr = np.frombuffer(payload, dtype=dtype).reshape(row_end - row_start, n_channel)
    return arr.copy(), shape


def _prediction_groups(csv_path: Path, selected: set[int], *, direction_filter: str, min_score: float, max_tracks: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {idx: {} for idx in selected}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sample_index = int(row["sample_index"])
            if sample_index not in selected:
                continue
            direction = str(row.get("direction", ""))
            if direction_filter != "all" and direction != direction_filter:
                continue
            score = float(row.get("score", "0") or 0.0)
            if score < float(min_score):
                continue
            track_id = int(row["pred_track_id"])
            track = grouped[sample_index].setdefault(
                track_id,
                {
                    "track_id": track_id,
                    "score": score,
                    "direction": direction,
                    "speed_kmh": float(row.get("speed_kmh", "nan") or "nan"),
                    "points": [],
                },
            )
            track["score"] = max(float(track["score"]), score)
            track["points"].append(
                {
                    "channel": int(row["channel"]),
                    "time_norm": float(row["time_norm"]),
                    "peak_prob": float(row.get("peak_prob", "nan") or "nan"),
                }
            )
    out: dict[int, list[dict[str, Any]]] = {}
    for sample_index, tracks in grouped.items():
        items = sorted(tracks.values(), key=lambda item: float(item["score"]), reverse=True)
        if int(max_tracks) > 0:
            items = items[: int(max_tracks)]
        out[sample_index] = items
    return out


def _robust_scale(arr: np.ndarray, quantile: float) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    centered = finite - float(np.median(finite))
    return max(float(np.quantile(np.abs(centered), float(quantile))), 1e-6)


def _track_xy(
    track: dict[str, Any],
    *,
    sample_start_s: float,
    window_seconds: float,
    channel_start: int,
    dx_m: float,
    waveform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    points = sorted(track["points"], key=lambda item: int(item["channel"]))
    channels = np.asarray([int(item["channel"]) + int(channel_start) for item in points], dtype=np.float64)
    times = np.asarray([sample_start_s + float(item["time_norm"]) * float(window_seconds) for item in points], dtype=np.float64)
    if waveform:
        return channels * float(dx_m) * 1e-3, times
    return times, channels


def _plot_sample(
    out_path: Path,
    *,
    raw_window: np.ndarray,
    sample_index: int,
    sample_start_s: float,
    window_seconds: float,
    fs: float,
    dx_m: float,
    channel_start: int,
    tracks: list[dict[str, Any]],
    plot_style: str,
    plot_downsample: int,
    normalize_traces: bool,
    vmax_quantile: float,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    step = max(1, int(plot_downsample))
    shown = raw_window[::step, :]
    n_time, n_ch = shown.shape
    t_axis = sample_start_s + (np.arange(n_time, dtype=np.float64) * step) / float(fs)
    colors = plt.get_cmap("tab20").colors

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.0, 7.5))
    style = str(plot_style).lower()
    if style == "waveform":
        x_axis = (np.arange(n_ch, dtype=np.float64) + int(channel_start)) * float(dx_m) * 1e-3
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 0.1
        global_scale = _robust_scale(shown, float(vmax_quantile))
        for local_ch in range(n_ch):
            trace = shown[:, local_ch].astype(np.float64)
            trace = trace - float(np.median(trace[np.isfinite(trace)])) if np.isfinite(trace).any() else trace
            scale = _robust_scale(trace, float(vmax_quantile)) if bool(normalize_traces) else global_scale
            ratio = np.clip(trace / scale, -1.35, 1.35)
            ax.plot(x_axis[local_ch] + ratio * 0.27 * spacing, t_axis, color="0.45", linewidth=0.65, alpha=0.85)
        for idx, track in enumerate(tracks):
            x, y = _track_xy(
                track,
                sample_start_s=sample_start_s,
                window_seconds=window_seconds,
                channel_start=channel_start,
                dx_m=dx_m,
                waveform=True,
            )
            if x.size <= 0:
                continue
            color = colors[idx % len(colors)]
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.95)
            ax.scatter(x, y, color=color, s=11, edgecolors="white", linewidths=0.25, zorder=3)
        pad = 0.5 * spacing
        ax.set_xlim(float(x_axis[0] - pad), float(x_axis[-1] + pad))
        ax.set_ylim(float(sample_start_s), float(sample_start_s + window_seconds))
        ax.invert_yaxis()
        ax.set_xlabel("Offset (km)")
        ax.set_ylabel("Original time (s)")
    else:
        centered = shown - np.nanmedian(shown, axis=0, keepdims=True)
        vmax = _robust_scale(centered, float(vmax_quantile))
        ax.imshow(
            centered.T,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            vmin=-vmax,
            vmax=vmax,
            extent=(float(t_axis[0]), float(t_axis[-1]), channel_start - 0.5, channel_start + n_ch - 0.5),
            interpolation="nearest",
        )
        for idx, track in enumerate(tracks):
            x, y = _track_xy(
                track,
                sample_start_s=sample_start_s,
                window_seconds=window_seconds,
                channel_start=channel_start,
                dx_m=dx_m,
                waveform=False,
            )
            if x.size <= 0:
                continue
            color = colors[idx % len(colors)]
            ax.plot(x, y, color=color, linewidth=1.2, alpha=0.95)
            ax.scatter(x, y, color=color, s=11, edgecolors="white", linewidths=0.25, zorder=3)
        ax.set_xlim(float(sample_start_s), float(sample_start_s + window_seconds))
        ax.set_ylim(channel_start - 0.5, channel_start + n_ch - 0.5)
        ax.set_xlabel("Original time (s)")
        ax.set_ylabel("Channel")
    ax.set_title(f"Sample {sample_index:06d} raw data with predicted tracks ({len(tracks)} tracks)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    data_dir = _resolve_data_dir(args.prediction_dir, args.data_dir)
    meta = _load_json(data_dir / "meta.json")
    sample_count = int(meta.get("num_samples", 0))
    selected = _parse_sample_indices(
        args.sample_indices,
        start_sample=int(args.start_sample),
        plot_samples=int(args.plot_samples),
        sample_count=sample_count,
    )
    if not selected:
        raise ValueError("No valid sample indices selected.")

    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    window_seconds = float(meta.get("window_seconds", 120.0))
    window_samples = int(meta.get("window_samples", round(window_seconds * fs)))
    window_starts = [int(item) for item in meta.get("window_start_samples", [])]
    if len(window_starts) < sample_count:
        stride_samples = int(meta.get("stride_samples", window_samples))
        window_starts = [idx * stride_samples for idx in range(sample_count)]
    channel_slice = meta.get("channel_slice", [0, int(meta.get("n_channels", 0))])
    channel_start = int(channel_slice[0])
    channel_end = int(channel_slice[1])
    n_channels = int(meta.get("n_channels", channel_end - channel_start))
    channel_end = min(channel_start + n_channels, channel_end)

    row_start = min(window_starts[idx] for idx in selected)
    row_end = max(window_starts[idx] + window_samples for idx in selected)
    npy_name = _zip_npy_name(args.zip_path, args.array_name)
    raw_range, source_shape = _load_time_channel_range(args.zip_path, npy_name, row_start=row_start, row_end=row_end)
    if raw_range.shape[1] < channel_end:
        raise ValueError(f"Raw array only has {raw_range.shape[1]} channels, but metadata needs channel {channel_end - 1}.")

    tracks_by_sample = _prediction_groups(
        args.prediction_dir / "predicted_tracks.csv",
        set(selected),
        direction_filter=str(args.direction_filter),
        min_score=float(args.min_score),
        max_tracks=int(args.max_tracks_per_sample),
    )
    plot_downsample = int(args.plot_downsample) if int(args.plot_downsample) > 0 else int(meta.get("time_downsample", 10))
    out_paths = []
    for sample_index in selected:
        start = int(window_starts[sample_index])
        local_start = start - row_start
        local_end = local_start + window_samples
        raw_window = raw_range[local_start:local_end, channel_start:channel_end]
        out_path = args.out_dir / "plots" / f"sample_{sample_index:06d}_{Path(npy_name).stem}.png"
        _plot_sample(
            out_path,
            raw_window=raw_window,
            sample_index=int(sample_index),
            sample_start_s=float(start) / fs,
            window_seconds=window_seconds,
            fs=fs,
            dx_m=dx_m,
            channel_start=channel_start,
            tracks=tracks_by_sample.get(sample_index, []),
            plot_style=str(args.plot_style),
            plot_downsample=plot_downsample,
            normalize_traces=bool(args.normalize_traces),
            vmax_quantile=float(args.vmax_quantile),
            dpi=int(args.dpi),
        )
        out_paths.append(out_path)

    summary = {
        "zip_path": args.zip_path,
        "zip_npy_name": npy_name,
        "source_shape": source_shape,
        "prediction_dir": args.prediction_dir,
        "data_dir": data_dir,
        "selected_samples": selected,
        "row_range_read": [row_start, row_end],
        "plot_style": args.plot_style,
        "plot_downsample": plot_downsample,
        "normalize_traces": bool(args.normalize_traces),
        "direction_filter": args.direction_filter,
        "min_score": float(args.min_score),
        "max_tracks_per_sample": int(args.max_tracks_per_sample),
        "plots": out_paths,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(f"Wrote {len(out_paths)} overlay plots to {args.out_dir / 'plots'}")


if __name__ == "__main__":
    main()
