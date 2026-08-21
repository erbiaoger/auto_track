"""Build a single-vehicle benchmark from `manual_labels.json` and a source `.npy`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.labeling.track_label_project import TrackLabelProject


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-vehicle benchmark from labeled real data.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output .pt benchmark file.")
    parser.add_argument("--source-npy", required=True, type=Path, help="Real DAS .npy file.")
    parser.add_argument("--labels-json", required=True, type=Path, help="manual_labels.json or compatible project JSON.")
    parser.add_argument("--array-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Input array layout.")
    parser.add_argument("--channel-start", type=int, default=0, help="First channel index to keep.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of channels to keep.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Benchmark window length in seconds.")
    parser.add_argument("--margin-seconds", type=float, default=5.0, help="Extra margin around each labeled track.")
    parser.add_argument("--windows-per-track", type=int, default=3, help="Number of windows generated per labeled track.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum benchmark windows; 0 means all tracks.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to extracted raw window.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Fallback sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by labels.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    return parser.parse_args(argv)


def _load_window(
    arr: np.ndarray,
    *,
    layout: str,
    channel_start: int,
    channel_count: int,
    start_t: int,
    window_samples: int,
    background_scale: float,
) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D .npy array, got shape {arr.shape}")
    end_ch = int(channel_start + channel_count)
    if layout == "time_channel":
        if start_t < 0 or start_t + window_samples > int(arr.shape[0]):
            raise ValueError("Window exceeds array bounds")
        if channel_start < 0 or end_ch > int(arr.shape[1]):
            raise ValueError("Channel slice exceeds array bounds")
        window = np.array(arr[start_t : start_t + window_samples, channel_start:end_ch], dtype=np.float32, copy=True).T
    else:
        if start_t < 0 or start_t + window_samples > int(arr.shape[1]):
            raise ValueError("Window exceeds array bounds")
        if channel_start < 0 or end_ch > int(arr.shape[0]):
            raise ValueError("Channel slice exceeds array bounds")
        window = np.array(arr[channel_start:end_ch, start_t : start_t + window_samples], dtype=np.float32, copy=True)
    window = np.nan_to_num(window, copy=False)
    if float(background_scale) != 1.0:
        window *= float(background_scale)
    return window


def _prepare_input(window: np.ndarray, time_downsample: int, clip_ratio: float) -> torch.Tensor:
    arr = np.asarray(window, dtype=np.float32)
    arr_ds = arr[:, :: int(max(1, time_downsample))]
    finite = np.abs(arr_ds[np.isfinite(arr_ds)])
    if finite.size == 0:
        scale = 1.0
    else:
        q995 = float(np.quantile(finite, 0.995))
        rms = float(np.sqrt(np.mean(finite * finite)))
        scale = max(q995, 3.0 * rms, 1e-6)
    clip = max(1e-6, float(clip_ratio))
    x = np.clip(arr_ds / scale, -clip, clip) / clip
    return torch.from_numpy(x[None, :, :].astype(np.float32, copy=False))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project = TrackLabelProject.from_json(args.labels_json)
    arr = np.load(str(args.source_npy), mmap_mode="r")
    fs = float(project.fs_hz if np.isfinite(project.fs_hz) and project.fs_hz > 0 else args.fs)
    dx_m = float(project.dx_m if np.isfinite(project.dx_m) and project.dx_m > 0 else args.dx_m)
    window_samples = int(round(float(args.window_seconds) * fs))
    margin_samples = int(round(float(args.margin_seconds) * fs))

    samples: list[dict[str, Any]] = []
    for track in project.tracks:
        if not track.points:
            continue
        t_min = min(int(p.t_idx) for p in track.points)
        t_max = max(int(p.t_idx) for p in track.points)
        centers = np.linspace(float(t_min), float(t_max), num=max(1, int(args.windows_per_track))).tolist()
        candidate_starts: list[int] = []
        max_start = max(0, (arr.shape[0] if str(args.array_layout) == "time_channel" else arr.shape[1]) - window_samples)
        for center in centers:
            start_t = int(round(float(center) - 0.5 * float(window_samples)))
            start_t = max(0, min(max_start, start_t))
            if candidate_starts and abs(start_t - candidate_starts[-1]) < max(1, window_samples // 8):
                continue
            candidate_starts.append(start_t)
        for start_t in candidate_starts:
            if any(int(p.t_idx) < start_t - margin_samples or int(p.t_idx) >= start_t + window_samples + margin_samples for p in track.points):
                continue
            window = _load_window(
                arr,
                layout=str(args.array_layout),
                channel_start=int(args.channel_start),
                channel_count=int(args.channel_count),
                start_t=int(start_t),
                window_samples=int(window_samples),
                background_scale=float(args.background_scale),
            )
            x = _prepare_input(window, int(args.time_downsample), float(args.clip_ratio))
            time = np.zeros((int(args.channel_count),), dtype=np.float32)
            visibility = np.zeros((int(args.channel_count),), dtype=np.float32)
            raw_window = window.astype(np.float32, copy=False)
            for point in track.points:
                ch = int(point.ch_idx) - int(args.channel_start)
                if 0 <= ch < int(args.channel_count):
                    rel_t = int(point.t_idx) - int(start_t)
                    if 0 <= rel_t < int(window_samples):
                        visibility[ch] = 1.0
                        time[ch] = float(rel_t) / float(max(1, window_samples - 1))
            samples.append(
                {
                    "x": x.cpu(),
                    "target": {
                        "time": torch.from_numpy(time[None, :]),
                        "visibility": torch.from_numpy(visibility[None, :]),
                        "direction": torch.tensor([0 if track.direction == "forward" else 1], dtype=torch.long),
                        "speed": torch.tensor([float(track.mean_speed_kmh) / max(1e-6, float(args.speed_norm_kmh))], dtype=torch.float32),
                        "raw_window": torch.from_numpy(raw_window),
                    },
                    "meta": {
                        "track_id": int(track.track_id),
                        "start_t": int(start_t),
                        "window_samples": int(window_samples),
                    },
                }
            )
            if int(args.max_samples) > 0 and len(samples) >= int(args.max_samples):
                break
        if int(args.max_samples) > 0 and len(samples) >= int(args.max_samples):
            break

    payload = {
        "format": "single_vehicle_benchmark_v1",
        "meta": _json_ready(
            {
                "source_npy": args.source_npy,
                "labels_json": args.labels_json,
                "array_layout": args.array_layout,
                "channel_start": int(args.channel_start),
                "channel_count": int(args.channel_count),
                "window_seconds": float(args.window_seconds),
                "margin_seconds": float(args.margin_seconds),
                "fs": float(fs),
                "dx_m": float(dx_m),
                "time_downsample": int(args.time_downsample),
                "speed_norm_kmh": float(args.speed_norm_kmh),
                "clip_ratio": float(args.clip_ratio),
            }
        ),
        "length": int(len(samples)),
        "samples": samples,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out_file))
    args.out_file.with_suffix(".json").write_text(json.dumps(_json_ready(payload["meta"]), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
