from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from dataclasses import fields

import matplotlib
import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.simple_vehicle_peak_dataset import SimpleLinearVehiclePeakDataset, SimplePeakSetDatasetConfig
from autotrack.dl.vehicle_peak_set_transformer import (
    DecodedPeakTrack,
    PeakSetInferenceConfig,
    decode_peak_guided_vehicle_tracks,
    decode_vehicle_peak_tracks,
    json_ready,
    load_checkpoint_model,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render overlays for the vehicle peak set transformer.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional exported shard dataset directory")
    parser.add_argument("--device", default="auto", help="Torch device")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of consecutive samples to render")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Synthetic window length")
    parser.add_argument("--predict-seconds", type=float, default=None, help="Requested prediction/plot duration in seconds")
    parser.add_argument("--allow-window-tiling", action="store_true", help="Allow shorter-window checkpoints to tile over longer inputs")
    parser.add_argument("--no-refine", action="store_true", help="Skip the refined overlay and show only raw model decode")
    parser.add_argument("--plot-style", default="waveform", choices=["waveform", "heatmap"], help="Overlay base style")
    parser.add_argument("--save-sheet", action="store_true", help="Also save one combined sheet containing all plotted samples.")
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def _parse_int_csv(value: str | None) -> list[int]:
    if value is None:
        return []
    items: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        try:
            items.append(int(token))
        except ValueError:
            continue
    return items


def _detect_dead_channels(raw: np.ndarray, configured: Any = None, *, energy_threshold: float = 1e-8) -> list[int]:
    if configured is not None:
        if isinstance(configured, str):
            return [idx for idx in _parse_int_csv(configured) if idx >= 0]
        if isinstance(configured, (list, tuple, np.ndarray)):
            out: list[int] = []
            for item in configured:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            return out
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    energy = np.mean(np.abs(arr), axis=1)
    dead = np.where(energy <= float(energy_threshold))[0].tolist()
    return [int(idx) for idx in dead]


def _draw_background(
    ax: Any,
    raw: np.ndarray,
    *,
    fs: float,
    x_axis_m: np.ndarray,
    plot_style: str,
    dead_channels: list[int] | None = None,
) -> tuple[np.ndarray, float, float]:
    arr = np.asarray(raw, dtype=np.float32)
    n_ch, n_t = int(arr.shape[0]), int(arr.shape[1])
    window_seconds = float(n_t) / float(max(1e-9, fs))
    style = str(plot_style).lower()
    if style == "waveform":
        if len(x_axis_m) == n_ch and np.ptp(x_axis_m) > 0.0:
            x_axis = np.asarray(x_axis_m, dtype=np.float64) * 1e-3
            x_label = "Offset [km]"
        else:
            x_axis = np.arange(n_ch, dtype=np.float64)
            x_label = "Channel index"
        finite = arr[np.isfinite(arr)]
        vmax = max(float(np.quantile(np.abs(finite), 0.995)), 1e-6) if finite.size else 1.0
        t_axis = np.linspace(0.0, window_seconds, n_t, dtype=np.float64)
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        wiggle_amp = 0.27 * spacing
        clip_ratio = 1.35
        dead = sorted({int(idx) for idx in (dead_channels or []) if 0 <= int(idx) < n_ch})
        if dead:
            half_band = 0.49 * spacing
            for ch in dead:
                ax.axvspan(
                    float(x_axis[ch]) - half_band,
                    float(x_axis[ch]) + half_band,
                    facecolor="#d9d9d9",
                    edgecolor="#a0a0a0",
                    linewidth=0.4,
                    alpha=0.25,
                    zorder=0,
                )
            ax.text(
                0.01,
                0.02,
                f"dead channels: {len(dead)}",
                transform=ax.transAxes,
                color="#666666",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.4},
                zorder=7,
            )
        for ch in range(n_ch):
            ratio = np.clip(arr[ch].astype(np.float64) / max(vmax, 1e-12), -clip_ratio, clip_ratio)
            ax.plot(x_axis[ch] + ratio * wiggle_amp, t_axis, color="0.45", linewidth=0.8, alpha=0.9)
        pad = 0.4 * spacing
        x_min = float(x_axis[0] - pad)
        x_max = float(x_axis[-1] + pad)
        x_span = max(1e-6, x_max - x_min)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, window_seconds)
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        return x_axis, x_max, x_span

    ax.imshow(
        arr.T,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(-0.5, float(n_ch) - 0.5, 0.0, window_seconds),
    )
    ax.set_xlabel("channel")
    ax.set_ylabel("time [s]")
    ax.invert_yaxis()
    return np.arange(n_ch, dtype=np.float64), float(n_ch) - 0.5, max(1e-6, float(n_ch))


def _draw_tracks(
    ax: Any,
    decoded: list[Any],
    *,
    x_axis_m: np.ndarray,
    plot_style: str,
    show_labels: bool = False,
) -> None:
    waveform = str(plot_style).lower() == "waveform"
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    x_axis_km = np.asarray(x_axis_m, dtype=np.float64) * 1e-3 if len(x_axis_m) else np.empty((0,), dtype=np.float64)
    for idx, item in enumerate(decoded):
        tr = item.track
        pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
        observed_mask = np.asarray(item.observed_valid, dtype=bool)
        if waveform:
            xs = [
                float(x_axis_km[int(p.ch_idx)]) if int(p.ch_idx) < len(x_axis_km) else float(p.offset_m) * 1e-3
                for p in pts
            ]
        else:
            xs = [int(p.ch_idx) for p in pts]
        ys = [float(p.time_s) for p in pts]
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, color=color, linewidth=2.0, alpha=0.9)
        obs_x = [x for x, keep in zip(xs, observed_mask) if keep]
        obs_y = [y for y, keep in zip(ys, observed_mask) if keep]
        miss_x = [x for x, keep in zip(xs, observed_mask) if not keep]
        miss_y = [y for y, keep in zip(ys, observed_mask) if not keep]
        if obs_x:
            ax.scatter(
                obs_x,
                obs_y,
                s=10,
                marker="s",
                color=color,
                linewidths=1.0,
                zorder=4,
            )
        if miss_x:
            ax.scatter(
                miss_x,
                miss_y,
                s=22,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                zorder=5,
            )
        if show_labels and xs:
            ax.text(xs[0], ys[0], f"{idx}", color=color, fontsize=7, ha="left", va="bottom")
        if waveform and xs and ys and math.isfinite(float(tr.mean_speed_kmh)):
            mid = len(xs) // 2
            x_span = max(1e-6, float(ax.get_xlim()[1] - ax.get_xlim()[0]))
            x_text = min(float(ax.get_xlim()[1]) - 0.02 * x_span, float(xs[mid]) + 0.01 * x_span)
            ax.text(
                x_text,
                float(ys[mid]),
                f"{float(tr.mean_speed_kmh):.1f} km/h",
                color=color,
                fontsize=8,
                ha="left",
                va="center",
                alpha=0.95,
                bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
                zorder=6,
            )


def _draw_ground_truth(
    ax: Any,
    target: dict[str, torch.Tensor],
    *,
    x_axis_m: np.ndarray,
    plot_style: str,
    label: str = "GT",
) -> int:
    if "full_time" not in target or "full_valid" not in target:
        return 0
    full_time = target["full_time"].detach().cpu().numpy()
    full_valid = target["full_valid"].detach().cpu().numpy() > 0.5
    waveform = str(plot_style).lower() == "waveform"
    x_axis_km = np.asarray(x_axis_m, dtype=np.float64) * 1e-3 if len(x_axis_m) else np.empty((0,), dtype=np.float64)
    colors = ["black", "#444444", "#666666", "#888888"] if waveform else ["#dddddd", "#bbbbbb", "#999999", "#777777"]
    count = 0
    for idx in range(int(full_time.shape[0])):
        mask = full_valid[idx]
        if int(mask.sum()) < 2:
            continue
        chs = np.where(mask)[0]
        if waveform:
            xs = np.asarray([float(x_axis_km[int(ch)]) if int(ch) < len(x_axis_km) else float(ch) for ch in chs], dtype=np.float64)
        else:
            xs = chs
        ys = full_time[idx, mask] * float(target.get("_window_seconds", 120.0))
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, color=color, linewidth=0.8 if waveform else 1.2, alpha=0.65 if waveform else 0.35, linestyle="--", zorder=1)
        ax.scatter(
            xs,
            ys,
            s=6,
            marker="o",
            color="black" if waveform else "white",
            edgecolors="black" if waveform else "white",
            linewidths=0.5,
            zorder=2,
        )
        count += 1
    if count > 0:
        ax.text(0.01, 0.99, label, transform=ax.transAxes, color="black" if waveform else "#dddddd", fontsize=8, ha="left", va="top")
    return count


def _filter_dataset_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(SimplePeakSetDatasetConfig)}
    return {key: value for key, value in raw_cfg.items() if key in allowed}


class _ExportedShardDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).expanduser()
        meta = json.loads((self.dataset_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.shard_paths = [self.dataset_dir / str(name) for name in meta.get("shards", [])]
        meta_sizes = meta.get("shard_sizes", [])
        if len(meta_sizes) == len(self.shard_paths):
            self.shard_sizes = [int(size) for size in meta_sizes]
        else:
            self.shard_sizes = []
            for shard_path in self.shard_paths:
                payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
                self.shard_sizes.append(int(payload["x"].shape[0]))
        self.total = int(sum(self.shard_sizes))
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return self.total

    def _resolve(self, global_index: int) -> tuple[int, int]:
        idx = int(global_index) % max(1, self.total)
        acc = 0
        for shard_idx, shard_size in enumerate(self.shard_sizes):
            next_acc = acc + shard_size
            if idx < next_acc:
                return shard_idx, idx - acc
            acc = next_acc
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(str(self.shard_paths[shard_idx]), map_location="cpu", weights_only=False)
        return self._cache[shard_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shard_idx, local_idx = self._resolve(index)
        payload = self._load_shard(shard_idx)
        x = payload["x"][local_idx].to(torch.float32)
        target = {key: value[local_idx].clone() for key, value in payload["targets"].items() if torch.is_tensor(value)}
        if "raw_window" in payload:
            target["raw_window"] = payload["raw_window"][local_idx].clone()
        elif "raw_window" in payload["targets"]:
            target["raw_window"] = payload["targets"]["raw_window"][local_idx].clone()
        return x, target


def _extract_window_seconds(dataset_dir: Path | None) -> float | None:
    if dataset_dir is None:
        return None
    meta_path = Path(dataset_dir).expanduser() / "meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    dataset_cfg = meta.get("dataset_config")
    if isinstance(dataset_cfg, dict) and "window_seconds" in dataset_cfg:
        try:
            return float(dataset_cfg["window_seconds"])
        except (TypeError, ValueError):
            return None
    if "window_seconds" in meta:
        try:
            return float(meta["window_seconds"])
        except (TypeError, ValueError):
            return None
    return None


def _extract_dataset_config(dataset_dir: Path | None) -> dict[str, Any]:
    if dataset_dir is None:
        return {}
    meta_path = Path(dataset_dir).expanduser() / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    cfg = meta.get("dataset_config")
    return dict(cfg) if isinstance(cfg, dict) else {}


def _checkpoint_train_window_seconds(checkpoint: dict[str, Any]) -> float | None:
    cfg = checkpoint.get("dataset_config", {})
    if isinstance(cfg, dict) and "window_seconds" in cfg:
        try:
            return float(cfg["window_seconds"])
        except (TypeError, ValueError):
            pass
    dataset_dir = cfg.get("dataset_dir") if isinstance(cfg, dict) else None
    if dataset_dir:
        return _extract_window_seconds(Path(str(dataset_dir)))
    return None


def _offset_decoded_track(item: DecodedPeakTrack, *, sample_offset: int, time_offset_s: float, track_id: int) -> DecodedPeakTrack:
    points = [
        TrackPoint(
            ch_idx=int(point.ch_idx),
            t_idx=int(point.t_idx) + int(sample_offset),
            time_s=float(point.time_s) + float(time_offset_s),
            offset_m=float(point.offset_m),
            amp=float(point.amp),
            score=float(point.score),
        )
        for point in item.track.points
    ]
    return DecodedPeakTrack(
        track=Track(
            track_id=int(track_id),
            direction=str(item.track.direction),
            points=points,
            total_score=float(item.track.total_score),
            mean_speed_kmh=float(item.track.mean_speed_kmh),
        ),
        objectness=float(item.objectness),
        query_index=int(item.query_index),
        channel_indices=np.asarray(item.channel_indices, dtype=np.int64),
        complete_valid=np.asarray(item.complete_valid, dtype=np.float32),
        observed_valid=np.asarray(item.observed_valid, dtype=np.float32),
        point_times_norm=np.asarray(item.point_times_norm, dtype=np.float32),
    )


def _decode_tracks_for_plot(
    model: Any,
    model_type: str,
    raw: np.ndarray,
    *,
    fs: float,
    x_axis_m: np.ndarray,
    config: PeakSetInferenceConfig,
    device: str,
    trained_window_seconds: float | None,
    allow_window_tiling: bool,
) -> tuple[list[DecodedPeakTrack], int, float]:
    raw_window_seconds = float(raw.shape[1]) / float(max(1e-9, fs))
    decode_fn = decode_peak_guided_vehicle_tracks if str(model_type) == "PeakGuidedVehicleSetTransformer" else decode_vehicle_peak_tracks
    if trained_window_seconds is None or float(trained_window_seconds) <= 0.0:
        tracks = decode_fn(model, raw, fs=fs, x_axis_m=x_axis_m, config=config, device=device)
        return tracks, 1, raw_window_seconds
    chunk_seconds = float(trained_window_seconds)
    if raw_window_seconds <= chunk_seconds * 1.10:
        tracks = decode_fn(model, raw, fs=fs, x_axis_m=x_axis_m, config=config, device=device)
        return tracks, 1, raw_window_seconds
    if not bool(allow_window_tiling):
        raise RuntimeError(
            f"Requested {raw_window_seconds:.1f}s prediction with a checkpoint trained on {chunk_seconds:.1f}s windows. "
            "Generate/train a matching checkpoint, or set ALLOW_WINDOW_TILING=1 for a boundary-unsafe fallback."
        )

    chunk_samples = int(round(chunk_seconds * float(fs)))
    chunk_samples = int(max(1, min(chunk_samples, int(raw.shape[1]))))
    decoded: list[DecodedPeakTrack] = []
    chunk_count = 0
    for start in range(0, int(raw.shape[1]), chunk_samples):
        end = min(int(raw.shape[1]), start + chunk_samples)
        if end - start < max(2, int(0.25 * chunk_samples)):
            break
        chunk_count += 1
        chunk = raw[:, start:end]
        chunk_tracks = decode_fn(model, chunk, fs=fs, x_axis_m=x_axis_m, config=config, device=device)
        for item in chunk_tracks:
            decoded.append(
                _offset_decoded_track(
                    item,
                    sample_offset=int(start),
                    time_offset_s=float(start) / float(fs),
                    track_id=len(decoded),
                )
            )
    return decoded, chunk_count, chunk_seconds


def _crop_to_predict_seconds(
    raw: np.ndarray,
    target: dict[str, torch.Tensor],
    *,
    fs: float,
    source_window_seconds: float,
    predict_seconds: float,
) -> tuple[np.ndarray, dict[str, torch.Tensor], float]:
    raw_seconds = float(raw.shape[1]) / float(max(1e-9, fs))
    keep_seconds = float(min(max(1.0 / float(max(1e-9, fs)), predict_seconds), raw_seconds))
    keep_samples = int(round(keep_seconds * float(fs)))
    keep_samples = int(max(1, min(keep_samples, int(raw.shape[1]))))
    if keep_samples >= int(raw.shape[1]):
        target["_window_seconds"] = torch.tensor(float(source_window_seconds))
        target["_predict_seconds"] = torch.tensor(float(raw_seconds))
        return raw, target, raw_seconds

    cropped_target = dict(target)
    cropped_raw = raw[:, :keep_samples]
    if "full_time" in cropped_target and "full_valid" in cropped_target:
        full_time = cropped_target["full_time"].clone()
        full_valid = cropped_target["full_valid"].clone()
        keep_norm = float(keep_seconds) / float(max(1e-9, source_window_seconds))
        full_valid = full_valid * (full_time <= float(keep_norm)).to(full_valid.dtype)
        cropped_target["full_valid"] = full_valid
        if "observed_visibility" in cropped_target:
            cropped_target["observed_visibility"] = cropped_target["observed_visibility"].clone() * full_valid.to(
                cropped_target["observed_visibility"].dtype
            )
    cropped_target["_window_seconds"] = torch.tensor(float(source_window_seconds))
    cropped_target["_predict_seconds"] = torch.tensor(float(keep_seconds))
    return cropped_raw, cropped_target, keep_seconds


def _load_sample_source(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    index: int,
    *,
    exported_dataset: _ExportedShardDataset | None = None,
    synthetic_dataset: SimpleLinearVehiclePeakDataset | None = None,
) -> tuple[np.ndarray, float, np.ndarray, dict[str, torch.Tensor]]:
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        checkpoint_dataset_dir = checkpoint.get("dataset_config", {}).get("dataset_dir")
        if checkpoint_dataset_dir:
            dataset_dir = Path(str(checkpoint_dataset_dir))
    if dataset_dir is not None and Path(dataset_dir).expanduser().exists():
        ds = exported_dataset or _ExportedShardDataset(Path(dataset_dir))
        _x, target = ds[int(index)]
        raw = target.get("raw_window")
        if raw is None:
            print(
                f"warning: shard sample {index} has no raw_window; falling back to synthetic reconstruction",
                flush=True,
            )
        else:
            raw_np = raw.to(torch.float32).cpu().numpy()
            exported_cfg = _extract_dataset_config(Path(dataset_dir))
            fs = float(exported_cfg.get("fs", checkpoint.get("dataset_config", {}).get("fs", 1000.0)))
            dx_m = float(exported_cfg.get("dx_m", checkpoint.get("dataset_config", {}).get("dx_m", 20.0)))
            x_axis_m = np.arange(raw_np.shape[0], dtype=np.float32) * dx_m
            dataset_window_seconds = _extract_window_seconds(Path(dataset_dir))
            if dataset_window_seconds is None:
                dataset_window_seconds = float(checkpoint.get("dataset_config", {}).get("window_seconds", args.window_seconds))
            target["_window_seconds"] = torch.tensor(float(dataset_window_seconds))
            predict_seconds = float(args.predict_seconds if args.predict_seconds is not None else dataset_window_seconds)
            raw_np, target, _ = _crop_to_predict_seconds(
                raw_np,
                target,
                fs=fs,
                source_window_seconds=float(dataset_window_seconds),
                predict_seconds=float(predict_seconds),
            )
            return raw_np, fs, x_axis_m, target

    dataset_cfg = _filter_dataset_config(dict(checkpoint.get("dataset_config", {})))
    dataset_cfg.setdefault("length", 1)
    dataset_cfg["return_raw_window"] = True
    dataset_cfg["window_seconds"] = float(args.predict_seconds if args.predict_seconds is not None else args.window_seconds)
    dataset_cfg["seed"] = int(dataset_cfg.get("seed", 42))
    ds = synthetic_dataset or SimpleLinearVehiclePeakDataset(config=SimplePeakSetDatasetConfig(**dataset_cfg))
    sample = ds[int(index) % len(ds)]
    _x, target = sample
    raw_np = target["raw_window"].to(torch.float32).cpu().numpy()
    fs = float(dataset_cfg.get("fs", 1000.0))
    dx_m = float(dataset_cfg.get("dx_m", 20.0))
    x_axis_m = np.arange(raw_np.shape[0], dtype=np.float32) * dx_m
    target["_window_seconds"] = torch.tensor(float(dataset_cfg.get("window_seconds", args.window_seconds)))
    return raw_np, fs, x_axis_m, target


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    model, checkpoint = load_checkpoint_model(Path(args.model).expanduser(), device=device)
    model_type = str(checkpoint.get("model_type", type(model).__name__))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    dataset_cfg = _filter_dataset_config(dict(checkpoint.get("dataset_config", {})))
    dataset_cfg.setdefault("window_seconds", float(args.window_seconds))
    dataset_cfg.setdefault("time_downsample", 10)
    dataset_cfg.setdefault("clip_ratio", 1.35)
    dataset_cfg.setdefault("min_visible_channels", 5)
    dataset_cfg.setdefault("speed_min_kmh", 60.0)
    dataset_cfg.setdefault("speed_max_kmh", 90.0)
    exported_dataset = None
    if args.dataset_dir is not None and Path(args.dataset_dir).expanduser().exists():
        exported_dataset = _ExportedShardDataset(Path(args.dataset_dir))
    synthetic_dataset = None
    if exported_dataset is None:
        synthetic_cfg = dict(dataset_cfg)
        synthetic_cfg.setdefault("length", 1)
        synthetic_cfg["return_raw_window"] = True
        synthetic_cfg["window_seconds"] = float(args.window_seconds)
        synthetic_cfg["seed"] = int(synthetic_cfg.get("seed", 42))
        synthetic_dataset = SimpleLinearVehiclePeakDataset(config=SimplePeakSetDatasetConfig(**synthetic_cfg))

    n_samples = int(max(1, args.num_samples))
    start_index = int(args.sample_index)
    sample_summaries: list[dict[str, Any]] = []
    summary_window_seconds = _extract_window_seconds(Path(args.dataset_dir)) if args.dataset_dir is not None else None
    if summary_window_seconds is None:
        summary_window_seconds = float(args.window_seconds)
    requested_predict_seconds = float(args.predict_seconds if args.predict_seconds is not None else summary_window_seconds)
    trained_window_seconds = _checkpoint_train_window_seconds(checkpoint)
    sheet_fig = None
    sheet_axes = None
    if bool(args.save_sheet):
        sheet_fig, sheet_axes = plt.subplots(n_samples, 2, figsize=(15, max(1, n_samples) * 6), dpi=int(args.plot_dpi), constrained_layout=True)
        if n_samples == 1:
            sheet_axes = np.asarray([sheet_axes])

    for row in range(n_samples):
        sample_index = start_index + row
        raw, fs, x_axis_m, target = _load_sample_source(
            args,
            checkpoint,
            sample_index,
            exported_dataset=exported_dataset,
            synthetic_dataset=synthetic_dataset,
        )
        configured_dead = _parse_int_csv(dataset_cfg.get("dead_channel_indices"))
        detected_dead = _detect_dead_channels(raw)
        dead_channels = sorted({*configured_dead, *detected_dead})
        actual_plot_seconds = float(raw.shape[1]) / float(max(1e-9, fs))
        base_cfg = PeakSetInferenceConfig(
            time_downsample=int(dataset_cfg.get("time_downsample", 10)),
            objectness_threshold=0.25,
            complete_valid_threshold=0.40,
            anchor_threshold=0.45,
            min_visible_channels=int(dataset_cfg.get("min_visible_channels", 5)),
            speed_min_kmh=float(dataset_cfg.get("speed_min_kmh", 60.0)),
            speed_max_kmh=float(dataset_cfg.get("speed_max_kmh", 90.0)),
            refine_radius_samples=0 if args.no_refine else 80,
            max_tracks=16,
            dedup_tolerance_samples=30,
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
            graph_refine=False,
        )
        refined_cfg = PeakSetInferenceConfig(
            time_downsample=int(dataset_cfg.get("time_downsample", 10)),
            objectness_threshold=0.25,
            complete_valid_threshold=0.40,
            anchor_threshold=0.45,
            min_visible_channels=int(dataset_cfg.get("min_visible_channels", 5)),
            speed_min_kmh=float(dataset_cfg.get("speed_min_kmh", 60.0)),
            speed_max_kmh=float(dataset_cfg.get("speed_max_kmh", 90.0)),
            refine_radius_samples=120,
            max_tracks=16,
            dedup_tolerance_samples=30,
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
            graph_refine=True,
        )

        raw_tracks, raw_chunk_count, raw_chunk_seconds = _decode_tracks_for_plot(
            model,
            model_type,
            raw,
            fs=fs,
            x_axis_m=x_axis_m,
            config=base_cfg,
            device=device,
            trained_window_seconds=trained_window_seconds,
            allow_window_tiling=bool(args.allow_window_tiling),
        )
        if args.no_refine:
            refined_tracks = raw_tracks
            refined_chunk_count = raw_chunk_count
        else:
            refined_tracks, refined_chunk_count, _ = _decode_tracks_for_plot(
                model,
                model_type,
                raw,
                fs=fs,
                x_axis_m=x_axis_m,
                config=refined_cfg,
                device=device,
                trained_window_seconds=trained_window_seconds,
                allow_window_tiling=bool(args.allow_window_tiling),
            )

        if sheet_axes is not None:
            panels = [
                (sheet_axes[row, 0], raw_tracks, "Model decode | raw completion"),
                (sheet_axes[row, 1], refined_tracks, "Postprocessed decode | line fit + local refine"),
            ]
            for ax, decoded, title in panels:
                _draw_background(ax, raw, fs=fs, x_axis_m=x_axis_m, plot_style=str(args.plot_style), dead_channels=dead_channels)
                _draw_ground_truth(ax, target, x_axis_m=x_axis_m, plot_style=str(args.plot_style), label="GT" if ax is panels[0][0] else "")
                _draw_tracks(ax, decoded, x_axis_m=x_axis_m, plot_style=str(args.plot_style), show_labels=False)
                ax.set_title(f"{title} | sample={sample_index} tracks={len(decoded)}")

        single_fig, single_axes = plt.subplots(1, 2, figsize=(15, 6), dpi=int(args.plot_dpi), constrained_layout=True)
        for ax, decoded, title in [
            (single_axes[0], raw_tracks, "Model decode | raw completion"),
            (single_axes[1], refined_tracks, "Postprocessed decode | line fit + local refine"),
        ]:
            _draw_background(ax, raw, fs=fs, x_axis_m=x_axis_m, plot_style=str(args.plot_style), dead_channels=dead_channels)
            _draw_ground_truth(ax, target, x_axis_m=x_axis_m, plot_style=str(args.plot_style), label="GT" if ax is single_axes[0] else "")
            _draw_tracks(ax, decoded, x_axis_m=x_axis_m, plot_style=str(args.plot_style), show_labels=False)
            ax.set_title(f"{title} | sample={sample_index} tracks={len(decoded)}")
        single_path = out_dir / f"vehicle_peak_set_overlay_{sample_index:06d}.png"
        single_fig.savefig(str(single_path))
        plt.close(single_fig)

        sample_summaries.append(
            {
                "sample_index": int(sample_index),
                "raw_tracks": int(len(raw_tracks)),
                "refined_tracks": int(len(refined_tracks)),
                "dead_channels": [int(x) for x in dead_channels],
                "decode_chunks": int(raw_chunk_count),
                "decode_chunk_seconds": float(raw_chunk_seconds),
                "actual_plot_seconds": float(actual_plot_seconds),
                "overlay": str(single_path),
            }
        )

    overlay = out_dir / "vehicle_peak_set_overlay_sheet.png"
    if sheet_fig is not None:
        sheet_fig.savefig(str(overlay))
        plt.close(sheet_fig)
    else:
        overlay = None

    summary = {
        "model": str(Path(args.model).expanduser()),
        "device": device,
        "sample_index": int(args.sample_index),
        "num_samples": int(n_samples),
        "samples": sample_summaries,
        "overlay": str(overlay) if overlay is not None else None,
        "save_sheet": bool(args.save_sheet),
        "window_seconds": float(args.window_seconds),
        "effective_window_seconds": float(summary_window_seconds),
        "requested_predict_seconds": float(requested_predict_seconds),
        "trained_window_seconds": None if trained_window_seconds is None else float(trained_window_seconds),
        "allow_window_tiling": bool(args.allow_window_tiling),
        "no_refine": bool(args.no_refine),
        "plot_style": str(args.plot_style),
        "dataset_dir": str(Path(args.dataset_dir).expanduser()) if args.dataset_dir is not None else None,
        "checkpoint_dataset_dir": checkpoint.get("dataset_config", {}).get("dataset_dir"),
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None
