"""Plot two-channel PeakSlotNet shards with U-Net prior and GT targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PeakSlotNet+U-Net-prior input/target samples.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing generated shard_*.pt files.")
    parser.add_argument("--meta-dir", type=Path, default=None, help="Directory containing meta.json; defaults to --data-dir.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output plot directory.")
    parser.add_argument("--sample-indices", default="0,1,2,3", help="Comma-separated global sample indices.")
    parser.add_argument("--plot-peaks", action="store_true", help="Draw all peak candidates in the target panel.")
    parser.add_argument("--max-gt-tracks", type=int, default=24, help="Maximum GT tracks to draw per sample.")
    parser.add_argument("--dpi", type=int, default=150, help="Output PNG DPI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    meta_dir = Path(args.meta_dir).expanduser() if args.meta_dir is not None else data_dir
    out_dir = Path(args.out_dir).expanduser()
    meta = _load_meta(meta_dir)
    shards = _resolve_shards(data_dir, meta)
    selected = _parse_indices(str(args.sample_indices))
    if not selected:
        raise ValueError("No sample indices selected.")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for global_idx, shard_name, local_idx, sample in _iter_samples(data_dir, shards, selected):
        out_path = out_dir / f"sample_{global_idx:06d}_input_target.png"
        _plot_sample(
            out_path,
            sample_index=global_idx,
            shard_name=shard_name,
            local_idx=local_idx,
            sample=sample,
            meta=meta,
            plot_peaks=bool(args.plot_peaks),
            max_gt_tracks=int(args.max_gt_tracks),
            dpi=int(args.dpi),
        )
        summary.append(
            {
                "sample_index": int(global_idx),
                "shard": str(shard_name),
                "local_index": int(local_idx),
                "path": str(out_path),
            }
        )
        print(f"wrote {out_path}", flush=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def _load_meta(meta_dir: Path) -> dict[str, Any]:
    path = meta_dir / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"meta.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_shards(data_dir: Path, meta: dict[str, Any]) -> list[str]:
    listed = [str(item) for item in meta.get("shards", [])]
    existing = {path.name for path in data_dir.glob("shard_*.pt")}
    if listed:
        shards = [name for name in listed if name in existing]
    else:
        shards = sorted(existing)
    if not shards:
        raise FileNotFoundError(f"No shard_*.pt files found in {data_dir}")
    return shards


def _parse_indices(text: str) -> set[int]:
    out: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if item:
            out.add(int(item))
    return out


def _iter_samples(
    data_dir: Path,
    shards: list[str],
    selected: set[int],
) -> Any:
    global_start = 0
    for shard_name in shards:
        payload = torch.load(str(data_dir / shard_name), map_location="cpu", weights_only=False)
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
        if global_start > max(selected):
            break


def _plot_sample(
    out_path: Path,
    *,
    sample_index: int,
    shard_name: str,
    local_idx: int,
    sample: dict[str, torch.Tensor],
    meta: dict[str, Any],
    plot_peaks: bool,
    max_gt_tracks: int,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = sample["x"].to(torch.float32).cpu().numpy()
    if x.ndim != 3 or x.shape[0] < 2:
        raise ValueError(f"Expected sample x shape [2,C,T], got {x.shape}")
    track = np.asarray(x[0], dtype=np.float32)
    prior = np.asarray(x[1], dtype=np.float32)
    window_seconds = float(meta.get("window_seconds", 120.0))
    n_ch, n_t = int(track.shape[0]), int(track.shape[1])
    extent = (0.0, window_seconds, -0.5, float(n_ch) - 0.5)
    vmax = _robust_vmax(track)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.0), dpi=int(dpi), sharex=True, sharey=True)
    axes = axes.ravel()
    panels = [
        (track, "input ch0: track Gaussian heatmap", "gray_r", 0.0, vmax),
        (prior, "input ch1: waveform-line U-Net prior", "magma", 0.0, max(1e-6, float(np.quantile(prior, 0.999)))),
        (_blend(track, prior), "two-channel input blend", None, 0.0, 1.0),
        (track, "PeakSlot target: candidates and GT-selected peaks", "gray_r", 0.0, vmax),
    ]
    for ax, (arr, title, cmap, vmin, vmax_panel) in zip(axes, panels):
        if arr.ndim == 3:
            ax.imshow(arr, origin="lower", aspect="auto", extent=extent, interpolation="nearest")
        else:
            ax.imshow(
                arr,
                origin="lower",
                aspect="auto",
                extent=extent,
                interpolation="nearest",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_panel,
            )
        ax.set_title(title)
        ax.set_ylabel("channel")
        ax.grid(color="white", alpha=0.12, linewidth=0.35)

    gt_tracks = _gt_tracks(sample, window_seconds=window_seconds, max_gt_tracks=max_gt_tracks)
    for ax_idx, ax in enumerate(axes):
        if ax_idx in {0, 2, 3}:
            _draw_gt_tracks(ax, gt_tracks)
    if bool(plot_peaks):
        _draw_peak_candidates(axes[3], sample, window_seconds=window_seconds)
    _draw_gt_selected_peaks(axes[3], gt_tracks)
    for ax in axes[2:]:
        ax.set_xlabel("time [s]")
    fig.suptitle(f"sample {sample_index} | {shard_name} local {local_idx}", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _robust_vmax(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    return max(float(np.quantile(np.abs(finite), 0.995)), 1e-6)


def _blend(track: np.ndarray, prior: np.ndarray) -> np.ndarray:
    gray = np.clip(track / _robust_vmax(track), 0.0, 1.0)
    p = np.clip(prior / max(float(np.quantile(prior, 0.999)), 1e-6), 0.0, 1.0)
    rgb = np.stack([gray, gray, gray], axis=-1)
    rgb[..., 1] = np.maximum(rgb[..., 1], 0.25 * p)
    rgb[..., 2] = np.maximum(rgb[..., 2], p)
    return np.clip(rgb, 0.0, 1.0)


def _gt_tracks(sample: dict[str, torch.Tensor], *, window_seconds: float, max_gt_tracks: int) -> list[dict[str, Any]]:
    peak_time = sample["peak_time"].to(torch.float32)
    gt_peak_index = sample["gt_peak_index"].to(torch.long)
    visibility = sample["visibility"].to(torch.float32)
    gt_valid = sample["gt_valid"].to(torch.bool)
    direction = sample.get("direction")
    k_count = int(peak_time.shape[1])
    tracks: list[dict[str, Any]] = []
    valid_gt = torch.where(gt_valid)[0].tolist()
    if int(max_gt_tracks) > 0:
        valid_gt = valid_gt[: int(max_gt_tracks)]
    for gt_idx in valid_gt:
        points = []
        for ch in torch.where(visibility[int(gt_idx)] > 0.5)[0].tolist():
            peak_idx = int(gt_peak_index[int(gt_idx), int(ch)].item())
            if not (0 <= peak_idx < k_count):
                continue
            t_s = float(peak_time[int(ch), peak_idx].item()) * float(window_seconds)
            points.append((t_s, int(ch), int(peak_idx)))
        if points:
            tracks.append(
                {
                    "gt_idx": int(gt_idx),
                    "direction": int(direction[int(gt_idx)].item()) if torch.is_tensor(direction) else -1,
                    "points": sorted(points, key=lambda item: item[1]),
                }
            )
    return tracks


def _draw_gt_tracks(ax: Any, tracks: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    for rank, track in enumerate(tracks):
        pts = track["points"]
        if not pts:
            continue
        color = cmap(rank % 20)
        times = [item[0] for item in pts]
        chs = [item[1] for item in pts]
        ax.plot(times, chs, color=color, linewidth=1.1, alpha=0.9)
        ax.scatter(times, chs, color=color, s=8, alpha=0.9, linewidths=0)


def _draw_peak_candidates(ax: Any, sample: dict[str, torch.Tensor], *, window_seconds: float) -> None:
    peak_time = sample["peak_time"].to(torch.float32)
    peak_valid = sample["peak_valid"].to(torch.bool)
    ts = []
    chs = []
    for ch in range(int(peak_valid.shape[0])):
        valid = torch.where(peak_valid[ch])[0].tolist()
        ts.extend([float(peak_time[ch, int(idx)].item()) * float(window_seconds) for idx in valid])
        chs.extend([int(ch)] * len(valid))
    if ts:
        ax.scatter(ts, chs, s=4, color="#1f77b4", alpha=0.18, linewidths=0)


def _draw_gt_selected_peaks(ax: Any, tracks: list[dict[str, Any]]) -> None:
    ts = []
    chs = []
    for track in tracks:
        for t_s, ch, _ in track["points"]:
            ts.append(float(t_s))
            chs.append(int(ch))
    if ts:
        ax.scatter(ts, chs, s=16, facecolors="none", edgecolors="#ff3333", linewidths=0.7, alpha=0.95)


if __name__ == "__main__":
    raise SystemExit(main())
