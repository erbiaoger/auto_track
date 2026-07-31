"""Evaluate the trajectory-energy model on a multi-vehicle benchmark file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.trajectory_energy_model import ModelConfig, TrajectoryEnergyNet, extract_vehicle_tracks_from_energy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trajectory-energy model on a multi-vehicle benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for metrics and overlays.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit evaluation to the first N samples; 0 means all.")
    parser.add_argument("--plot-samples", type=int, default=8, help="Number of overlay figures to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample used by the model.")
    parser.add_argument("--min-track-channels", type=int, default=4, help="Minimum channels for a decoded track.")
    parser.add_argument("--min-track-score", type=float, default=0.8, help="Minimum track score for decoder.")
    parser.add_argument("--max-skip-channels", type=int, default=8, help="Graph search skip limit.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Peak prominence for decoder.")
    parser.add_argument("--peak-min-height", type=float, default=0.04, help="Peak min height for decoder.")
    parser.add_argument("--seed-threshold", type=float, default=0.08, help="Seed threshold for line scan.")
    parser.add_argument("--dedup-time-tol", type=int, default=800, help="Track deduplication tolerance in samples.")
    parser.add_argument("--dedup-channel-overlap", type=int, default=3, help="Track deduplication minimum shared channels.")
    parser.add_argument("--dedup-overlap-ratio", type=float, default=0.65, help="Track deduplication overlap ratio.")
    parser.add_argument("--decoder-profile", default="strict", choices=["auto", "balanced", "strict", "recall"], help="Decoder profile used by the extractor.")
    parser.add_argument("--scene-active-ratio-threshold", type=float, default=0.033, help="Active-ratio threshold used when decoder-profile=auto.")
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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _gt_tracks(sample: dict[str, Any]) -> list[Track]:
    target = sample["target"]
    raw_window = target.get("raw_window")
    raw_len = int(raw_window.shape[1]) if torch.is_tensor(raw_window) and raw_window.ndim == 2 else int(sample["x"].shape[-1])
    if "gt_valid" in target:
        gt_indices = torch.where(target["gt_valid"].to(torch.bool))[0].tolist()
    else:
        gt_indices = list(range(int(target["time"].shape[0])))
    tracks: list[Track] = []
    for local_id, g_idx in enumerate(gt_indices):
        time = target["time"][g_idx]
        visibility = target["visibility"][g_idx]
        direction = "forward" if int(target["direction"][g_idx].item()) == 0 else "reverse"
        speed = float(target["speed"][g_idx].item())
        points: list[TrackPoint] = []
        for ch in torch.where(visibility > 0.5)[0].tolist():
            t_idx = int(round(float(time[ch].item()) * max(1, raw_len - 1)))
            points.append(TrackPoint(ch_idx=int(ch), t_idx=int(t_idx), time_s=float(t_idx), offset_m=float(ch), amp=1.0, score=1.0))
        if points:
            tracks.append(Track(track_id=int(local_id), direction=direction, points=points, total_score=float(len(points)), mean_speed_kmh=speed))
    return tracks


def _scale_tracks(tracks: list[Track], scale: int, *, fs: float) -> list[Track]:
    factor = int(max(1, scale))
    scaled: list[Track] = []
    for tr in tracks:
        points = [
            TrackPoint(
                ch_idx=int(p.ch_idx),
                t_idx=int(p.t_idx * factor),
                time_s=float(p.t_idx * factor) / float(fs),
                offset_m=float(p.offset_m),
                amp=float(p.amp),
                score=float(p.score),
            )
            for p in tr.points
        ]
        scaled.append(
            Track(
                track_id=int(tr.track_id),
                direction=str(tr.direction),
                points=points,
                total_score=float(tr.total_score),
                mean_speed_kmh=float(tr.mean_speed_kmh),
            )
        )
    return scaled


def _track_overlap(a: Track, b: Track, *, tol_samples: int, min_overlap_channels: int) -> tuple[float, int]:
    amap = {int(p.ch_idx): int(p.t_idx) for p in a.points}
    bmap = {int(p.ch_idx): int(p.t_idx) for p in b.points}
    common = sorted(set(amap).intersection(bmap))
    if len(common) < int(min_overlap_channels):
        return 0.0, 0
    diffs = [abs(float(amap[ch] - bmap[ch])) for ch in common]
    if float(np.median(diffs)) > float(tol_samples):
        return 0.0, len(common)
    ratio = len(common) / float(max(1, min(len(a.points), len(b.points))))
    return float(ratio), len(common)


def _match_tracks(
    pred_tracks: list[Track],
    gt_tracks: list[Track],
    *,
    tol_samples: int,
    min_overlap_channels: int,
    min_overlap_ratio: float,
) -> tuple[int, int, int]:
    scored: list[tuple[float, int, int]] = []
    for p_idx, pred in enumerate(pred_tracks):
        for g_idx, gt in enumerate(gt_tracks):
            ratio, common = _track_overlap(pred, gt, tol_samples=int(tol_samples), min_overlap_channels=int(min_overlap_channels))
            if common >= int(min_overlap_channels) and ratio >= float(min_overlap_ratio):
                scored.append((ratio, p_idx, g_idx))
    scored.sort(reverse=True)
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    tp = 0
    for _, p_idx, g_idx in scored:
        if p_idx in used_pred or g_idx in used_gt:
            continue
        used_pred.add(int(p_idx))
        used_gt.add(int(g_idx))
        tp += 1
    return int(tp), int(len(pred_tracks) - tp), int(len(gt_tracks) - tp)


def _plot_sample(out_path: Path, *, raw_window: np.ndarray, pred_tracks: list[Track], gt_tracks: list[Track], dpi: int) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True, dpi=int(dpi))
    ax.imshow(raw_window, aspect="auto", origin="lower", cmap="magma")
    colors = ["cyan", "lime", "yellow", "orange", "deepskyblue", "white", "red", "violet"]
    for idx, tr in enumerate(pred_tracks):
        pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
        ax.plot([float(p.t_idx) for p in pts], [int(p.ch_idx) for p in pts], lw=2.0, color=colors[idx % len(colors)], label=f"pred {idx}")
    for tr in gt_tracks:
        pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
        ax.plot([float(p.t_idx) for p in pts], [int(p.ch_idx) for p in pts], lw=1.2, color="lime", alpha=0.65)
    ax.set_title(f"trajectory-energy overlay | gt={len(gt_tracks)} pred={len(pred_tracks)}")
    ax.set_xlabel("time [sample idx]")
    ax.set_ylabel("channel")
    if pred_tracks:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.75, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    samples = list(payload.get("samples", []))
    if int(args.max_samples) > 0:
        samples = samples[: int(args.max_samples)]
    if not samples:
        raise ValueError("benchmark contains no samples")

    checkpoint = torch.load(str(Path(args.model).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**dict(checkpoint.get("model_config", {})))
    model = TrajectoryEnergyNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    meta = dict(payload.get("meta", {}))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_cfg = {
        "time_downsample": int(args.time_downsample),
        "decoder_profile": str(args.decoder_profile),
        "scene_active_ratio_threshold": float(args.scene_active_ratio_threshold),
        "min_track_channels": int(args.min_track_channels),
        "min_track_score": float(args.min_track_score),
        "max_skip_channels": int(args.max_skip_channels),
        "peak_prominence": float(args.peak_prominence),
        "peak_min_height": float(args.peak_min_height),
        "seed_threshold": float(args.seed_threshold),
        "dedup_time_tol": int(args.dedup_time_tol),
        "dedup_channel_overlap": int(args.dedup_channel_overlap),
        "dedup_overlap_ratio": float(args.dedup_overlap_ratio),
    }

    rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    total_count_err = 0.0
    for idx, sample in enumerate(samples):
        raw_window = sample["target"].get("raw_window")
        if raw_window is None:
            raw_window = sample["x"][0]
        raw_np = raw_window.to(torch.float32).cpu().numpy()
        x = sample["x"].to(torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            outputs = model(x.to(device))
        energy = torch.sigmoid(outputs["energy_logits"][0]).detach().cpu().numpy()
        pred_tracks = extract_vehicle_tracks_from_energy(
            energy,
            fs=float(fs),
            dx_m=float(dx_m),
            direction="both",
            vmin_kmh=float(meta.get("speed_min_kmh", 70.0)),
            vmax_kmh=float(meta.get("speed_max_kmh", 90.0)),
            config=decode_cfg,
        )
        pred_tracks = _scale_tracks(pred_tracks, int(meta.get("time_downsample", 10)), fs=float(fs))
        gt_tracks = _gt_tracks(sample)
        tp, fp, fn = _match_tracks(
            pred_tracks,
            gt_tracks,
            tol_samples=int(args.dedup_time_tol),
            min_overlap_channels=int(args.dedup_channel_overlap),
            min_overlap_ratio=float(args.dedup_overlap_ratio),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_count_err += abs(len(pred_tracks) - len(gt_tracks))
        rows.append(
            {
                "sample_index": int(idx),
                "gt_count": int(len(gt_tracks)),
                "pred_count": int(len(pred_tracks)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }
        )
        if int(args.plot_samples) > 0 and idx < int(args.plot_samples):
            _plot_sample(out_dir / f"sample_{idx:02d}.png", raw_window=raw_np, pred_tracks=pred_tracks, gt_tracks=gt_tracks, dpi=int(args.plot_dpi))

    precision = float(total_tp / max(1, total_tp + total_fp))
    recall = float(total_tp / max(1, total_tp + total_fn))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    summary = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model": str(Path(args.model).expanduser()),
        "device": device,
        "samples": int(len(samples)),
        "track_precision": precision,
        "track_recall": recall,
        "track_f1": f1,
        "count_mae": float(total_count_err / max(1, len(samples))),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "per_sample": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_index", "gt_count", "pred_count", "tp", "fp", "fn"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
