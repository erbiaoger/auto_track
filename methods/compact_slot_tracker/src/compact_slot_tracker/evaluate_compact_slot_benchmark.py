"""Evaluate the compact slot model on a multi-vehicle benchmark file."""

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
from autotrack.dl.compact_slot_model import LABEL_TO_DIRECTION, InferenceConfig, load_checkpoint_model, predict_tracks_from_window
from autotrack.dl.multi_vehicle_benchmark_dataset import MultiVehicleBenchmarkDataset, benchmark_json_ready


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the compact slot model on a multi-vehicle benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for metrics and overlays.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit evaluation to the first N samples; 0 means all.")
    parser.add_argument("--plot-samples", type=int, default=8, help="Number of overlay figures to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI.")
    parser.add_argument("--objectness-threshold", type=float, default=0.15, help="Slot objectness threshold.")
    parser.add_argument("--candidate-objectness-floor", type=float, default=0.05, help="Candidate slot floor.")
    parser.add_argument("--objectness-count-scale", type=float, default=1.05, help="Soft count scale for slot retention.")
    parser.add_argument("--visibility-threshold", type=float, default=0.35, help="Per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for a predicted track.")
    parser.add_argument("--point-threshold", type=float, default=0.12, help="Normalized time error threshold for a TP track.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample factor used by the checkpoint.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization denominator.")
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
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _gt_tracks(sample: dict[str, Any], sample_index: int) -> list[Track]:
    target = sample["target"]
    if "gt_valid" in target:
        gt_indices = torch.where(target["gt_valid"].to(torch.bool))[0].tolist()
    else:
        gt_indices = list(range(int(target["time"].shape[0])))
    tracks: list[Track] = []
    for local_id, g_idx in enumerate(gt_indices):
        time = target["time"][g_idx]
        visibility = target["visibility"][g_idx]
        direction = LABEL_TO_DIRECTION.get(int(target["direction"][g_idx].item()), "forward")
        speed = float(target["speed"][g_idx].item())
        points: list[TrackPoint] = []
        for ch in torch.where(visibility > 0.5)[0].tolist():
            t_idx = int(round(float(time[ch].item()) * max(1, int(time.shape[0]) - 1)))
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=int(t_idx),
                    time_s=float(t_idx),
                    offset_m=float(ch),
                    amp=1.0,
                    score=1.0,
                )
            )
        if points:
            tracks.append(Track(track_id=int(local_id), direction=direction, points=points, total_score=float(len(points)), mean_speed_kmh=speed))
    return tracks


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
    ax.set_title(f"compact slot benchmark overlay | gt={len(gt_tracks)} pred={len(pred_tracks)}")
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
    benchmark = MultiVehicleBenchmarkDataset(args.benchmark_file, max_samples=int(args.max_samples))
    model, checkpoint = load_checkpoint_model(args.model, device=device)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    infer_cfg = InferenceConfig(
        time_downsample=int(args.time_downsample),
        objectness_threshold=float(args.objectness_threshold),
        visibility_threshold=float(args.visibility_threshold),
        min_visible_channels=int(args.min_visible_channels),
        max_tracks=int(model.config.max_tracks),
        candidate_objectness_floor=float(args.candidate_objectness_floor),
        objectness_count_scale=float(args.objectness_count_scale),
        speed_norm_kmh=float(args.speed_norm_kmh),
    )

    rows: list[dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    total_count_err = 0.0
    for idx, sample in enumerate(benchmark.samples):
        raw_window = sample["target"].get("raw_window")
        if raw_window is None:
            raw_window = sample["x"][0]
        raw_np = raw_window.to(torch.float32).cpu().numpy()
        gt_tracks = _gt_tracks(sample, idx)
        pred_tracks = predict_tracks_from_window(
            model,
            raw_np,
            fs=float(benchmark.meta.payload.get("fs", 1000.0)),
            x_axis_m=np.arange(int(raw_np.shape[0]), dtype=np.float32) * float(benchmark.meta.payload.get("dx_m", 100.0)),
            config=infer_cfg,
            device=device,
        )
        tp, fp, fn = _match_tracks(
            pred_tracks,
            gt_tracks,
            tol_samples=180,
            min_overlap_channels=3,
            min_overlap_ratio=0.7,
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
        "samples": int(len(benchmark)),
        "track_precision": precision,
        "track_recall": recall,
        "track_f1": f1,
        "count_mae": float(total_count_err / max(1, len(benchmark))),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
        "per_sample": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "per_sample.csv").write_text("", encoding="utf-8")
    with (out_dir / "per_sample.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["sample_index", "gt_count", "pred_count", "tp", "fp", "fn"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
