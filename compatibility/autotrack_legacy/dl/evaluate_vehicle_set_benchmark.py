"""Evaluate the query-based vehicle set network on a benchmark file."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.vehicle_set_net import VehicleSetModelConfig, VehicleSetNet, decode_vehicle_set_tracks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the vehicle set network on a multi_vehicle_benchmark_v1 file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Benchmark .pt file.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit to the first N samples; 0 means all.")
    parser.add_argument("--plot-samples", type=int, default=4, help="How many overlay figures to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI.")
    parser.add_argument("--objectness-threshold", type=float, default=0.2, help="Decoder objectness threshold.")
    parser.add_argument("--visibility-threshold", type=float, default=0.45, help="Decoder visibility threshold.")
    parser.add_argument("--max-output-tracks", type=int, default=0, help="Maximum decoded tracks per sample; 0 keeps all.")
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


def _gt_tracks(sample: dict[str, Any], *, raw_len: int) -> list[Track]:
    target = sample["target"]
    gt_valid = target.get("gt_valid")
    if gt_valid is None:
        gt_valid = torch.ones((target["time"].shape[0],), dtype=torch.bool)
    tracks: list[Track] = []
    for local_id, g_idx in enumerate(torch.where(gt_valid.to(torch.bool))[0].tolist()):
        time = target["time"][g_idx]
        visibility = target["visibility"][g_idx]
        direction = "forward" if int(target["direction"][g_idx].item()) == 0 else "reverse"
        speed = float(target["speed"][g_idx].item())
        points: list[TrackPoint] = []
        for ch in torch.where(visibility > 0.5)[0].tolist():
            t_idx = int(round(float(time[ch].item()) * float(max(1, raw_len - 1))))
            points.append(TrackPoint(ch_idx=int(ch), t_idx=int(t_idx), time_s=float(t_idx), offset_m=float(ch), amp=1.0, score=1.0))
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


def _match_tracks(pred_tracks: list[Track], gt_tracks: list[Track], *, tol_samples: int, min_overlap_channels: int, min_overlap_ratio: float) -> tuple[int, int, int]:
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
    ax.set_title(f"vehicle-set overlay | gt={len(gt_tracks)} pred={len(pred_tracks)}")
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
    model_config = VehicleSetModelConfig(**dict(checkpoint.get("model_config", {})))
    model = VehicleSetNet(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    meta = dict(payload.get("meta", {}))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

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
        pred_tracks = decode_vehicle_set_tracks(
            {key: value[0] for key, value in outputs.items()},
            raw_time_bins=int(raw_np.shape[1]),
            fs=float(fs),
            dx_m=float(dx_m),
            raw_window=raw_np,
            snap_search_radius=6000,
            config={
                **asdict(model_config),
                "objectness_threshold": float(args.objectness_threshold),
                "visibility_threshold": float(args.visibility_threshold),
                "max_output_tracks": None if int(args.max_output_tracks) <= 0 else int(args.max_output_tracks),
            },
        )
        gt_tracks = _gt_tracks(sample, raw_len=int(raw_np.shape[1]))
        tp, fp, fn = _match_tracks(
            pred_tracks,
            gt_tracks,
            tol_samples=int(model_config.dedup_time_tol),
            min_overlap_channels=int(model_config.dedup_channel_overlap),
            min_overlap_ratio=float(model_config.dedup_overlap_ratio),
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_count_err += abs(len(pred_tracks) - len(gt_tracks))
        rows.append({"sample_index": int(idx), "gt_count": int(len(gt_tracks)), "pred_count": int(len(pred_tracks)), "tp": int(tp), "fp": int(fp), "fn": int(fn)})
        if idx < int(args.plot_samples):
            _plot_sample(out_dir / f"sample_{idx:02d}.png", raw_window=raw_np, pred_tracks=pred_tracks, gt_tracks=gt_tracks, dpi=int(args.plot_dpi))

    precision = float(total_tp / max(1, total_tp + total_fp))
    recall = float(total_tp / max(1, total_tp + total_fn))
    f1 = float(2.0 * precision * recall / max(1e-6, precision + recall))
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
    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_index", "gt_count", "pred_count", "tp", "fp", "fn"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
