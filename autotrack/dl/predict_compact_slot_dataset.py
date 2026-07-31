"""Evaluate the compact slot model on shard datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track, TrackPoint
from autotrack.dl.compact_slot_model import (
    LABEL_TO_DIRECTION,
    load_checkpoint_model,
    track_slot_detection_metrics,
    track_slot_set_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the compact slot model on shard datasets.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing meta.json and shard_*.pt.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary and CSV files.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum evaluated samples; 0 evaluates all samples.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of heatmap overlay figures to write; 0 disables plots.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for overlay figures.")
    parser.add_argument("--objectness-threshold", type=float, default=0.15, help="Predicted slot objectness threshold.")
    parser.add_argument("--candidate-objectness-floor", type=float, default=0.05, help="Floor for candidate slots.")
    parser.add_argument("--objectness-count-scale", type=float, default=1.05, help="Soft count scale used to keep a stable number of slots.")
    parser.add_argument("--visibility-threshold", type=float, default=0.35, help="Predicted per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for a predicted track.")
    parser.add_argument("--point-threshold", type=float, default=0.12, help="Normalized time error threshold for a true-positive track.")
    parser.add_argument("--no-ground-truth-csv", action="store_true", help="Do not write ground_truth_tracks.csv.")
    return parser.parse_args()


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    max_samples: int,
) -> Iterator[tuple[list[int], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]]:
    emitted = 0
    global_start = 0
    for shard in shards:
        payload = torch.load(str(data_dir / shard), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        for start in range(0, n, int(batch_size)):
            if int(max_samples) > 0 and emitted >= int(max_samples):
                return
            take = min(int(batch_size), n - start)
            if int(max_samples) > 0:
                take = min(take, int(max_samples) - emitted)
            if take <= 0:
                return
            idx = torch.arange(start, start + take)
            sample_indices = list(range(global_start + start, global_start + start + take))
            emitted += take
            targets = {
                "time": payload["time"][idx].to(torch.float32),
                "visibility": payload["visibility"][idx].to(torch.float32),
                "direction": payload["direction"][idx].to(torch.long),
                "speed": payload["speed"][idx].to(torch.float32),
                "gt_valid": payload["gt_valid"][idx].to(torch.bool),
            }
            yield sample_indices, payload["x"][idx].to(torch.float32), targets, payload["x"][idx].to(torch.float32)
        global_start += n


def _track_rows(tracks: list[Track], sample_index: int, kind: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tr in tracks:
        for point in tr.points:
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "kind": kind,
                    "track_id": int(tr.track_id),
                    "direction": str(tr.direction),
                    "ch_idx": int(point.ch_idx),
                    "t_idx": int(point.t_idx),
                    "time_s": float(point.time_s),
                    "offset_m": float(point.offset_m),
                    "amp": float(point.amp),
                    "score": float(point.score),
                    "track_score": float(tr.total_score),
                    "mean_speed_kmh": float(tr.mean_speed_kmh),
                }
            )
    return rows


def _plot_sample(
    out_path: Path,
    *,
    x: torch.Tensor,
    pred_tracks: list[Track],
    gt_tracks: list[Track],
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), dpi=int(dpi))
    img = x[0].detach().cpu().numpy()
    im = ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
    fig.colorbar(im, ax=ax, label="input value")
    for tr in gt_tracks:
        ax.plot([p.time_s for p in tr.points], [p.ch_idx for p in tr.points], color="lime", linewidth=1.8)
    for tr in pred_tracks:
        ax.plot([p.time_s for p in tr.points], [p.ch_idx for p in tr.points], color="cyan", linewidth=1.8)
    ax.set_xlabel("time index")
    ax.set_ylabel("channel")
    ax.set_title(f"compact slot eval | GT={len(gt_tracks)} Pred={len(pred_tracks)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _gt_tracks_from_targets(targets: dict[str, torch.Tensor], sample_idx: int) -> list[Track]:
    tracks: list[Track] = []
    gt_valid = targets["gt_valid"][sample_idx]
    visible_tracks = torch.where(gt_valid.to(torch.bool))[0].tolist()
    for local_id, g_idx in enumerate(visible_tracks):
        time = targets["time"][sample_idx, g_idx]
        vis = targets["visibility"][sample_idx, g_idx]
        direction = LABEL_TO_DIRECTION.get(int(targets["direction"][sample_idx, g_idx].item()), "forward")
        speed = float(targets["speed"][sample_idx, g_idx].item())
        points: list[TrackPoint] = []
        for ch in torch.where(vis > 0.5)[0].tolist():
            t_idx = int(round(float(time[ch].item()) * max(1, int(time.shape[0]) - 1)))
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=t_idx,
                    time_s=float(t_idx),
                    offset_m=float(ch),
                    amp=1.0,
                    score=1.0,
                )
            )
        if points:
            tracks.append(Track(track_id=local_id, direction=direction, points=points, total_score=float(len(points)), mean_speed_kmh=speed))
    return tracks


def _pred_tracks_from_outputs(
    outputs: dict[str, torch.Tensor],
    x: torch.Tensor,
    sample_idx: int,
    *,
    objectness_threshold: float,
    candidate_objectness_floor: float,
    objectness_count_scale: float,
    visibility_threshold: float,
    min_visible_channels: int,
) -> list[Track]:
    obj = torch.sigmoid(outputs["objectness_logits"][sample_idx]).detach().cpu()
    vis = torch.sigmoid(outputs["visibility_logits"][sample_idx]).detach().cpu()
    time_norm = outputs["time"][sample_idx].detach().cpu()
    direction = torch.argmax(outputs["direction_logits"][sample_idx], dim=-1).detach().cpu()
    speed = outputs["speed"][sample_idx].detach().cpu() * 150.0
    soft_keep = int(round(float(obj.sum().item()) * float(objectness_count_scale)))
    keep_limit = max(1, min(int(obj.shape[0]), soft_keep))
    floor = min(float(objectness_threshold), float(candidate_objectness_floor))
    pool = torch.where(obj >= floor)[0]
    if pool.numel() == 0:
        pool = torch.arange(int(obj.shape[0]))
    order = pool[torch.argsort(obj[pool], descending=True)].tolist()
    if len(order) < keep_limit:
        remaining = [int(i) for i in torch.argsort(obj, descending=True).tolist() if int(i) not in set(order)]
        order = order + remaining
    order = order[:keep_limit]
    tracks: list[Track] = []
    for q_idx in order:
        chs = torch.where(vis[q_idx] >= float(visibility_threshold))[0].tolist()
        if len(chs) < int(min_visible_channels):
            continue
        points: list[TrackPoint] = []
        for ch in chs:
            t_idx = int(round(float(time_norm[q_idx, ch].item()) * max(1, int(time_norm.shape[1]) - 1)))
            t_idx = max(0, min(int(time_norm.shape[1]) - 1, t_idx))
            amp = float(x[sample_idx, 0, ch, max(0, min(int(x.shape[-1]) - 1, t_idx))].item())
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=t_idx,
                    time_s=float(t_idx),
                    offset_m=float(ch),
                    amp=amp,
                    score=float(obj[q_idx].item() * vis[q_idx, ch].item()),
                )
            )
        if len(points) < int(min_visible_channels):
            continue
        tracks.append(
            Track(
                track_id=len(tracks),
                direction=LABEL_TO_DIRECTION.get(int(direction[q_idx].item()), "forward"),
                points=points,
                total_score=float(obj[q_idx].item()),
                mean_speed_kmh=float(speed[q_idx].item()),
            )
        )
    return tracks


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    model, checkpoint = load_checkpoint_model(args.model, device=device)

    sample_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    gt_rows: list[dict[str, Any]] = []
    metrics_items: list[dict[str, float]] = []
    plotted = 0
    sample_count = 0

    with torch.no_grad():
        for sample_indices, x, targets, raw_x in _iter_batches(
            data_dir,
            shards,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
        ):
            x_dev = x.to(device)
            targets_dev = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets.items()}
            outputs = model(x_dev)
            loss, metrics = track_slot_set_loss(
                outputs,
                targets_dev,
                collect_metrics=True,
            )
            metrics.update(
                track_slot_detection_metrics(
                    outputs,
                    targets_dev,
                    objectness_threshold=float(args.objectness_threshold),
                    point_threshold=float(args.point_threshold),
                )
            )
            metrics_items.append(metrics)
            for local_i, sample_idx in enumerate(sample_indices):
                pred_tracks = _pred_tracks_from_outputs(
                    outputs,
                    x_dev,
                    local_i,
                    objectness_threshold=float(args.objectness_threshold),
                    candidate_objectness_floor=float(args.candidate_objectness_floor),
                    objectness_count_scale=float(args.objectness_count_scale),
                    visibility_threshold=float(args.visibility_threshold),
                    min_visible_channels=int(args.min_visible_channels),
                )
                gt_tracks = _gt_tracks_from_targets(targets_dev, local_i)
                sample_rows.append(
                    {
                        "sample_index": int(sample_idx),
                        "pred_track_count": int(len(pred_tracks)),
                        "gt_track_count": int(len(gt_tracks)),
                        "loss": float(loss.detach().cpu()),
                        "track_f1": float(metrics.get("track_f1", float("nan"))),
                        "count_mae": float(metrics.get("count_mae", float("nan"))),
                    }
                )
                pred_rows.extend(_track_rows(pred_tracks, int(sample_idx), "pred"))
                if not bool(args.no_ground_truth_csv):
                    gt_rows.extend(_track_rows(gt_tracks, int(sample_idx), "gt"))
                if plotted < int(args.plot_samples):
                    _plot_sample(out_dir / "plots" / f"sample_{sample_count:06d}.png", x=raw_x[local_i], pred_tracks=pred_tracks, gt_tracks=gt_tracks, dpi=int(args.plot_dpi))
                    plotted += 1
                sample_count += 1

    def _mean(key: str) -> float:
        vals = [float(item[key]) for item in metrics_items if key in item and np.isfinite(float(item[key]))]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    summary = {
        "model": str(args.model),
        "data_dir": str(data_dir),
        "device": device,
        "sample_count": int(sample_count),
        "loss": _mean("loss"),
        "track_f1": _mean("track_f1"),
        "count_mae": _mean("count_mae"),
        "track_precision": _mean("track_precision"),
        "track_recall": _mean("track_recall"),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
        "outputs": {
            "sample_summary_csv": str(out_dir / "sample_summary.csv"),
            "predicted_tracks_csv": str(out_dir / "predicted_tracks.csv"),
            "ground_truth_tracks_csv": str(out_dir / "ground_truth_tracks.csv") if not bool(args.no_ground_truth_csv) else None,
            "plots_dir": str(out_dir / "plots"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    with (out_dir / "sample_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(sample_rows[0].keys()) if sample_rows else ["sample_index"])
        writer.writeheader()
        for row in sample_rows:
            writer.writerow(row)

    with (out_dir / "predicted_tracks.csv").open("w", encoding="utf-8", newline="") as fp:
        fieldnames = ["sample_index", "kind", "track_id", "direction", "ch_idx", "t_idx", "time_s", "offset_m", "amp", "score", "track_score", "mean_speed_kmh"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in pred_rows:
            writer.writerow(row)

    if not bool(args.no_ground_truth_csv):
        with (out_dir / "ground_truth_tracks.csv").open("w", encoding="utf-8", newline="") as fp:
            fieldnames = ["sample_index", "kind", "track_id", "direction", "ch_idx", "t_idx", "time_s", "offset_m", "amp", "score", "track_score", "mean_speed_kmh"]
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in gt_rows:
                writer.writerow(row)

    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
