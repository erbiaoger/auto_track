"""Predict and evaluate TrackSlotNet checkpoints on tensor shard datasets.

Purpose:
    This script validates a trained `track_slot` checkpoint directly on the
    `.pt` shards created by `generate_track_slot_dataset.py`. It does not read
    SAC files. Use it when the training data is the clean Gaussian heatmap
    tensor dataset and you want to inspect predicted vehicle slots.

Example:
    uv run python -m autotrack.dl.predict_track_slot_dataset \
        --data-dir datasets/track_slot/train \
        --model models/track_slot_cuda/checkpoint_best.pt \
        --out-dir /tmp/track_slot_prediction_check \
        --device cuda \
        --max-samples 128 \
        --objectness-threshold 0.3 \
        --visibility-threshold 0.3

Arguments:
    --data-dir must contain `meta.json` and `shard_*.pt`.
    --model points to a TrackSlotNet checkpoint produced by `train_track_slot.py`.
    --max-samples limits how many shard samples are evaluated; use 0 for all.
    --max-csv-samples limits detailed per-channel CSV output; use 0 to disable.
    --objectness-threshold and --visibility-threshold control which predicted
    slots and channel points are written to CSV.

Outputs:
    <out-dir>/summary.json
        Loss, detection metrics, count metrics, checkpoint metadata, and paths.
    <out-dir>/sample_summary.csv
        One row per evaluated sample with GT count and filtered prediction count.
    <out-dir>/predicted_tracks.csv
        Per-channel rows for predicted vehicle slots on the first CSV samples.
    <out-dir>/ground_truth_tracks.csv
        Per-channel rows for GT vehicle trajectories on the first CSV samples,
        unless `--no-ground-truth-csv` is set.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from autotrack.dl.track_slot_model import (
    load_checkpoint_model,
    track_slot_detection_metrics,
    track_slot_set_loss,
)
from autotrack.dl.trajectory_set_model import LABEL_TO_DIRECTION, auto_torch_device, move_targets_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict TrackSlotNet outputs from generated .pt tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing meta.json and .pt shards.")
    parser.add_argument("--model", required=True, type=Path, help="TrackSlotNet checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary and CSV files.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum evaluated samples; 0 evaluates all samples.")
    parser.add_argument("--max-csv-samples", type=int, default=32, help="Detailed CSV sample limit; 0 disables detailed CSV.")
    parser.add_argument("--objectness-threshold", type=float, default=0.5, help="Predicted slot objectness threshold.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5, help="Predicted per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for a predicted track.")
    parser.add_argument("--max-predicted-tracks", type=int, default=96, help="Maximum tracks kept per sample for CSV/counts.")
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "greedy"], help="Metric matching strategy.")
    parser.add_argument("--no-object-weight", type=float, default=0.05, help="Unmatched slot weight for set loss.")
    parser.add_argument("--metric-point-threshold", type=float, default=0.05, help="Normalized time-error threshold for TP.")
    parser.add_argument(
        "--speed-norm-kmh",
        type=float,
        default=0.0,
        help="Speed normalization denominator; <=0 reads checkpoint/meta and falls back to 150.",
    )
    parser.add_argument("--no-ground-truth-csv", action="store_true", help="Do not write ground_truth_tracks.csv.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


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


def _targets_from_payload(payload: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    targets = {
        "time": payload["time"][idx].to(torch.float32),
        "visibility": payload["visibility"][idx].to(torch.float32),
        "direction": payload["direction"][idx].to(torch.long),
        "speed": payload["speed"][idx].to(torch.float32),
        "gt_valid": payload["gt_valid"][idx].to(torch.bool),
    }
    targets["gt_count"] = targets["gt_valid"].sum(dim=1).to(torch.long)
    return targets


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    max_samples: int,
) -> Iterator[tuple[list[int], torch.Tensor, dict[str, torch.Tensor]]]:
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
            yield sample_indices, payload["x"][idx].to(torch.float32), _targets_from_payload(payload, idx)
        global_start += n


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _speed_norm(args: argparse.Namespace, checkpoint: dict[str, Any], meta: dict[str, Any]) -> float:
    if float(args.speed_norm_kmh) > 0.0:
        return float(args.speed_norm_kmh)
    dataset_config = dict(checkpoint.get("dataset_config", {}))
    if "speed_norm_kmh" in dataset_config:
        return float(dataset_config["speed_norm_kmh"])
    return float(meta.get("speed_norm_kmh", 150.0))


def _active_predictions(
    outputs_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    objectness_threshold: float,
    visibility_threshold: float,
    min_visible_channels: int,
    max_predicted_tracks: int,
    speed_norm_kmh: float,
) -> list[dict[str, Any]]:
    obj = torch.sigmoid(outputs_cpu["objectness_logits"][batch_index])
    vis = torch.sigmoid(outputs_cpu["visibility_logits"][batch_index])
    time_norm = outputs_cpu["time"][batch_index]
    direction = torch.argmax(outputs_cpu["direction_logits"][batch_index], dim=-1)
    speed = outputs_cpu["speed"][batch_index]
    limit = min(int(max_predicted_tracks), int(obj.shape[0]))
    order = torch.argsort(obj, descending=True)[:limit]

    predictions: list[dict[str, Any]] = []
    for rank, slot_tensor in enumerate(order.tolist()):
        slot = int(slot_tensor)
        score = float(obj[slot].item())
        if score < float(objectness_threshold):
            continue
        channels = torch.where(vis[slot] >= float(visibility_threshold))[0].tolist()
        if len(channels) < int(min_visible_channels):
            continue
        direction_label = int(direction[slot].item())
        predictions.append(
            {
                "rank": int(rank),
                "slot": slot,
                "score": score,
                "direction_label": direction_label,
                "direction": LABEL_TO_DIRECTION.get(direction_label, str(direction_label)),
                "speed_kmh": float(speed[slot].item() * float(speed_norm_kmh)),
                "channels": [int(ch) for ch in channels],
                "time_norm": time_norm[slot],
                "visibility_prob": vis[slot],
            }
        )
    return predictions


def _write_prediction_rows(
    writer: csv.DictWriter,
    *,
    sample_index: int,
    predictions: list[dict[str, Any]],
) -> None:
    for pred_id, pred in enumerate(predictions):
        for ch in pred["channels"]:
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "pred_track_id": int(pred_id),
                    "slot": int(pred["slot"]),
                    "rank": int(pred["rank"]),
                    "score": f"{float(pred['score']):.6f}",
                    "direction": str(pred["direction"]),
                    "speed_kmh": f"{float(pred['speed_kmh']):.6f}",
                    "channel": int(ch),
                    "time_norm": f"{float(pred['time_norm'][ch].item()):.8f}",
                    "visibility_prob": f"{float(pred['visibility_prob'][ch].item()):.6f}",
                }
            )


def _write_ground_truth_rows(
    writer: csv.DictWriter,
    *,
    sample_index: int,
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    speed_norm_kmh: float,
) -> None:
    valid = targets_cpu["gt_valid"][batch_index]
    valid_indices = torch.where(valid)[0].tolist()
    for gt_id, gt_idx in enumerate(valid_indices):
        direction_label = int(targets_cpu["direction"][batch_index, gt_idx].item())
        direction = LABEL_TO_DIRECTION.get(direction_label, str(direction_label))
        speed_kmh = float(targets_cpu["speed"][batch_index, gt_idx].item() * float(speed_norm_kmh))
        visible_channels = torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist()
        for ch in visible_channels:
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "gt_track_id": int(gt_id),
                    "direction": direction,
                    "speed_kmh": f"{speed_kmh:.6f}",
                    "channel": int(ch),
                    "time_norm": f"{float(targets_cpu['time'][batch_index, gt_idx, ch].item()):.8f}",
                }
            )


def _weighted_add(
    sums: defaultdict[str, float],
    weights: defaultdict[str, float],
    metrics: dict[str, float],
    weight: int,
) -> None:
    for key, value in metrics.items():
        value_f = float(value)
        if math.isfinite(value_f):
            sums[str(key)] += value_f * float(weight)
            weights[str(key)] += float(weight)


def _weighted_mean(sums: defaultdict[str, float], weights: defaultdict[str, float]) -> dict[str, float]:
    return {key: float(sums[key] / max(1e-12, weights[key])) for key in sorted(sums) if weights[key] > 0.0}


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {data_dir / 'meta.json'}")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be > 0")

    device = _resolve_device(str(args.device))
    model, checkpoint = load_checkpoint_model(args.model, device=device)
    if str(checkpoint.get("model_family", "track_slot")) != "track_slot":
        raise ValueError(f"Checkpoint is not track_slot: model_family={checkpoint.get('model_family')}")
    speed_norm_kmh = _speed_norm(args, checkpoint, meta)

    summary_path = out_dir / "summary.json"
    sample_summary_path = out_dir / "sample_summary.csv"
    pred_csv_path = out_dir / "predicted_tracks.csv"
    gt_csv_path = out_dir / "ground_truth_tracks.csv"

    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_weights: defaultdict[str, float] = defaultdict(float)
    loss_sums: defaultdict[str, float] = defaultdict(float)
    loss_weights: defaultdict[str, float] = defaultdict(float)
    aggregate_tp = 0.0
    aggregate_pred_obj = 0.0
    aggregate_gt = 0.0
    filtered_pred_total = 0
    filtered_gt_total = 0
    filtered_count_abs_error = 0.0
    filtered_count_exact = 0
    sample_count = 0
    batch_count = 0
    csv_sample_count = 0
    t0 = time.perf_counter()

    pred_fields = [
        "sample_index",
        "pred_track_id",
        "slot",
        "rank",
        "score",
        "direction",
        "speed_kmh",
        "channel",
        "time_norm",
        "visibility_prob",
    ]
    gt_fields = ["sample_index", "gt_track_id", "direction", "speed_kmh", "channel", "time_norm"]
    sample_fields = [
        "sample_index",
        "gt_count",
        "pred_count",
        "max_objectness",
        "mean_objectness",
        "top_score",
    ]

    with sample_summary_path.open("w", newline="", encoding="utf-8") as sample_fp, pred_csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as pred_fp:
        sample_writer = csv.DictWriter(sample_fp, fieldnames=sample_fields)
        pred_writer = csv.DictWriter(pred_fp, fieldnames=pred_fields)
        sample_writer.writeheader()
        pred_writer.writeheader()
        gt_fp = None
        gt_writer: Optional[csv.DictWriter] = None
        if not bool(args.no_ground_truth_csv):
            gt_fp = gt_csv_path.open("w", newline="", encoding="utf-8")
            gt_writer = csv.DictWriter(gt_fp, fieldnames=gt_fields)
            gt_writer.writeheader()
        try:
            model.eval()
            with torch.no_grad():
                for sample_indices, x_cpu, targets_cpu in _iter_batches(
                    data_dir,
                    shards,
                    batch_size=int(args.batch_size),
                    max_samples=int(args.max_samples),
                ):
                    batch_n = int(x_cpu.shape[0])
                    non_blocking = str(device).startswith("cuda")
                    x = x_cpu.to(device=device, non_blocking=non_blocking)
                    targets = move_targets_to_device(targets_cpu, device, non_blocking=non_blocking)
                    outputs = model(x)
                    _, loss_metrics = track_slot_set_loss(
                        outputs,
                        targets,
                        no_object_weight=float(args.no_object_weight),
                        matcher=str(args.matcher),
                        collect_metrics=True,
                    )
                    det_metrics = track_slot_detection_metrics(
                        outputs,
                        targets,
                        objectness_threshold=float(args.objectness_threshold),
                        point_threshold=float(args.metric_point_threshold),
                        matcher=str(args.matcher),
                    )
                    _weighted_add(loss_sums, loss_weights, loss_metrics, batch_n)
                    _weighted_add(metric_sums, metric_weights, det_metrics, batch_n)
                    aggregate_tp += float(det_metrics.get("track_tp", 0.0))
                    aggregate_pred_obj += float(det_metrics.get("pred_count", 0.0))
                    aggregate_gt += float(det_metrics.get("gt_count", 0.0))

                    outputs_cpu = {
                        key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()
                    }
                    obj_cpu = torch.sigmoid(outputs_cpu["objectness_logits"])
                    for b, sample_index in enumerate(sample_indices):
                        predictions = _active_predictions(
                            outputs_cpu,
                            b,
                            objectness_threshold=float(args.objectness_threshold),
                            visibility_threshold=float(args.visibility_threshold),
                            min_visible_channels=int(args.min_visible_channels),
                            max_predicted_tracks=int(args.max_predicted_tracks),
                            speed_norm_kmh=float(speed_norm_kmh),
                        )
                        gt_count = int(targets_cpu["gt_valid"][b].sum().item())
                        pred_count = int(len(predictions))
                        filtered_gt_total += gt_count
                        filtered_pred_total += pred_count
                        filtered_count_abs_error += abs(pred_count - gt_count)
                        filtered_count_exact += int(pred_count == gt_count)
                        top_score = float(predictions[0]["score"]) if predictions else float("nan")
                        sample_writer.writerow(
                            {
                                "sample_index": int(sample_index),
                                "gt_count": gt_count,
                                "pred_count": pred_count,
                                "max_objectness": f"{float(obj_cpu[b].max().item()):.6f}",
                                "mean_objectness": f"{float(obj_cpu[b].mean().item()):.6f}",
                                "top_score": "" if not math.isfinite(top_score) else f"{top_score:.6f}",
                            }
                        )
                        if int(args.max_csv_samples) > 0 and csv_sample_count < int(args.max_csv_samples):
                            _write_prediction_rows(pred_writer, sample_index=int(sample_index), predictions=predictions)
                            if gt_writer is not None:
                                _write_ground_truth_rows(
                                    gt_writer,
                                    sample_index=int(sample_index),
                                    targets_cpu=targets_cpu,
                                    batch_index=b,
                                    speed_norm_kmh=float(speed_norm_kmh),
                                )
                            csv_sample_count += 1
                    sample_count += batch_n
                    batch_count += 1
        finally:
            if gt_fp is not None:
                gt_fp.close()

    precision = float(aggregate_tp / max(1.0, aggregate_pred_obj))
    recall = float(aggregate_tp / max(1.0, aggregate_gt))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    filtered_count_mae = float(filtered_count_abs_error / max(1, sample_count))
    filtered_count_acc = float(filtered_count_exact / max(1, sample_count))

    summary = {
        "mode": "track_slot_tensor_shard_prediction",
        "data_dir": str(data_dir),
        "model": str(Path(args.model).expanduser()),
        "model_family": str(checkpoint.get("model_family", "track_slot")),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "device": device,
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "thresholds": {
            "objectness_threshold": float(args.objectness_threshold),
            "visibility_threshold": float(args.visibility_threshold),
            "min_visible_channels": int(args.min_visible_channels),
            "metric_point_threshold": float(args.metric_point_threshold),
        },
        "loss_metrics": _weighted_mean(loss_sums, loss_weights),
        "batch_mean_detection_metrics": _weighted_mean(metric_sums, metric_weights),
        "aggregate_detection_metrics": {
            "track_precision": precision,
            "track_recall": recall,
            "track_f1": f1,
            "track_tp": float(aggregate_tp),
            "pred_count_objectness_only": float(aggregate_pred_obj),
            "gt_count": float(aggregate_gt),
        },
        "filtered_count_metrics": {
            "pred_count": int(filtered_pred_total),
            "gt_count": int(filtered_gt_total),
            "count_mae": filtered_count_mae,
            "count_acc": filtered_count_acc,
        },
        "outputs": {
            "summary_json": str(summary_path),
            "sample_summary_csv": str(sample_summary_path),
            "predicted_tracks_csv": str(pred_csv_path),
            "ground_truth_tracks_csv": None if bool(args.no_ground_truth_csv) else str(gt_csv_path),
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "done: "
        f"samples={sample_count}, filtered_pred={filtered_pred_total}, gt={filtered_gt_total}, "
        f"f1={f1:.3f}, count_mae={filtered_count_mae:.2f}, out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
