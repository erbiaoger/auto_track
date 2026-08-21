"""Train the compact slot model directly on a multi-vehicle benchmark file."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

from autotrack.dl.compact_slot_model import (
    ModelConfig,
    TrackSlotPredictor,
    auto_torch_device,
    save_checkpoint,
    track_slot_detection_metrics,
    track_slot_set_loss,
)
from autotrack.dl.multi_vehicle_benchmark_dataset import MultiVehicleBenchmarkDataset, benchmark_json_ready, stack_benchmark_batch
from autotrack.dl.trajectory_set_model import WindowDatasetConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the compact slot model on a multi-vehicle benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max-tracks", type=int, default=32, help="Maximum predicted slots.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dimension.")
    parser.add_argument("--num-heads", type=int, default=4, help="Slot attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Model dropout.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--amp", default="auto", choices=["auto", "on", "off"], help="Use CUDA AMP.")
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"], help="CUDA AMP dtype.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Fraction of samples reserved for validation.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split and use the whole benchmark for training.")
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "greedy"], help="Slot-to-GT assignment.")
    parser.add_argument("--no-object-weight", type=float, default=0.05, help="Object loss weight for unmatched slots.")
    parser.add_argument("--count-loss-weight", type=float, default=0.05, help="Soft count loss weight.")
    parser.add_argument("--monotonic-loss-weight", type=float, default=0.5, help="Monotonicity penalty weight.")
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.1, help="Second-difference penalty weight.")
    parser.add_argument("--visibility-negative-weight", type=float, default=2.0, help="Visibility BCE weight for invisible channels.")
    parser.add_argument("--metric-objectness-threshold", type=float, default=0.5, help="Objectness threshold for metrics.")
    parser.add_argument("--metric-point-threshold", type=float, default=0.12, help="Normalized time error threshold for TP metrics.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--resume-model-only", action="store_true", help="Load model weights but reset optimizer.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from <out-dir>/checkpoint_last.pt if present.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=10, help="Print batch progress every N batches; 0 prints epoch summaries only.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit benchmark samples; 0 means all.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _append_history_row(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    out: dict[str, float] = {}
    for key in keys:
        values = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
        if values:
            out[key] = float(sum(values) / len(values))
    return out


def _move_batch_to_device(
    x: torch.Tensor,
    targets: dict[str, torch.Tensor],
    device: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    non_blocking = str(device).startswith("cuda")
    x = x.to(device=device, non_blocking=non_blocking)
    moved = {key: value.to(device=device, non_blocking=non_blocking) if torch.is_tensor(value) else value for key, value in targets.items()}
    if "gt_valid" in moved:
        moved["gt_valid"] = moved["gt_valid"].to(torch.bool)
    return x, moved


def _resolve_resume_path(args: argparse.Namespace) -> Path | None:
    if args.resume is not None:
        path = Path(args.resume).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    if not bool(args.auto_resume):
        return None
    candidate = Path(args.out_dir).expanduser() / "checkpoint_last.pt"
    return candidate if candidate.is_file() else None


def _checkpoint_loss(checkpoint: dict[str, Any], fallback: float = float("inf")) -> float:
    metrics = dict(checkpoint.get("metrics", {}))
    for key in ("val_loss", "loss"):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            loss = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(loss):
            return loss
    return float(fallback)


def _initial_best_loss(out_dir: Path, resume_checkpoint: dict[str, Any] | None) -> float:
    best_loss = _checkpoint_loss(resume_checkpoint) if resume_checkpoint is not None else float("inf")
    best_path = out_dir / "checkpoint_best.pt"
    if best_path.is_file():
        try:
            best_checkpoint = torch.load(str(best_path), map_location="cpu", weights_only=False)
            best_loss = min(best_loss, _checkpoint_loss(best_checkpoint, best_loss))
        except Exception as exc:  # pragma: no cover
            print(f"Could not read existing best checkpoint {best_path}: {exc}", flush=True)
    return float(best_loss)


def _evaluate(
    model: TrackSlotPredictor,
    loader: DataLoader,
    args: argparse.Namespace,
    device: str,
) -> dict[str, float]:
    model.eval()
    metrics_items: list[dict[str, float]] = []
    with torch.no_grad():
        for x, targets in loader:
            x, targets = _move_batch_to_device(x, targets, device)
            outputs = model(x)
            _, metrics = track_slot_set_loss(
                outputs,
                targets,
                no_object_weight=float(args.no_object_weight),
                matcher=str(args.matcher),
                count_loss_weight=float(args.count_loss_weight),
                monotonic_loss_weight=float(args.monotonic_loss_weight),
                smoothness_loss_weight=float(args.smoothness_loss_weight),
                visibility_negative_weight=float(args.visibility_negative_weight),
                collect_metrics=True,
            )
            metrics.update(
                track_slot_detection_metrics(
                    outputs,
                    targets,
                    objectness_threshold=float(args.metric_objectness_threshold),
                    point_threshold=float(args.metric_point_threshold),
                    matcher=str(args.matcher),
                )
            )
            metrics_items.append(metrics)
    return _mean_metrics(metrics_items)


def main() -> int:
    args = parse_args()
    benchmark = MultiVehicleBenchmarkDataset(args.benchmark_file, max_samples=int(args.max_samples))
    all_samples = list(range(len(benchmark)))
    max_gt = 0
    for sample in benchmark.samples:
        target_time = sample["target"].get("time")
        if torch.is_tensor(target_time):
            max_gt = max(max_gt, int(target_time.shape[0]))
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))

    if bool(args.no_val):
        train_indices = all_samples
        val_indices: list[int] = []
    else:
        val_count = int(round(len(all_samples) * max(0.0, min(0.9, float(args.val_fraction)))))
        val_count = min(max(0, len(all_samples) - 1), val_count)
        if val_count <= 0:
            train_indices = all_samples
            val_indices = []
        else:
            split = len(all_samples) - val_count
            train_indices = all_samples[:split]
            val_indices = all_samples[split:]

    sample_x = benchmark[0][0]
    if sample_x.ndim != 3:
        raise ValueError(f"Expected x to have shape [1, C, T], got {tuple(sample_x.shape)}")
    model_config = ModelConfig(
        n_channels=int(sample_x.shape[1]),
        in_channels=int(sample_x.shape[0]),
        max_tracks=int(max(int(args.max_tracks), max_gt + 4)),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        dropout=float(args.dropout),
    )
    dataset_config = WindowDatasetConfig(
        window_seconds=float(benchmark.meta.payload.get("window_seconds", 120.0)),
        time_downsample=int(benchmark.meta.payload.get("time_downsample", 10)),
        samples_per_folder=int(len(benchmark)),
        min_visible_channels=int(benchmark.meta.payload.get("min_visible_channels", 3)),
        speed_norm_kmh=float(benchmark.meta.payload.get("speed_norm_kmh", 150.0)),
        clip_ratio=float(benchmark.meta.payload.get("clip_ratio", 1.35)),
        input_mode=str(benchmark.meta.payload.get("input_mode", "raw")),
        seed=int(benchmark.meta.payload.get("seed", args.seed)),
    )

    train_subset = Subset(benchmark, train_indices)
    val_subset = Subset(benchmark, val_indices) if val_indices else None
    train_loader = DataLoader(train_subset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=stack_benchmark_batch)
    val_loader = DataLoader(val_subset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=stack_benchmark_batch) if val_subset is not None else None

    resume_path = _resolve_resume_path(args)
    resume_checkpoint: dict[str, Any] | None = None
    resume_epoch = 0
    if resume_path is not None:
        resume_checkpoint = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        model_config = ModelConfig(**dict(resume_checkpoint.get("model_config", {})))

    model = TrackSlotPredictor(model_config).to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if resume_checkpoint is not None and not bool(args.resume_model_only) and "optimizer_state" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
    use_amp = (str(args.amp) == "on") or (str(args.amp) == "auto" and str(device).startswith("cuda"))
    amp_dtype = torch.float16 if str(args.amp_dtype) == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and amp_dtype == torch.float16))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "compact_slot_benchmark",
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "benchmark_meta": benchmark_json_ready(benchmark.meta.payload),
        "model_config": asdict(model_config),
        "train_args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "resolved_resume": str(resume_path) if resume_path is not None else "",
        "device": device,
        "created_at_unix": time.time(),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_loss = _initial_best_loss(out_dir, resume_checkpoint)
    start_epoch = resume_epoch + 1 if resume_checkpoint is not None else 1
    if start_epoch > int(args.epochs):
        print(f"Checkpoint epoch={resume_epoch} already satisfies target epochs={int(args.epochs)}", flush=True)
        return 0

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_metrics: list[dict[str, float]] = []
        for batch_idx, (x, targets) in enumerate(train_loader, start=1):
            x, targets = _move_batch_to_device(x, targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp and str(device).startswith("cuda"))):
                outputs = model(x, targets=targets)
                loss, metrics = track_slot_set_loss(
                    outputs,
                    targets,
                    no_object_weight=float(args.no_object_weight),
                    matcher=str(args.matcher),
                    count_loss_weight=float(args.count_loss_weight),
                    monotonic_loss_weight=float(args.monotonic_loss_weight),
                    smoothness_loss_weight=float(args.smoothness_loss_weight),
                    visibility_negative_weight=float(args.visibility_negative_weight),
                    collect_metrics=True,
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            metrics.update(
                track_slot_detection_metrics(
                    outputs,
                    targets,
                    objectness_threshold=float(args.metric_objectness_threshold),
                    point_threshold=float(args.metric_point_threshold),
                    matcher=str(args.matcher),
                )
            )
            epoch_metrics.append(metrics)
            if int(args.log_every) > 0 and (batch_idx == 1 or batch_idx % int(args.log_every) == 0 or batch_idx == len(train_loader)):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(train_loader):04d} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"f1={metrics.get('track_f1', float('nan')):.3f} "
                    f"cnt_mae={metrics.get('count_mae', float('nan')):.2f} "
                    f"matched={metrics.get('matched', 0.0):.0f} gt={metrics.get('gt', 0.0):.0f}",
                    flush=True,
                )

        train_metrics = _mean_metrics(epoch_metrics)
        train_metrics["epoch"] = float(epoch)
        train_metrics["elapsed_seconds"] = float(time.perf_counter() - t0)
        print(
            f"epoch={epoch:03d} loss={train_metrics.get('loss', float('nan')):.4f} "
            f"f1={train_metrics.get('track_f1', float('nan')):.3f} "
            f"cnt_mae={train_metrics.get('count_mae', float('nan')):.2f} "
            f"gt={train_metrics.get('gt', 0.0):.1f} matched={train_metrics.get('matched', 0.0):.1f}",
            flush=True,
        )

        val_metrics: dict[str, float] = {}
        if val_loader is not None:
            val_metrics = _evaluate(model, val_loader, args, device)
            print(
                f"epoch={epoch:03d} val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"val_f1={val_metrics.get('track_f1', float('nan')):.3f} "
                f"val_cnt_mae={val_metrics.get('count_mae', float('nan')):.2f}",
                flush=True,
            )

        history_row: dict[str, float | int | str] = {"epoch": int(epoch), "elapsed_seconds": float(train_metrics["elapsed_seconds"])}
        history_row.update({f"train_{key}": float(value) for key, value in train_metrics.items() if np.isfinite(value)})
        history_row.update({f"val_{key}": float(value) for key, value in val_metrics.items() if np.isfinite(value)})
        _append_history_row(out_dir / "train_history.jsonl", history_row)

        if (epoch % int(max(1, args.checkpoint_every)) == 0) or epoch == int(args.epochs):
            checkpoint_metrics = dict(train_metrics)
            checkpoint_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            last_path = out_dir / "checkpoint_last.pt"
            save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
            print(f"Saved checkpoint: {last_path}", flush=True)
            current_loss = float(val_metrics.get("loss", train_metrics.get("loss", float("inf"))))
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = out_dir / "checkpoint_best.pt"
                save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
                print(f"Saved new best checkpoint: {best_path}", flush=True)

    print(f"Done. Best loss={best_loss:.4f}. Output: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
