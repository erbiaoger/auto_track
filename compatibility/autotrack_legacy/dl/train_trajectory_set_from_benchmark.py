"""Train the trajectory-set predictor from a multi-vehicle benchmark file."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from autotrack.dl.trajectory_set_model import (
    ModelConfig,
    TrajectorySetPredictor,
    auto_torch_device,
    save_checkpoint,
    WindowDatasetConfig,
    targets_to_batched,
    trajectory_detection_metrics,
    trajectory_set_loss,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trajectory-set model from a multi-vehicle benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="multi_vehicle_benchmark_v1 .pt file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, cpu, mps, auto")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden size")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder layers")
    parser.add_argument("--max-queries", type=int, default=64, help="Maximum trajectory queries")
    parser.add_argument("--trajectory-points", type=int, default=32, help="Polyline points per query")
    parser.add_argument("--pooled-channels", type=int, default=8, help="Backbone pooled channel height")
    parser.add_argument("--pooled-time", type=int, default=128, help="Backbone pooled time length")
    parser.add_argument("--denoising-queries", type=int, default=32, help="Maximum denoising queries")
    parser.add_argument("--dn-point-noise", type=float, default=0.04, help="Noise applied to denoising polylines")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N batches")
    parser.add_argument("--no-object-weight", type=float, default=0.02, help="Loss weight for unmatched queries")
    parser.add_argument("--duplicate-loss-weight", type=float, default=0.2, help="Duplicate query penalty")
    parser.add_argument("--duplicate-distance-tau", type=float, default=0.04, help="Duplicate distance scale")
    parser.add_argument("--denoising-loss-weight", type=float, default=1.0, help="Auxiliary denoising loss weight")
    parser.add_argument("--line-loss-weight", type=float, default=1.0, help="Polyline linearity loss weight")
    parser.add_argument("--slope-smooth-loss-weight", type=float, default=0.25, help="Polyline slope smoothness loss weight")
    parser.add_argument("--matcher", default="greedy", choices=["greedy", "hungarian"], help="Matching strategy")
    return parser.parse_args(argv)


class _BenchmarkDataset(Dataset):
    def __init__(self, payload: dict[str, Any]):
        self.samples = list(payload.get("samples", []))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[int(index)]
        x = sample["x"].to(torch.float32)
        target = dict(sample["target"])
        if "gt_valid" not in target:
            n_gt = int(target["time"].shape[0])
            target["gt_valid"] = torch.ones((n_gt,), dtype=torch.bool)
        return x, target


def _collate(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raise RuntimeError("collate factory must bind trajectory_points")


def _collate_with_points(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    *,
    trajectory_points: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    n_channels = int(targets[0]["time"].shape[1]) if targets and targets[0]["time"].ndim == 2 else 0
    out = targets_to_batched(targets, n_channels=n_channels, trajectory_points=int(trajectory_points))
    return xs, out


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(sum(item.get(key, 0.0) for item in items) / len(items)) for key in keys}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    device = str(args.device).strip() or auto_torch_device()
    print(f"Using torch device: {device}")

    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    dataset = _BenchmarkDataset(payload)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=partial(_collate_with_points, trajectory_points=int(args.trajectory_points)),
        drop_last=False,
    )
    sample0 = dataset[0]
    n_channels = int(sample0[0].shape[-2])
    model_config = ModelConfig(
        n_channels=n_channels,
        in_channels=int(sample0[0].shape[0]),
        max_queries=int(args.max_queries),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        decoder_layers=int(args.decoder_layers),
        pooled_channels=int(args.pooled_channels),
        pooled_time=int(args.pooled_time),
        trajectory_points=int(args.trajectory_points),
        denoising_queries=int(args.denoising_queries),
        dn_point_noise=float(args.dn_point_noise),
    )
    model = TrajectorySetPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "model_config": asdict(model_config),
        "device": device,
        "created_at_unix": time.time(),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = dict(payload.get("meta", {}))
    dataset_config = WindowDatasetConfig(
        window_seconds=float(meta.get("window_seconds", 120.0)),
        time_downsample=int(meta.get("time_downsample", 10)),
        samples_per_folder=int(max(1, len(dataset))),
        min_visible_channels=int(meta.get("min_visible_channels", 3)),
        speed_norm_kmh=float(meta.get("speed_norm_kmh", 150.0)),
        clip_ratio=float(meta.get("clip_ratio", 1.35)),
        input_mode=str(meta.get("input_mode", "raw")),
        seed=int(meta.get("seed", 42)),
    )

    best_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_metrics: list[dict[str, float]] = []
        t0 = time.perf_counter()
        for batch_idx, (x, targets) in enumerate(loader, start=1):
            x = x.to(device)
            targets = {key: value.to(device) if torch.is_tensor(value) else value for key, value in targets.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, targets=targets)
            loss, metrics = trajectory_set_loss(
                outputs,
                targets,
                no_object_weight=float(args.no_object_weight),
                duplicate_loss_weight=float(args.duplicate_loss_weight),
                duplicate_distance_tau=float(args.duplicate_distance_tau),
                denoising_loss_weight=float(args.denoising_loss_weight),
                line_loss_weight=float(args.line_loss_weight),
                slope_smooth_loss_weight=float(args.slope_smooth_loss_weight),
                matcher=str(args.matcher),
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            epoch_metrics.append(metrics)
            if batch_idx == 1 or batch_idx % int(max(1, args.log_every)) == 0 or batch_idx == len(loader):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"obj={metrics.get('loss_obj', float('nan')):.4f} "
                    f"point={metrics.get('loss_point', float('nan')):.4f} "
                    f"vis={metrics.get('loss_valid', float('nan')):.4f} "
                    f"dir={metrics.get('loss_dir', float('nan')):.4f} "
                    f"speed={metrics.get('loss_speed', float('nan')):.4f} "
                    f"dup={metrics.get('loss_duplicate', float('nan')):.4f} "
                    f"dn={metrics.get('loss_dn', float('nan')):.4f} "
                    f"gt={metrics.get('gt', 0.0):.0f} matched={metrics.get('matched', 0.0):.0f}",
                    flush=True,
                )
        mean_metrics = _mean_metrics(epoch_metrics)
        elapsed = time.perf_counter() - t0
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["elapsed_seconds"] = float(elapsed)
        print(
            f"epoch={epoch:03d} "
            f"loss={mean_metrics.get('loss', float('nan')):.4f} "
            f"obj={mean_metrics.get('loss_obj', float('nan')):.4f} "
            f"point={mean_metrics.get('loss_point', float('nan')):.4f} "
            f"vis={mean_metrics.get('loss_valid', float('nan')):.4f} "
            f"gt={mean_metrics.get('gt', 0.0):.1f} "
            f"matched={mean_metrics.get('matched', 0.0):.1f} "
            f"elapsed={elapsed:.1f}s"
        )
        last_path = out_dir / "checkpoint_last.pt"
        save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, mean_metrics)
        current_loss = float(mean_metrics.get("loss", float("inf")))
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = out_dir / "checkpoint_best.pt"
            save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, mean_metrics)

    summary = {
        "best_loss": float(best_loss),
        "epochs": int(args.epochs),
        "device": device,
        "checkpoint_best": str(out_dir / "checkpoint_best.pt"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
