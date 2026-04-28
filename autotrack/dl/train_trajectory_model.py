"""Train the DAS vehicle trajectory-query model.

用途：
    使用 `simulate_vehicle_sac.py` 生成的 SAC 文件和 `tracks.json` 真值轨迹，
    训练一个 CNN/FPN + Transformer query 模型。模型输入一个 `[channel, time]`
    窗口，输出不定数量车辆轨迹；每个 query 以 MapTR 风格输出一组
    `(channel, time)` polyline 点，代表一辆车。

用例：
    uv run python KF/auto_track/train_trajectory_model.py \
        --data-folder KF/auto_track/data \
        --out-dir KF/auto_track/models/trajectory_query_demo \
        --epochs 20 \
        --batch-size 2 \
        --window-seconds 60 \
        --time-downsample 10

参数说明：
    --data-folder 可重复传入多个模拟数据目录，每个目录必须包含 CH*.sac 和 tracks.json。
    --window-seconds 是训练裁窗长度；--time-downsample 是模型输入的时间降采样倍率。
    --max-queries 是每个窗口最多候选车辆数，实际车辆数由 objectness 自动决定。
    --trajectory-points 是每辆候选车输出的固定 polyline 点数，valid mask 决定哪些点有效。

输出：
    - <out-dir>/checkpoint_last.pt：最近一个 epoch 的模型。
    - <out-dir>/checkpoint_best.pt：训练 loss 最低的模型。
    - <out-dir>/train_config.json：训练配置和模型配置。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from autotrack.dl.trajectory_set_model import (
    ModelConfig,
    SimulatedSacTrajectoryDataset,
    TrajectorySetPredictor,
    WindowDatasetConfig,
    auto_torch_device,
    save_checkpoint,
    trajectory_collate,
    trajectory_set_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a DAS vehicle trajectory-query model.")
    parser.add_argument(
        "--data-folder",
        action="append",
        required=True,
        help="Simulated SAC folder containing CH*.sac and tracks.json. Repeat for multiple folders.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Random training window length in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time-axis stride for model input.")
    parser.add_argument("--samples-per-folder", type=int, default=512, help="Random windows sampled per folder per epoch.")
    parser.add_argument("--min-visible-channels", type=int, default=2, help="Minimum visible points for a GT track.")
    parser.add_argument("--max-queries", type=int, default=128, help="Maximum predicted trajectory queries per window.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Transformer hidden dimension.")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder layer count.")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument(
        "--pooled-channels",
        type=int,
        default=8,
        help="Backbone feature height after pooling. Keep <= 8 on MPS for 50-channel data.",
    )
    parser.add_argument("--pooled-time", type=int, default=128, help="Backbone feature time length after pooling.")
    parser.add_argument("--trajectory-points", type=int, default=32, help="Polyline points predicted per trajectory query.")
    parser.add_argument("--device", default="", help="Torch device: cuda, mps, cpu, or empty for auto.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=10, help="Print training progress every N batches.")
    parser.add_argument("--no-object-weight", type=float, default=0.02, help="Loss weight for unmatched/no-object queries.")
    parser.add_argument("--duplicate-loss-weight", type=float, default=0.2, help="Penalty weight for high-confidence duplicate trajectory queries.")
    parser.add_argument("--duplicate-distance-tau", type=float, default=0.04, help="Normalized trajectory distance scale for duplicate penalty.")
    parser.add_argument("--denoising-loss-weight", type=float, default=1.0, help="Auxiliary loss weight for denoising trajectory queries.")
    parser.add_argument("--line-loss-weight", type=float, default=1.0, help="Soft loss weight for fitting each predicted trajectory to a line.")
    parser.add_argument("--slope-smooth-loss-weight", type=float, default=0.25, help="Soft loss weight for penalizing abrupt local speed changes.")
    parser.add_argument("--denoising-queries", type=int, default=32, help="Maximum denoising GT queries appended during training.")
    parser.add_argument("--dn-point-noise", type=float, default=0.04, help="Normalized coordinate noise added to denoising GT polyline inputs.")
    return parser.parse_args()


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(sum(item.get(key, 0.0) for item in items) / len(items)) for key in keys}


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = str(args.device).strip() or auto_torch_device()
    print(f"Using torch device: {device}")

    dataset_config = WindowDatasetConfig(
        window_seconds=float(args.window_seconds),
        time_downsample=int(max(1, args.time_downsample)),
        samples_per_folder=int(max(1, args.samples_per_folder)),
        min_visible_channels=int(max(1, args.min_visible_channels)),
        seed=int(args.seed),
    )
    dataset = SimulatedSacTrajectoryDataset(args.data_folder, config=dataset_config)
    n_channels = int(dataset.records[0]["data"].shape[0])
    for rec in dataset.records:
        if int(rec["data"].shape[0]) != n_channels:
            raise ValueError("All training folders must have the same channel count for this first model version.")

    model_config = ModelConfig(
        n_channels=n_channels,
        in_channels=2,
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
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=trajectory_collate,
        drop_last=False,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.out_dir / "train_config.json"
    config_payload = {
        "data_folders": [str(Path(p).expanduser()) for p in args.data_folder],
        "dataset_config": asdict(dataset_config),
        "model_config": asdict(model_config),
        "device": device,
        "created_at_unix": time.time(),
    }
    config_path.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        "Dataset loaded: "
        f"folders={len(dataset.records)}, windows_per_epoch={len(dataset)}, "
        f"batch_size={int(args.batch_size)}, batches_per_epoch={len(loader)}, "
        f"channels={n_channels}, window_seconds={dataset_config.window_seconds}, "
        f"time_downsample={dataset_config.time_downsample}"
    )
    print(
        "Model: "
        f"queries={model_config.max_queries}, hidden_dim={model_config.hidden_dim}, "
        f"decoder_layers={model_config.decoder_layers}, pooled=({model_config.pooled_channels}, {model_config.pooled_time}), "
        f"trajectory_points={model_config.trajectory_points}"
    )
    print(f"Writing checkpoints to: {args.out_dir}")

    best_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_metrics: list[dict[str, float]] = []
        log_every = int(max(1, args.log_every))
        for batch_idx, (x, targets) in enumerate(loader, start=1):
            x = x.to(device)
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
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            epoch_metrics.append(metrics)
            if batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"obj={metrics.get('loss_obj', float('nan')):.4f} "
                    f"time={metrics.get('loss_time', float('nan')):.4f} "
                    f"vis={metrics.get('loss_vis', float('nan')):.4f} "
                    f"dup={metrics.get('loss_duplicate', float('nan')):.4f} "
                    f"dn={metrics.get('loss_dn', float('nan')):.4f} "
                    f"line={metrics.get('loss_line', float('nan')):.4f} "
                    f"slope={metrics.get('loss_slope_smooth', float('nan')):.4f} "
                    f"gt={metrics.get('gt', 0.0):.0f} "
                    f"matched={metrics.get('matched', 0.0):.0f}",
                    flush=True,
                )

        mean_metrics = _mean_metrics(epoch_metrics)
        elapsed = time.perf_counter() - t0
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["elapsed_seconds"] = float(elapsed)
        print(
            f"epoch={epoch:03d} "
            f"loss={mean_metrics.get('loss', float('nan')):.4f} "
            f"time={mean_metrics.get('loss_time', float('nan')):.4f} "
            f"obj={mean_metrics.get('loss_obj', float('nan')):.4f} "
            f"dup={mean_metrics.get('loss_duplicate', float('nan')):.4f} "
            f"dn={mean_metrics.get('loss_dn', float('nan')):.4f} "
            f"line={mean_metrics.get('loss_line', float('nan')):.4f} "
            f"slope={mean_metrics.get('loss_slope_smooth', float('nan')):.4f} "
            f"gt={mean_metrics.get('gt', 0.0):.1f} "
            f"matched={mean_metrics.get('matched', 0.0):.1f} "
            f"elapsed={elapsed:.1f}s"
        )

        last_path = args.out_dir / "checkpoint_last.pt"
        save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, mean_metrics)
        current_loss = float(mean_metrics.get("loss", float("inf")))
        print(f"Saved checkpoint: {last_path}", flush=True)
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = args.out_dir / "checkpoint_best.pt"
            save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, mean_metrics)
            print(f"Saved new best checkpoint: {best_path}", flush=True)

    print(f"Done. Best loss={best_loss:.4f}. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
