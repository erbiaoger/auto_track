"""Train the query-based vehicle set network on a multi-vehicle benchmark."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autotrack.dl.vehicle_set_net import (
    VehicleSetModelConfig,
    VehicleSetNet,
    save_checkpoint,
    vehicle_set_loss,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vehicle set network on a benchmark .pt file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--lr", type=float, default=8e-4, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--base-dim", type=int, default=32, help="Backbone base dimension.")
    parser.add_argument("--hidden-dim", type=int, default=96, help="Backbone hidden dimension.")
    parser.add_argument("--query-dim", type=int, default=128, help="Decoder feedforward dimension.")
    parser.add_argument("--num-queries", type=int, default=32, help="Number of set queries.")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder layers.")
    parser.add_argument("--decoder-heads", type=int, default=4, help="Transformer decoder heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Validation split fraction.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable CUDA AMP.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=10, help="Print batch progress every N batches.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit benchmark samples; 0 means all.")
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


class _BenchmarkDataset(Dataset):
    def __init__(self, payload: dict[str, Any], *, max_samples: int = 0):
        samples = list(payload.get("samples", []))
        if int(max_samples) > 0:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError("benchmark contains no samples")
        self.samples = samples
        self.meta = dict(payload.get("meta", {}))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        return {
            "x": sample["x"].to(torch.float32),
            "target": {k: v.to(torch.float32) if v.dtype.is_floating_point else v.clone() for k, v in sample["target"].items()},
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    xs = torch.stack([item["x"] for item in batch], dim=0)
    targets = [item["target"] for item in batch]
    return {"x": xs, "targets": targets}


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
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def _save_checkpoint(path: Path, model: VehicleSetNet, config: VehicleSetModelConfig, meta: dict[str, Any], optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "meta": _json_ready(meta),
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    dataset = _BenchmarkDataset(payload, max_samples=int(args.max_samples))
    n_channels = int(dataset[0]["x"].shape[-2])
    model_config = VehicleSetModelConfig(
        n_channels=n_channels,
        base_dim=int(args.base_dim),
        hidden_dim=int(args.hidden_dim),
        query_dim=int(args.query_dim),
        num_queries=int(args.num_queries),
        decoder_layers=int(args.decoder_layers),
        decoder_heads=int(args.decoder_heads),
        dropout=float(args.dropout),
    )
    model = VehicleSetNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)

    if not bool(args.no_val):
        val_count = int(round(len(dataset) * max(0.0, min(0.9, float(args.val_fraction)))))
        val_count = min(max(0, len(dataset) - 1), val_count)
        if val_count > 0:
            split = len(dataset) - val_count
            train_set, val_set = torch.utils.data.random_split(dataset, [split, val_count], generator=torch.Generator().manual_seed(int(args.seed)))
            loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
            val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
        else:
            val_loader = None
    else:
        val_loader = None

    use_amp = bool(args.amp and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "vehicle_set_benchmark",
        "benchmark_file": str(Path(args.benchmark_file).expanduser()),
        "benchmark_meta": _json_ready(dataset.meta),
        "model_config": asdict(model_config),
        "train_args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "device": device,
        "created_at_unix": time.time(),
        "num_samples": len(dataset),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(x)
                loss, metrics = vehicle_set_loss(outputs, batch["targets"], speed_norm_kmh=float(model_config.speed_norm_kmh))
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * int(x.shape[0])
            count += int(x.shape[0])
            if int(args.log_every) > 0 and (batch_idx == 1 or batch_idx % int(args.log_every) == 0 or batch_idx == len(loader)):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} loss={float(loss.item()):.4f} "
                    f"obj={float(metrics['loss_objectness']):.4f} vis={float(metrics['loss_visibility']):.4f} "
                    f"time={float(metrics['loss_time']):.4f}",
                    flush=True,
                )
        train_loss = total_loss / max(1, count)
        val_loss = None
        if val_loader is not None:
            model.eval()
            total = 0.0
            n = 0
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["x"].to(device)
                    if x.ndim == 3:
                        x = x.unsqueeze(1)
                    outputs = model(x)
                    loss, _ = vehicle_set_loss(outputs, batch["targets"], speed_norm_kmh=float(model_config.speed_norm_kmh))
                    total += float(loss.item()) * int(x.shape[0])
                    n += int(x.shape[0])
            val_loss = total / max(1, n)

        row = {"epoch": int(epoch), "train_loss": float(train_loss)}
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
        history.append(row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        current = float(val_loss if val_loss is not None else train_loss)
        if (epoch % int(max(1, args.checkpoint_every)) == 0) or epoch == int(args.epochs):
            _save_checkpoint(out_dir / "checkpoint_last.pt", model, model_config, dataset.meta, optimizer, epoch, {"loss": float(train_loss), "val_loss": float(val_loss) if val_loss is not None else None})
            print(f"Saved checkpoint: {out_dir / 'checkpoint_last.pt'}", flush=True)
            if current < best_loss:
                best_loss = current
                _save_checkpoint(out_dir / "checkpoint_best.pt", model, model_config, dataset.meta, optimizer, epoch, {"loss": float(train_loss), "val_loss": float(val_loss) if val_loss is not None else None})
                print(f"Saved new best checkpoint: {out_dir / 'checkpoint_best.pt'}", flush=True)

    summary = {"best_loss": float(best_loss), "epochs": int(args.epochs), "device": device, "checkpoint_best": str(out_dir / "checkpoint_best.pt")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
