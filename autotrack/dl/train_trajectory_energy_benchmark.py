"""Train the trajectory-energy model on a multi-vehicle benchmark file."""

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

from autotrack.dl.trajectory_energy_model import ModelConfig, TrajectoryEnergyNet, trajectory_energy_loss


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the trajectory-energy model on a multi-vehicle benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--base-dim", type=int, default=32, help="Model base dimension.")
    parser.add_argument("--hidden-dim", type=int, default=96, help="Model hidden dimension.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--val-fraction", type=float, default=0.25, help="Validation split fraction.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation split.")
    parser.add_argument("--amp", default="auto", choices=["auto", "on", "off"], help="Use CUDA AMP.")
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"], help="CUDA AMP dtype.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=10, help="Print batch progress every N batches; 0 prints epoch summaries only.")
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
        x = sample["x"].to(torch.float32)
        energy = sample["energy"].to(torch.float32)
        union_visibility = sample["union_visibility"].to(torch.float32)
        return {
            "x": x,
            "energy": energy,
            "visibility": union_visibility,
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch], dim=0) for key in keys}


def _save_checkpoint(path: Path, model: TrajectoryEnergyNet, config: ModelConfig, meta: dict[str, Any], optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, float]) -> None:
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
    sample0 = dataset[0]
    n_channels = int(sample0["x"].shape[-2])
    model_config = ModelConfig(
        n_channels=n_channels,
        base_dim=int(args.base_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    )
    model = TrajectoryEnergyNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
    val_loader = None
    if not bool(args.no_val):
        val_count = int(round(len(dataset) * max(0.0, min(0.9, float(args.val_fraction)))))
        val_count = min(max(0, len(dataset) - 1), val_count)
        if val_count > 0:
            split = len(dataset) - val_count
            train_set, val_set = torch.utils.data.random_split(dataset, [split, val_count], generator=torch.Generator().manual_seed(int(args.seed)))
            loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)
            val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)

    use_amp = (str(args.amp) == "on") or (str(args.amp) == "auto" and str(device).startswith("cuda"))
    amp_dtype = torch.float16 if str(args.amp_dtype) == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and amp_dtype == torch.float16))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "trajectory_energy_benchmark",
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
        epoch_loss = 0.0
        count = 0
        for batch_idx, batch in enumerate(loader, start=1):
            x = batch["x"].to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            if x.ndim != 4:
                raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
            energy_tgt = batch["energy"].to(device)
            vis_tgt = batch["visibility"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp and str(device).startswith("cuda"))):
                outputs = model(x)
                loss, metrics = trajectory_energy_loss(
                    outputs,
                    target_energy=energy_tgt,
                    target_visibility=vis_tgt,
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item()) * int(x.shape[0])
            count += int(x.shape[0])
            if int(args.log_every) > 0 and (batch_idx == 1 or batch_idx % int(args.log_every) == 0 or batch_idx == len(loader)):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={float(loss.item()):.4f} energy={float(metrics['loss_energy']):.4f} vis={float(metrics['loss_visibility']):.4f}",
                    flush=True,
                )
        train_loss = epoch_loss / max(1, count)
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
                    loss, _ = trajectory_energy_loss(outputs, target_energy=batch["energy"].to(device), target_visibility=batch["visibility"].to(device))
                    total += float(loss.item()) * int(x.shape[0])
                    n += int(x.shape[0])
            val_loss = total / max(1, n)

        row = {"epoch": int(epoch), "train_loss": float(train_loss)}
        if val_loss is not None:
            row["val_loss"] = float(val_loss)
        history.append(row)
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

        ckpt_metrics = {"loss": float(train_loss)}
        if val_loss is not None:
            ckpt_metrics["val_loss"] = float(val_loss)
        if (epoch % int(max(1, args.checkpoint_every)) == 0) or epoch == int(args.epochs):
            _save_checkpoint(out_dir / "checkpoint_last.pt", model, model_config, dataset.meta, optimizer, epoch, ckpt_metrics)
            print(f"Saved checkpoint: {out_dir / 'checkpoint_last.pt'}", flush=True)
            current = float(val_loss if val_loss is not None else train_loss)
            if current < best_loss:
                best_loss = current
                _save_checkpoint(out_dir / "checkpoint_best.pt", model, model_config, dataset.meta, optimizer, epoch, ckpt_metrics)
                print(f"Saved new best checkpoint: {out_dir / 'checkpoint_best.pt'}", flush=True)

    summary = {"best_loss": float(best_loss), "epochs": int(args.epochs), "device": device, "checkpoint_best": str(out_dir / "checkpoint_best.pt")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
