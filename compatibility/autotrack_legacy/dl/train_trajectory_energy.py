"""Train the trajectory-energy mainline on a single-vehicle benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autotrack.dl.trajectory_energy_model import (
    ModelConfig,
    TrajectoryEnergyNet,
    build_energy_target,
    trajectory_energy_loss,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the trajectory-energy network on a benchmark file.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="single_vehicle_benchmark_v1 .pt file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, cpu, mps, auto")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden-dim", type=int, default=96, help="Model hidden size")
    parser.add_argument("--base-dim", type=int, default=32, help="Model base size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--time-downsample", type=int, default=10, help="Time downsample used by the benchmark")
    parser.add_argument("--time-bins", type=int, default=12000, help="Expected temporal bins in the benchmark target")
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
    def __init__(self, payload: dict[str, Any]):
        self.samples = list(payload.get("samples", []))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        target = sample["target"]
        x = sample["x"].to(torch.float32)
        gt_masks = target.get("gt_masks", None)
        if gt_masks is not None and int(gt_masks.numel()) > 0 and int(gt_masks.shape[0]) > 1:
            energy = sample.get("energy", gt_masks.to(torch.float32).amax(dim=0))
            visibility = sample.get("union_visibility", gt_masks.to(torch.float32).amax(dim=0).amax(dim=-1))
            time = torch.zeros((0, int(x.shape[-2])), dtype=torch.float32)
            direction = torch.zeros((0,), dtype=torch.long)
            speed = torch.zeros((0,), dtype=torch.float32)
        else:
            time = target["time"].to(torch.float32).squeeze(0)
            visibility = target["visibility"].to(torch.float32).squeeze(0)
            direction = target["direction"].to(torch.long).squeeze(0)
            speed = target["speed"].to(torch.float32).squeeze(0)
            energy = target.get("gt_masks", None)
            if energy is None:
                energy = build_energy_target(time, visibility, n_channels=int(time.shape[0]), time_bins=int(sample["x"].shape[-1]))
            else:
                energy = energy.to(torch.float32).squeeze(0)
        return {
            "x": x,
            "time": time,
            "visibility": visibility,
            "direction": direction,
            "speed": speed,
            "energy": energy,
        }


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


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out


def _save_checkpoint(path: Path, model: TrajectoryEnergyNet, config: ModelConfig, meta: dict[str, Any]) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "meta": _json_ready(meta),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    dataset = _BenchmarkDataset(payload)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=0, collate_fn=_collate)

    sample0 = dataset[0]
    n_channels = int(sample0["x"].shape[-2])
    time_bins = int(sample0["energy"].shape[-1])
    model_config = ModelConfig(n_channels=n_channels, base_dim=int(args.base_dim), hidden_dim=int(args.hidden_dim), dropout=float(args.dropout))
    model = TrajectoryEnergyNet(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(int(args.epochs)):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in loader:
            x = batch["x"].to(device)
            if x.ndim == 3:
                x = x.unsqueeze(1)
            elif x.ndim != 4:
                raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
            outputs = model(x)
            has_track_targets = int(batch["time"].shape[1]) > 0
            loss, items = trajectory_energy_loss(
                outputs,
                target_energy=batch["energy"].to(device),
                target_visibility=batch["visibility"].to(device),
                target_time=batch["time"].to(device) if has_track_targets else None,
                target_direction=batch["direction"].to(device) if has_track_targets else None,
                target_speed=batch["speed"].to(device) if has_track_targets else None,
                speed_norm_kmh=float(payload.get("meta", {}).get("speed_norm_kmh", 150.0)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * int(x.shape[0])
            count += int(x.shape[0])
        avg_loss = total_loss / max(1, count)
        history.append({"epoch": int(epoch), "loss": float(avg_loss)})
        (out_dir / "train_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        _save_checkpoint(out_dir / "checkpoint_last.pt", model, model_config, {"epoch": int(epoch), "loss": float(avg_loss), "time_bins": time_bins})
        if avg_loss < best_loss:
            best_loss = avg_loss
            _save_checkpoint(out_dir / "checkpoint_best.pt", model, model_config, {"epoch": int(epoch), "loss": float(avg_loss), "time_bins": time_bins})

    summary = {"best_loss": float(best_loss), "epochs": int(args.epochs), "device": device, "checkpoint_best": str(out_dir / "checkpoint_best.pt")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
