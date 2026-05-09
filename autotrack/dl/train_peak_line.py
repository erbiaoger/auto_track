"""Train PeakLineNet from generated sparse-point/line-mask tensor shards.

Purpose:
    Train the `peak_line` semantic line segmentation model on `.pt` shards
    produced by `generate_peak_line_dataset.py`. Training reads pre-generated
    tensors only; it does not read SAC files and does not synthesize samples
    online.

Example:
    uv run python -m autotrack.dl.train_peak_line \
        --data-dir datasets/peak_line/train \
        --out-dir models/peak_line_cuda \
        --device cuda \
        --amp on \
        --epochs 50 \
        --batch-size 32

    Continue an interrupted run from <out-dir>/checkpoint_last.pt:
    uv run python -m autotrack.dl.train_peak_line \
        --data-dir datasets/peak_line/train \
        --out-dir models/peak_line_cuda \
        --device cuda \
        --epochs 200 \
        --auto-resume

Arguments:
    --pos-weight controls the BCE positive class weight for sparse labels.
    --val-fraction reserves a fraction of shards for validation.
    --resume or --auto-resume continues checkpoint training.

Outputs:
    <out-dir>/checkpoint_last.pt
    <out-dir>/checkpoint_best.pt
    <out-dir>/train_config.json
    <out-dir>/train_history.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from autotrack.dl.peak_line_model import ModelConfig, PeakLineNet, peak_line_loss, save_checkpoint
from autotrack.dl.trajectory_set_model import WindowDatasetConfig, auto_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PeakLineNet from generated tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing meta.json and .pt shards.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50, help="Target total training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--base-channels", type=int, default=8, help="Base U-Net channel count.")
    parser.add_argument("--dropout", type=float, default=0.05, help="Dropout2d probability in convolution blocks.")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class BCE weight.")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight.")
    parser.add_argument("--focal-weight", type=float, default=0.25, help="Focal loss weight.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma.")
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Probability threshold for pixel metrics.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--amp", default="auto", choices=["auto", "on", "off"], help="Use CUDA automatic mixed precision.")
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"], help="CUDA AMP dtype.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last layout on CUDA.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of shards reserved for validation.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs when val shards exist.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from <out-dir>/checkpoint_last.pt if it exists.")
    parser.add_argument("--resume-model-only", action="store_true", help="Load model weights but reset optimizer.")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional maximum samples per epoch; 0 uses all.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=20, help="Print batch progress every N batches; 0 prints epoch summaries only.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_shards(shards: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    if not shards:
        raise ValueError("Dataset contains no shards")
    fraction = max(0.0, min(0.9, float(val_fraction)))
    val_count = int(round(len(shards) * fraction))
    if fraction > 0.0 and len(shards) > 1:
        val_count = max(1, val_count)
    if val_count <= 0:
        return shards, []
    val_count = min(len(shards) - 1, val_count)
    return shards[:-val_count], shards[-val_count:]


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _append_history_row(path: Path, row: dict[str, float | int | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resolve_resume_path(args: argparse.Namespace) -> Optional[Path]:
    if args.resume is not None:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        return resume_path
    if not bool(args.auto_resume):
        return None
    candidate = Path(args.out_dir).expanduser() / "checkpoint_last.pt"
    if candidate.is_file():
        return candidate
    print(f"Auto-resume requested, but no checkpoint found at {candidate}; starting a new run.", flush=True)
    return None


def _checkpoint_loss(checkpoint: dict, fallback: float = float("inf")) -> float:
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


def _initial_best_loss(out_dir: Path, resume_checkpoint: Optional[dict]) -> float:
    best_loss = _checkpoint_loss(resume_checkpoint) if resume_checkpoint is not None else float("inf")
    best_path = out_dir / "checkpoint_best.pt"
    if best_path.is_file():
        try:
            best_checkpoint = torch.load(str(best_path), map_location="cpu", weights_only=False)
            best_loss = min(best_loss, _checkpoint_loss(best_checkpoint, best_loss))
        except Exception as exc:
            print(f"Could not read existing best checkpoint {best_path}: {exc}", flush=True)
    return float(best_loss)


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


def _shard_batch_count(meta: dict, all_shards: list[str], shards: list[str], batch_size: int, max_samples: int) -> int:
    total = 0
    shard_size = int(meta.get("shard_size", 1))
    sample_count = int(meta.get("num_samples", 0))
    for shard in shards:
        shard_idx = all_shards.index(shard)
        n = max(0, sample_count - shard_idx * shard_size) if shard_idx == len(all_shards) - 1 else shard_size
        total += int(n)
    if int(max_samples) > 0:
        total = min(total, int(max_samples))
    return int(math.ceil(int(total) / max(1, int(batch_size))))


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    max_samples: int = 0,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(epoch) * 97_531)
    shard_order = list(shards)
    if shuffle and len(shard_order) > 1:
        perm = torch.randperm(len(shard_order), generator=gen).tolist()
        shard_order = [shard_order[i] for i in perm]
    emitted = 0
    for shard in shard_order:
        payload = torch.load(str(data_dir / shard), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        order = torch.randperm(n, generator=gen) if shuffle and n > 1 else torch.arange(n)
        for start in range(0, n, int(batch_size)):
            if int(max_samples) > 0 and emitted >= int(max_samples):
                return
            take = min(int(batch_size), n - start)
            if int(max_samples) > 0:
                take = min(take, int(max_samples) - emitted)
            if take <= 0:
                return
            idx = order[start : start + take]
            emitted += int(take)
            targets = {
                "line_mask": payload["line_mask"][idx].to(torch.float32),
                "point_target": payload["point_target"][idx].to(torch.float32),
            }
            yield payload["x"][idx].to(torch.float32), targets


def _batch_to_device(
    x: torch.Tensor,
    targets: dict[str, torch.Tensor],
    device: str,
    *,
    channels_last: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    non_blocking = str(device).startswith("cuda")
    x = x.to(device=device, non_blocking=non_blocking)
    if bool(channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    targets = {key: value.to(device=device, non_blocking=non_blocking) for key, value in targets.items()}
    return x, targets


def _evaluate(model: PeakLineNet, data_dir: Path, shards: list[str], device: str, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    metrics_items: list[dict[str, float]] = []
    with torch.no_grad():
        for x, targets in _iter_batches(
            data_dir,
            shards,
            batch_size=int(args.batch_size),
            shuffle=False,
            seed=int(args.seed),
            epoch=0,
            max_samples=int(args.max_samples),
        ):
            x, targets = _batch_to_device(x, targets, device, channels_last=bool(args.channels_last and str(device).startswith("cuda")))
            outputs = model(x)
            _, metrics = peak_line_loss(
                outputs,
                targets,
                pos_weight=float(args.pos_weight),
                dice_weight=float(args.dice_weight),
                focal_weight=float(args.focal_weight),
                focal_gamma=float(args.focal_gamma),
                threshold=float(args.metric_threshold),
            )
            metrics_items.append(metrics)
    return _mean_metrics(metrics_items)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    args.data_dir = data_dir
    args.out_dir = Path(args.out_dir).expanduser()
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    train_shards, val_shards = _split_shards(shards, float(args.val_fraction))
    raw_device = str(args.device).strip()
    device = auto_torch_device() if raw_device in {"", "auto", "None"} else raw_device
    torch.manual_seed(int(args.seed))
    print(f"Using torch device: {device}")

    resume_checkpoint = None
    resume_epoch = 0
    resume_path = _resolve_resume_path(args)
    if resume_path is not None:
        resume_checkpoint = torch.load(str(resume_path), map_location="cpu", weights_only=False)
        resume_epoch = int(resume_checkpoint.get("epoch", 0))
        model_config = ModelConfig(**dict(resume_checkpoint.get("model_config", {})))
        print(f"Resuming from checkpoint: {resume_path} at epoch={resume_epoch}", flush=True)
    else:
        model_config = ModelConfig(
            n_channels=int(meta["n_channels"]),
            in_channels=int(meta.get("in_channels", 1)),
            out_channels=int(meta.get("out_channels", 1)),
            base_channels=int(args.base_channels),
            dropout=float(args.dropout),
        )

    dataset_config = WindowDatasetConfig(
        window_seconds=float(meta["window_seconds"]),
        time_downsample=int(meta["time_downsample"]),
        samples_per_folder=int(meta["num_samples"]),
        min_visible_channels=int(meta.get("generator_args", {}).get("min_visible_channels", 2)),
        speed_norm_kmh=150.0,
        clip_ratio=1.0,
        input_mode="peak_point_image",
        seed=int(meta.get("generator_args", {}).get("seed", args.seed)),
    )
    model = PeakLineNet(model_config).to(device)
    if bool(args.channels_last) and str(device).startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if resume_checkpoint is not None and not bool(args.resume_model_only) and "optimizer_state" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        _move_optimizer_state_to_device(optimizer, device)
        print("Loaded optimizer state from checkpoint.", flush=True)
    elif resume_checkpoint is not None:
        print("Loaded model weights only; optimizer starts from scratch.", flush=True)

    use_amp = (str(args.amp) == "on") or (str(args.amp) == "auto" and str(device).startswith("cuda"))
    amp_dtype = torch.float16 if str(args.amp_dtype) == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and amp_dtype == torch.float16))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "peak_line_shards",
        "model_family": "peak_line",
        "data_dir": str(data_dir),
        "train_shards": train_shards,
        "val_shards": val_shards,
        "dataset_meta": meta,
        "dataset_config": asdict(dataset_config),
        "model_config": asdict(model_config),
        "train_args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "resolved_resume": str(resume_path) if resume_path is not None else "",
        "device": device,
        "created_at_unix": time.time(),
    }
    (args.out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    batches_per_epoch = _shard_batch_count(meta, shards, train_shards, int(args.batch_size), int(args.max_samples))
    print(
        "Dataset: "
        f"samples={meta.get('num_samples')}, train_shards={len(train_shards)}, val_shards={len(val_shards)}, "
        f"batch_size={int(args.batch_size)}, batches_per_epoch={batches_per_epoch}, "
        f"window_seconds={dataset_config.window_seconds}, time_downsample={dataset_config.time_downsample}"
    )
    print(
        "Model: "
        f"base_channels={model_config.base_channels}, dropout={model_config.dropout}, "
        f"amp={use_amp}, loss=BCE(pos_weight={float(args.pos_weight)})+Dice+{float(args.focal_weight)}*Focal"
    )

    best_loss = _initial_best_loss(args.out_dir, resume_checkpoint)
    start_epoch = resume_epoch + 1 if resume_checkpoint is not None else 1
    if start_epoch > int(args.epochs):
        print(
            f"Checkpoint epoch={resume_epoch} is already >= target epochs={int(args.epochs)}; nothing to train. "
            "Increase --epochs to continue for more total epochs."
        )
        return 0

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_metrics: list[dict[str, float]] = []
        log_every = int(args.log_every)
        for batch_idx, (x, targets) in enumerate(
            _iter_batches(
                data_dir,
                train_shards,
                batch_size=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                epoch=epoch,
                max_samples=int(args.max_samples),
            ),
            start=1,
        ):
            should_log = log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == batches_per_epoch)
            x, targets = _batch_to_device(x, targets, device, channels_last=bool(args.channels_last and str(device).startswith("cuda")))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp and str(device).startswith("cuda"))):
                outputs = model(x)
                loss, metrics = peak_line_loss(
                    outputs,
                    targets,
                    pos_weight=float(args.pos_weight),
                    dice_weight=float(args.dice_weight),
                    focal_weight=float(args.focal_weight),
                    focal_gamma=float(args.focal_gamma),
                    threshold=float(args.metric_threshold),
                )
            scaler.scale(loss).backward()
            if float(args.grad_clip) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            epoch_metrics.append(metrics)
            if should_log:
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{batches_per_epoch:04d} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"bce={metrics.get('loss_bce', float('nan')):.4f} "
                    f"dice={metrics.get('loss_dice', float('nan')):.4f} "
                    f"f1={metrics.get('pixel_f1', float('nan')):.3f} "
                    f"iou={metrics.get('pixel_iou', float('nan')):.3f} "
                    f"pred+={metrics.get('pred_positive_frac', float('nan')):.4f}",
                    flush=True,
                )

        mean_metrics = _mean_metrics(epoch_metrics)
        elapsed = time.perf_counter() - t0
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["elapsed_seconds"] = float(elapsed)
        print(
            f"epoch={epoch:03d} loss={mean_metrics.get('loss', float('nan')):.4f} "
            f"bce={mean_metrics.get('loss_bce', float('nan')):.4f} "
            f"dice={mean_metrics.get('loss_dice', float('nan')):.4f} "
            f"f1={mean_metrics.get('pixel_f1', float('nan')):.3f} "
            f"iou={mean_metrics.get('pixel_iou', float('nan')):.3f} "
            f"elapsed={elapsed:.1f}s"
        )

        val_metrics: dict[str, float] = {}
        if val_shards and (epoch % int(max(1, args.val_every)) == 0):
            val_t0 = time.perf_counter()
            val_metrics = _evaluate(model, data_dir, val_shards, device, args)
            print(
                f"epoch={epoch:03d} val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"val_f1={val_metrics.get('pixel_f1', float('nan')):.3f} "
                f"val_iou={val_metrics.get('pixel_iou', float('nan')):.3f} "
                f"val_elapsed={time.perf_counter() - val_t0:.1f}s",
                flush=True,
            )

        history_row: dict[str, float | int | str] = {"epoch": int(epoch), "elapsed_seconds": float(elapsed)}
        history_row.update({f"train_{key}": float(value) for key, value in mean_metrics.items() if np.isfinite(value)})
        history_row.update({f"val_{key}": float(value) for key, value in val_metrics.items() if np.isfinite(value)})
        _append_history_row(args.out_dir / "train_history.jsonl", history_row)

        save_this_epoch = (epoch % int(max(1, args.checkpoint_every)) == 0) or (epoch == int(args.epochs))
        if save_this_epoch:
            checkpoint_metrics = dict(mean_metrics)
            checkpoint_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            last_path = args.out_dir / "checkpoint_last.pt"
            save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
            print(f"Saved checkpoint: {last_path}", flush=True)
            current_loss = float(val_metrics.get("loss", mean_metrics.get("loss", float("inf"))))
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = args.out_dir / "checkpoint_best.pt"
                save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
                print(f"Saved new best checkpoint: {best_path}", flush=True)

    print(f"Done. Best loss={best_loss:.4f}. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
