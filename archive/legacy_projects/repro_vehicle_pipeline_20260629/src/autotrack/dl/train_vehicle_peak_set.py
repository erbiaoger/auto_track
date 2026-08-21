from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from autotrack.dl.simple_vehicle_peak_dataset import (
    SimplePeakSetDatasetConfig,
    SimpleLinearVehiclePeakDataset,
    peakset_collate,
)
from autotrack.dl.vehicle_peak_set_transformer import (
    PeakSetModelConfig,
    auto_torch_device,
    json_ready,
    peak_guided_set_loss,
    PeakGuidedVehicleSetTransformer,
    save_checkpoint,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the vehicle peak set transformer on simple synthetic traffic.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Optional exported shard dataset directory")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction when using shard dataset")
    parser.add_argument("--max-train-samples", type=int, default=0, help="Optional training sample cap for shard dataset")
    parser.add_argument("--max-val-samples", type=int, default=0, help="Optional validation sample cap for shard dataset")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, cpu, mps, auto")
    parser.add_argument("--torch-threads", type=int, default=1, help="Torch intra-op CPU threads")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples during training")
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the model")
    parser.add_argument(
        "--freeze-except-observed-valid",
        action="store_true",
        help="Freeze all parameters except the observed_valid_head; useful for auxiliary-head warmup.",
    )
    parser.add_argument(
        "--freeze-except-observed-valid-and-pair-tail",
        action="store_true",
        help="Freeze all parameters except observed_valid_head and the last trainable complete_pair_head layer.",
    )
    parser.add_argument(
        "--reset-best-val",
        action="store_true",
        help="Do not inherit best_val_loss from the resume checkpoint when selecting checkpoint_best.pt.",
    )
    parser.add_argument("--train-samples", type=int, default=512, help="Training samples")
    parser.add_argument("--val-samples", type=int, default=128, help="Validation samples")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N batches")
    parser.add_argument("--no-object-weight", type=float, default=0.02, help="Loss weight for unmatched queries")
    parser.add_argument("--complete-time-weight", type=float, default=8.0, help="Completion time regression weight")
    parser.add_argument("--complete-valid-weight", type=float, default=1.0, help="Completion visibility classification weight")
    parser.add_argument("--observed-valid-weight", type=float, default=1.0, help="Observed-vs-completed channel classification weight")
    parser.add_argument("--missing-complete-time-weight", type=float, default=0.0, help="Missing-channel completion time weight")
    parser.add_argument("--missing-complete-valid-weight", type=float, default=0.0, help="Missing-channel completion visibility weight")
    parser.add_argument("--dead-complete-time-weight", type=float, default=0.0, help="Dead-channel completion time weight")
    parser.add_argument("--dead-complete-valid-weight", type=float, default=0.0, help="Dead-channel completion visibility weight")
    parser.add_argument("--anchor-weight", type=float, default=1.0, help="Peak anchor classification weight")
    parser.add_argument("--anchor-time-weight", type=float, default=0.5, help="Observed anchor time regression weight")
    parser.add_argument("--direction-weight", type=float, default=0.2, help="Direction classification weight")
    parser.add_argument("--speed-weight", type=float, default=0.2, help="Speed regression weight")
    parser.add_argument("--count-weight", type=float, default=0.02, help="Count regularization weight")
    parser.add_argument("--jump-weight", type=float, default=0.20, help="Adjacent complete-time jump penalty weight")
    parser.add_argument("--slope-variation-weight", type=float, default=0.30, help="Local slope variation penalty weight")
    parser.add_argument("--inertia-weight", type=float, default=0.60, help="Anchor-time inertia regularization weight")
    parser.add_argument("--linearity-weight", type=float, default=0.40, help="Linearity regularization weight")
    parser.add_argument("--smoothness-weight", type=float, default=0.30, help="Curvature regularization weight")
    parser.add_argument("--save-every", type=int, default=10, help="Save an intermediate checkpoint every N epochs")
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "greedy"], help="Matching strategy")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden size")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer heads")
    parser.add_argument("--encoder-layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder layers")
    parser.add_argument("--max-queries", type=int, default=32, help="Maximum queries")
    parser.add_argument("--pooled-time", type=int, default=96, help="Temporal pooling length")
    parser.add_argument("--in-channels", type=int, default=1, help="Input channels")
    parser.add_argument("--residual-completion", action=argparse.BooleanOptionalAction, default=True, help="Predict completion as a global trajectory plus bounded residual")
    parser.add_argument("--max-residual-norm", type=float, default=0.005, help="Maximum normalized residual around the global trajectory")
    parser.add_argument("--max-base-slope-norm", type=float, default=0.75, help="Maximum normalized linear trajectory slope")
    parser.add_argument("--max-base-curve-norm", type=float, default=0.35, help="Maximum normalized quadratic trajectory curvature")
    parser.add_argument("--vehicles-min", type=int, default=4, help="Minimum vehicles per scene")
    parser.add_argument("--vehicles-max", type=int, default=12, help="Maximum vehicles per scene")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Additive noise std")
    parser.add_argument("--amp-min", type=float, default=4.0, help="Minimum pulse amplitude")
    parser.add_argument("--amp-max", type=float, default=8.0, help="Maximum pulse amplitude")
    parser.add_argument("--sigma-min-s", type=float, default=0.05, help="Minimum pulse sigma in seconds")
    parser.add_argument("--sigma-max-s", type=float, default=0.09, help="Maximum pulse sigma in seconds")
    parser.add_argument("--min-visible-channels", type=int, default=5, help="Minimum visible channels per vehicle")
    parser.add_argument("--primary-ratio", type=float, default=0.8, help="Primary direction probability")
    parser.add_argument("--same-direction-ratio", type=float, default=0.75, help="Same-direction vehicle probability")
    parser.add_argument("--crossing-ratio", type=float, default=0.25, help="Opposite-direction vehicle probability")
    parser.add_argument("--missing-random-ratio-min", type=float, default=0.05, help="Lower bound for random missing ratio")
    parser.add_argument("--missing-random-ratio-max", type=float, default=0.20, help="Upper bound for random missing ratio")
    parser.add_argument("--missing-segment-count-max", type=int, default=2, help="Maximum number of missing segments")
    parser.add_argument("--missing-segment-min-len", type=int, default=2, help="Minimum missing segment length")
    parser.add_argument("--missing-segment-max-len", type=int, default=6, help="Maximum missing segment length")
    parser.add_argument("--dead-channel-indices", default="", help="Comma-separated fixed dead channel indices")
    parser.add_argument("--input-mode", default="raw", choices=["raw", "raw_abs"], help="Input representation")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio")
    return parser.parse_args(argv)


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(sum(item.get(key, 0.0) for item in items) / len(items)) for key in keys}


class _ExportedShardDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        *,
        split: str,
        val_fraction: float = 0.2,
        max_samples: int = 0,
    ):
        self.dataset_dir = Path(dataset_dir).expanduser()
        meta_path = self.dataset_dir / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        self.meta = payload
        self.shard_paths = [self.dataset_dir / str(name) for name in payload.get("shards", [])]
        meta_sizes = payload.get("shard_sizes", [])
        if len(meta_sizes) == len(self.shard_paths):
            self.shard_sizes = [int(size) for size in meta_sizes]
        else:
            self.shard_sizes = []
            for shard_path in self.shard_paths:
                shard = torch.load(str(shard_path), map_location="cpu", weights_only=False)
                self.shard_sizes.append(int(shard["x"].shape[0]))
        self.total = int(sum(self.shard_sizes))
        self.val_fraction = float(min(0.9, max(0.0, val_fraction)))
        split = str(split).lower()
        if split not in {"train", "val"}:
            raise ValueError("split must be train or val")
        cut = int(round(self.total * (1.0 - self.val_fraction)))
        self.start = 0 if split == "train" else max(0, cut)
        self.end = max(0, cut) if split == "train" else self.total
        if int(max_samples) > 0:
            self.end = min(self.end, self.start + int(max_samples))
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def _resolve(self, global_index: int) -> tuple[int, int]:
        idx = int(global_index)
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        acc = 0
        for shard_idx, shard_size in enumerate(self.shard_sizes):
            next_acc = acc + shard_size
            if idx < next_acc:
                return shard_idx, idx - acc
            acc = next_acc
        raise IndexError(idx)

    def _load_shard(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(str(self.shard_paths[shard_idx]), map_location="cpu", weights_only=False)
        return self._cache[shard_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        global_index = self.start + int(index)
        shard_idx, local_idx = self._resolve(global_index)
        payload = self._load_shard(shard_idx)
        x = payload["x"][local_idx].to(torch.float32)
        targets = payload["targets"]
        target = {key: value[local_idx].clone() for key, value in targets.items() if torch.is_tensor(value)}
        if "gt_valid" not in target:
            n_gt = int(target["full_time"].shape[0]) if "full_time" in target else 0
            target["gt_valid"] = torch.ones((n_gt,), dtype=torch.bool)
        return x, target


def _build_dataset(args: argparse.Namespace, *, length: int, seed: int) -> SimpleLinearVehiclePeakDataset:
    cfg = SimplePeakSetDatasetConfig(
        length=int(length),
        n_channels=50,
        fs=1000.0,
        window_seconds=60.0,
        time_downsample=10,
        dx_m=20.0,
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        min_visible_channels=int(args.min_visible_channels),
        primary_ratio=float(args.primary_ratio),
        same_direction_ratio=float(args.same_direction_ratio),
        crossing_ratio=float(args.crossing_ratio),
        missing_random_ratio_min=float(args.missing_random_ratio_min),
        missing_random_ratio_max=float(args.missing_random_ratio_max),
        missing_segment_count_max=int(args.missing_segment_count_max),
        missing_segment_min_len=int(args.missing_segment_min_len),
        missing_segment_max_len=int(args.missing_segment_max_len),
        dead_channel_indices=str(args.dead_channel_indices),
        clip_ratio=float(args.clip_ratio),
        input_mode=str(args.input_mode),
        seed=int(seed),
    )
    return SimpleLinearVehiclePeakDataset(config=cfg)


def _build_shard_dataset(args: argparse.Namespace, *, split: str) -> _ExportedShardDataset:
    if args.dataset_dir is None:
        raise ValueError("--dataset-dir is required for shard-backed training")
    max_samples = int(args.max_train_samples if split == "train" else args.max_val_samples)
    return _ExportedShardDataset(
        Path(args.dataset_dir),
        split=split,
        val_fraction=float(args.val_fraction),
        max_samples=max_samples,
    )


def _dataset_payload(dataset: object, args: argparse.Namespace, split: str) -> dict[str, Any]:
    if hasattr(dataset, "config"):
        return asdict(getattr(dataset, "config"))
    if args.dataset_dir is not None:
        payload: dict[str, Any] = {
            "dataset_dir": str(Path(args.dataset_dir).expanduser()),
            "split": str(split),
            "val_fraction": float(args.val_fraction),
        }
        if hasattr(dataset, "meta"):
            dataset_cfg = getattr(dataset, "meta", {}).get("dataset_config", {})
            if isinstance(dataset_cfg, dict):
                payload.update(dataset_cfg)
        if hasattr(dataset, "start") and hasattr(dataset, "end"):
            payload["sample_range"] = [int(getattr(dataset, "start")), int(getattr(dataset, "end"))]
        return payload
    return {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    device = str(args.device).strip() or auto_torch_device()
    print(f"Using torch device: {device}")
    try:
        torch.set_num_threads(int(getattr(args, "torch_threads", 1)))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    if str(device).startswith("cuda"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.dataset_dir is not None:
        train_dataset = _build_shard_dataset(args, split="train")
        val_dataset = _build_shard_dataset(args, split="val")
    else:
        train_dataset = _build_dataset(args, length=int(args.train_samples), seed=int(args.seed))
        val_dataset = _build_dataset(args, length=int(args.val_samples), seed=int(args.seed) + 10_000)

    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(str(Path(args.resume).expanduser()), map_location="cpu", weights_only=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=bool(args.shuffle),
        num_workers=int(args.num_workers),
        collate_fn=peakset_collate,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=peakset_collate,
        drop_last=False,
        pin_memory=str(device).startswith("cuda"),
        persistent_workers=int(args.num_workers) > 0,
        prefetch_factor=int(args.prefetch_factor) if int(args.num_workers) > 0 else None,
    )

    if resume_checkpoint is not None:
        resume_model_config = dict(resume_checkpoint.get("model_config", {}))
        resume_model_config.setdefault("residual_completion", bool(args.residual_completion))
        resume_model_config.setdefault("max_residual_norm", float(args.max_residual_norm))
        resume_model_config.setdefault("max_base_slope_norm", float(args.max_base_slope_norm))
        resume_model_config.setdefault("max_base_curve_norm", float(args.max_base_curve_norm))
        model_config = PeakSetModelConfig(**resume_model_config)
    else:
        sample0 = train_dataset[0]
        n_channels = int(sample0[0].shape[-2])
        model_config = PeakSetModelConfig(
            n_channels=n_channels,
            in_channels=int(sample0[0].shape[0]),
            max_queries=int(args.max_queries),
            hidden_dim=int(args.hidden_dim),
            num_heads=int(args.num_heads),
            encoder_layers=int(args.encoder_layers),
            decoder_layers=int(args.decoder_layers),
            pooled_time=int(args.pooled_time),
            trajectory_time_bins=n_channels,
            residual_completion=bool(args.residual_completion),
            max_residual_norm=float(args.max_residual_norm),
            max_base_slope_norm=float(args.max_base_slope_norm),
            max_base_curve_norm=float(args.max_base_curve_norm),
        )
    sample0 = train_dataset[0]
    if int(sample0[0].shape[-2]) != int(model_config.n_channels):
        raise SystemExit(f"channel mismatch: checkpoint expects {model_config.n_channels}, dataset has {sample0[0].shape[-2]}")
    if int(sample0[0].shape[0]) != int(model_config.in_channels):
        raise SystemExit(f"input-channel mismatch: checkpoint expects {model_config.in_channels}, dataset has {sample0[0].shape[0]}")
    raw_model = PeakGuidedVehicleSetTransformer(model_config).to(device)
    model = raw_model
    start_epoch = 1
    best_val_loss = float("inf")

    if resume_checkpoint is not None:
        load_result = raw_model.load_state_dict(resume_checkpoint["model_state"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "Loaded checkpoint with architecture differences: "
                f"missing={list(load_result.missing_keys)} unexpected={list(load_result.unexpected_keys)}",
            flush=True,
        )
    if bool(args.freeze_except_observed_valid) and bool(args.freeze_except_observed_valid_and_pair_tail):
        raise SystemExit("choose only one freeze mode")
    if bool(args.freeze_except_observed_valid) or bool(args.freeze_except_observed_valid_and_pair_tail):
        trainable_names: list[str] = []
        for name, param in raw_model.named_parameters():
            trainable = name.startswith("observed_valid_head.")
            if bool(args.freeze_except_observed_valid_and_pair_tail):
                trainable = trainable or name.startswith("complete_pair_head.3.")
            param.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise SystemExit("freeze mode requested, but no trainable parameters were found")
        print(
            "Training only selected parameters: "
            + ", ".join(trainable_names),
            flush=True,
        )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if resume_checkpoint is not None:
        freeze_mode = bool(args.freeze_except_observed_valid) or bool(args.freeze_except_observed_valid_and_pair_tail)
        if "optimizer_state" in resume_checkpoint and not freeze_mode and not (load_result.missing_keys or load_result.unexpected_keys):
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_val_loss = float("inf") if bool(args.reset_best_val) else float(resume_checkpoint.get("metrics", {}).get("loss", float("inf")))
        print(
            f"Resuming from {Path(args.resume).expanduser()} at epoch {start_epoch}, "
            f"best_val_loss={best_val_loss:.6f}",
            flush=True,
        )
    if bool(args.compile) and hasattr(torch, "compile"):
        model = torch.compile(model)
    use_amp = bool(args.amp) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "device": device,
        "created_at_unix": time.time(),
        "model_config": asdict(model_config),
        "train_dataset": _dataset_payload(train_dataset, args, "train"),
        "val_dataset": _dataset_payload(val_dataset, args, "val"),
        "args": vars(args),
    }
    (out_dir / "train_config.json").write_text(json.dumps(json_ready(config_payload), indent=2, ensure_ascii=False), encoding="utf-8")

    history_path = out_dir / "train_history.jsonl"
    history_mode = "a" if resume_checkpoint is not None and history_path.exists() else "w"
    with history_path.open(history_mode, encoding="utf-8") as history_fp:
        for epoch in range(int(start_epoch), int(args.epochs) + 1):
            model.train()
            train_metrics: list[dict[str, float]] = []
            t0 = time.perf_counter()
            for batch_idx, (x, targets) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                targets = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in targets.items()}
                optimizer.zero_grad(set_to_none=True)
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else contextlib.nullcontext()
                with autocast_ctx:
                    outputs = model(
                        x,
                        peak_time=targets.get("peak_time"),
                        peak_amp=targets.get("peak_amp"),
                        peak_valid=targets.get("peak_valid"),
                        peak_index=targets.get("peak_index"),
                    )
                    loss, metrics = peak_guided_set_loss(
                        outputs,
                        targets,
                        no_object_weight=float(args.no_object_weight),
                        complete_time_weight=float(args.complete_time_weight),
                        complete_valid_weight=float(args.complete_valid_weight),
                        observed_valid_weight=float(args.observed_valid_weight),
                        missing_complete_time_weight=float(args.missing_complete_time_weight),
                        missing_complete_valid_weight=float(args.missing_complete_valid_weight),
                        anchor_weight=float(args.anchor_weight),
                        anchor_time_weight=float(args.anchor_time_weight),
                        inertia_weight=float(args.inertia_weight),
                        line_weight=float(args.linearity_weight),
                        jump_weight=float(args.jump_weight),
                        slope_variation_weight=float(args.slope_variation_weight),
                        matcher=str(args.matcher),
                    )
                scaler.scale(loss).backward()
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                train_metrics.append(metrics)
                if batch_idx == 1 or batch_idx % int(max(1, args.log_every)) == 0 or batch_idx == len(train_loader):
                    print(
                        f"epoch={epoch:03d} batch={batch_idx:04d}/{len(train_loader):04d} "
                        f"loss={metrics.get('loss', float('nan')):.4f} "
                        f"complete_t={metrics.get('loss_complete_time', float('nan')):.4f} "
                        f"missing_t={metrics.get('loss_missing_complete_time', float('nan')):.4f} "
                        f"anchor={metrics.get('loss_anchor', float('nan')):.4f} "
                        f"jump={metrics.get('loss_jump', float('nan')):.4f} "
                        f"slope={metrics.get('loss_slope_variation', float('nan')):.4f} "
                        f"complete_v={metrics.get('loss_complete_valid', float('nan')):.4f} "
                        f"observed_v={metrics.get('loss_observed_valid', float('nan')):.4f} "
                        f"missing_v={metrics.get('loss_missing_complete_valid', float('nan')):.4f} "
                        f"gt={metrics.get('gt', 0.0):.0f} matched={metrics.get('matched', 0.0):.0f}",
                        flush=True,
                    )

            model.eval()
            val_metrics: list[dict[str, float]] = []
            with torch.inference_mode():
                for x, targets in val_loader:
                    x = x.to(device, non_blocking=True)
                    targets = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in targets.items()}
                    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else contextlib.nullcontext()
                    with autocast_ctx:
                        outputs = model(
                            x,
                            peak_time=targets.get("peak_time"),
                            peak_amp=targets.get("peak_amp"),
                            peak_valid=targets.get("peak_valid"),
                            peak_index=targets.get("peak_index"),
                        )
                        _loss, metrics = peak_guided_set_loss(
                            outputs,
                            targets,
                            no_object_weight=float(args.no_object_weight),
                            complete_time_weight=float(args.complete_time_weight),
                            complete_valid_weight=float(args.complete_valid_weight),
                            observed_valid_weight=float(args.observed_valid_weight),
                            missing_complete_time_weight=float(args.missing_complete_time_weight),
                            missing_complete_valid_weight=float(args.missing_complete_valid_weight),
                            anchor_weight=float(args.anchor_weight),
                            anchor_time_weight=float(args.anchor_time_weight),
                            inertia_weight=float(args.inertia_weight),
                            line_weight=float(args.linearity_weight),
                            jump_weight=float(args.jump_weight),
                            slope_variation_weight=float(args.slope_variation_weight),
                            matcher=str(args.matcher),
                        )
                    val_metrics.append(metrics)

            train_mean = _mean_metrics(train_metrics)
            val_mean = _mean_metrics(val_metrics)
            elapsed = time.perf_counter() - t0
            row = {
                "epoch": int(epoch),
                "elapsed_seconds": float(elapsed),
                "train": train_mean,
                "val": val_mean,
            }
            history_fp.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")
            history_fp.flush()
            print(
                f"epoch={epoch:03d} train_loss={train_mean.get('loss', float('nan')):.4f} "
                f"val_loss={val_mean.get('loss', float('nan')):.4f} "
                f"train_matched={train_mean.get('matched', 0.0):.1f} val_matched={val_mean.get('matched', 0.0):.1f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

            last_path = out_dir / "checkpoint_last.pt"
            save_checkpoint(last_path, raw_model, optimizer, model_config, _dataset_payload(train_dataset, args, "train"), epoch, val_mean)
            save_every = int(max(0, args.save_every))
            if save_every > 0 and epoch % save_every == 0:
                epoch_path = out_dir / f"checkpoint_epoch_{epoch:03d}.pt"
                save_checkpoint(epoch_path, raw_model, optimizer, model_config, _dataset_payload(train_dataset, args, "train"), epoch, val_mean)
            if float(val_mean.get("loss", float("inf"))) < best_val_loss:
                best_val_loss = float(val_mean.get("loss", float("inf")))
                best_path = out_dir / "checkpoint_best.pt"
                save_checkpoint(best_path, raw_model, optimizer, model_config, _dataset_payload(train_dataset, args, "train"), epoch, val_mean)

    summary = {
        "best_val_loss": float(best_val_loss),
        "epochs": int(args.epochs),
        "device": device,
        "checkpoint_best": str(out_dir / "checkpoint_best.pt"),
        "dataset_dir": str(Path(args.dataset_dir).expanduser()) if args.dataset_dir is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
