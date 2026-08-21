"""Train the dense multi-vehicle proposal network on synthetic windows."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.vehicle_proposal_net import (
    ProposalModelConfig,
    VehicleProposalNet,
    build_multi_vehicle_heatmap_target,
    build_multi_vehicle_objectness_target,
    save_checkpoint,
    vehicle_proposal_loss,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the dense multi-vehicle proposal network.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--train-samples", type=int, default=4096, help="Number of synthetic training windows.")
    parser.add_argument("--val-samples", type=int, default=512, help="Number of synthetic validation windows.")
    parser.add_argument("--benchmark-file", type=Path, default=None, help="Optional fixed benchmark .pt file.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dimension.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--vehicles-min", type=int, default=8, help="Minimum vehicles per synthetic window.")
    parser.add_argument("--vehicles-max", type=int, default=22, help="Maximum vehicles per synthetic window.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--noise-std", type=float, default=0.08, help="Background noise standard deviation.")
    parser.add_argument("--amp-min", type=float, default=1.0, help="Minimum pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=2.8, help="Maximum pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.05, help="Minimum pulse width in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.14, help="Maximum pulse width in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=0.5, help="Probability of forward-direction samples.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels per sample.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--input-mode", default="raw", choices=["raw"], help="Synthetic input mode.")
    parser.add_argument(
        "--background-pt",
        type=Path,
        default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/xi_gauss_50_120s_large/test/shard_000000.pt"),
        help="Optional real shard .pt used as background texture.",
    )
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy to sample windows from.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Layout of the real background .npy.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="First channel index to slice from the real background.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to the real background window.")
    parser.add_argument("--artifact-dropout-ratio", type=float, default=0.35, help="Chance to remove a contiguous block of visible channels.")
    parser.add_argument("--artifact-dropout-min-channels", type=int, default=2, help="Minimum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-dropout-max-channels", type=int, default=8, help="Maximum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-decoy-ratio", type=float, default=0.25, help="Chance to inject a decoy branch or spike cluster.")
    parser.add_argument("--artifact-decoy-min-points", type=int, default=1, help="Minimum decoy points per sample.")
    parser.add_argument("--artifact-decoy-max-points", type=int, default=4, help="Maximum decoy points per sample.")
    parser.add_argument("--artifact-decoy-amp-scale-min", type=float, default=1.1, help="Minimum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-amp-scale-max", type=float, default=2.0, help="Maximum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-time-jitter-s", type=float, default=0.18, help="Random decoy time jitter in seconds.")
    parser.add_argument("--artifact-competing-ratio", type=float, default=0.45, help="Chance to inject an unlabeled competing vehicle track.")
    parser.add_argument("--artifact-competing-time-jitter-s", type=float, default=0.8, help="Random time jitter for the competing vehicle anchor.")
    parser.add_argument("--artifact-competing-amp-scale-min", type=float, default=0.8, help="Minimum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-amp-scale-max", type=float, default=1.6, help="Maximum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-speed-ratio-min", type=float, default=0.88, help="Minimum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-speed-ratio-max", type=float, default=1.12, help="Maximum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-channel-offset-max", type=int, default=5, help="Maximum channel offset for the competing vehicle anchor relative to the target anchor.")
    parser.add_argument("--artifact-competing-opposite-direction-ratio", type=float, default=0.5, help="Probability of assigning the competing vehicle the opposite direction.")
    parser.add_argument("--heatmap-positive-weight", type=float, default=0.0, help="Positive-pixel weight for heatmap loss; 0 means auto.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory when transferring to CUDA.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--heatmap-weight", type=float, default=1.0, help="Weight for the heatmap loss.")
    parser.add_argument("--dice-weight", type=float, default=0.25, help="Weight for the dice loss component.")
    parser.add_argument("--objectness-weight", type=float, default=0.2, help="Weight for the objectness loss.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping threshold.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Enable CUDA AMP training.")
    parser.add_argument("--log_every", type=int, default=20, help="Print training stats every N steps.")
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


def _collate(items: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    xs = torch.stack([item[0] for item in items], dim=0).contiguous()
    targets = [{key: value for key, value in item[1].items()} for item in items]
    return xs, targets


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
            n_gt = int(target["gt_masks"].shape[0]) if "gt_masks" in target else int(target["time"].shape[0])
            target["gt_valid"] = torch.ones((n_gt,), dtype=torch.bool)
        return x, target


def _build_dataset(args: argparse.Namespace, *, length: int, seed: int) -> OnlineSyntheticTrajectoryDataset:
    return OnlineSyntheticTrajectoryDataset(
        length=int(max(1, length)),
        n_channels=int(args.n_channels),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=0.0,
        slow_speed_min_kmh=float(args.speed_min_kmh),
        slow_speed_max_kmh=float(args.speed_max_kmh),
        fast_speed_min_kmh=float(args.speed_min_kmh),
        fast_speed_max_kmh=float(args.speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        speed_norm_kmh=150.0,
        clip_ratio=float(args.clip_ratio),
        input_mode=str(args.input_mode),
        seed=int(seed),
        scene_mode="realistic_traffic",
        vehicle_count_profile="mixed_density",
        speed_variation_ratio=0.08,
        same_direction_cluster_ratio=0.35,
        crossing_ratio=0.35,
        parallel_close_ratio=0.20,
        multi_gap_dropout_ratio=0.55,
        mask_sigma_ch=0.8,
        mask_sigma_t=2.0,
        cache_dataset=False,
        return_raw_window=True,
        background_pt=args.background_pt if args.background_pt is not None and Path(args.background_pt).expanduser().is_file() else None,
        background_npy=args.background_npy,
        background_layout=str(args.background_layout),
        background_channel_start=int(args.background_channel_start),
        background_scale=float(args.background_scale),
        artifact_dropout_ratio=float(args.artifact_dropout_ratio),
        artifact_dropout_min_channels=int(args.artifact_dropout_min_channels),
        artifact_dropout_max_channels=int(args.artifact_dropout_max_channels),
        artifact_decoy_ratio=float(args.artifact_decoy_ratio),
        artifact_decoy_min_points=int(args.artifact_decoy_min_points),
        artifact_decoy_max_points=int(args.artifact_decoy_max_points),
        artifact_decoy_amp_scale_min=float(args.artifact_decoy_amp_scale_min),
        artifact_decoy_amp_scale_max=float(args.artifact_decoy_amp_scale_max),
        artifact_decoy_time_jitter_s=float(args.artifact_decoy_time_jitter_s),
        artifact_competing_ratio=float(args.artifact_competing_ratio),
        artifact_competing_time_jitter_s=float(args.artifact_competing_time_jitter_s),
        artifact_competing_amp_scale_min=float(args.artifact_competing_amp_scale_min),
        artifact_competing_amp_scale_max=float(args.artifact_competing_amp_scale_max),
        artifact_competing_speed_ratio_min=float(args.artifact_competing_speed_ratio_min),
        artifact_competing_speed_ratio_max=float(args.artifact_competing_speed_ratio_max),
        artifact_competing_channel_offset_max=int(args.artifact_competing_channel_offset_max),
        artifact_competing_opposite_direction_ratio=float(args.artifact_competing_opposite_direction_ratio),
    )


def _sample_targets(targets: dict[str, torch.Tensor], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    gt_masks = targets["gt_masks"].to(device)
    heatmap = build_multi_vehicle_heatmap_target(gt_masks)
    objectness = build_multi_vehicle_objectness_target(gt_masks)
    return heatmap, objectness


def _run_epoch(
    model: VehicleProposalNet,
    loader: DataLoader,
    device: str,
    *,
    amp: bool,
    scaler: Optional[torch.amp.GradScaler],
    heatmap_weight: float,
    dice_weight: float,
    objectness_weight: float,
    heatmap_positive_weight: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    log_every: int = 20,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    seen = 0
    t0 = time.perf_counter()
    for step, (x, target_items) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=(device == "cuda"))
        heatmaps: list[torch.Tensor] = []
        objectness: list[torch.Tensor] = []
        valid_rows: list[int] = []
        for idx, target in enumerate(target_items):
            gt_masks = target["gt_masks"].to(device, non_blocking=(device == "cuda"))
            if int(gt_masks.shape[0]) <= 0:
                continue
            heatmap, obj = _sample_targets({"gt_masks": gt_masks}, device=x.device)
            heatmaps.append(heatmap)
            objectness.append(obj)
            valid_rows.append(int(idx))
        if not valid_rows:
            continue

        idx = torch.tensor(valid_rows, device=x.device, dtype=torch.long)
        x = x.index_select(0, idx)
        heatmap_tgt = torch.stack(heatmaps, dim=0).to(device)
        objectness_tgt = torch.stack(objectness, dim=0).to(device)

        use_amp = bool(amp and device == "cuda")
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else nullcontext()
        with autocast_ctx:
            outputs = model(x)
            loss, loss_parts = vehicle_proposal_loss(
                outputs,
                target_heatmap=heatmap_tgt,
                target_objectness=objectness_tgt,
                heatmap_weight=float(heatmap_weight),
                dice_weight=float(dice_weight),
                objectness_weight=float(objectness_weight),
                heatmap_positive_weight=heatmap_positive_weight,
            )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
                if float(grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

        batch_size = int(x.shape[0])
        seen += batch_size
        pred = torch.sigmoid(outputs["heatmap_logits"])
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        for key, value in loss_parts.items():
            totals[str(key)] = totals.get(str(key), 0.0) + float(value.detach().cpu()) * batch_size
        iou_pred = pred >= 0.5
        iou_tgt = heatmap_tgt >= 0.5
        inter = (iou_pred & iou_tgt).float().sum(dim=(1, 2))
        union = (iou_pred | iou_tgt).float().sum(dim=(1, 2)).clamp_min(1.0)
        totals["heatmap_iou"] = totals.get("heatmap_iou", 0.0) + float(torch.mean(inter / union).detach().cpu()) * batch_size
        totals["objectness_acc"] = totals.get("objectness_acc", 0.0) + float(((torch.sigmoid(outputs["objectness_logits"]) >= 0.5) == (objectness_tgt >= 0.5)).float().mean().detach().cpu()) * batch_size
        if log_every > 0 and (step % int(log_every) == 0):
            elapsed = time.perf_counter() - t0
            print(
                f"step={step:04d} seen={seen} loss={float(loss.detach().cpu()):.4f} "
                f"heatmap={float(loss_parts['loss_heatmap'].detach().cpu()):.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    if seen <= 0:
        return {}
    return {key: float(value / seen) for key, value in totals.items()}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_file is not None:
        payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
        train_ds = _BenchmarkDataset(payload)
        val_ds = _BenchmarkDataset(payload)
    else:
        train_ds = _build_dataset(args, length=int(max(1, args.train_samples)), seed=int(args.seed))
        val_ds = _build_dataset(args, length=int(max(1, args.val_samples)), seed=int(args.seed) + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=_collate,
        worker_init_fn=None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=_collate,
        worker_init_fn=None,
    )

    sample_x = train_ds[0][0]
    model_cfg = ProposalModelConfig(n_channels=int(sample_x.shape[1]), in_channels=int(sample_x.shape[0]), hidden_dim=int(args.hidden_dim))
    model = VehicleProposalNet(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(args.amp and device == "cuda"))

    if args.resume is not None:
        resume_ckpt = torch.load(str(Path(args.resume).expanduser()), map_location="cpu", weights_only=False)
        resume_state = resume_ckpt.get("model_state")
        if resume_state is not None:
            missing, unexpected = model.load_state_dict(resume_state, strict=False)
            if missing:
                print(f"resume loaded with newly initialized keys: {missing}", flush=True)
            if unexpected:
                print(f"resume ignored unexpected keys: {unexpected}", flush=True)
        resume_opt = resume_ckpt.get("optimizer_state")
        if resume_opt is not None:
            try:
                optimizer.load_state_dict(resume_opt)
            except Exception as exc:  # pragma: no cover - defensive for mismatch across runs
                print(f"resume optimizer state skipped: {exc}", flush=True)
        print(f"resumed_from={Path(args.resume).expanduser()}", flush=True)

    train_config = {"device": device, **vars(args), "model_config": asdict(model_cfg)}
    (out_dir / "train_config.json").write_text(json.dumps(_json_ready(train_config), indent=2, ensure_ascii=False), encoding="utf-8")

    best_val = float("inf")
    history_path = out_dir / "train_history.jsonl"
    for epoch in range(1, int(args.epochs) + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            amp=bool(args.amp),
        scaler=scaler,
        heatmap_weight=float(args.heatmap_weight),
        dice_weight=float(args.dice_weight),
        objectness_weight=float(args.objectness_weight),
        heatmap_positive_weight=float(args.heatmap_positive_weight) if float(args.heatmap_positive_weight) > 0 else None,
        optimizer=optimizer,
        grad_clip=float(args.grad_clip),
        log_every=int(args.log_every),
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            amp=False,
            scaler=None,
            heatmap_weight=float(args.heatmap_weight),
            dice_weight=float(args.dice_weight),
            objectness_weight=float(args.objectness_weight),
            heatmap_positive_weight=float(args.heatmap_positive_weight) if float(args.heatmap_positive_weight) > 0 else None,
            optimizer=None,
            grad_clip=0.0,
            log_every=0,
        )
        row = {"epoch": int(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        with history_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True) + "\n")

        metrics = {**{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        save_checkpoint(out_dir / "checkpoint_last.pt", model, optimizer, model_cfg, metrics)
        if float(val_metrics.get("loss", float("inf"))) < best_val:
            best_val = float(val_metrics["loss"])
            save_checkpoint(out_dir / "checkpoint_best.pt", model, optimizer, model_cfg, metrics)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics.get('loss', float('nan')):.4f} "
            f"val_loss={val_metrics.get('loss', float('nan')):.4f} best_val={best_val:.4f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
