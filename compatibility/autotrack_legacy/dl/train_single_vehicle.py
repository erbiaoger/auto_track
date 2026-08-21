"""Train the new single-vehicle tracker front-end on synthetic windows."""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.single_vehicle_net import (
    ModelConfig,
    SingleVehiclePeakNet,
    build_single_vehicle_heatmap_target,
    build_single_vehicle_line_target,
    build_single_vehicle_trajectory_target,
    estimate_direction_index_from_outputs,
    save_checkpoint,
    single_vehicle_supervised_loss,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the single-vehicle heatmap model on synthetic windows.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--train-samples", type=int, default=4096, help="Number of synthetic training windows.")
    parser.add_argument("--val-samples", type=int, default=512, help="Number of synthetic validation windows.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Model hidden dimension.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--vehicles-min", type=int, default=1, help="Minimum vehicles per synthetic window.")
    parser.add_argument("--vehicles-max", type=int, default=1, help="Maximum vehicles per synthetic window.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=100.0, help="Maximum vehicle speed.")
    parser.add_argument("--noise-std", type=float, default=0.12, help="Background noise standard deviation.")
    parser.add_argument("--amp-min", type=float, default=0.8, help="Minimum pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=2.0, help="Maximum pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.03, help="Minimum pulse width in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.08, help="Maximum pulse width in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=1.0, help="Probability of forward-direction samples.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels per sample.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by synthetic labels.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--input-mode", default="raw", choices=["raw"], help="Synthetic input mode.")
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy to sample windows from.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Layout of the real background .npy.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="First channel index to slice from the real background.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to the real background window.")
    parser.add_argument("--artifact-dropout-ratio", type=float, default=0.0, help="Chance to remove a contiguous block of visible channels.")
    parser.add_argument("--artifact-dropout-min-channels", type=int, default=2, help="Minimum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-dropout-max-channels", type=int, default=6, help="Maximum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-decoy-ratio", type=float, default=0.0, help="Chance to inject a decoy branch or spike cluster.")
    parser.add_argument("--artifact-decoy-min-points", type=int, default=1, help="Minimum decoy points per sample.")
    parser.add_argument("--artifact-decoy-max-points", type=int, default=3, help="Maximum decoy points per sample.")
    parser.add_argument("--artifact-decoy-amp-scale-min", type=float, default=1.1, help="Minimum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-amp-scale-max", type=float, default=2.2, help="Maximum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-time-jitter-s", type=float, default=0.18, help="Random decoy time jitter in seconds.")
    parser.add_argument("--artifact-competing-ratio", type=float, default=0.0, help="Chance to inject an unlabeled competing vehicle track.")
    parser.add_argument("--artifact-competing-time-jitter-s", type=float, default=0.8, help="Random time jitter for the competing vehicle anchor.")
    parser.add_argument("--artifact-competing-amp-scale-min", type=float, default=0.8, help="Minimum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-amp-scale-max", type=float, default=1.6, help="Maximum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-speed-ratio-min", type=float, default=0.88, help="Minimum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-speed-ratio-max", type=float, default=1.12, help="Maximum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-channel-offset-max", type=int, default=5, help="Maximum channel offset for the competing vehicle anchor relative to the target anchor.")
    parser.add_argument("--artifact-competing-opposite-direction-ratio", type=float, default=0.0, help="Probability of assigning the competing vehicle the opposite direction.")
    parser.add_argument("--benchmark-file", type=Path, default=None, help="Optional fixed benchmark .pt file for finetuning.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--mask-sigma-ch", type=float, default=0.8, help="Target heatmap sigma in channel units.")
    parser.add_argument("--mask-sigma-t", type=float, default=2.0, help="Target heatmap sigma in downsampled time bins.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory when transferring to CUDA.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--log-every", type=int, default=20, help="Print progress every N batches.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision on CUDA.")
    parser.add_argument("--heatmap-weight", type=float, default=1.0, help="Heatmap loss weight.")
    parser.add_argument("--time-consistency-weight", type=float, default=0.5, help="Adjacent-channel time slope consistency weight.")
    parser.add_argument("--competitor-weight", type=float, default=0.75, help="Hard-negative weight for the unlabeled competing vehicle.")
    parser.add_argument("--objectness-weight", type=float, default=0.5, help="Objectness loss weight.")
    parser.add_argument("--direction-weight", type=float, default=0.8, help="Direction loss weight.")
    parser.add_argument("--speed-weight", type=float, default=0.2, help="Speed loss weight.")
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


def _seed_worker(worker_id: int) -> None:
    torch.set_num_threads(1)
    np.random.seed((torch.initial_seed() + int(worker_id)) % (2**32))


def _collate(items: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in items], dim=0).contiguous()
    common_keys = set(items[0][1].keys())
    for _, target in items[1:]:
        common_keys &= set(target.keys())
    common_keys.discard("raw_window")
    keys = sorted(common_keys)
    targets = {key: torch.stack([item[1][key] for item in items], dim=0).contiguous() for key in keys}
    return xs, targets


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
        speed_norm_kmh=float(args.speed_norm_kmh),
        clip_ratio=float(args.clip_ratio),
        input_mode=str(args.input_mode),
        seed=int(seed),
        mask_sigma_ch=float(args.mask_sigma_ch),
        mask_sigma_t=float(args.mask_sigma_t),
        cache_dataset=False,
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


class _BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any]):
        self.samples = list(payload.get("samples", []))
        self.mask_sigma_ch = 0.8
        self.mask_sigma_t = 2.0
        self.sample_weights = [
            float(sample.get("target", {}).get("sample_weight", torch.tensor([1.0])).item())
            if isinstance(sample.get("target", {}).get("sample_weight", None), torch.Tensor)
            else float(sample.get("target", {}).get("sample_weight", 1.0))
            for sample in self.samples
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[int(index)]
        x = sample["x"].to(torch.float32)
        target = {key: value.clone() for key, value in sample["target"].items()}
        return x, target


def _sample_targets(
    targets: dict[str, torch.Tensor],
    *,
    n_channels: int,
    time_bins: int,
    mask_sigma_ch: float,
    mask_sigma_t: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(targets["time"].shape[0]) <= 0:
        raise ValueError("synthetic sample unexpectedly has no vehicle")
    time = targets["time"][0]
    visibility = targets["visibility"][0]
    heatmap = build_single_vehicle_heatmap_target(
        time,
        visibility,
        n_channels=int(n_channels),
        time_bins=int(time_bins),
        sigma_ch=float(mask_sigma_ch),
        sigma_t=float(mask_sigma_t),
    )
    objectness = torch.tensor(1.0, dtype=torch.float32)
    direction = targets["direction"][0].to(torch.long)
    speed = targets["speed"][0].to(torch.float32)
    line = build_single_vehicle_line_target(time, visibility)
    trajectory = build_single_vehicle_trajectory_target(time, visibility)
    comp_heatmap = torch.zeros_like(heatmap)
    comp_time = targets.get("artifact_competing_time")
    comp_vis = targets.get("artifact_competing_visibility")
    if comp_time is not None and comp_vis is not None:
        if bool(torch.as_tensor(comp_vis).max().item() > 0.0):
            comp_heatmap = build_single_vehicle_heatmap_target(
                comp_time,
                comp_vis,
                n_channels=int(n_channels),
                time_bins=int(time_bins),
                sigma_ch=float(mask_sigma_ch),
                sigma_t=float(mask_sigma_t),
            )
    return heatmap, objectness, direction, speed, visibility, comp_heatmap, line, trajectory


def _run_epoch(
    model: SingleVehiclePeakNet,
    loader: DataLoader,
    device: str,
    *,
    amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    heatmap_weight: float,
    time_consistency_weight: float,
    competitor_weight: float,
    objectness_weight: float,
    direction_weight: float,
    speed_weight: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    log_every: int = 20,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    seen = 0
    t0 = time.perf_counter()
    for step, (x, targets) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=(device == "cuda"))
        targets = {
            key: value.to(device, non_blocking=(device == "cuda"))
            for key, value in targets.items()
            if key != "raw_window"
        }
        heatmaps: list[torch.Tensor] = []
        competitor_heatmaps: list[torch.Tensor] = []
        line_targets: list[torch.Tensor] = []
        trajectory_targets: list[torch.Tensor] = []
        objectness: list[torch.Tensor] = []
        directions: list[torch.Tensor] = []
        speeds: list[torch.Tensor] = []
        valid_rows: list[int] = []
        for idx in range(int(x.shape[0])):
            if int(targets["time"][idx].shape[0]) <= 0:
                continue
            heatmap, obj, direction, speed, _, comp_heatmap, line, trajectory = _sample_targets(
                {key: value[idx] for key, value in targets.items()},
                n_channels=int(x.shape[2]),
                time_bins=int(x.shape[3]),
                mask_sigma_ch=float(loader.dataset.mask_sigma_ch),  # type: ignore[attr-defined]
                mask_sigma_t=float(loader.dataset.mask_sigma_t),  # type: ignore[attr-defined]
            )
            heatmaps.append(heatmap)
            competitor_heatmaps.append(comp_heatmap)
            line_targets.append(line)
            trajectory_targets.append(trajectory)
            objectness.append(obj)
            directions.append(direction)
            speeds.append(speed)
            valid_rows.append(int(idx))

        if not valid_rows:
            continue

        idx = torch.tensor(valid_rows, device=x.device, dtype=torch.long)
        x = x.index_select(0, idx)
        heatmap_tgt = torch.stack(heatmaps, dim=0).to(device)
        competitor_heatmap_tgt = torch.stack(competitor_heatmaps, dim=0).to(device)
        line_tgt = torch.stack(line_targets, dim=0).to(device)
        trajectory_tgt = torch.stack(trajectory_targets, dim=0).to(device)
        objectness_tgt = torch.stack(objectness, dim=0).to(device)
        direction_tgt = torch.stack(directions, dim=0).to(device)
        speed_tgt = torch.stack(speeds, dim=0).to(device)
        visibility_tgt = torch.stack([targets["visibility"][int(i)][0] for i in valid_rows], dim=0).to(device)
        time_tgt = torch.stack([targets["time"][int(i)][0] for i in valid_rows], dim=0).to(device)

        use_amp = bool(amp and device == "cuda")
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else nullcontext()
        with autocast_ctx:
            outputs = model(x)
            loss, loss_parts = single_vehicle_supervised_loss(
                outputs,
                target_heatmap=heatmap_tgt,
                target_competitor_heatmap=competitor_heatmap_tgt,
                target_line=line_tgt,
                target_trajectory=trajectory_tgt,
                target_visibility=visibility_tgt,
                target_time=time_tgt,
                target_objectness=objectness_tgt,
                target_direction=direction_tgt,
                target_speed=speed_tgt,
                heatmap_weight=float(heatmap_weight),
                visibility_weight=max(0.25, float(heatmap_weight) * 0.5),
                time_weight=max(1.0, float(heatmap_weight)),
                time_consistency_weight=float(time_consistency_weight),
                competitor_weight=float(competitor_weight),
                objectness_weight=float(objectness_weight),
                direction_weight=float(direction_weight),
                speed_weight=float(speed_weight),
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
        pred_dir = torch.tensor([estimate_direction_index_from_outputs({key: value[idx : idx + 1] for key, value in outputs.items()}) for idx in range(int(outputs["direction_logits"].shape[0]))], device=outputs["direction_logits"].device)
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch_size
        for key, value in loss_parts.items():
            totals[str(key)] = totals.get(str(key), 0.0) + float(value.detach().cpu()) * batch_size
        totals["direction_acc"] = totals.get("direction_acc", 0.0) + float((pred_dir == direction_tgt).float().mean().detach().cpu()) * batch_size
        totals["speed_mae"] = totals.get("speed_mae", 0.0) + float(torch.mean(torch.abs(outputs["speed"] - speed_tgt)).detach().cpu()) * batch_size
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
        train_len = int(max(1, args.train_samples))
        val_len = int(max(1, args.val_samples))
        train_ds = _build_dataset(args, length=train_len, seed=int(args.seed))
        val_ds = _build_dataset(args, length=val_len, seed=int(args.seed) + 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=False if args.benchmark_file is not None else True,
        sampler=WeightedRandomSampler(
            weights=[max(1e-6, float(w)) for w in getattr(train_ds, "sample_weights", [1.0] * len(train_ds))],
            num_samples=max(len(train_ds), int(len(train_ds) * 2)) if args.benchmark_file is not None else len(train_ds),
            replacement=True,
        )
        if args.benchmark_file is not None
        else None,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=_collate,
        worker_init_fn=_seed_worker if int(args.num_workers) > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=None,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        collate_fn=_collate,
        worker_init_fn=_seed_worker if int(args.num_workers) > 0 else None,
    )

    if args.benchmark_file is not None and len(train_ds) > 0:
        sample_x = train_ds[0][0]
        model_cfg = ModelConfig(n_channels=int(sample_x.shape[1]), in_channels=int(sample_x.shape[0]), hidden_dim=int(args.hidden_dim))
    else:
        model_cfg = ModelConfig(n_channels=int(args.n_channels), in_channels=1, hidden_dim=int(args.hidden_dim))
    model = SingleVehiclePeakNet(model_cfg).to(device)
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
            time_consistency_weight=float(args.time_consistency_weight),
            competitor_weight=float(args.competitor_weight),
            objectness_weight=float(args.objectness_weight),
            direction_weight=float(args.direction_weight),
            speed_weight=float(args.speed_weight),
            optimizer=optimizer,
            grad_clip=float(args.grad_clip),
            log_every=int(args.log_every),
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            device,
            amp=False,
            heatmap_weight=float(args.heatmap_weight),
            time_consistency_weight=float(args.time_consistency_weight),
            competitor_weight=float(args.competitor_weight),
            objectness_weight=float(args.objectness_weight),
            direction_weight=float(args.direction_weight),
            speed_weight=float(args.speed_weight),
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
