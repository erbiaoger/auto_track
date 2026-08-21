"""Train PeakSlotNet from peak-candidate tensor shards.

Purpose:
    Train `peak_slot` models on shards produced by
    `convert_track_slot_to_peak_slot.py`. The model assigns per-channel peak
    candidates to fixed vehicle slots. It does not regress arbitrary times.

Example:
    uv run python -m autotrack.dl.train_peak_slot \
        --data-dir datasets/peak_slot/train \
        --out-dir models/peak_slot_cuda \
        --device cuda \
        --amp on \
        --epochs 50 \
        --batch-size 32 \
        --val-fraction 0.1 \
        --val-every 5

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
import os
import time
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from autotrack.dl.peak_slot_model import (
    ModelConfig,
    PeakSlotPredictor,
    peak_slot_detection_metrics,
    peak_slot_physics_metrics,
    peak_slot_set_loss,
    save_checkpoint,
)
from autotrack.dl.trajectory_set_model import WindowDatasetConfig, auto_torch_device, move_targets_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PeakSlotNet from generated tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Dataset directory containing peak-slot meta.json and shards.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints and logs.")
    parser.add_argument("--epochs", type=int, default=50, help="Target total training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per epoch for smoke tests; 0 uses all.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. 0 uses the legacy synchronous shard iterator.")
    parser.add_argument("--pin-memory", action="store_true", help="Use pinned CPU memory for CUDA input transfer.")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when num-workers > 0.")
    parser.add_argument("--worker-shard-cache-size", type=int, default=2, help="Maximum loaded shards kept in each DataLoader worker; <=0 disables caching.")
    parser.add_argument(
        "--multiprocessing-context",
        default="auto",
        choices=["auto", "fork", "spawn", "forkserver"],
        help="DataLoader worker start method. auto uses fork on Unix when available.",
    )
    parser.add_argument("--dataloader-timeout", type=float, default=0.0, help="Seconds before DataLoader raises when a worker stalls; 0 disables.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--max-tracks", type=int, default=96, help="Output slots Q.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Model hidden dimension.")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Transformer decoder layers.")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--pooled-channels", type=int, default=8, help="Pooled channel tokens.")
    parser.add_argument("--pooled-time", type=int, default=128, help="Pooled time tokens.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Decoder dropout.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--amp", default="auto", choices=["auto", "on", "off"], help="Use CUDA automatic mixed precision.")
    parser.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"], help="CUDA AMP dtype.")
    parser.add_argument(
        "--input-transfer-dtype",
        default="auto",
        choices=["auto", "preserve", "float32", "float16"],
        help="CPU shard x dtype before GPU transfer. auto preserves shard dtype only for CUDA AMP training.",
    )
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last input/model layout on CUDA.")
    parser.add_argument(
        "--matcher",
        default="independent",
        choices=["hungarian", "greedy", "auction", "independent"],
        help="Slot-to-GT assignment. independent is the fastest approximate GPU matcher for throughput training.",
    )
    parser.add_argument("--no-object-weight", type=float, default=0.15, help="Object loss weight for unmatched slots.")
    parser.add_argument("--none-weight", type=float, default=0.35, help="Peak CE weight for GT-none channel targets.")
    parser.add_argument("--object-loss-weight", type=float, default=1.25, help="Overall objectness loss multiplier.")
    parser.add_argument("--count-loss-weight", type=float, default=0.15, help="Soft count loss weight.")
    parser.add_argument("--monotonic-loss-weight", type=float, default=1.0, help="Monotonic loss weight.")
    parser.add_argument("--smoothness-loss-weight", type=float, default=0.2, help="Smoothness loss weight.")
    parser.add_argument("--time-prior-loss-weight", type=float, default=2.0, help="Time prior supervision loss weight.")
    parser.add_argument("--visibility-prior-loss-weight", type=float, default=0.75, help="Visibility prior supervision loss weight.")
    parser.add_argument("--slot-competition-loss-weight", type=float, default=0.15, help="Penalty for multiple slots selecting the same peaks.")
    parser.add_argument("--crossing-loss-weight", type=float, default=0.2, help="Margin loss against switching to competing peaks.")
    parser.add_argument("--gt-coverage-loss-weight", type=float, default=0.5, help="Loss weight requiring every GT to be explainable by at least one slot.")
    parser.add_argument("--gt-coverage-temperature", type=float, default=0.2, help="Softmin temperature for GT coverage loss.")
    parser.add_argument("--close-pair-separation-loss-weight", type=float, default=0.3, help="Loss weight for separating close GT vehicle pairs.")
    parser.add_argument("--close-pair-margin", type=float, default=0.5, help="Margin for close-pair slot separation loss.")
    parser.add_argument("--close-pair-min-common-channels", type=int, default=8, help="Minimum shared visible channels for a close GT pair.")
    parser.add_argument("--close-pair-min-gap-s", type=float, default=0.15, help="Minimum mean time gap for close-pair metrics/loss.")
    parser.add_argument("--close-pair-max-gap-s", type=float, default=1.5, help="Maximum mean time gap for close-pair metrics/loss.")
    parser.add_argument("--physics-speed-min-kmh", type=float, default=60.0, help="Minimum speed for physics violation metrics.")
    parser.add_argument("--physics-speed-max-kmh", type=float, default=100.0, help="Maximum speed for physics violation metrics.")
    parser.add_argument("--metric-objectness-threshold", type=float, default=0.35, help="Objectness threshold for metrics.")
    parser.add_argument("--metric-point-threshold", type=float, default=0.05, help="Normalized mean time error threshold for TP metrics.")
    parser.add_argument("--val-data-dir", type=Path, default=None, help="Optional separate peak-slot validation dataset directory. Overrides --val-fraction when provided.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Fraction of shards reserved for validation.")
    parser.add_argument("--val-every", type=int, default=5, help="Run validation every N epochs when val shards exist.")
    parser.add_argument("--val-max-samples", type=int, default=0, help="Limit validation samples; 0 evaluates all validation samples.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument("--metrics-every", type=int, default=0, help="Collect detailed metrics every N batches; 0 disables intermediate metrics.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    parser.add_argument("--auto-resume", action="store_true", help="Resume from <out-dir>/checkpoint_last.pt if present.")
    parser.add_argument("--resume-model-only", action="store_true", help="Load model weights but reset optimizer.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=20, help="Print batch progress every N batches.")
    parser.add_argument("--timing-every", type=int, default=5, help="Print and log epoch-level step timing every N epochs; 0 disables.")
    parser.add_argument("--profile-steps", type=int, default=0, help="Print step timing for the first N profiled batches; 0 disables.")
    parser.add_argument("--profile-warmup", type=int, default=2, help="Skip this many batches before printing step timing.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _split_shards(shards: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    if not shards:
        raise ValueError("Dataset contains no shards")
    val_count = int(round(len(shards) * max(0.0, min(0.9, float(val_fraction)))))
    if val_count <= 0:
        return shards, []
    val_count = min(len(shards) - 1, val_count)
    return shards[:-val_count], shards[-val_count:]


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _sync_for_timing(device: str, enabled: bool) -> None:
    if enabled and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def _resolve_resume_path(args: argparse.Namespace) -> Optional[Path]:
    if args.resume is not None:
        path = Path(args.resume).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
        return path
    if bool(args.auto_resume):
        path = Path(args.out_dir).expanduser() / "checkpoint_last.pt"
        if path.is_file():
            return path
        print(f"Auto-resume requested, but no checkpoint found at {path}; starting a new run.", flush=True)
    return None


def _model_config_from_meta(args: argparse.Namespace, meta: dict) -> ModelConfig:
    return ModelConfig(
        n_channels=int(meta["n_channels"]),
        in_channels=int(meta["in_channels"]),
        max_tracks=max(int(args.max_tracks), int(meta.get("max_gt", 0))),
        peak_candidates=int(meta["peak_candidates_per_channel"]),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        decoder_layers=int(args.decoder_layers),
        pooled_channels=int(args.pooled_channels),
        pooled_time=int(args.pooled_time),
        dropout=float(args.dropout),
    )


def _resume_config_with_dataset_input(checkpoint_config: ModelConfig, args: argparse.Namespace, meta: dict) -> ModelConfig:
    if not bool(args.resume_model_only):
        return checkpoint_config
    target_in_channels = int(meta["in_channels"])
    target_n_channels = int(meta["n_channels"])
    target_peak_candidates = int(meta["peak_candidates_per_channel"])
    if (
        int(checkpoint_config.in_channels) == target_in_channels
        and int(checkpoint_config.n_channels) == target_n_channels
        and int(checkpoint_config.peak_candidates) == target_peak_candidates
    ):
        return checkpoint_config
    cfg = ModelConfig(**asdict(checkpoint_config))
    cfg.in_channels = target_in_channels
    cfg.n_channels = target_n_channels
    cfg.peak_candidates = target_peak_candidates
    cfg.max_tracks = max(int(checkpoint_config.max_tracks), int(args.max_tracks), int(meta.get("max_gt", 0)))
    return cfg


def _adapt_resume_state_dict(model: PeakSlotPredictor, checkpoint_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    target_state = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        if key not in target_state:
            adapted[key] = value
            continue
        target = target_state[key]
        if tuple(value.shape) == tuple(target.shape):
            adapted[key] = value
            continue
        if key == "backbone.0.net.0.weight" and value.ndim == 4 and target.ndim == 4:
            new_value = target.detach().clone()
            common_in = min(int(value.shape[1]), int(target.shape[1]))
            new_value[:, :common_in] = value[:, :common_in].to(dtype=new_value.dtype)
            if int(target.shape[1]) > int(value.shape[1]):
                new_value[:, int(value.shape[1]) :] = 0.0
            adapted[key] = new_value
            print(
                f"Adapted {key} from shape={tuple(value.shape)} to shape={tuple(target.shape)}; "
                "new input channels initialized to 0.",
                flush=True,
            )
            continue
        skipped.append(f"{key}: checkpoint={tuple(value.shape)} target={tuple(target.shape)}")
    if skipped:
        print(f"Skipped incompatible resume tensors: {skipped}", flush=True)
    return adapted


def _resolve_x_transfer_dtype(args: argparse.Namespace, *, device: str, use_amp: bool) -> str:
    requested = str(args.input_transfer_dtype).lower()
    if requested != "auto":
        return requested
    if str(device).startswith("cuda") and bool(use_amp):
        return "preserve"
    return "float32"


def _shard_batch_count(meta: dict, all_shards: list[str], shards: list[str], batch_size: int, max_samples: int) -> int:
    if int(max_samples) > 0:
        return int(math.ceil(int(max_samples) / max(1, int(batch_size))))
    total = 0
    shard_size = int(meta.get("shard_size", 1))
    sample_count = int(meta.get("num_samples", 0))
    for shard in shards:
        shard_idx = all_shards.index(shard)
        n = max(0, sample_count - shard_idx * shard_size) if shard_idx == len(all_shards) - 1 else shard_size
        total += int(math.ceil(int(n) / max(1, int(batch_size))))
    return total


def _shard_sample_count(meta: dict, all_shards: list[str], shard: str) -> int:
    shard_idx = all_shards.index(shard)
    shard_size = int(meta.get("shard_size", 1))
    sample_count = int(meta.get("num_samples", 0))
    if shard_idx == len(all_shards) - 1:
        return max(0, sample_count - shard_idx * shard_size)
    return int(shard_size)


def _convert_x_dtype(x: torch.Tensor, x_transfer_dtype: str) -> torch.Tensor:
    if str(x_transfer_dtype) == "float32":
        return x.to(torch.float32)
    if str(x_transfer_dtype) == "float16":
        return x.to(torch.float16)
    return x


class EpochShuffleSampler(Sampler[int]):
    """Deterministic per-epoch sampler compatible with persistent workers.

    When shard offsets are available, keep accesses shard-local so worker shard
    caches can actually help. We still shuffle shard order per epoch and shuffle
    sample order inside each shard.
    """

    def __init__(
        self,
        length: int,
        *,
        seed: int,
        epoch: int,
        shuffle: bool,
        shard_offsets: Optional[list[tuple[int, int]]] = None,
    ):
        self.length = int(length)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle = bool(shuffle)
        self.shard_offsets = list(shard_offsets or [])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle or self.length <= 1:
            return iter(range(self.length))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + self.epoch * 97_531)
        if self.shard_offsets:
            order: list[int] = []
            for shard_id in torch.randperm(len(self.shard_offsets), generator=gen).tolist():
                start, stop = self.shard_offsets[int(shard_id)]
                shard_len = max(0, int(stop) - int(start))
                if shard_len <= 0:
                    continue
                local_order = torch.randperm(shard_len, generator=gen).tolist()
                order.extend(int(start) + int(offset) for offset in local_order)
            return iter(order)
        return iter(torch.randperm(self.length, generator=gen).tolist())

    def __len__(self) -> int:
        return self.length


class PeakSlotShardDataset(Dataset):
    """Map-style dataset backed by peak-slot shard .pt files.

    Each worker keeps a small per-process shard cache, so shuffled sample access
    does not reload the same shard for every item.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        meta: dict,
        all_shards: list[str],
        shards: list[str],
        max_samples: int,
        x_transfer_dtype: str,
        worker_shard_cache_size: int,
    ):
        self.data_dir = Path(data_dir)
        self.x_transfer_dtype = str(x_transfer_dtype)
        self.worker_shard_cache_size = int(worker_shard_cache_size)
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        refs: list[tuple[str, int]] = []
        shard_offsets: list[tuple[int, int]] = []
        limit = int(max_samples)
        for shard in shards:
            shard_start = len(refs)
            n = _shard_sample_count(meta, all_shards, shard)
            for idx in range(n):
                if limit > 0 and len(refs) >= limit:
                    break
                refs.append((str(shard), int(idx)))
            shard_stop = len(refs)
            if shard_stop > shard_start:
                shard_offsets.append((shard_start, shard_stop))
            if limit > 0 and len(refs) >= limit:
                break
        self.refs = refs
        self.shard_offsets = shard_offsets

    def __len__(self) -> int:
        return len(self.refs)

    def _payload(self, shard: str) -> dict[str, torch.Tensor]:
        payload = self._cache.get(shard)
        if payload is not None:
            self._cache.move_to_end(shard)
            return payload
        payload = torch.load(str(self.data_dir / shard), map_location="cpu", weights_only=False)
        if self.worker_shard_cache_size > 0:
            self._cache[shard] = payload
            while len(self._cache) > self.worker_shard_cache_size:
                self._cache.popitem(last=False)
        return payload

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shard, sample_idx = self.refs[int(index)]
        payload = self._payload(shard)
        x = _convert_x_dtype(payload["x"][sample_idx], self.x_transfer_dtype)
        target = {
            "peak_time": payload["peak_time"][sample_idx].to(torch.float32),
            "peak_amp": payload["peak_amp"][sample_idx].to(torch.float32),
            "peak_valid": payload["peak_valid"][sample_idx].to(torch.bool),
            "peak_index": payload["peak_index"][sample_idx].to(torch.long),
            "gt_peak_index": payload["gt_peak_index"][sample_idx].to(torch.long),
            "visibility": payload["visibility"][sample_idx].to(torch.float32),
            "direction": payload["direction"][sample_idx].to(torch.long),
            "speed": payload["speed"][sample_idx].to(torch.float32),
            "gt_valid": payload["gt_valid"][sample_idx].to(torch.bool),
        }
        target["gt_count"] = target["gt_valid"].sum().to(torch.long)
        return x, target


def _collate_peak_slot_batch(items: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in items], dim=0).contiguous()
    keys = list(items[0][1].keys())
    targets = {key: torch.stack([item[1][key] for item in items], dim=0).contiguous() for key in keys}
    return xs, targets


def _seed_worker(worker_id: int) -> None:
    torch.set_num_threads(1)
    np.random.seed((torch.initial_seed() + int(worker_id)) % (2**32))


def _shutdown_dataloader(loader: Optional[DataLoader]) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    _shutdown_dataloader_iterator(iterator)


def _shutdown_dataloader_iterator(iterator: object) -> None:
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def _make_batch_iter(
    data_dir: Path,
    *,
    meta: dict,
    all_shards: list[str],
    shards: list[str],
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    max_samples: int,
    x_transfer_dtype: str,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    multiprocessing_context: str,
    dataloader_timeout: float,
    worker_shard_cache_size: int,
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    if int(num_workers) <= 0:
        return _iter_batches(
            data_dir,
            shards=shards,
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            seed=int(seed),
            epoch=int(epoch),
            max_samples=int(max_samples),
            x_transfer_dtype=str(x_transfer_dtype),
        )
    loader = _make_dataloader(
        data_dir,
        meta=meta,
        all_shards=all_shards,
        shards=shards,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        epoch=epoch,
        max_samples=int(max_samples),
        x_transfer_dtype=str(x_transfer_dtype),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=str(multiprocessing_context),
        dataloader_timeout=float(dataloader_timeout),
        worker_shard_cache_size=int(worker_shard_cache_size),
    )
    return iter(loader)


def _resolve_multiprocessing_context(requested: str, *, device: str) -> str:
    requested = str(requested).strip().lower()
    if requested != "auto":
        return requested
    return "fork" if hasattr(os, "fork") else "spawn"


def _make_dataloader(
    data_dir: Path,
    *,
    meta: dict,
    all_shards: list[str],
    shards: list[str],
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    max_samples: int,
    x_transfer_dtype: str,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    multiprocessing_context: str,
    dataloader_timeout: float,
    worker_shard_cache_size: int,
) -> DataLoader:
    dataset = PeakSlotShardDataset(
        data_dir,
        meta=meta,
        all_shards=all_shards,
        shards=shards,
        max_samples=int(max_samples),
        x_transfer_dtype=str(x_transfer_dtype),
        worker_shard_cache_size=int(worker_shard_cache_size),
    )
    sampler = EpochShuffleSampler(
        len(dataset),
        seed=int(seed),
        epoch=int(epoch),
        shuffle=bool(shuffle),
        shard_offsets=dataset.shard_offsets,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers),
        "prefetch_factor": max(1, int(prefetch_factor)),
        "collate_fn": _collate_peak_slot_batch,
        "worker_init_fn": _seed_worker,
        "timeout": max(0.0, float(dataloader_timeout)),
    }
    if int(num_workers) > 0:
        loader_kwargs["multiprocessing_context"] = str(multiprocessing_context)
    return DataLoader(**loader_kwargs)


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
    max_samples: int,
    x_transfer_dtype: str = "float32",
) -> Iterator[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + int(epoch) * 97_531)
    emitted = 0
    shard_order = list(shards)
    if shuffle and len(shard_order) > 1:
        perm = torch.randperm(len(shard_order), generator=gen).tolist()
        shard_order = [shard_order[i] for i in perm]
    for shard in shard_order:
        payload = torch.load(str(data_dir / shard), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        order = torch.randperm(n, generator=gen) if shuffle and n > 1 else torch.arange(n)
        for start in range(0, n, int(batch_size)):
            if int(max_samples) > 0 and emitted >= int(max_samples):
                return
            idx = order[start : start + int(batch_size)]
            if int(max_samples) > 0:
                idx = idx[: max(0, int(max_samples) - emitted)]
            if int(idx.numel()) <= 0:
                return
            emitted += int(idx.numel())
            targets = {
                "peak_time": payload["peak_time"][idx].to(torch.float32),
                "peak_amp": payload["peak_amp"][idx].to(torch.float32),
                "peak_valid": payload["peak_valid"][idx].to(torch.bool),
                "peak_index": payload["peak_index"][idx].to(torch.long),
                "gt_peak_index": payload["gt_peak_index"][idx].to(torch.long),
                "visibility": payload["visibility"][idx].to(torch.float32),
                "direction": payload["direction"][idx].to(torch.long),
                "speed": payload["speed"][idx].to(torch.float32),
                "gt_valid": payload["gt_valid"][idx].to(torch.bool),
            }
            targets["gt_count"] = targets["gt_valid"].sum(dim=1).to(torch.long)
            x = _convert_x_dtype(payload["x"][idx], str(x_transfer_dtype))
            yield x, targets


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
    return x, move_targets_to_device(targets, device, non_blocking=non_blocking)  # type: ignore[return-value]


def _forward(model: PeakSlotPredictor, x: torch.Tensor, targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return model(x, targets["peak_time"], targets["peak_amp"], targets["peak_valid"])


def _evaluate(
    model: PeakSlotPredictor,
    data_dir: Path,
    shards: list[str],
    device: str,
    args: argparse.Namespace,
    meta: dict,
    *,
    max_samples: int,
) -> dict[str, float]:
    model.eval()
    metrics_items: list[dict[str, float]] = []
    with torch.no_grad():
        all_shards = [str(item) for item in meta.get("shards", [])]
        batch_iter = _make_batch_iter(
            data_dir,
            meta=meta,
            all_shards=all_shards,
            shards=shards,
            batch_size=int(args.batch_size),
            shuffle=False,
            seed=int(args.seed),
            epoch=0,
            max_samples=int(max_samples),
            x_transfer_dtype="float32",
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory and str(device).startswith("cuda")),
            persistent_workers=bool(args.persistent_workers and int(args.num_workers) > 0),
            prefetch_factor=int(args.prefetch_factor),
            multiprocessing_context=str(args.multiprocessing_context),
            dataloader_timeout=float(args.dataloader_timeout),
            worker_shard_cache_size=int(args.worker_shard_cache_size),
        )
        try:
            for x, targets in batch_iter:
                x, targets = _batch_to_device(x, targets, device, channels_last=bool(args.channels_last and str(device).startswith("cuda")))
                outputs = _forward(model, x, targets)
                _, metrics = peak_slot_set_loss(
                    outputs,
                    targets,
                    no_object_weight=float(args.no_object_weight),
                    none_weight=float(args.none_weight),
                    matcher=str(args.matcher),
                    object_loss_weight=float(args.object_loss_weight),
                    count_loss_weight=float(args.count_loss_weight),
                    monotonic_loss_weight=float(args.monotonic_loss_weight),
                    smoothness_loss_weight=float(args.smoothness_loss_weight),
                    time_prior_loss_weight=float(args.time_prior_loss_weight),
                    visibility_prior_loss_weight=float(args.visibility_prior_loss_weight),
                    slot_competition_loss_weight=float(args.slot_competition_loss_weight),
                    crossing_loss_weight=float(args.crossing_loss_weight),
                    gt_coverage_loss_weight=float(args.gt_coverage_loss_weight),
                    gt_coverage_temperature=float(args.gt_coverage_temperature),
                    close_pair_separation_loss_weight=float(args.close_pair_separation_loss_weight),
                    close_pair_margin=float(args.close_pair_margin),
                    close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                    close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                    close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                    close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                    collect_metrics=True,
                )
                metrics.update(
                    peak_slot_detection_metrics(
                        outputs,
                        targets,
                        objectness_threshold=float(args.metric_objectness_threshold),
                        point_threshold=float(args.metric_point_threshold),
                        matcher=str(args.matcher),
                        close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                        close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                        close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                        close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                    )
                )
                metrics.update(
                    peak_slot_physics_metrics(
                        outputs,
                        targets,
                        objectness_threshold=float(args.metric_objectness_threshold),
                        peak_threshold=0.4,
                        speed_min_kmh=float(args.physics_speed_min_kmh),
                        speed_max_kmh=float(args.physics_speed_max_kmh),
                        time_downsample=int(meta.get("time_downsample", 10)),
                        fs=float(meta.get("fs", 1000.0)),
                        dx_m=float(meta.get("dx_m", 100.0)),
                    )
                )
                metrics_items.append(metrics)
        finally:
            _shutdown_dataloader_iterator(batch_iter)
    return _mean_metrics(metrics_items)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    val_data_dir = Path(args.val_data_dir).expanduser() if args.val_data_dir is not None else None
    args.data_dir = data_dir
    args.out_dir = out_dir
    args.val_data_dir = val_data_dir
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    val_meta: dict = meta
    if val_data_dir is not None:
        val_meta = _load_meta(val_data_dir)
        train_shards = shards
        val_shards = [str(item) for item in val_meta.get("shards", [])]
        if not val_shards:
            raise ValueError(f"Validation dataset contains no shards: {val_data_dir}")
    else:
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
        checkpoint_config = ModelConfig(**dict(resume_checkpoint.get("model_config", {})))
        model_config = _resume_config_with_dataset_input(checkpoint_config, args, meta)
        print(
            f"Resuming from checkpoint: {resume_path} at epoch={resume_epoch}; "
            f"checkpoint_in_channels={checkpoint_config.in_channels}, target_in_channels={model_config.in_channels}",
            flush=True,
        )
        if bool(args.resume_model_only):
            print("Resume-model-only requested; resetting training epoch counter to 0.", flush=True)
            resume_epoch = 0
    else:
        model_config = _model_config_from_meta(args, meta)
    dataset_config = WindowDatasetConfig(
        window_seconds=float(meta["window_seconds"]),
        time_downsample=int(meta["time_downsample"]),
        samples_per_folder=int(meta["num_samples"]),
        min_visible_channels=int(meta.get("generator_args", {}).get("min_visible_channels", 2)),
        speed_norm_kmh=float(meta.get("speed_norm_kmh", 150.0)),
        clip_ratio=float(meta.get("clip_ratio", 1.35)),
        input_mode=str(meta.get("input_mode", "raw")),
        seed=int(meta.get("generator_args", {}).get("seed", args.seed)),
    )
    model = PeakSlotPredictor(model_config).to(device)
    if bool(args.channels_last) and str(device).startswith("cuda"):
        model = model.to(memory_format=torch.channels_last)
    if resume_checkpoint is not None:
        resume_state = _adapt_resume_state_dict(model, resume_checkpoint["model_state"])
        missing, unexpected = model.load_state_dict(resume_state, strict=False)
        if missing:
            print(f"Resume checkpoint missing newly initialized keys: {missing}", flush=True)
        if unexpected:
            print(f"Resume checkpoint ignored unexpected keys: {unexpected}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    if resume_checkpoint is not None and not bool(args.resume_model_only) and "optimizer_state" in resume_checkpoint:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
            _move_optimizer_state_to_device(optimizer, device)
            print("Loaded optimizer state from checkpoint.", flush=True)
        except ValueError as exc:
            print(f"Skipped optimizer state because model parameters changed: {exc}", flush=True)
    elif resume_checkpoint is not None:
        print("Loaded model weights only; optimizer starts from scratch.", flush=True)
    use_amp = (str(args.amp) == "on") or (str(args.amp) == "auto" and str(device).startswith("cuda"))
    amp_dtype = torch.float16 if str(args.amp_dtype) == "float16" else torch.bfloat16
    x_transfer_dtype = _resolve_x_transfer_dtype(args, device=device, use_amp=bool(use_amp))
    multiprocessing_context = _resolve_multiprocessing_context(str(args.multiprocessing_context), device=device)
    args.multiprocessing_context = multiprocessing_context
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and amp_dtype == torch.float16))
    out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "peak_slot_shards",
        "model_family": "peak_slot",
        "data_dir": str(data_dir),
        "val_data_dir": "" if val_data_dir is None else str(val_data_dir),
        "train_shards": train_shards,
        "val_shards": val_shards,
        "dataset_meta": meta,
        "val_dataset_meta": val_meta if val_data_dir is not None else {},
        "dataset_config": asdict(dataset_config),
        "model_config": asdict(model_config),
        "train_args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "resolved_resume": str(resume_path) if resume_path is not None else "",
        "device": device,
        "created_at_unix": time.time(),
    }
    (out_dir / "train_config.json").write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    batches_per_epoch = _shard_batch_count(meta, shards, train_shards, int(args.batch_size), int(args.max_samples))
    print(
        "Dataset: "
        f"samples={meta.get('num_samples')}, train_shards={len(train_shards)}, val_shards={len(val_shards)}, "
        f"batch_size={int(args.batch_size)}, batches_per_epoch={batches_per_epoch}, peaks={model_config.peak_candidates}, "
        f"workers={int(args.num_workers)}, pin_memory={bool(args.pin_memory and str(device).startswith('cuda'))}, "
        f"mp_context={multiprocessing_context}, dataloader_timeout={float(args.dataloader_timeout):.1f}s, "
        f"worker_shard_cache_size={int(args.worker_shard_cache_size)}"
    )
    if val_shards:
        val_source = str(val_data_dir) if val_data_dir is not None else f"{float(args.val_fraction):.3f} tail split"
        val_limit = int(args.val_max_samples)
        print(
            "Validation: "
            f"source={val_source}, val_every={int(args.val_every)}, "
            f"val_samples={val_meta.get('num_samples')}, val_max_samples={val_limit if val_limit > 0 else 'all'}",
            flush=True,
        )
    print(
        "Model: "
        f"slots={model_config.max_tracks}, hidden_dim={model_config.hidden_dim}, "
        f"decoder_layers={model_config.decoder_layers}, pooled=({model_config.pooled_channels}, {model_config.pooled_time}), "
        f"amp={use_amp}, matcher={args.matcher}, x_transfer_dtype={x_transfer_dtype}"
    )
    best_loss = float("inf")
    if resume_checkpoint is not None and not bool(args.resume_model_only):
        metrics = dict(resume_checkpoint.get("metrics", {}))
        best_loss = float(metrics.get("val_loss", metrics.get("loss", best_loss)))
    start_epoch = resume_epoch + 1 if resume_checkpoint is not None else 1
    if start_epoch > int(args.epochs):
        print(f"Checkpoint epoch={resume_epoch} is already >= target epochs={int(args.epochs)}; nothing to train.")
        return 0

    train_loader: Optional[DataLoader] = None
    if int(args.num_workers) > 0:
        train_loader = _make_dataloader(
            data_dir,
            meta=meta,
            all_shards=shards,
            shards=train_shards,
            batch_size=int(args.batch_size),
            shuffle=True,
            seed=int(args.seed),
            epoch=int(start_epoch),
            max_samples=int(args.max_samples),
            x_transfer_dtype=str(x_transfer_dtype),
            num_workers=int(args.num_workers),
            pin_memory=bool(args.pin_memory and str(device).startswith("cuda")),
            persistent_workers=bool(args.persistent_workers),
            prefetch_factor=int(args.prefetch_factor),
            multiprocessing_context=str(multiprocessing_context),
            dataloader_timeout=float(args.dataloader_timeout),
            worker_shard_cache_size=int(args.worker_shard_cache_size),
        )

    for epoch in range(start_epoch, int(args.epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_metrics: list[dict[str, float]] = []
        epoch_samples = 0
        epoch_batches = 0
        timing_enabled = int(args.timing_every) > 0 and epoch % int(args.timing_every) == 0
        timing_sums = {
            "data_wait": 0.0,
            "h2d": 0.0,
            "zero_grad": 0.0,
            "forward": 0.0,
            "loss": 0.0,
            "backward": 0.0,
            "update": 0.0,
            "metrics": 0.0,
            "total": 0.0,
        }
        timing_batches = 0
        fetch_start = time.perf_counter()
        print(f"Starting epoch={epoch:03d}; waiting for first batch...", flush=True)
        if train_loader is None:
            batch_iter = _make_batch_iter(
                data_dir,
                meta=meta,
                all_shards=shards,
                shards=train_shards,
                batch_size=int(args.batch_size),
                shuffle=True,
                seed=int(args.seed),
                epoch=epoch,
                max_samples=int(args.max_samples),
                x_transfer_dtype=str(x_transfer_dtype),
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=int(args.prefetch_factor),
                multiprocessing_context=str(multiprocessing_context),
                dataloader_timeout=float(args.dataloader_timeout),
                worker_shard_cache_size=int(args.worker_shard_cache_size),
            )
        else:
            sampler = getattr(train_loader, "sampler", None)
            set_epoch = getattr(sampler, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)
            batch_iter = iter(train_loader)
        try:
            for batch_idx, (x, targets) in enumerate(batch_iter, start=1):
                batch_samples = int(x.shape[0])
                epoch_samples += batch_samples
                epoch_batches += 1
                profile_enabled = int(args.profile_steps) > 0 and batch_idx > int(args.profile_warmup)
                profile_active = profile_enabled and batch_idx <= int(args.profile_warmup) + int(args.profile_steps)
                sync_timing = bool(profile_active or timing_enabled)
                data_wait_s = time.perf_counter() - fetch_start
                _sync_for_timing(device, sync_timing)
                step_t0 = time.perf_counter()
                should_log = int(args.log_every) > 0 and (batch_idx == 1 or batch_idx % int(args.log_every) == 0 or batch_idx == batches_per_epoch)
                should_collect_heavy_metrics = int(args.metrics_every) > 0 and batch_idx % int(args.metrics_every) == 0
                collect_loss_metrics = should_log or batch_idx == 1 or batch_idx == batches_per_epoch or should_collect_heavy_metrics
                x, targets = _batch_to_device(x, targets, device, channels_last=bool(args.channels_last and str(device).startswith("cuda")))
                _sync_for_timing(device, sync_timing)
                h2d_s = time.perf_counter() - step_t0
                optim_t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                _sync_for_timing(device, sync_timing)
                zero_grad_s = time.perf_counter() - optim_t0
                forward_t0 = time.perf_counter()
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=bool(use_amp and str(device).startswith("cuda"))):
                    outputs = _forward(model, x, targets)
                    _sync_for_timing(device, sync_timing)
                    forward_s = time.perf_counter() - forward_t0
                    loss_t0 = time.perf_counter()
                    loss, metrics = peak_slot_set_loss(
                        outputs,
                        targets,
                        no_object_weight=float(args.no_object_weight),
                        none_weight=float(args.none_weight),
                        matcher=str(args.matcher),
                        object_loss_weight=float(args.object_loss_weight),
                        count_loss_weight=float(args.count_loss_weight),
                        monotonic_loss_weight=float(args.monotonic_loss_weight),
                        smoothness_loss_weight=float(args.smoothness_loss_weight),
                        time_prior_loss_weight=float(args.time_prior_loss_weight),
                        visibility_prior_loss_weight=float(args.visibility_prior_loss_weight),
                        slot_competition_loss_weight=float(args.slot_competition_loss_weight),
                        crossing_loss_weight=float(args.crossing_loss_weight),
                        gt_coverage_loss_weight=float(args.gt_coverage_loss_weight),
                        gt_coverage_temperature=float(args.gt_coverage_temperature),
                        close_pair_separation_loss_weight=float(args.close_pair_separation_loss_weight),
                        close_pair_margin=float(args.close_pair_margin),
                        close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                        close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                        close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                        close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                        collect_metrics=bool(collect_loss_metrics),
                    )
                    _sync_for_timing(device, sync_timing)
                    loss_s = time.perf_counter() - loss_t0
                backward_t0 = time.perf_counter()
                scaler.scale(loss).backward()
                _sync_for_timing(device, sync_timing)
                backward_s = time.perf_counter() - backward_t0
                step_update_t0 = time.perf_counter()
                if float(args.grad_clip) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
                scaler.step(optimizer)
                scaler.update()
                _sync_for_timing(device, sync_timing)
                update_s = time.perf_counter() - step_update_t0
                if metrics:
                    metrics_t0 = time.perf_counter()
                    if should_collect_heavy_metrics:
                        metrics.update(
                            peak_slot_detection_metrics(
                                outputs,
                                targets,
                                objectness_threshold=float(args.metric_objectness_threshold),
                                point_threshold=float(args.metric_point_threshold),
                                matcher=str(args.matcher),
                                close_pair_window_seconds=float(meta.get("window_seconds", 120.0)),
                                close_pair_min_common_channels=int(args.close_pair_min_common_channels),
                                close_pair_min_gap_s=float(args.close_pair_min_gap_s),
                                close_pair_max_gap_s=float(args.close_pair_max_gap_s),
                            )
                        )
                        metrics.update(
                            peak_slot_physics_metrics(
                                outputs,
                                targets,
                                objectness_threshold=float(args.metric_objectness_threshold),
                                peak_threshold=0.4,
                                speed_min_kmh=float(args.physics_speed_min_kmh),
                                speed_max_kmh=float(args.physics_speed_max_kmh),
                                time_downsample=int(meta.get("time_downsample", 10)),
                                fs=float(meta.get("fs", 1000.0)),
                                dx_m=float(meta.get("dx_m", 100.0)),
                            )
                        )
                    _sync_for_timing(device, sync_timing)
                    metrics_s = time.perf_counter() - metrics_t0
                    epoch_metrics.append(metrics)
                else:
                    metrics_s = 0.0
                total_s = data_wait_s + h2d_s + zero_grad_s + forward_s + loss_s + backward_s + update_s + metrics_s
                if timing_enabled:
                    timing_sums["data_wait"] += float(data_wait_s)
                    timing_sums["h2d"] += float(h2d_s)
                    timing_sums["zero_grad"] += float(zero_grad_s)
                    timing_sums["forward"] += float(forward_s)
                    timing_sums["loss"] += float(loss_s)
                    timing_sums["backward"] += float(backward_s)
                    timing_sums["update"] += float(update_s)
                    timing_sums["metrics"] += float(metrics_s)
                    timing_sums["total"] += float(total_s)
                    timing_batches += 1
                if profile_active:
                    print(
                        f"profile epoch={epoch:03d} batch={batch_idx:04d} "
                        f"data_wait={data_wait_s * 1000.0:.1f}ms "
                        f"h2d={h2d_s * 1000.0:.1f}ms "
                        f"zero={zero_grad_s * 1000.0:.1f}ms "
                        f"forward={forward_s * 1000.0:.1f}ms "
                        f"loss={loss_s * 1000.0:.1f}ms "
                        f"backward={backward_s * 1000.0:.1f}ms "
                        f"update={update_s * 1000.0:.1f}ms "
                        f"metrics={metrics_s * 1000.0:.1f}ms "
                        f"total={total_s * 1000.0:.1f}ms",
                        flush=True,
                    )
                if should_log and metrics:
                    print(
                        f"epoch={epoch:03d} batch={batch_idx:04d}/{batches_per_epoch:04d} "
                        f"loss={metrics.get('loss', float('nan')):.4f} "
                        f"peak={metrics.get('loss_peak', float('nan')):.4f} "
                        f"obj={metrics.get('loss_obj', float('nan')):.4f} "
                        f"cnt_loss={metrics.get('loss_count', float('nan')):.2f} "
                        f"f1={metrics.get('track_f1', float('nan')):.3f} "
                        f"cnt_mae={metrics.get('count_mae', float('nan')):.2f} "
                        f"spd_bad={metrics.get('speed_window_violation_rate', float('nan')):.3f}",
                        flush=True,
                    )
                fetch_start = time.perf_counter()
        finally:
            if train_loader is None or not bool(args.persistent_workers):
                _shutdown_dataloader_iterator(batch_iter)
        mean_metrics = _mean_metrics(epoch_metrics)
        elapsed = time.perf_counter() - t0
        mean_metrics["samples_per_second"] = float(epoch_samples / max(elapsed, 1e-12))
        mean_metrics["steps_per_second"] = float(epoch_batches / max(elapsed, 1e-12))
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["elapsed_seconds"] = float(elapsed)
        timing_metrics: dict[str, float] = {}
        if timing_enabled and timing_batches > 0:
            timing_metrics = {f"timing_{key}_mean_ms": 1000.0 * value / float(timing_batches) for key, value in timing_sums.items()}
            timing_metrics.update({f"timing_{key}_total_s": float(value) for key, value in timing_sums.items()})
            timing_metrics["timing_batches"] = float(timing_batches)
        print(
            f"epoch={epoch:03d} loss={mean_metrics.get('loss', float('nan')):.4f} "
            f"peak={mean_metrics.get('loss_peak', float('nan')):.4f} "
            f"f1={mean_metrics.get('track_f1', float('nan')):.3f} "
            f"cnt_mae={mean_metrics.get('count_mae', float('nan')):.2f} "
            f"spd_bad={mean_metrics.get('speed_window_violation_rate', float('nan')):.3f} "
            f"samples/s={mean_metrics.get('samples_per_second', float('nan')):.2f} "
            f"steps/s={mean_metrics.get('steps_per_second', float('nan')):.3f} "
            f"elapsed={elapsed:.1f}s"
        )
        if timing_metrics:
            print(
                f"epoch={epoch:03d} timing mean_ms "
                f"data={timing_metrics['timing_data_wait_mean_ms']:.1f} "
                f"h2d={timing_metrics['timing_h2d_mean_ms']:.1f} "
                f"forward={timing_metrics['timing_forward_mean_ms']:.1f} "
                f"loss={timing_metrics['timing_loss_mean_ms']:.1f} "
                f"backward={timing_metrics['timing_backward_mean_ms']:.1f} "
                f"update={timing_metrics['timing_update_mean_ms']:.1f} "
                f"metrics={timing_metrics['timing_metrics_mean_ms']:.1f} "
                f"total={timing_metrics['timing_total_mean_ms']:.1f}",
                flush=True,
            )
        val_metrics: dict[str, float] = {}
        if val_shards and epoch % int(max(1, args.val_every)) == 0:
            val_metrics = _evaluate(
                model,
                val_data_dir or data_dir,
                val_shards,
                device,
                args,
                val_meta,
                max_samples=int(args.val_max_samples),
            )
            print(
                f"epoch={epoch:03d} val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"val_f1={val_metrics.get('track_f1', float('nan')):.3f} "
                f"val_cnt_mae={val_metrics.get('count_mae', float('nan')):.2f}",
                flush=True,
            )
        history_row: dict[str, float | int | str] = {"epoch": int(epoch), "elapsed_seconds": float(elapsed)}
        history_row.update({f"train_{key}": float(value) for key, value in mean_metrics.items() if np.isfinite(value)})
        history_row.update({f"train_{key}": float(value) for key, value in timing_metrics.items() if np.isfinite(value)})
        history_row.update({f"val_{key}": float(value) for key, value in val_metrics.items() if np.isfinite(value)})
        _append_history_row(out_dir / "train_history.jsonl", history_row)
        save_this_epoch = epoch % int(max(1, args.checkpoint_every)) == 0 or epoch == int(args.epochs)
        if save_this_epoch:
            checkpoint_metrics = dict(mean_metrics)
            checkpoint_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            last_path = out_dir / "checkpoint_last.pt"
            save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics, dataset_meta=meta)
            print(f"Saved checkpoint: {last_path}", flush=True)
            current_loss = float(val_metrics.get("loss", mean_metrics.get("loss", float("inf"))))
            if current_loss < best_loss:
                best_loss = current_loss
                best_path = out_dir / "checkpoint_best.pt"
                save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics, dataset_meta=meta)
                print(f"Saved new best checkpoint: {best_path}", flush=True)
    _shutdown_dataloader(train_loader)
    print(f"Done. Best loss={best_loss:.4f}. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
