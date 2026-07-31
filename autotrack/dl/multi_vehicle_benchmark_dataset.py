"""Utilities for loading multi-vehicle benchmark files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    return value


@dataclass(frozen=True)
class MultiVehicleBenchmarkMeta:
    path: Path
    payload: dict[str, Any]

    @property
    def samples(self) -> list[dict[str, Any]]:
        return list(self.payload.get("samples", []))


class MultiVehicleBenchmarkDataset(Dataset):
    def __init__(self, benchmark_file: str | Path, *, max_samples: int = 0):
        self.path = Path(benchmark_file).expanduser()
        payload = torch.load(str(self.path), map_location="cpu", weights_only=False)
        samples = list(payload.get("samples", []))
        if int(max_samples) > 0:
            samples = samples[: int(max_samples)]
        if not samples:
            raise ValueError(f"benchmark contains no samples: {self.path}")
        self.meta = MultiVehicleBenchmarkMeta(path=self.path, payload=dict(payload.get("meta", {})))
        self.samples = samples
        self.format = str(payload.get("format", ""))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[int(index)]
        x = sample["x"].to(torch.float32)
        target = {key: value.to(torch.float32) if torch.is_tensor(value) and value.dtype.is_floating_point else value.clone() if torch.is_tensor(value) else value for key, value in sample["target"].items()}
        if "gt_valid" in target:
            target["gt_valid"] = target["gt_valid"].to(torch.bool)
        return x, target


def stack_benchmark_batch(batch: Iterable[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs: list[torch.Tensor] = []
    targets_list: list[dict[str, torch.Tensor]] = []
    for x, target in batch:
        xs.append(x)
        targets_list.append(target)
    if not xs:
        raise ValueError("empty batch")
    x = torch.stack(xs, dim=0)
    stacked: dict[str, torch.Tensor] = {}
    keys = sorted({key for item in targets_list for key in item.keys()})
    for key in keys:
        values = [item[key] for item in targets_list if key in item]
        if not values:
            continue
        if not torch.is_tensor(values[0]):
            raise TypeError(f"Unsupported target type for key={key!r}")
        shapes = [tuple(value.shape) for value in values]
        if all(shape == shapes[0] for shape in shapes):
            stacked[key] = torch.stack([value.clone() for value in values], dim=0)
            continue
        rank = values[0].ndim
        if any(value.ndim != rank for value in values):
            raise ValueError(f"Incompatible tensor ranks for key={key!r}: {shapes}")
        max_shape = [max(value.shape[dim] for value in values) for dim in range(rank)]
        fill_value = False if values[0].dtype == torch.bool else 0
        padded: list[torch.Tensor] = []
        for value in values:
            out = torch.full(tuple(max_shape), fill_value=fill_value, dtype=value.dtype)
            slices = tuple(slice(0, int(size)) for size in value.shape)
            out[slices] = value
            padded.append(out)
        stacked[key] = torch.stack(padded, dim=0)
    if "gt_valid" in stacked:
        stacked["gt_valid"] = stacked["gt_valid"].to(torch.bool)
    return x, stacked


def benchmark_json_ready(payload: Any) -> Any:
    return _json_ready(payload)
