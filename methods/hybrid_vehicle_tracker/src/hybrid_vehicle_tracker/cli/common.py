from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type != "cuda":
        raise RuntimeError("training is GPU-only by project requirement; set device: cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA training was requested but CUDA is unavailable; no CPU fallback is permitted"
        )
    return device
