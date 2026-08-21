from __future__ import annotations

from pathlib import Path

import numpy as np


def load_window(array: np.ndarray, start_s: float, duration_s: float, sample_rate_hz: float) -> np.ndarray:
    begin = int(round(float(start_s) * float(sample_rate_hz)))
    count = int(round(float(duration_s) * float(sample_rate_hz)))
    end = begin + count
    if begin < 0 or end > int(array.shape[0]):
        raise ValueError(f"window [{start_s}, {start_s + duration_s}) is outside input")
    return np.asarray(array[begin:end], dtype=np.float32)


def open_mmap(path: str | Path) -> np.ndarray:
    return np.load(str(Path(path).expanduser()), mmap_mode="r")
