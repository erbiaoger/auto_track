from __future__ import annotations

from pathlib import Path

import numpy as np


def load_window(
    path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """Load a [time, station] slice without reading the full NPY into memory."""
    array = np.load(Path(path), mmap_mode="r")
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D [time, station] array, got {array.shape}")
    begin = int(round(start_s * sample_rate_hz))
    length = int(round(duration_s * sample_rate_hz))
    end = begin + length
    if begin < 0 or end > array.shape[0]:
        raise ValueError(
            f"requested samples [{begin}:{end}] exceed array length {array.shape[0]}"
        )
    return np.asarray(array[begin:end], dtype=np.float32)


def load_modal_window(
    raw_path: str | Path,
    pre_path: str | Path,
    gauss_path: str | Path,
    *,
    start_s: float,
    duration_s: float,
    sample_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = tuple(
        load_window(
            path,
            start_s=start_s,
            duration_s=duration_s,
            sample_rate_hz=sample_rate_hz,
        )
        for path in (raw_path, pre_path, gauss_path)
    )
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"modal arrays must be aligned, got shapes {sorted(shapes)}")
    return arrays  # type: ignore[return-value]

