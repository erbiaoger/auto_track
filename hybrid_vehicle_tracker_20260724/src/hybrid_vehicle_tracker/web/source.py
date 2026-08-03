from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DataChunk:
    """One absolute-time chunk exposed to the replay loop and browser."""

    start_s: float
    sample_rate_hz: float
    raw: np.ndarray
    pre: np.ndarray
    gauss: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.raw.shape[0] / self.sample_rate_hz)


class DataSource(Protocol):
    sample_rate_hz: float
    duration_s: float
    station_count: int

    def read(self, start_s: float, duration_s: float) -> DataChunk:
        """Read an aligned, absolute-time modal chunk."""

    def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
        """Return a browser-sized downsampled frame."""


def _slice(array: np.ndarray, start_s: float, duration_s: float, rate: float) -> np.ndarray:
    begin = int(round(start_s * rate))
    end = begin + int(round(duration_s * rate))
    if begin < 0 or end > array.shape[0]:
        raise ValueError(f"requested [{start_s}, {start_s + duration_s}) outside source")
    return np.asarray(array[begin:end], dtype=np.float32)


def _pool(values: np.ndarray, bins: int, mode: str) -> np.ndarray:
    usable = values.shape[0] - values.shape[0] % bins
    if usable <= 0:
        return values[:0]
    view = values[:usable].reshape(-1, bins, values.shape[1])
    if mode == "max":
        return np.max(view, axis=1)
    if mode == "min":
        return np.min(view, axis=1)
    if mode == "mean":
        return np.mean(view, axis=1, dtype=np.float32)
    raise ValueError(mode)


class NpyReplaySource:
    """Memory-mapped DAY11 threshold cache; no modal array is copied at startup."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        display_rate_hz: float = 20.0,
        waveform_downsample: int = 20,
    ) -> None:
        manifest_path = Path(manifest_path)
        payload = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
        self.manifest_path = manifest_path
        self.sample_rate_hz = float(payload["sample_rate_hz"])
        self.duration_s = float(payload["duration_s"])
        self.display_rate_hz = float(display_rate_hz)
        self.waveform_downsample = int(waveform_downsample)
        if self.display_rate_hz <= 0 or self.waveform_downsample < 1:
            raise ValueError("display_rate_hz must be positive and waveform_downsample >= 1")
        def resolve(value: str | Path) -> Path:
            path = Path(value)
            if path.is_absolute():
                return path
            candidates = [Path.cwd() / path, manifest_path.parent.parent.parent / path, manifest_path.parent / path]
            return next((item for item in candidates if item.exists()), candidates[0])

        self.raw_path = resolve(payload["raw_path"])
        self.pre_path = resolve(payload["pre_path"])
        self.gauss_path = resolve(payload["gauss_path"])
        self.raw = np.load(self.raw_path, mmap_mode="r")
        self.pre = np.load(self.pre_path, mmap_mode="r")
        self.gauss = np.load(self.gauss_path, mmap_mode="r")
        shapes = {self.raw.shape, self.pre.shape, self.gauss.shape}
        if len(shapes) != 1 or self.raw.ndim != 2:
            raise ValueError(f"cache arrays must be aligned 2-D arrays, got {sorted(shapes)}")
        expected_samples = int(round(self.duration_s * self.sample_rate_hz))
        if self.raw.shape[0] < expected_samples:
            raise ValueError("cache duration exceeds array length")
        self.station_count = int(self.raw.shape[1])

    def set_waveform_downsample(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError("waveform_downsample must be >= 1")
        self.waveform_downsample = value

    def read(self, start_s: float, duration_s: float) -> DataChunk:
        return DataChunk(
            start_s=float(start_s),
            sample_rate_hz=self.sample_rate_hz,
            raw=_slice(self.raw, start_s, duration_s, self.sample_rate_hz),
            pre=_slice(self.pre, start_s, duration_s, self.sample_rate_hz),
            gauss=_slice(self.gauss, start_s, duration_s, self.sample_rate_hz),
        )

    def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
        chunk = self.read(start_s, duration_s)
        bins = max(1, int(round(chunk.sample_rate_hz / self.display_rate_hz)))
        gauss = _pool(chunk.gauss, bins, "max")
        raw_min = _pool(chunk.raw, bins, "min")
        raw_max = _pool(chunk.raw, bins, "max")
        waveform_bins = self.waveform_downsample
        raw_wave = _pool(chunk.raw, waveform_bins, "mean")
        actual_rate = chunk.sample_rate_hz / bins
        actual_waveform_rate = chunk.sample_rate_hz / waveform_bins
        times = (float(start_s) + np.arange(gauss.shape[0], dtype=np.float32) / actual_rate)
        return {
            "event": "frame",
            "start_s": float(start_s),
            "duration_s": float(duration_s),
            "sample_rate_hz": float(actual_rate),
            "waveform_rate_hz": float(actual_waveform_rate),
            "station_count": self.station_count,
            "times_s": times.tolist(),
            # Time-major [time_bin, station] layout matches the Float32 frame decoder.
            "gauss": gauss.astype(np.float32).ravel().tolist(),
            "raw_min": raw_min.astype(np.float32).ravel().tolist(),
            "raw_max": raw_max.astype(np.float32).ravel().tolist(),
            "raw_wave": raw_wave.astype(np.float32).ravel().tolist(),
        }


class DirectoryStreamSource:
    """Future live source contract; intentionally refuses unsupported protocols."""

    def __init__(self, directory: str | Path, **_: object) -> None:
        self.directory = Path(directory)
        raise NotImplementedError(
            "DirectoryStreamSource is an interface placeholder; provide a site-specific reader"
        )
