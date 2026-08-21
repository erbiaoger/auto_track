from __future__ import annotations

import hashlib
import math
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class OnlineSyntheticTrajectoryDataset(Dataset):
    """Online synthetic dataset with optional disk-first cache.

    When `cache_dataset=True` and `cache_dir` is set, samples are first generated
    onto disk and then loaded per-index during training (`disk_cache_only=True`).
    This avoids preloading the whole dataset into RAM.
    """

    def __init__(
        self,
        *,
        length: int,
        n_channels: int,
        fs: float,
        window_seconds: float,
        time_downsample: int,
        dx_m: float,
        vehicles_min: int,
        vehicles_max: int,
        speed_min_kmh: float,
        speed_max_kmh: float,
        speed_outlier_ratio: float,
        slow_speed_min_kmh: float,
        slow_speed_max_kmh: float,
        fast_speed_min_kmh: float,
        fast_speed_max_kmh: float,
        noise_std: float,
        amp_min: float,
        amp_max: float,
        sigma_min_s: float,
        sigma_max_s: float,
        primary_ratio: float,
        min_visible_channels: int,
        speed_norm_kmh: float,
        clip_ratio: float,
        input_mode: str,
        seed: int,
        scene_mode: str = "realistic_traffic",
        vehicle_count_profile: str = "mixed_density",
        speed_variation_ratio: float = 0.08,
        same_direction_cluster_ratio: float = 0.35,
        crossing_ratio: float = 0.35,
        parallel_close_ratio: float = 0.20,
        multi_gap_dropout_ratio: float = 0.55,
        mask_sigma_ch: float = 0.8,
        mask_sigma_t: float = 2.0,
        cache_dataset: bool = False,
        cache_dtype: str = "float16",
        cache_build_workers: int = 0,
        cache_dir: Optional[Path] = None,
        cache_rebuild: bool = False,
        disk_cache_only: bool = True,
        return_raw_window: bool = False,
        background_npy: Optional[Path] = None,
        background_pt: Optional[Path] = None,
        background_layout: str = "time_channel",
        background_channel_start: int = 0,
        background_scale: float = 1.0,
        artifact_dropout_ratio: float = 0.0,
        artifact_dropout_min_channels: int = 2,
        artifact_dropout_max_channels: int = 6,
        artifact_decoy_ratio: float = 0.0,
        artifact_decoy_min_points: int = 1,
        artifact_decoy_max_points: int = 3,
        artifact_decoy_amp_scale_min: float = 1.1,
        artifact_decoy_amp_scale_max: float = 2.2,
        artifact_decoy_time_jitter_s: float = 0.18,
        artifact_competing_ratio: float = 0.0,
        artifact_competing_time_jitter_s: float = 0.8,
        artifact_competing_amp_scale_min: float = 0.8,
        artifact_competing_amp_scale_max: float = 1.6,
        artifact_competing_speed_ratio_min: float = 0.88,
        artifact_competing_speed_ratio_max: float = 1.12,
        artifact_competing_channel_offset_max: int = 5,
        artifact_competing_opposite_direction_ratio: float = 0.0,
    ):
        self.length = int(length)
        self.n_channels = int(n_channels)
        self.fs = float(fs)
        self.window_seconds = float(window_seconds)
        self.window_samples = int(round(self.window_seconds * self.fs))
        self.time_downsample = int(max(1, time_downsample))
        self.dx_m = float(dx_m)
        self.vehicles_min = int(vehicles_min)
        self.vehicles_max = int(max(vehicles_min, vehicles_max))
        self.speed_min_kmh = float(speed_min_kmh)
        self.speed_max_kmh = float(speed_max_kmh)
        self.speed_outlier_ratio = float(min(1.0, max(0.0, speed_outlier_ratio)))
        self.slow_speed_min_kmh = float(slow_speed_min_kmh)
        self.slow_speed_max_kmh = float(max(slow_speed_min_kmh, slow_speed_max_kmh))
        self.fast_speed_min_kmh = float(fast_speed_min_kmh)
        self.fast_speed_max_kmh = float(max(fast_speed_min_kmh, fast_speed_max_kmh))
        self.noise_std = float(noise_std)
        self.amp_min = float(amp_min)
        self.amp_max = float(max(amp_min, amp_max))
        self.sigma_min_s = float(sigma_min_s)
        self.sigma_max_s = float(max(sigma_min_s, sigma_max_s))
        self.primary_ratio = float(primary_ratio)
        self.min_visible_channels = int(min_visible_channels)
        self.speed_norm_kmh = float(speed_norm_kmh)
        self.clip_ratio = float(clip_ratio)
        self.input_mode = str(input_mode).lower()
        self.seed = int(seed)
        self.scene_mode = str(scene_mode).lower()
        self.vehicle_count_profile = str(vehicle_count_profile).lower()
        self.speed_variation_ratio = float(max(0.0, speed_variation_ratio))
        self.same_direction_cluster_ratio = float(min(1.0, max(0.0, same_direction_cluster_ratio)))
        self.crossing_ratio = float(min(1.0, max(0.0, crossing_ratio)))
        self.parallel_close_ratio = float(min(1.0, max(0.0, parallel_close_ratio)))
        self.multi_gap_dropout_ratio = float(min(1.0, max(0.0, multi_gap_dropout_ratio)))
        self.mask_sigma_ch = float(max(1e-3, mask_sigma_ch))
        self.mask_sigma_t = float(max(1e-3, mask_sigma_t))
        self.ds_samples = int(max(1, len(range(0, self.window_samples, self.time_downsample))))
        self.cache_dataset = bool(cache_dataset)
        self.cache_dtype = str(cache_dtype).lower()
        self.cache_build_workers = int(max(0, cache_build_workers))
        self.cache_rebuild = bool(cache_rebuild)
        self.disk_cache_only = bool(disk_cache_only)
        self.return_raw_window = bool(return_raw_window)
        self.background_npy = Path(background_npy).expanduser() if background_npy is not None else None
        self.background_pt = Path(background_pt).expanduser() if background_pt is not None else None
        self.background_layout = str(background_layout).lower()
        self.background_channel_start = int(background_channel_start)
        self.background_scale = float(background_scale)
        self.artifact_dropout_ratio = float(min(1.0, max(0.0, artifact_dropout_ratio)))
        self.artifact_dropout_min_channels = int(max(1, artifact_dropout_min_channels))
        self.artifact_dropout_max_channels = int(max(self.artifact_dropout_min_channels, artifact_dropout_max_channels))
        self.artifact_decoy_ratio = float(min(1.0, max(0.0, artifact_decoy_ratio)))
        self.artifact_decoy_min_points = int(max(0, artifact_decoy_min_points))
        self.artifact_decoy_max_points = int(max(self.artifact_decoy_min_points, artifact_decoy_max_points))
        self.artifact_decoy_amp_scale_min = float(max(1.0, artifact_decoy_amp_scale_min))
        self.artifact_decoy_amp_scale_max = float(max(self.artifact_decoy_amp_scale_min, artifact_decoy_amp_scale_max))
        self.artifact_decoy_time_jitter_s = float(max(0.0, artifact_decoy_time_jitter_s))
        self.artifact_competing_ratio = float(min(1.0, max(0.0, artifact_competing_ratio)))
        self.artifact_competing_time_jitter_s = float(max(0.0, artifact_competing_time_jitter_s))
        self.artifact_competing_amp_scale_min = float(max(1.0, artifact_competing_amp_scale_min))
        self.artifact_competing_amp_scale_max = float(max(self.artifact_competing_amp_scale_min, artifact_competing_amp_scale_max))
        self.artifact_competing_speed_ratio_min = float(max(0.05, artifact_competing_speed_ratio_min))
        self.artifact_competing_speed_ratio_max = float(
            max(self.artifact_competing_speed_ratio_min, artifact_competing_speed_ratio_max)
        )
        self.artifact_competing_channel_offset_max = int(max(0, artifact_competing_channel_offset_max))
        self.artifact_competing_opposite_direction_ratio = float(
            min(1.0, max(0.0, artifact_competing_opposite_direction_ratio))
        )
        self._background_array: Optional[np.ndarray] = None
        self._background_pt_tensor: Optional[torch.Tensor] = None
        self._cache: Optional[list[tuple[torch.Tensor, dict[str, torch.Tensor]]]] = None

        self.cache_root: Optional[Path] = None
        if cache_dir is not None:
            self.cache_root = Path(cache_dir).expanduser() / self._cache_signature()

        if self.cache_dataset:
            self.build_cache()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        i = int(index)
        if self._cache is not None:
            x_cached, target = self._cache[i]
            return x_cached.to(torch.float32), {key: value.clone() for key, value in target.items()}
        if self.cache_dataset and self.cache_root is not None and self._item_path(i).is_file():
            payload = torch.load(str(self._item_path(i)), map_location="cpu", weights_only=False)
            x = payload["x"].to(torch.float32)
            target = {k: v.clone() for k, v in payload["target"].items()}
            return x, target
        return self._generate_item(i, cache_x=False)

    def build_cache(self) -> None:
        if self.cache_root is not None:
            manifest = self.cache_root / "manifest.json"
            if (not self.cache_rebuild) and manifest.is_file():
                data = json.loads(manifest.read_text(encoding="utf-8"))
                if int(data.get("length", -1)) == self.length:
                    print(f"cache_dataset: loaded disk cache <- {self.cache_root}", flush=True)
                    if not self.disk_cache_only:
                        self._load_all_from_disk_into_ram()
                    return

        workers = min(int(self.cache_build_workers), int(self.length))
        t0 = time.perf_counter()
        print(
            f"cache_dataset: building {self.length} windows to disk, dtype={self.cache_dtype}, "
            f"build_workers={workers if workers > 1 else 0}",
            flush=True,
        )
        if self.cache_root is not None:
            (self.cache_root / "items").mkdir(parents=True, exist_ok=True)

        if workers <= 1:
            ram_cache = []
            for index in range(self.length):
                x, target = self._generate_item(index, cache_x=True)
                self._save_item(index, x, target)
                if not self.disk_cache_only:
                    ram_cache.append((x, target))
                if (index + 1) % 500 == 0 or index + 1 == self.length:
                    elapsed = time.perf_counter() - t0
                    print(f"cache_dataset: {index + 1}/{self.length} windows, elapsed={elapsed:.1f}s", flush=True)
            self._cache = ram_cache if not self.disk_cache_only else None
            self._write_manifest()
            return

        worker_kwargs = self._cache_worker_kwargs()
        chunksize = max(1, min(8, self.length // max(1, workers * 8)))
        ram_parallel: list[Optional[tuple[torch.Tensor, dict[str, torch.Tensor]]]] = [None] * self.length
        with ProcessPoolExecutor(max_workers=workers) as executor:
            jobs = ((worker_kwargs, index) for index in range(self.length))
            for done, (index, item_np) in enumerate(executor.map(_generate_cached_online_item, jobs, chunksize=chunksize), start=1):
                x_np, target_np = item_np
                item = (
                    torch.from_numpy(x_np).contiguous(),
                    {key: torch.from_numpy(value).contiguous() for key, value in target_np.items()},
                )
                self._save_item(index, item[0], item[1])
                if not self.disk_cache_only:
                    ram_parallel[index] = item
                if done % 500 == 0 or done == self.length:
                    elapsed = time.perf_counter() - t0
                    print(f"cache_dataset: {done}/{self.length} windows, elapsed={elapsed:.1f}s", flush=True)
        self._cache = [x for x in ram_parallel if x is not None] if not self.disk_cache_only else None
        self._write_manifest()

    def _write_manifest(self) -> None:
        if self.cache_root is None:
            return
        payload = {
            "signature": self._cache_signature(),
            "length": self.length,
            "cache_dtype": self.cache_dtype,
            "disk_cache_only": self.disk_cache_only,
        }
        (self.cache_root / "manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"cache_dataset: saved disk cache -> {self.cache_root}", flush=True)

    def _load_all_from_disk_into_ram(self) -> None:
        cache: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        for i in range(self.length):
            payload = torch.load(str(self._item_path(i)), map_location="cpu", weights_only=False)
            cache.append((payload["x"], payload["target"]))
        self._cache = cache

    def _item_path(self, index: int) -> Path:
        if self.cache_root is None:
            raise RuntimeError("cache_root is not configured")
        return self.cache_root / "items" / f"item_{int(index):06d}.pt"

    def _save_item(self, index: int, x: torch.Tensor, target: dict[str, torch.Tensor]) -> None:
        if self.cache_root is None:
            return
        torch.save({"x": x.cpu(), "target": {k: v.cpu() for k, v in target.items()}}, self._item_path(index))

    def _cache_signature(self) -> str:
        payload = {
            "length": int(self.length),
            "n_channels": int(self.n_channels),
            "fs": float(self.fs),
            "window_seconds": float(self.window_seconds),
            "time_downsample": int(self.time_downsample),
            "dx_m": float(self.dx_m),
            "vehicles_min": int(self.vehicles_min),
            "vehicles_max": int(self.vehicles_max),
            "speed_min_kmh": float(self.speed_min_kmh),
            "speed_max_kmh": float(self.speed_max_kmh),
            "speed_outlier_ratio": float(self.speed_outlier_ratio),
            "slow_speed_min_kmh": float(self.slow_speed_min_kmh),
            "slow_speed_max_kmh": float(self.slow_speed_max_kmh),
            "fast_speed_min_kmh": float(self.fast_speed_min_kmh),
            "fast_speed_max_kmh": float(self.fast_speed_max_kmh),
            "noise_std": float(self.noise_std),
            "amp_min": float(self.amp_min),
            "amp_max": float(self.amp_max),
            "sigma_min_s": float(self.sigma_min_s),
            "sigma_max_s": float(self.sigma_max_s),
            "primary_ratio": float(self.primary_ratio),
            "min_visible_channels": int(self.min_visible_channels),
            "speed_norm_kmh": float(self.speed_norm_kmh),
            "clip_ratio": float(self.clip_ratio),
            "input_mode": str(self.input_mode),
            "seed": int(self.seed),
            "scene_mode": str(self.scene_mode),
            "vehicle_count_profile": str(self.vehicle_count_profile),
            "speed_variation_ratio": float(self.speed_variation_ratio),
            "same_direction_cluster_ratio": float(self.same_direction_cluster_ratio),
            "crossing_ratio": float(self.crossing_ratio),
            "parallel_close_ratio": float(self.parallel_close_ratio),
            "multi_gap_dropout_ratio": float(self.multi_gap_dropout_ratio),
            "mask_sigma_ch": float(self.mask_sigma_ch),
            "mask_sigma_t": float(self.mask_sigma_t),
            "cache_dtype": str(self.cache_dtype),
            "return_raw_window": bool(self.return_raw_window),
            "background_npy": str(self.background_npy) if self.background_npy is not None else None,
            "background_pt": str(self.background_pt) if self.background_pt is not None else None,
            "background_layout": str(self.background_layout),
            "background_channel_start": int(self.background_channel_start),
            "background_scale": float(self.background_scale),
            "artifact_dropout_ratio": float(self.artifact_dropout_ratio),
            "artifact_dropout_min_channels": int(self.artifact_dropout_min_channels),
            "artifact_dropout_max_channels": int(self.artifact_dropout_max_channels),
            "artifact_decoy_ratio": float(self.artifact_decoy_ratio),
            "artifact_decoy_min_points": int(self.artifact_decoy_min_points),
            "artifact_decoy_max_points": int(self.artifact_decoy_max_points),
            "artifact_decoy_amp_scale_min": float(self.artifact_decoy_amp_scale_min),
            "artifact_decoy_amp_scale_max": float(self.artifact_decoy_amp_scale_max),
            "artifact_decoy_time_jitter_s": float(self.artifact_decoy_time_jitter_s),
            "artifact_competing_ratio": float(self.artifact_competing_ratio),
            "artifact_competing_time_jitter_s": float(self.artifact_competing_time_jitter_s),
            "artifact_competing_amp_scale_min": float(self.artifact_competing_amp_scale_min),
            "artifact_competing_amp_scale_max": float(self.artifact_competing_amp_scale_max),
            "artifact_competing_speed_ratio_min": float(self.artifact_competing_speed_ratio_min),
            "artifact_competing_speed_ratio_max": float(self.artifact_competing_speed_ratio_max),
            "artifact_competing_channel_offset_max": int(self.artifact_competing_channel_offset_max),
            "artifact_competing_opposite_direction_ratio": float(self.artifact_competing_opposite_direction_ratio),
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _cache_worker_kwargs(self) -> dict[str, object]:
        return {
            "n_channels": self.n_channels,
            "fs": self.fs,
            "window_seconds": self.window_seconds,
            "time_downsample": self.time_downsample,
            "dx_m": self.dx_m,
            "vehicles_min": self.vehicles_min,
            "vehicles_max": self.vehicles_max,
            "speed_min_kmh": self.speed_min_kmh,
            "speed_max_kmh": self.speed_max_kmh,
            "speed_outlier_ratio": self.speed_outlier_ratio,
            "slow_speed_min_kmh": self.slow_speed_min_kmh,
            "slow_speed_max_kmh": self.slow_speed_max_kmh,
            "fast_speed_min_kmh": self.fast_speed_min_kmh,
            "fast_speed_max_kmh": self.fast_speed_max_kmh,
            "noise_std": self.noise_std,
            "amp_min": self.amp_min,
            "amp_max": self.amp_max,
            "sigma_min_s": self.sigma_min_s,
            "sigma_max_s": self.sigma_max_s,
            "primary_ratio": self.primary_ratio,
            "min_visible_channels": self.min_visible_channels,
            "speed_norm_kmh": self.speed_norm_kmh,
            "clip_ratio": self.clip_ratio,
            "input_mode": self.input_mode,
            "seed": self.seed,
            "scene_mode": self.scene_mode,
            "vehicle_count_profile": self.vehicle_count_profile,
            "speed_variation_ratio": self.speed_variation_ratio,
            "same_direction_cluster_ratio": self.same_direction_cluster_ratio,
            "crossing_ratio": self.crossing_ratio,
            "parallel_close_ratio": self.parallel_close_ratio,
            "multi_gap_dropout_ratio": self.multi_gap_dropout_ratio,
            "mask_sigma_ch": self.mask_sigma_ch,
            "mask_sigma_t": self.mask_sigma_t,
            "cache_dtype": self.cache_dtype,
            "return_raw_window": self.return_raw_window,
            "background_npy": str(self.background_npy) if self.background_npy is not None else None,
            "background_pt": str(self.background_pt) if self.background_pt is not None else None,
            "background_layout": self.background_layout,
            "background_channel_start": self.background_channel_start,
            "background_scale": self.background_scale,
            "artifact_dropout_ratio": self.artifact_dropout_ratio,
            "artifact_dropout_min_channels": self.artifact_dropout_min_channels,
            "artifact_dropout_max_channels": self.artifact_dropout_max_channels,
            "artifact_decoy_ratio": self.artifact_decoy_ratio,
            "artifact_decoy_min_points": self.artifact_decoy_min_points,
            "artifact_decoy_max_points": self.artifact_decoy_max_points,
            "artifact_decoy_amp_scale_min": self.artifact_decoy_amp_scale_min,
            "artifact_decoy_amp_scale_max": self.artifact_decoy_amp_scale_max,
            "artifact_decoy_time_jitter_s": self.artifact_decoy_time_jitter_s,
            "artifact_competing_ratio": self.artifact_competing_ratio,
            "artifact_competing_time_jitter_s": self.artifact_competing_time_jitter_s,
            "artifact_competing_amp_scale_min": self.artifact_competing_amp_scale_min,
            "artifact_competing_amp_scale_max": self.artifact_competing_amp_scale_max,
            "artifact_competing_speed_ratio_min": self.artifact_competing_speed_ratio_min,
            "artifact_competing_speed_ratio_max": self.artifact_competing_speed_ratio_max,
            "artifact_competing_channel_offset_max": self.artifact_competing_channel_offset_max,
            "artifact_competing_opposite_direction_ratio": self.artifact_competing_opposite_direction_ratio,
        }

    def _render_instance_mask(self, center_idx: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        ch_axis = torch.arange(self.n_channels, dtype=torch.float32).view(-1, 1)
        t_axis = torch.arange(self.ds_samples, dtype=torch.float32).view(1, -1)
        mask = torch.zeros((self.n_channels, self.ds_samples), dtype=torch.float32)
        idx_ds = torch.div(center_idx.to(torch.long), int(max(1, self.time_downsample)), rounding_mode="floor")
        idx_ds = idx_ds.clamp(0, self.ds_samples - 1)
        for ch in torch.where(visible)[0].tolist():
            gc = torch.exp(-0.5 * ((ch_axis - float(ch)) / self.mask_sigma_ch) ** 2)
            gt = torch.exp(-0.5 * ((t_axis - float(idx_ds[ch].item())) / self.mask_sigma_t) ** 2)
            mask = torch.maximum(mask, gc * gt)
        return mask.clamp(0.0, 1.0)

    def _background_array_view(self) -> Optional[np.ndarray]:
        if self.background_npy is None:
            return None
        if self._background_array is None:
            self._background_array = np.load(str(self.background_npy), mmap_mode="r")
        return self._background_array

    def _background_pt_view(self) -> Optional[torch.Tensor]:
        if self.background_pt is None:
            return None
        if self._background_pt_tensor is None:
            payload = torch.load(str(self.background_pt), map_location="cpu", weights_only=False)
            if isinstance(payload, dict) and "x" in payload:
                data = payload["x"]
            elif torch.is_tensor(payload):
                data = payload
            else:
                raise ValueError(f"Unsupported background_pt payload: {type(payload)}")
            if not torch.is_tensor(data):
                data = torch.as_tensor(data)
            self._background_pt_tensor = data.to(torch.float32).contiguous()
        return self._background_pt_tensor

    def _background_window(self, index: int, gen: torch.Generator) -> tuple[torch.Tensor, dict[str, float]]:
        if self.background_pt is not None:
            data = self._background_pt_view()
            if data is None:
                raise RuntimeError("background_pt tensor unexpectedly unavailable")
            if data.ndim == 4:
                sample_idx = int(torch.randint(0, data.shape[0], (1,), generator=gen).item())
                window = data[sample_idx]
            elif data.ndim == 3:
                sample_idx = int(torch.randint(0, data.shape[0], (1,), generator=gen).item())
                window = data[sample_idx]
            elif data.ndim == 2:
                window = data
                sample_idx = 0
            else:
                raise ValueError(f"background_pt must be 2-D, 3-D, or 4-D, got shape={tuple(data.shape)}")
            if window.ndim == 3 and int(window.shape[0]) == 1:
                window = window[0]
            if window.ndim != 2:
                raise ValueError(f"background_pt sample must reduce to 2-D, got shape={tuple(window.shape)}")
            if int(window.shape[0]) != self.n_channels and int(window.shape[1]) == self.n_channels:
                window = window.transpose(0, 1)
            if int(window.shape[0]) != self.n_channels:
                raise ValueError(f"background_pt channel mismatch: expected {self.n_channels}, got {tuple(window.shape)}")
            if int(window.shape[1]) != self.window_samples:
                if int(window.shape[1]) == max(1, self.window_samples // self.time_downsample):
                    window = window.repeat_interleave(self.time_downsample, dim=1)
                else:
                    window = F.interpolate(window.unsqueeze(0), size=self.window_samples, mode="linear", align_corners=False).squeeze(0)
            window = window.to(torch.float32).contiguous()
            if float(self.background_scale) != 1.0:
                window = window * float(self.background_scale)
            if float(self.noise_std) > 0.0:
                window = window + torch.normal(
                    mean=0.0,
                    std=float(self.noise_std) * 0.25,
                    size=window.shape,
                    generator=gen,
                    dtype=torch.float32,
                )
            return window, {"source": 2.0, "start_t": float(sample_idx)}
        if self.background_npy is None:
            return (
                torch.normal(
                    mean=0.0,
                    std=self.noise_std,
                    size=(self.n_channels, self.window_samples),
                    generator=gen,
                    dtype=torch.float32,
                ),
                {"source": 0.0},
            )
        arr = self._background_array_view()
        if arr is None:
            raise RuntimeError("background array unexpectedly unavailable")
        if arr.ndim != 2:
            raise ValueError(f"background_npy must be 2-D, got shape={arr.shape}")
        layout = self.background_layout
        if layout not in {"time_channel", "channel_time"}:
            raise ValueError("background_layout must be time_channel or channel_time")
        n_time = int(arr.shape[0] if layout == "time_channel" else arr.shape[1])
        n_channels_all = int(arr.shape[1] if layout == "time_channel" else arr.shape[0])
        start_ch = int(self.background_channel_start)
        end_ch = int(start_ch + self.n_channels)
        if start_ch < 0 or end_ch > n_channels_all:
            raise ValueError(
                f"background channel slice [{start_ch}, {end_ch}) outside source shape {arr.shape}"
            )
        max_start = int(n_time - self.window_samples)
        if max_start < 0:
            raise ValueError(f"background_npy is shorter than one window: n_time={n_time}, window={self.window_samples}")
        start_t = int(torch.randint(0, max_start + 1, (1,), generator=gen).item()) if max_start > 0 else 0
        if layout == "time_channel":
            window = np.array(arr[start_t : start_t + self.window_samples, start_ch:end_ch], dtype=np.float32, copy=True).T
        else:
            window = np.array(arr[start_ch:end_ch, start_t : start_t + self.window_samples], dtype=np.float32, copy=True)
        window = np.nan_to_num(window, copy=False)
        if float(self.background_scale) != 1.0:
            window *= float(self.background_scale)
        return torch.from_numpy(window), {"source": 1.0, "start_t": float(start_t)}

    def _sample_vehicle_count(self, gen: torch.Generator) -> int:
        lo = int(max(1, self.vehicles_min))
        hi = int(max(lo, self.vehicles_max))
        if hi <= lo:
            return lo
        profile = self.vehicle_count_profile
        r = float(torch.rand((), generator=gen).item())
        if profile in {"mixed_density", "mixed", "traffic"}:
            span = hi - lo
            low_hi = min(hi, lo + max(1, span // 4))
            mid_lo = min(hi, low_hi + 1)
            mid_hi = min(hi, lo + max(2, (span * 2) // 3))
            high_lo = min(hi, max(mid_hi, mid_lo))
            buckets = [
                (lo, low_hi),
                (mid_lo, mid_hi),
                (high_lo, hi),
            ]
            weights = (0.25, 0.50, 0.25)
            if r < weights[0]:
                bucket = buckets[0]
            elif r < weights[0] + weights[1]:
                bucket = buckets[1]
            else:
                bucket = buckets[2]
            b_lo, b_hi = bucket
            if b_hi < b_lo:
                b_lo, b_hi = lo, hi
            return int(torch.randint(int(b_lo), int(b_hi) + 1, (1,), generator=gen).item())
        if profile in {"dense", "high"}:
            b_lo = max(lo, hi - max(2, (hi - lo) // 3))
            return int(torch.randint(int(b_lo), int(hi) + 1, (1,), generator=gen).item())
        if profile in {"sparse", "low"}:
            b_hi = min(hi, lo + max(1, (hi - lo) // 3))
            return int(torch.randint(int(lo), int(b_hi) + 1, (1,), generator=gen).item())
        return int(torch.randint(int(lo), int(hi) + 1, (1,), generator=gen).item())

    def _sample_base_speed_kmh(self, gen: torch.Generator) -> float:
        return float(self.speed_min_kmh + torch.rand((), generator=gen).item() * (self.speed_max_kmh - self.speed_min_kmh))

    def _speed_profile_factor(self, progress: float, drift: float, wobble: float, phase: float) -> float:
        p = float(np.clip(progress, 0.0, 1.0))
        factor = 1.0 + float(drift) * (p - 0.5) + float(wobble) * math.sin(2.0 * math.pi * p + float(phase))
        return float(np.clip(factor, 0.72, 1.32))

    def _sample_related_speed_kmh(self, gen: torch.Generator, base_speed_kmh: float, *, tight: bool = False) -> float:
        scale = 0.05 if tight else 0.10
        jitter = 1.0 + float((2.0 * torch.rand((), generator=gen).item() - 1.0) * scale)
        speed = float(base_speed_kmh) * jitter
        return float(np.clip(speed, self.speed_min_kmh, self.speed_max_kmh))

    def _cache_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dtype in {"float16", "fp16", "half"}:
            return x.to(torch.float16).contiguous()
        if self.cache_dtype in {"bfloat16", "bf16"}:
            return x.to(torch.bfloat16).contiguous()
        return x.to(torch.float32).contiguous()

    def _generate_item(self, index: int, cache_x: bool):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + int(index) * 1000003)

        data, background_meta = self._background_window(index, gen)
        n_veh = int(self._sample_vehicle_count(gen))
        time_rows: list[torch.Tensor] = []
        vis_rows: list[torch.Tensor] = []
        dir_rows: list[int] = []
        speed_rows: list[float] = []
        track_ids: list[int] = []
        mask_rows: list[torch.Tensor] = []
        artifact_dropout_total = 0
        artifact_decoy_total = 0
        artifact_competing_total = 0
        artifact_competing_direction = -1

        def _draw_vehicle(
            *,
            direction_label: int,
            speed_kmh: float,
            sigma_s: float,
            amp: float,
            label_track: bool,
            anchor_time: float | None = None,
            anchor_ch: int | None = None,
            amp_scale: float = 1.0,
            time_jitter_s: float = 0.0,
            allow_dropout: bool = True,
        ) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor] | None:
            nonlocal artifact_dropout_total
            speed_mps = float(speed_kmh) / 3.6
            if speed_mps <= 1e-6:
                return None
            if anchor_ch is None:
                local_anchor_ch = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
            else:
                local_anchor_ch = int(max(0, min(self.n_channels - 1, int(anchor_ch))))
            if anchor_time is None:
                local_anchor_time = float(torch.rand((), generator=gen).item() * self.window_seconds)
            else:
                local_anchor_time = float(anchor_time)
            local_anchor_time = float(local_anchor_time)
            if time_jitter_s > 0.0:
                local_anchor_time += float(2.0 * torch.rand((), generator=gen).item() - 1.0) * float(time_jitter_s)
            trend_sign = 1.0 if int(direction_label) == 0 else -1.0
            drift = float(self.speed_variation_ratio) * float(2.0 * torch.rand((), generator=gen).item() - 1.0)
            wobble = 0.35 * float(self.speed_variation_ratio) * float(2.0 * torch.rand((), generator=gen).item() - 1.0)
            phase = float(2.0 * math.pi * torch.rand((), generator=gen).item())
            t_center = torch.zeros((self.n_channels,), dtype=torch.float32)
            t_center[local_anchor_ch] = float(local_anchor_time)
            for ch in range(local_anchor_ch + 1, self.n_channels):
                progress = float(ch - local_anchor_ch) / float(max(1, self.n_channels - 1))
                factor = self._speed_profile_factor(progress, drift, wobble, phase)
                step_speed = max(1e-6, speed_mps * factor)
                t_center[ch] = t_center[ch - 1] + float(trend_sign * self.dx_m / step_speed)
            for ch in range(local_anchor_ch - 1, -1, -1):
                progress = float(local_anchor_ch - ch) / float(max(1, self.n_channels - 1))
                factor = self._speed_profile_factor(progress, drift, wobble, phase)
                step_speed = max(1e-6, speed_mps * factor)
                t_center[ch] = t_center[ch + 1] - float(trend_sign * self.dx_m / step_speed)
            visible = (t_center >= 0.0) & (t_center < self.window_seconds)
            if int(visible.sum().item()) < self.min_visible_channels:
                return None

            channel_amp_scale = torch.ones((self.n_channels,), dtype=torch.float32)
            dropout_prob = max(float(self.artifact_dropout_ratio), float(self.multi_gap_dropout_ratio))
            if allow_dropout and dropout_prob > 0.0 and int(visible.sum().item()) > self.min_visible_channels + 1:
                if float(torch.rand((), generator=gen).item()) < dropout_prob:
                    vis_idx = torch.where(visible)[0]
                    min_drop = int(self.artifact_dropout_min_channels)
                    max_drop = int(min(self.artifact_dropout_max_channels, max(1, int(vis_idx.numel()) - self.min_visible_channels)))
                    if max_drop >= min_drop:
                        segment_count = 1
                        if self.scene_mode in {"realistic_traffic", "realistic", "traffic"}:
                            segment_count = 1 + int(torch.randint(0, 3, (1,), generator=gen).item())
                        for _ in range(segment_count):
                            current_vis = torch.where(visible)[0]
                            if int(current_vis.numel()) <= self.min_visible_channels + 1:
                                break
                            drop_cap = min(int(max_drop), int(current_vis.numel()) - self.min_visible_channels)
                            if drop_cap < min_drop:
                                break
                            dropout_count = int(
                                torch.randint(
                                    int(min_drop),
                                    int(drop_cap) + 1,
                                    (1,),
                                    generator=gen,
                                ).item()
                            )
                            start_offset = int(
                                torch.randint(0, int(current_vis.numel()) - dropout_count + 1, (1,), generator=gen).item()
                            )
                            drop_idx = current_vis[start_offset : start_offset + dropout_count]
                            if float(torch.rand((), generator=gen).item()) < 0.55:
                                visible[drop_idx] = False
                                channel_amp_scale[drop_idx] = 0.0
                            else:
                                weaken = float(0.25 + 0.45 * torch.rand((), generator=gen).item())
                                channel_amp_scale[drop_idx] *= weaken
                            artifact_dropout_total += int(dropout_count)

            center_idx = torch.round(t_center * self.fs).to(torch.long).clamp(0, self.window_samples - 1)
            half_width = int(max(1, round(4.0 * sigma_s * self.fs)))
            for ch in torch.where(visible)[0].tolist():
                center = int(center_idx[ch].item())
                left = max(0, center - half_width)
                right = min(self.window_samples - 1, center + half_width)
                idx = torch.arange(left, right + 1, dtype=torch.float32)
                dt = idx / self.fs - float(t_center[ch].item())
                pulse = float(amp) * float(amp_scale) * float(channel_amp_scale[ch].item()) * torch.exp(-0.5 * (dt / sigma_s) ** 2)
                data[int(ch), left : right + 1] += pulse

            time_norm = torch.zeros((self.n_channels,), dtype=torch.float32)
            vis_float = visible.to(torch.float32)
            time_norm[visible] = (center_idx[visible].to(torch.float32) / float(max(1, self.window_samples - 1))).clamp(0, 1)
            if not label_track:
                return visible, center_idx, int(visible.sum().item()), time_norm, vis_float

            time_rows.append(time_norm)
            vis_rows.append(vis_float)
            dir_rows.append(int(direction_label))
            speed_rows.append(float(speed_kmh) / max(1e-6, self.speed_norm_kmh))
            track_ids.append(track_id)
            mask_rows.append(self._render_instance_mask(center_idx=center_idx, visible=visible))
            return visible, center_idx, int(visible.sum().item()), time_norm, vis_float

        track_id = 0
        attempts = 0
        max_attempts = max(32, n_veh * 64)
        while track_id < n_veh and attempts < max_attempts:
            attempts += 1
            is_primary = bool(torch.rand((), generator=gen).item() < self.primary_ratio)
            direction_label = 0 if is_primary else 1
            if torch.rand((), generator=gen).item() < self.speed_outlier_ratio:
                if torch.rand((), generator=gen).item() < 0.5:
                    speed_lo, speed_hi = self.slow_speed_min_kmh, self.slow_speed_max_kmh
                else:
                    speed_lo, speed_hi = self.fast_speed_min_kmh, self.fast_speed_max_kmh
            else:
                speed_lo, speed_hi = self.speed_min_kmh, self.speed_max_kmh
            speed_kmh = float(speed_lo + torch.rand((), generator=gen).item() * (speed_hi - speed_lo))
            speed_mps = speed_kmh / 3.6
            sigma_s = float(self.sigma_min_s + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s))
            amp = float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min))
            anchor_ch = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
            anchor_time = float(torch.rand((), generator=gen).item() * self.window_seconds)
            rendered = _draw_vehicle(
                direction_label=direction_label,
                speed_kmh=speed_kmh,
                sigma_s=sigma_s,
                amp=amp,
                label_track=True,
                anchor_time=anchor_time,
                anchor_ch=anchor_ch,
                amp_scale=1.0,
                allow_dropout=True,
            )
            if rendered is None:
                continue
            visible, center_idx, _, _, _ = rendered

            decoy_points = 0
            if self.artifact_decoy_ratio > 0.0 and float(torch.rand((), generator=gen).item()) < self.artifact_decoy_ratio:
                decoy_points = int(
                    torch.randint(
                        int(self.artifact_decoy_min_points),
                        int(self.artifact_decoy_max_points) + 1,
                        (1,),
                        generator=gen,
                    ).item()
                )
                decoy_amp_scale = float(
                    self.artifact_decoy_amp_scale_min
                    + torch.rand((), generator=gen).item() * (self.artifact_decoy_amp_scale_max - self.artifact_decoy_amp_scale_min)
                )
                decoy_anchor = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
                decoy_direction = 1 if torch.rand((), generator=gen).item() < 0.5 else -1
                decoy_base_time = float(torch.rand((), generator=gen).item() * self.window_seconds)
                decoy_slope = float(
                    (self.dx_m / max(1e-6, speed_mps))
                    * (1.0 + 0.25 * (2.0 * torch.rand((), generator=gen).item() - 1.0))
                )
                for step in range(int(decoy_points)):
                    ch = int(decoy_anchor + decoy_direction * step)
                    if ch < 0 or ch >= self.n_channels:
                        continue
                    t_c = decoy_base_time + float(decoy_direction * step) * decoy_slope
                    if self.artifact_decoy_time_jitter_s > 0.0:
                        t_c += float(2.0 * torch.rand((), generator=gen).item() - 1.0) * float(self.artifact_decoy_time_jitter_s)
                    if not (0.0 <= t_c < self.window_seconds):
                        continue
                    center = int(round(t_c * self.fs))
                    half = int(max(1, round(3.0 * sigma_s * self.fs)))
                    left = max(0, center - half)
                    right = min(self.window_samples - 1, center + half)
                    idx = torch.arange(left, right + 1, dtype=torch.float32)
                    dt = idx / self.fs - float(t_c)
                    pulse = (amp * decoy_amp_scale) * torch.exp(-0.5 * (dt / (sigma_s * 0.9)) ** 2)
                    data[ch, left : right + 1] += pulse
            artifact_decoy_total += int(decoy_points)
            track_id += 1

        def _append_related_vehicle(
            *,
            source_track: int,
            relation: str,
            direction_label: int,
            base_speed_kmh: float,
            anchor_time: float,
            anchor_ch: int,
            allow_dropout: bool,
        ) -> bool:
            nonlocal track_id
            if source_track < 0 or source_track >= len(speed_rows):
                return False
            if relation == "same":
                speed_kmh = self._sample_related_speed_kmh(gen, base_speed_kmh, tight=True)
                is_parallel_close = float(torch.rand((), generator=gen).item()) < float(self.parallel_close_ratio)
                if is_parallel_close:
                    time_jitter = 0.12 * float(self.window_seconds)
                    channel_jitter = max(1, int(round(self.n_channels * 0.06)))
                    amp_scale = 0.92 + 0.16 * float(torch.rand((), generator=gen).item())
                else:
                    time_jitter = 0.25 * float(self.window_seconds)
                    channel_jitter = max(1, int(round(self.n_channels * 0.12)))
                    amp_scale = 0.95 + 0.25 * float(torch.rand((), generator=gen).item())
                direction = int(direction_label)
            elif relation == "cross":
                speed_kmh = self._sample_related_speed_kmh(gen, base_speed_kmh, tight=False)
                time_jitter = 0.40 * float(self.window_seconds)
                channel_jitter = max(1, int(round(self.n_channels * 0.20)))
                amp_scale = 0.85 + 0.40 * float(torch.rand((), generator=gen).item())
                direction = 1 - int(direction_label)
            else:
                return False
            related_anchor_time = float(anchor_time)
            if time_jitter > 0.0:
                related_anchor_time += float(2.0 * torch.rand((), generator=gen).item() - 1.0) * float(time_jitter)
            related_anchor_ch = int(anchor_ch)
            if channel_jitter > 0:
                related_anchor_ch = int(
                    max(
                        0,
                        min(
                            self.n_channels - 1,
                            related_anchor_ch
                            + int(
                                torch.randint(
                                    -int(channel_jitter),
                                    int(channel_jitter) + 1,
                                    (1,),
                                    generator=gen,
                                ).item()
                            ),
                        ),
                    )
                )
            sigma_s = float(self.sigma_min_s + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s))
            amp = float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min))
            rendered = _draw_vehicle(
                direction_label=direction,
                speed_kmh=speed_kmh,
                sigma_s=sigma_s,
                amp=amp,
                label_track=True,
                anchor_time=related_anchor_time,
                anchor_ch=related_anchor_ch,
                amp_scale=amp_scale,
                time_jitter_s=0.0,
                allow_dropout=allow_dropout,
            )
            if rendered is None:
                return False
            track_id += 1
            return True

        comp_time_norm = torch.zeros((self.n_channels,), dtype=torch.float32)
        comp_vis_float = torch.zeros((self.n_channels,), dtype=torch.float32)
        if time_rows and self.artifact_competing_ratio > 0.0 and float(torch.rand((), generator=gen).item()) < self.artifact_competing_ratio:
            target_speed_kmh = float(speed_rows[0]) * float(self.speed_norm_kmh)
            target_direction = int(dir_rows[0])
            target_anchor_ch = int(torch.argmax(vis_rows[0]).item())
            target_anchor_time = float(time_rows[0][target_anchor_ch].item() * float(self.window_samples - 1) / float(self.fs))
            competitor_direction = target_direction
            if float(torch.rand((), generator=gen).item()) < self.artifact_competing_opposite_direction_ratio:
                competitor_direction = 1 - int(target_direction)
            comp_speed_kmh = max(
                self.speed_min_kmh,
                min(
                    self.speed_max_kmh,
                    float(target_speed_kmh)
                    * (
                        float(self.artifact_competing_speed_ratio_min)
                        + float(torch.rand((), generator=gen).item())
                        * (float(self.artifact_competing_speed_ratio_max) - float(self.artifact_competing_speed_ratio_min))
                    ),
                ),
            )
            comp_anchor_time = float(target_anchor_time)
            if self.artifact_competing_time_jitter_s > 0.0:
                comp_anchor_time += float(2.0 * torch.rand((), generator=gen).item() - 1.0) * float(self.artifact_competing_time_jitter_s)
            comp_anchor_ch = int(target_anchor_ch)
            if self.artifact_competing_channel_offset_max > 0:
                comp_anchor_ch = int(
                    max(
                        0,
                        min(
                            self.n_channels - 1,
                            int(
                                comp_anchor_ch
                                + torch.randint(
                                    -int(self.artifact_competing_channel_offset_max),
                                    int(self.artifact_competing_channel_offset_max) + 1,
                                    (1,),
                                    generator=gen,
                                ).item()
                            ),
                        ),
                    )
                )
            comp_sigma_s = float(self.sigma_min_s + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s))
            comp_amp = float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min))
            comp_amp_scale = float(
                self.artifact_competing_amp_scale_min
                + torch.rand((), generator=gen).item() * (self.artifact_competing_amp_scale_max - self.artifact_competing_amp_scale_min)
            )
            comp_rendered = _draw_vehicle(
                direction_label=competitor_direction,
                speed_kmh=comp_speed_kmh,
                sigma_s=comp_sigma_s,
                amp=comp_amp,
                label_track=True,
                anchor_time=comp_anchor_time,
                anchor_ch=comp_anchor_ch,
                amp_scale=comp_amp_scale,
                time_jitter_s=0.0,
                allow_dropout=True,
            )
            if comp_rendered is not None:
                _, _, _, comp_time_norm, comp_vis_float = comp_rendered
                artifact_competing_direction = int(competitor_direction)
                artifact_competing_total += 1

        if self.scene_mode in {"realistic_traffic", "realistic", "traffic"} and time_rows:
            base_track_count = len(time_rows)
            same_target = int(round(base_track_count * float(self.same_direction_cluster_ratio)))
            cross_target = int(round(base_track_count * float(self.crossing_ratio)))
            same_added = 0
            cross_added = 0
            source_indices = list(range(base_track_count))
            for source_idx in source_indices:
                if same_added < same_target and float(torch.rand((), generator=gen).item()) < self.same_direction_cluster_ratio:
                    anchor_ch = int(torch.argmax(vis_rows[source_idx]).item())
                    anchor_time = float(time_rows[source_idx][anchor_ch].item() * float(self.window_samples - 1) / float(self.fs))
                    if _append_related_vehicle(
                        source_track=source_idx,
                        relation="same",
                        direction_label=int(dir_rows[source_idx]),
                        base_speed_kmh=float(speed_rows[source_idx]) * float(self.speed_norm_kmh),
                        anchor_time=anchor_time,
                        anchor_ch=anchor_ch,
                        allow_dropout=True,
                    ):
                        same_added += 1
                if cross_added < cross_target and float(torch.rand((), generator=gen).item()) < self.crossing_ratio:
                    anchor_ch = int(torch.argmax(vis_rows[source_idx]).item())
                    anchor_time = float(time_rows[source_idx][anchor_ch].item() * float(self.window_samples - 1) / float(self.fs))
                    if _append_related_vehicle(
                        source_track=source_idx,
                        relation="cross",
                        direction_label=int(dir_rows[source_idx]),
                        base_speed_kmh=float(speed_rows[source_idx]) * float(self.speed_norm_kmh),
                        anchor_time=anchor_time,
                        anchor_ch=anchor_ch,
                        allow_dropout=True,
                    ):
                        cross_added += 1

        x = self._prepare_input(data)
        if cache_x:
            x = self._cache_x(x)
        if time_rows:
            target = {
                "time": torch.stack(time_rows, dim=0),
                "visibility": torch.stack(vis_rows, dim=0),
                "direction": torch.tensor(dir_rows, dtype=torch.long),
                "speed": torch.tensor(speed_rows, dtype=torch.float32),
                "track_id": torch.tensor(track_ids, dtype=torch.long),
                "gt_masks": torch.stack(mask_rows, dim=0),
            }
        else:
            target = {
                "time": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "visibility": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "direction": torch.zeros((0,), dtype=torch.long),
                "speed": torch.zeros((0,), dtype=torch.float32),
                "track_id": torch.zeros((0,), dtype=torch.long),
                "gt_masks": torch.zeros((0, self.n_channels, self.ds_samples), dtype=torch.float32),
            }
        if self.return_raw_window:
            target["raw_window"] = data.to(torch.float32)
        target["background_meta"] = torch.tensor([background_meta.get("source", 0.0), background_meta.get("start_t", 0.0)], dtype=torch.float32)
        target["artifact_meta"] = torch.tensor(
            [float(artifact_dropout_total), float(artifact_decoy_total), float(artifact_competing_total)],
            dtype=torch.float32,
        )
        target["artifact_competing_time"] = comp_time_norm
        target["artifact_competing_visibility"] = comp_vis_float
        target["artifact_competing_direction"] = torch.tensor([float(artifact_competing_direction)], dtype=torch.float32)
        return x, target

    def _prepare_input(self, data: torch.Tensor) -> torch.Tensor:
        arr = data[:, :: self.time_downsample]
        abs_vals = torch.abs(arr)
        q995 = torch.quantile(abs_vals.flatten(), 0.995)
        rms = torch.sqrt(torch.mean(abs_vals * abs_vals))
        scale = torch.clamp(torch.maximum(q995, 3.0 * rms), min=1e-6)
        clip = float(max(1e-6, self.clip_ratio))
        raw = torch.clamp(arr / scale, -clip, clip) / clip
        if self.input_mode == "raw":
            return raw.unsqueeze(0).to(torch.float32)
        if self.input_mode == "raw_abs":
            abs_feat = torch.clamp(abs_vals / scale, 0.0, clip) / clip
            return torch.stack([raw, abs_feat], dim=0).to(torch.float32)
        raise ValueError(f"Unsupported input_mode={self.input_mode!r}; expected raw or raw_abs")


def _generate_cached_online_item(args: tuple[dict[str, object], int]) -> tuple[int, tuple[np.ndarray, dict[str, np.ndarray]]]:
    torch.set_num_threads(1)
    dataset_kwargs, index = args
    dataset = OnlineSyntheticTrajectoryDataset(length=1, cache_dataset=False, cache_build_workers=0, cache_dir=None, disk_cache_only=True, **dataset_kwargs)
    x, target = dataset._generate_item(int(index), cache_x=True)
    if x.dtype == torch.bfloat16:
        x = x.to(torch.float32)
    x_np = np.ascontiguousarray(x.cpu().numpy())
    target_np = {key: np.ascontiguousarray(value.cpu().numpy()) for key, value in target.items()}
    return int(index), (x_np, target_np)
