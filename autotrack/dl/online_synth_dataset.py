from __future__ import annotations

import hashlib
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
        mask_sigma_ch: float = 0.8,
        mask_sigma_t: float = 2.0,
        cache_dataset: bool = False,
        cache_dtype: str = "float16",
        cache_build_workers: int = 0,
        cache_dir: Optional[Path] = None,
        cache_rebuild: bool = False,
        disk_cache_only: bool = True,
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
        self.mask_sigma_ch = float(max(1e-3, mask_sigma_ch))
        self.mask_sigma_t = float(max(1e-3, mask_sigma_t))
        self.ds_samples = int(max(1, len(range(0, self.window_samples, self.time_downsample))))
        self.cache_dataset = bool(cache_dataset)
        self.cache_dtype = str(cache_dtype).lower()
        self.cache_build_workers = int(max(0, cache_build_workers))
        self.cache_rebuild = bool(cache_rebuild)
        self.disk_cache_only = bool(disk_cache_only)
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
            "mask_sigma_ch": float(self.mask_sigma_ch),
            "mask_sigma_t": float(self.mask_sigma_t),
            "cache_dtype": str(self.cache_dtype),
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
            "mask_sigma_ch": self.mask_sigma_ch,
            "mask_sigma_t": self.mask_sigma_t,
            "cache_dtype": self.cache_dtype,
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

    def _cache_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dtype in {"float16", "fp16", "half"}:
            return x.to(torch.float16).contiguous()
        if self.cache_dtype in {"bfloat16", "bf16"}:
            return x.to(torch.bfloat16).contiguous()
        return x.to(torch.float32).contiguous()

    def _generate_item(self, index: int, cache_x: bool):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + int(index) * 1000003)

        data = torch.normal(mean=0.0, std=self.noise_std, size=(self.n_channels, self.window_samples), generator=gen, dtype=torch.float32)
        n_veh = int(torch.randint(self.vehicles_min, self.vehicles_max + 1, (1,), generator=gen).item())
        time_rows: list[torch.Tensor] = []
        vis_rows: list[torch.Tensor] = []
        dir_rows: list[int] = []
        speed_rows: list[float] = []
        track_ids: list[int] = []
        mask_rows: list[torch.Tensor] = []

        channel_index = torch.arange(self.n_channels, dtype=torch.float32)
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

            dist_m = channel_index * self.dx_m if is_primary else (self.n_channels - 1 - channel_index) * self.dx_m
            anchor_ch = int(torch.randint(0, self.n_channels, (1,), generator=gen).item())
            anchor_time = float(torch.rand((), generator=gen).item() * self.window_seconds)
            t_entry = anchor_time - float(dist_m[anchor_ch].item()) / max(1e-6, speed_mps)
            t_center = t_entry + dist_m / max(1e-6, speed_mps)
            visible = (t_center >= 0.0) & (t_center < self.window_seconds)
            if int(visible.sum().item()) < self.min_visible_channels:
                continue

            center_idx = torch.round(t_center * self.fs).to(torch.long).clamp(0, self.window_samples - 1)
            half_width = int(max(1, round(4.0 * sigma_s * self.fs)))
            for ch in torch.where(visible)[0].tolist():
                center = int(center_idx[ch].item())
                left = max(0, center - half_width)
                right = min(self.window_samples - 1, center + half_width)
                idx = torch.arange(left, right + 1, dtype=torch.float32)
                dt = idx / self.fs - float(t_center[ch].item())
                pulse = amp * torch.exp(-0.5 * (dt / sigma_s) ** 2)
                data[int(ch), left : right + 1] += pulse

            time_norm = torch.zeros((self.n_channels,), dtype=torch.float32)
            vis_float = visible.to(torch.float32)
            time_norm[visible] = (center_idx[visible].to(torch.float32) / float(max(1, self.window_samples - 1))).clamp(0, 1)
            time_rows.append(time_norm)
            vis_rows.append(vis_float)
            dir_rows.append(direction_label)
            speed_rows.append(speed_kmh / max(1e-6, self.speed_norm_kmh))
            track_ids.append(track_id)
            mask_rows.append(self._render_instance_mask(center_idx=center_idx, visible=visible))
            track_id += 1

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
