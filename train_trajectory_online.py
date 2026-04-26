"""Train trajectory-query model with online synthetic windows.

用途：
    训练时直接在内存中合成 `[channel, time]` 窗口和车辆轨迹标签，不写 SAC，
    也不从磁盘读取 SAC。适合在 NVIDIA GPU 上快速预训练 MapTR-style
    轨迹 query 模型，避免训练集制作和训练读取的 IO 瓶颈。

用例：
    uv run python train_trajectory_online.py \
        --out-dir models/trajectory_query_online_v1 \
        --device cuda \
        --epochs 20 \
        --steps-per-epoch 10000 \
        --batch-size 8 \
        --window-seconds 60 \
        --fs 1000 \
        --time-downsample 20

参数说明：
    数据集是“固定样本池 + 每个 epoch 打乱顺序”：同一个 index 对应同一个
    合成窗口，模型会反复看到同一批结构；不同 index 仍覆盖不同车流密度、
    方向、速度、脉宽、噪声和窗口边界截断情况。`--steps-per-epoch` 控制
    固定样本池大小。`--vehicles-min/--vehicles-max` 控制单窗口车辆数范围；
    模型的 `--max-queries` 应大于 `vehicles-max`。`--trajectory-points`
    控制每个 query 输出多少个 `(channel, time)` polyline 点。

输出：
    - <out-dir>/checkpoint_last.pt
    - <out-dir>/checkpoint_best.pt
    - <out-dir>/train_config.json

限制：
    第一版在线合成器生成常速轨迹，主要用于绕开 IO 做快速预训练；如果需要
    stop-go/加减速，可继续用离线 SAC 数据做 finetune。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from trajectory_set_model import (
    ModelConfig,
    TrajectorySetPredictor,
    WindowDatasetConfig,
    auto_torch_device,
    save_checkpoint,
    trajectory_collate,
    trajectory_set_loss,
)

plt.rcParams["font.family"] = "Times New Roman"


class OnlineSyntheticTrajectoryDataset(Dataset):
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
        seed: int,
        cache_dataset: bool = False,
        cache_dtype: str = "float16",
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
        self.seed = int(seed)
        self.cache_dataset = bool(cache_dataset)
        self.cache_dtype = str(cache_dtype).lower()
        self._cache: Optional[list[tuple[torch.Tensor, dict[str, torch.Tensor]]]] = None
        if self.cache_dataset:
            self.build_cache()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self._cache is not None:
            x_cached, target = self._cache[int(index)]
            return x_cached.to(torch.float32), {key: value.clone() for key, value in target.items()}
        return self._generate_item(int(index), cache_x=False)

    def build_cache(self) -> None:
        cache: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
        t0 = time.perf_counter()
        for index in range(self.length):
            x, target = self._generate_item(index, cache_x=True)
            if (index + 1) % 500 == 0 or index + 1 == self.length:
                elapsed = time.perf_counter() - t0
                print(f"cache_dataset: {index + 1}/{self.length} windows, elapsed={elapsed:.1f}s", flush=True)
            cache.append((x, target))
        self._cache = cache

    def _cache_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_dtype in {"float16", "fp16", "half"}:
            return x.to(torch.float16).contiguous()
        if self.cache_dtype in {"bfloat16", "bf16"}:
            return x.to(torch.bfloat16).contiguous()
        return x.to(torch.float32).contiguous()

    def _generate_item(self, index: int, cache_x: bool):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed + int(index) * 1000003)

        data = torch.normal(
            mean=0.0,
            std=self.noise_std,
            size=(self.n_channels, self.window_samples),
            generator=gen,
            dtype=torch.float32,
        )
        n_veh = int(torch.randint(self.vehicles_min, self.vehicles_max + 1, (1,), generator=gen).item())
        time_rows: list[torch.Tensor] = []
        vis_rows: list[torch.Tensor] = []
        dir_rows: list[int] = []
        speed_rows: list[float] = []
        track_ids: list[int] = []

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
                    speed_lo = self.slow_speed_min_kmh
                    speed_hi = self.slow_speed_max_kmh
                else:
                    speed_lo = self.fast_speed_min_kmh
                    speed_hi = self.fast_speed_max_kmh
            else:
                speed_lo = self.speed_min_kmh
                speed_hi = self.speed_max_kmh
            speed_kmh = float(speed_lo + torch.rand((), generator=gen).item() * (speed_hi - speed_lo))
            speed_mps = speed_kmh / 3.6
            sigma_s = float(
                self.sigma_min_s
                + torch.rand((), generator=gen).item() * (self.sigma_max_s - self.sigma_min_s)
            )
            amp = float(self.amp_min + torch.rand((), generator=gen).item() * (self.amp_max - self.amp_min))

            if is_primary:
                dist_m = channel_index * self.dx_m
            else:
                dist_m = (self.n_channels - 1 - channel_index) * self.dx_m
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
            }
        else:
            target = {
                "time": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "visibility": torch.zeros((0, self.n_channels), dtype=torch.float32),
                "direction": torch.zeros((0,), dtype=torch.long),
                "speed": torch.zeros((0,), dtype=torch.float32),
                "track_id": torch.zeros((0,), dtype=torch.long),
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
        abs_feat = torch.clamp(abs_vals / scale, 0.0, clip) / clip
        return torch.stack([raw, abs_feat], dim=0).to(torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train trajectory-query model with online synthetic windows.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10000,
        help="Fixed online synthetic window pool size reused and shuffled each epoch.",
    )
    parser.add_argument("--val-steps", type=int, default=200, help="Online validation windows per validation run; 0 disables validation.")
    parser.add_argument("--val-every", type=int, default=1, help="Run validation every N epochs.")
    parser.add_argument("--plot-every", type=int, default=10, help="Save prediction plot every N epochs; 0 disables.")
    parser.add_argument("--plot-window-seconds", type=float, default=240.0, help="Window length for periodic prediction plots.")
    parser.add_argument("--plot-seed", type=int, default=987654, help="Fixed seed for periodic prediction plots.")
    parser.add_argument("--plot-objectness-threshold", type=float, default=0.35, help="Objectness threshold for periodic plots.")
    parser.add_argument("--plot-visibility-threshold", type=float, default=0.5, help="Visibility threshold for periodic plots.")
    parser.add_argument("--plot-top-k", type=int, default=20, help="Maximum predicted trajectories to draw in periodic plots.")
    parser.add_argument(
        "--plot-display-floor",
        type=float,
        default=0.08,
        help="Set normalized display values below this floor to zero in periodic plots.",
    )
    parser.add_argument("--no-object-weight", type=float, default=0.05, help="Loss weight for unmatched/no-object queries.")
    parser.add_argument("--duplicate-loss-weight", type=float, default=0.2, help="Penalty weight for high-confidence duplicate trajectory queries.")
    parser.add_argument("--duplicate-distance-tau", type=float, default=0.04, help="Normalized trajectory distance scale for duplicate penalty.")
    parser.add_argument("--denoising-loss-weight", type=float, default=1.0, help="Auxiliary loss weight for denoising trajectory queries.")
    parser.add_argument("--line-loss-weight", type=float, default=1.0, help="Soft loss weight for fitting each predicted trajectory to a line.")
    parser.add_argument("--slope-smooth-loss-weight", type=float, default=0.25, help="Soft loss weight for penalizing abrupt local speed changes.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--n-ch", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Training window length.")
    parser.add_argument("--time-downsample", type=int, default=20, help="Time-axis input stride.")
    parser.add_argument("--vehicles-min", type=int, default=0, help="Minimum vehicles per window.")
    parser.add_argument("--vehicles-max", type=int, default=32, help="Maximum vehicles per window.")
    parser.add_argument("--speed-min-kmh", type=float, default=70.0, help="Typical minimum speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=85.0, help="Typical maximum speed.")
    parser.add_argument("--speed-outlier-ratio", type=float, default=0.12, help="Fraction of vehicles sampled from slow/fast outlier speed ranges.")
    parser.add_argument("--slow-speed-min-kmh", type=float, default=45.0, help="Slow outlier minimum speed.")
    parser.add_argument("--slow-speed-max-kmh", type=float, default=60.0, help="Slow outlier maximum speed.")
    parser.add_argument("--fast-speed-min-kmh", type=float, default=95.0, help="Fast outlier minimum speed.")
    parser.add_argument("--fast-speed-max-kmh", type=float, default=120.0, help="Fast outlier maximum speed.")
    parser.add_argument("--noise-std", type=float, default=0.3, help="Gaussian noise std.")
    parser.add_argument("--amp-min", type=float, default=4.0, help="Minimum vehicle pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=8.0, help="Maximum vehicle pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.06, help="Minimum pulse sigma in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.18, help="Maximum pulse sigma in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=0.75, help="Forward/primary direction ratio.")
    parser.add_argument("--min-visible-channels", type=int, default=2, help="Minimum visible points per GT track.")
    parser.add_argument("--max-queries", type=int, default=96, help="Maximum predicted trajectory queries.")
    parser.add_argument("--hidden-dim", type=int, default=96, help="Transformer hidden dimension.")
    parser.add_argument("--decoder-layers", type=int, default=1, help="Transformer decoder layer count.")
    parser.add_argument("--num-heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--pooled-channels", type=int, default=8, help="Pooled feature channel dimension.")
    parser.add_argument("--pooled-time", type=int, default=128, help="Pooled feature time dimension.")
    parser.add_argument("--trajectory-points", type=int, default=32, help="Polyline points predicted per trajectory query.")
    parser.add_argument("--denoising-queries", type=int, default=32, help="Maximum denoising GT queries appended during training.")
    parser.add_argument("--dn-point-noise", type=float, default=0.04, help="Normalized coordinate noise added to denoising GT polyline inputs.")
    parser.add_argument("--device", default="", help="Torch device: cuda, mps, cpu, or empty for auto.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--cache-dataset", action="store_true", help="Precompute the fixed online dataset pool into RAM before training.")
    parser.add_argument("--cache-dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="RAM cache dtype for input tensors.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm; <=0 disables.")
    parser.add_argument("--log-every", type=int, default=0, help="Print batch progress every N batches; 0 prints epoch summaries only.")
    return parser.parse_args()


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    return {key: float(sum(item.get(key, 0.0) for item in items) / len(items)) for key in keys}


def _build_online_dataset(args: argparse.Namespace, *, length: int, seed: int) -> OnlineSyntheticTrajectoryDataset:
    dataset_config = WindowDatasetConfig(
        window_seconds=float(args.window_seconds),
        time_downsample=int(max(1, args.time_downsample)),
        samples_per_folder=int(max(1, length)),
        min_visible_channels=int(max(1, args.min_visible_channels)),
        seed=int(seed),
    )
    return OnlineSyntheticTrajectoryDataset(
        length=int(max(1, length)),
        n_channels=int(args.n_ch),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=float(args.speed_outlier_ratio),
        slow_speed_min_kmh=float(args.slow_speed_min_kmh),
        slow_speed_max_kmh=float(args.slow_speed_max_kmh),
        fast_speed_min_kmh=float(args.fast_speed_min_kmh),
        fast_speed_max_kmh=float(args.fast_speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        speed_norm_kmh=float(dataset_config.speed_norm_kmh),
        clip_ratio=float(dataset_config.clip_ratio),
        seed=int(seed),
        cache_dataset=bool(args.cache_dataset),
        cache_dtype=str(args.cache_dtype),
    )


def _evaluate(
    model: TrajectorySetPredictor,
    loader: DataLoader,
    device: str,
    no_object_weight: float,
    duplicate_loss_weight: float,
    duplicate_distance_tau: float,
    denoising_loss_weight: float,
    line_loss_weight: float,
    slope_smooth_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    metrics_items: list[dict[str, float]] = []
    with torch.no_grad():
        for x, targets in loader:
            x = x.to(device, non_blocking=(device == "cuda"))
            outputs = model(x)
            _, metrics = trajectory_set_loss(
                outputs,
                targets,
                no_object_weight=float(no_object_weight),
                duplicate_loss_weight=float(duplicate_loss_weight),
                duplicate_distance_tau=float(duplicate_distance_tau),
                denoising_loss_weight=float(denoising_loss_weight),
                line_loss_weight=float(line_loss_weight),
                slope_smooth_loss_weight=float(slope_smooth_loss_weight),
            )
            metrics_items.append(metrics)
    return _mean_metrics(metrics_items)


def _target_to_tracks(target: dict[str, torch.Tensor], fs: float, window_samples: int, dx_m: float) -> list[dict]:
    tracks = []
    times = target["time"].detach().cpu()
    vis = target["visibility"].detach().cpu()
    dirs = target["direction"].detach().cpu()
    n_tracks = int(times.shape[0])
    for i in range(n_tracks):
        chs = torch.where(vis[i] > 0.5)[0].tolist()
        pts = []
        for ch in chs:
            t_idx = int(round(float(times[i, ch]) * float(max(1, window_samples - 1))))
            pts.append((float(ch) * float(dx_m) * 1e-3, float(t_idx) / float(fs), int(ch), int(t_idx)))
        if pts:
            tracks.append({"track_id": i, "direction": int(dirs[i]), "points": pts})
    return tracks


def _predict_plot_tracks(
    model: TrajectorySetPredictor,
    x: torch.Tensor,
    fs: float,
    window_samples: int,
    dx_m: float,
    device: str,
    objectness_threshold: float,
    visibility_threshold: float,
    top_k: int,
) -> tuple[list[dict], dict[str, float]]:
    model.eval()
    with torch.no_grad():
        outputs = model(x.unsqueeze(0).to(device))
    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu()
    dirs = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu()
    if "points" in outputs and "point_valid_logits" in outputs:
        valid_prob = torch.sigmoid(outputs["point_valid_logits"][0]).detach().cpu()
        point_coords = outputs["points"][0].detach().cpu()
        visibility_source = valid_prob
    else:
        valid_prob = torch.sigmoid(outputs["visibility_logits"][0]).detach().cpu()
        point_coords = outputs["time"][0].detach().cpu()
        visibility_source = valid_prob
    order = torch.argsort(obj, descending=True).tolist()
    stats = {
        "max_objectness": float(torch.max(obj).item()) if obj.numel() else 0.0,
        "mean_objectness": float(torch.mean(obj).item()) if obj.numel() else 0.0,
        "max_visibility": float(torch.max(visibility_source).item()) if visibility_source.numel() else 0.0,
        "mean_visibility": float(torch.mean(visibility_source).item()) if visibility_source.numel() else 0.0,
    }
    tracks = []
    for q_idx in order:
        if len(tracks) >= int(top_k):
            break
        if float(obj[q_idx]) < float(objectness_threshold):
            continue
        pts = []
        if point_coords.ndim == 3:
            keep = torch.where(valid_prob[q_idx] >= float(visibility_threshold))[0].tolist()
            if len(keep) < 2:
                continue
            by_ch = {}
            for p_idx in keep:
                ch = int(round(float(point_coords[q_idx, p_idx, 0].clamp(0, 1)) * float(model.config.n_channels - 1)))
                t_idx = int(round(float(point_coords[q_idx, p_idx, 1].clamp(0, 1)) * float(max(1, window_samples - 1))))
                score = float(obj[q_idx] * valid_prob[q_idx, p_idx])
                old = by_ch.get(ch)
                if old is None or score > old[-1]:
                    by_ch[ch] = (float(ch) * float(dx_m) * 1e-3, float(t_idx) / float(fs), int(ch), int(t_idx), score)
            pts = [item[:4] for item in sorted(by_ch.values(), key=lambda item: item[2])]
        else:
            chs = torch.where(valid_prob[q_idx] >= float(visibility_threshold))[0].tolist()
            if len(chs) < 2:
                continue
            for ch in chs:
                t_idx = int(round(float(point_coords[q_idx, ch].clamp(0, 1)) * float(max(1, window_samples - 1))))
                pts.append((float(ch) * float(dx_m) * 1e-3, float(t_idx) / float(fs), int(ch), int(t_idx)))
        if len(pts) < 2:
            continue
        tracks.append(
            {
                "track_id": len(tracks),
                "query_id": int(q_idx),
                "score": float(obj[q_idx]),
                "direction": int(dirs[q_idx]),
                "points": pts,
            }
        )
    return tracks, stats


def _save_prediction_plot(
    model: TrajectorySetPredictor,
    args: argparse.Namespace,
    out_dir: Path,
    epoch: int,
    device: str,
) -> None:
    plot_dataset = OnlineSyntheticTrajectoryDataset(
        length=1,
        n_channels=int(args.n_ch),
        fs=float(args.fs),
        window_seconds=float(args.plot_window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=float(args.speed_outlier_ratio),
        slow_speed_min_kmh=float(args.slow_speed_min_kmh),
        slow_speed_max_kmh=float(args.slow_speed_max_kmh),
        fast_speed_min_kmh=float(args.fast_speed_min_kmh),
        fast_speed_max_kmh=float(args.fast_speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        speed_norm_kmh=150.0,
        clip_ratio=1.35,
        seed=int(args.plot_seed),
    )
    x, target = plot_dataset[0]
    window_samples = int(round(float(args.plot_window_seconds) * float(args.fs)))
    gt_tracks = _target_to_tracks(target, float(args.fs), window_samples, float(args.dx_m))
    pred_tracks, pred_stats = _predict_plot_tracks(
        model=model,
        x=x,
        fs=float(args.fs),
        window_samples=window_samples,
        dx_m=float(args.dx_m),
        device=device,
        objectness_threshold=float(args.plot_objectness_threshold),
        visibility_threshold=float(args.plot_visibility_threshold),
        top_k=int(args.plot_top_k),
    )

    shown = x[1].detach().cpu().numpy()
    display_floor = float(max(0.0, args.plot_display_floor))
    shown_plot = np.where(shown >= display_floor, shown, 0.0) if display_floor > 0.0 else shown
    vmax = float(max(np.percentile(shown_plot[np.isfinite(shown_plot)], 99.5), 1e-6))
    extent = [0.0, (int(args.n_ch) - 1) * float(args.dx_m) * 1e-3, 0.0, float(args.plot_window_seconds)]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(
        np.clip(shown_plot.T, 0.0, vmax),
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=extent,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
    )
    gt_cmap = plt.get_cmap("tab20", max(1, len(gt_tracks)))
    for i, tr in enumerate(gt_tracks):
        pts = sorted(tr["points"], key=lambda p: p[2])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=gt_cmap(i), linewidth=1.5, alpha=0.9)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], color=[gt_cmap(i)], s=8, alpha=0.9)
    for tr in pred_tracks:
        pts = sorted(tr["points"], key=lambda p: p[2])
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color="cyan",
            linewidth=1.2,
            linestyle="--",
            alpha=0.85,
        )
    ax.set_xlabel("Offset [km]")
    ax.set_ylabel("Time [s]")
    ax.set_title(
        f"Epoch {epoch}: GT={len(gt_tracks)}, Pred={len(pred_tracks)}, "
        f"Window={float(args.plot_window_seconds):.0f}s, "
        f"max_obj={pred_stats['max_objectness']:.3f}, max_vis={pred_stats['max_visibility']:.3f}"
    )
    ax.invert_yaxis()
    fig.tight_layout()
    plot_dir = out_dir / "prediction_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_png = plot_dir / f"epoch_{epoch:04d}.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"Saved prediction plot: {out_png}", flush=True)


def main() -> int:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = str(args.device).strip() or auto_torch_device()
    print(f"Using torch device: {device}")
    if bool(args.cache_dataset) and int(args.num_workers) != 0:
        print("cache_dataset is enabled; forcing num_workers=0 to avoid copying the RAM cache into worker processes.")
        args.num_workers = 0

    dataset_config = WindowDatasetConfig(
        window_seconds=float(args.window_seconds),
        time_downsample=int(max(1, args.time_downsample)),
        samples_per_folder=int(max(1, args.steps_per_epoch)),
        min_visible_channels=int(max(1, args.min_visible_channels)),
        seed=int(args.seed),
    )
    dataset = _build_online_dataset(args, length=int(args.steps_per_epoch), seed=int(args.seed))
    val_dataset = None
    val_loader = None
    if int(args.val_steps) > 0:
        val_dataset = _build_online_dataset(args, length=int(args.val_steps), seed=int(args.seed) + 10_000_000)
    model_config = ModelConfig(
        n_channels=int(args.n_ch),
        in_channels=2,
        max_queries=int(args.max_queries),
        hidden_dim=int(args.hidden_dim),
        num_heads=int(args.num_heads),
        decoder_layers=int(args.decoder_layers),
        pooled_channels=int(args.pooled_channels),
        pooled_time=int(args.pooled_time),
        trajectory_points=int(args.trajectory_points),
        denoising_queries=int(args.denoising_queries),
        dn_point_noise=float(args.dn_point_noise),
    )
    model = TrajectorySetPredictor(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=trajectory_collate,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            collate_fn=trajectory_collate,
            drop_last=False,
            pin_memory=(device == "cuda"),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "mode": "online_synthetic",
        "dataset_config": asdict(dataset_config),
        "model_config": asdict(model_config),
        "online_args": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "device": device,
        "created_at_unix": time.time(),
    }
    (args.out_dir / "train_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "Online dataset: "
        f"fixed_pool_windows={len(dataset)}, shuffle_each_epoch=True, batch_size={int(args.batch_size)}, "
        f"batches_per_epoch={len(loader)}, vehicles=[{args.vehicles_min}, {args.vehicles_max}], "
        f"window_seconds={args.window_seconds}, time_downsample={args.time_downsample}, "
        f"val_windows={int(args.val_steps)}, cache_dataset={bool(args.cache_dataset)}, cache_dtype={args.cache_dtype}"
    )
    print(
        "Model: "
        f"queries={model_config.max_queries}, hidden_dim={model_config.hidden_dim}, "
        f"decoder_layers={model_config.decoder_layers}, pooled=({model_config.pooled_channels}, {model_config.pooled_time}), "
        f"trajectory_points={model_config.trajectory_points}"
    )

    best_loss = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        t0 = time.perf_counter()
        epoch_metrics: list[dict[str, float]] = []
        log_every = int(args.log_every)
        for batch_idx, (x, targets) in enumerate(loader, start=1):
            x = x.to(device, non_blocking=(device == "cuda"))
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x, targets=targets)
            loss, metrics = trajectory_set_loss(
                outputs,
                targets,
                no_object_weight=float(args.no_object_weight),
                duplicate_loss_weight=float(args.duplicate_loss_weight),
                duplicate_distance_tau=float(args.duplicate_distance_tau),
                denoising_loss_weight=float(args.denoising_loss_weight),
                line_loss_weight=float(args.line_loss_weight),
                slope_smooth_loss_weight=float(args.slope_smooth_loss_weight),
            )
            loss.backward()
            if float(args.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()
            epoch_metrics.append(metrics)
            if log_every > 0 and (batch_idx == 1 or batch_idx % log_every == 0 or batch_idx == len(loader)):
                print(
                    f"epoch={epoch:03d} batch={batch_idx:04d}/{len(loader):04d} "
                    f"loss={metrics.get('loss', float('nan')):.4f} "
                    f"obj={metrics.get('loss_obj', float('nan')):.4f} "
                    f"time={metrics.get('loss_time', float('nan')):.4f} "
                    f"vis={metrics.get('loss_vis', float('nan')):.4f} "
                    f"dup={metrics.get('loss_duplicate', float('nan')):.4f} "
                    f"dn={metrics.get('loss_dn', float('nan')):.4f} "
                    f"line={metrics.get('loss_line', float('nan')):.4f} "
                    f"slope={metrics.get('loss_slope_smooth', float('nan')):.4f} "
                    f"gt={metrics.get('gt', 0.0):.0f} "
                    f"matched={metrics.get('matched', 0.0):.0f} "
                    f"max_obj={metrics.get('max_objectness', 0.0):.3f}",
                    flush=True,
                )

        mean_metrics = _mean_metrics(epoch_metrics)
        elapsed = time.perf_counter() - t0
        mean_metrics["epoch"] = float(epoch)
        mean_metrics["elapsed_seconds"] = float(elapsed)
        print(
            f"epoch={epoch:03d} loss={mean_metrics.get('loss', float('nan')):.4f} "
            f"time={mean_metrics.get('loss_time', float('nan')):.4f} "
            f"obj={mean_metrics.get('loss_obj', float('nan')):.4f} "
            f"dup={mean_metrics.get('loss_duplicate', float('nan')):.4f} "
            f"dn={mean_metrics.get('loss_dn', float('nan')):.4f} "
            f"line={mean_metrics.get('loss_line', float('nan')):.4f} "
            f"slope={mean_metrics.get('loss_slope_smooth', float('nan')):.4f} "
            f"gt={mean_metrics.get('gt', 0.0):.1f} "
            f"matched={mean_metrics.get('matched', 0.0):.1f} "
            f"max_obj={mean_metrics.get('max_objectness', 0.0):.3f} "
            f"elapsed={elapsed:.1f}s"
        )

        val_metrics: dict[str, float] = {}
        if val_loader is not None and (epoch % int(max(1, args.val_every)) == 0):
            val_t0 = time.perf_counter()
            val_metrics = _evaluate(
                model,
                val_loader,
                device,
                no_object_weight=float(args.no_object_weight),
                duplicate_loss_weight=float(args.duplicate_loss_weight),
                duplicate_distance_tau=float(args.duplicate_distance_tau),
                denoising_loss_weight=0.0,
                line_loss_weight=float(args.line_loss_weight),
                slope_smooth_loss_weight=float(args.slope_smooth_loss_weight),
            )
            print(
                f"epoch={epoch:03d} val_loss={val_metrics.get('loss', float('nan')):.4f} "
                f"val_time={val_metrics.get('loss_time', float('nan')):.4f} "
                f"val_obj={val_metrics.get('loss_obj', float('nan')):.4f} "
                f"val_gt={val_metrics.get('gt', 0.0):.1f} "
                f"val_matched={val_metrics.get('matched', 0.0):.1f} "
                f"val_max_obj={val_metrics.get('max_objectness', 0.0):.3f} "
                f"val_elapsed={time.perf_counter() - val_t0:.1f}s",
                flush=True,
            )

        last_path = args.out_dir / "checkpoint_last.pt"
        checkpoint_metrics = dict(mean_metrics)
        checkpoint_metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        save_checkpoint(last_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
        print(f"Saved checkpoint: {last_path}", flush=True)
        current_loss = float(val_metrics.get("loss", mean_metrics.get("loss", float("inf"))))
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = args.out_dir / "checkpoint_best.pt"
            save_checkpoint(best_path, model, optimizer, model_config, dataset_config, epoch, checkpoint_metrics)
            print(f"Saved new best checkpoint: {best_path}", flush=True)

        if int(args.plot_every) > 0 and epoch % int(args.plot_every) == 0:
            _save_prediction_plot(model, args, args.out_dir, epoch, device)

    print(f"Done. Best loss={best_loss:.4f}. Output: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
