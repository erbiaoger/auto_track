"""Trajectory set prediction utilities for simulated DAS vehicle tracks.

This module implements the deep-learning path planned for the auto-track GUI:
it learns to predict a variable-size set of vehicle trajectories from one
`[channel, time]` DAS window. Each transformer query represents one possible
vehicle and directly regresses one time coordinate per channel plus a channel
visibility mask.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from obspy import read
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from track_extractor_graph import Track, TrackPoint


DIRECTION_TO_LABEL = {"forward": 0, "reverse": 1, "primary": 0, "secondary": 1}
LABEL_TO_DIRECTION = {0: "forward", 1: "reverse"}


@dataclass
class WindowDatasetConfig:
    window_seconds: float = 60.0
    time_downsample: int = 10
    samples_per_folder: int = 512
    min_visible_channels: int = 2
    speed_norm_kmh: float = 150.0
    clip_ratio: float = 1.35
    seed: int = 42


@dataclass
class ModelConfig:
    n_channels: int = 50
    in_channels: int = 2
    max_queries: int = 128
    hidden_dim: int = 128
    num_heads: int = 4
    decoder_layers: int = 2
    pooled_channels: int = 8
    pooled_time: int = 128
    dropout: float = 0.1


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.5
    visibility_threshold: float = 0.5
    min_visible_channels: int = 3
    refine_radius_samples: int = 120
    max_tracks: int = 128
    dedup_tolerance_samples: int = 180
    speed_norm_kmh: float = 150.0
    clip_ratio: float = 1.35


def load_sac_matrix(folder: str | Path) -> tuple[np.ndarray, float, np.ndarray, float]:
    folder_path = Path(folder).expanduser()
    sac_files = sorted(folder_path.glob("*.sac"))
    if not sac_files:
        raise FileNotFoundError(f"No .sac files found in folder: {folder_path}")

    traces = []
    for sac_file in sac_files:
        for tr in read(str(sac_file)):
            fallback = float(len(traces))
            try:
                distance = float(getattr(tr.stats, "distance", fallback))
            except Exception:  # noqa: BLE001
                distance = fallback
            traces.append((distance, len(traces), tr))
    if not traces:
        raise FileNotFoundError(f"Read failed: no valid SAC traces in folder: {folder_path}")

    traces.sort(key=lambda item: (item[0], item[1]))
    ordered = [tr for _, _, tr in traces]
    data = np.vstack([np.asarray(tr.data, dtype=np.float32) for tr in ordered])
    deltas = np.array([float(getattr(tr.stats, "delta", 0.001)) for tr in ordered], dtype=np.float64)
    delta = float(np.median(deltas))
    if delta <= 0:
        raise ValueError("Invalid SAC sampling interval")
    fs = 1.0 / delta

    x_axis = np.array([float(item[0]) for item in sorted(traces, key=lambda item: (item[0], item[1]))], dtype=np.float64)
    if x_axis.size >= 2 and np.any(np.diff(x_axis) <= 0):
        x_axis = np.arange(len(ordered), dtype=np.float64) * 100.0
    dx_m = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 100.0
    if dx_m < 10.0:
        dx_m = 100.0
        x_axis = np.arange(len(ordered), dtype=np.float64) * dx_m
    return data, float(fs), x_axis, float(dx_m)


def load_tracks_json(folder: str | Path) -> dict[str, Any]:
    path = Path(folder).expanduser() / "tracks.json"
    if not path.exists():
        raise FileNotFoundError(f"tracks.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _robust_scale(data: np.ndarray) -> float:
    finite = np.asarray(data[np.isfinite(data)], dtype=np.float32)
    if finite.size == 0:
        return 1.0
    abs_vals = np.abs(finite)
    q995 = float(np.quantile(abs_vals, 0.995))
    rms = float(np.sqrt(np.mean(abs_vals * abs_vals)))
    return max(q995, 3.0 * rms, 1e-6)


def prepare_window_input(
    data_window: np.ndarray,
    time_downsample: int,
    clip_ratio: float = 1.35,
) -> torch.Tensor:
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    stride = int(max(1, time_downsample))
    arr_ds = arr[:, ::stride]
    scale = _robust_scale(arr_ds)
    clip = float(max(clip_ratio, 1e-6))
    raw = np.clip(arr_ds / scale, -clip, clip) / clip
    abs_feat = np.clip(np.abs(arr_ds) / scale, 0.0, clip) / clip
    features = np.stack([raw, abs_feat], axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(features)


def build_window_target(
    tracks_payload: dict[str, Any],
    start_sample: int,
    window_samples: int,
    n_channels: int,
    min_visible_channels: int = 2,
    speed_norm_kmh: float = 150.0,
) -> dict[str, torch.Tensor]:
    end_sample = int(start_sample + window_samples)
    times: list[np.ndarray] = []
    visibilities: list[np.ndarray] = []
    directions: list[int] = []
    speeds: list[float] = []
    track_ids: list[int] = []

    denom = float(max(1, window_samples - 1))
    for track in tracks_payload.get("tracks", []):
        time_arr = np.zeros((n_channels,), dtype=np.float32)
        vis_arr = np.zeros((n_channels,), dtype=np.float32)
        for point in track.get("points", []):
            ch = int(point.get("ch_idx", -1))
            t_idx = int(point.get("t_idx", -1))
            if ch < 0 or ch >= n_channels:
                continue
            if start_sample <= t_idx < end_sample:
                vis_arr[ch] = 1.0
                time_arr[ch] = float(t_idx - start_sample) / denom
        if int(np.sum(vis_arr)) < int(min_visible_channels):
            continue
        direction = str(track.get("direction", track.get("sim_direction", "forward")))
        directions.append(int(DIRECTION_TO_LABEL.get(direction, 0)))
        speed_kmh = float(track.get("speed_kmh", 0.0))
        speeds.append(speed_kmh / max(1e-6, float(speed_norm_kmh)))
        times.append(time_arr)
        visibilities.append(vis_arr)
        track_ids.append(int(track.get("track_id", len(track_ids))))

    if not times:
        shape = (0, n_channels)
        return {
            "time": torch.zeros(shape, dtype=torch.float32),
            "visibility": torch.zeros(shape, dtype=torch.float32),
            "direction": torch.zeros((0,), dtype=torch.long),
            "speed": torch.zeros((0,), dtype=torch.float32),
            "track_id": torch.zeros((0,), dtype=torch.long),
        }

    return {
        "time": torch.from_numpy(np.stack(times, axis=0).astype(np.float32, copy=False)),
        "visibility": torch.from_numpy(np.stack(visibilities, axis=0).astype(np.float32, copy=False)),
        "direction": torch.tensor(directions, dtype=torch.long),
        "speed": torch.tensor(speeds, dtype=torch.float32),
        "track_id": torch.tensor(track_ids, dtype=torch.long),
    }


class SimulatedSacTrajectoryDataset(Dataset):
    def __init__(
        self,
        data_folders: Sequence[str | Path],
        config: Optional[WindowDatasetConfig] = None,
    ):
        self.config = config or WindowDatasetConfig()
        if not data_folders:
            raise ValueError("At least one data folder is required")
        self.records: list[dict[str, Any]] = []
        for folder in data_folders:
            data, fs, x_axis_m, dx_m = load_sac_matrix(folder)
            tracks_payload = load_tracks_json(folder)
            if int(data.shape[0]) != int(tracks_payload.get("n_ch", data.shape[0])):
                raise ValueError(f"Channel count mismatch between SAC and tracks.json in {folder}")
            self.records.append(
                {
                    "folder": str(Path(folder).expanduser()),
                    "data": data,
                    "fs": float(fs),
                    "x_axis_m": x_axis_m,
                    "dx_m": float(dx_m),
                    "tracks": tracks_payload,
                }
            )

    def __len__(self) -> int:
        return int(max(1, self.config.samples_per_folder) * len(self.records))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec_idx = int(index) % len(self.records)
        rec = self.records[rec_idx]
        data = rec["data"]
        fs = float(rec["fs"])
        window_samples = int(round(float(self.config.window_seconds) * fs))
        window_samples = max(1, min(window_samples, int(data.shape[1])))
        max_start = max(0, int(data.shape[1]) - window_samples)
        rng = np.random.default_rng(int(self.config.seed) + int(index) * 7919)
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        window = data[:, start : start + window_samples]
        x = prepare_window_input(
            window,
            time_downsample=int(self.config.time_downsample),
            clip_ratio=float(self.config.clip_ratio),
        )
        target = build_window_target(
            rec["tracks"],
            start_sample=start,
            window_samples=window_samples,
            n_channels=int(data.shape[0]),
            min_visible_channels=int(self.config.min_visible_channels),
            speed_norm_kmh=float(self.config.speed_norm_kmh),
        )
        return x, target


def trajectory_collate(batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    xs = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return xs, targets


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: tuple[int, int] = (1, 1)):
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectorySetPredictor(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        hidden = int(c.hidden_dim)
        self.backbone = nn.Sequential(
            ConvBlock(int(c.in_channels), 32, stride=(1, 2)),
            ConvBlock(32, 64, stride=(2, 2)),
            ConvBlock(64, hidden, stride=(2, 2)),
            ConvBlock(hidden, hidden, stride=(1, 2)),
        )
        self.pool_size = (int(c.pooled_channels), int(c.pooled_time))
        self.pos_embed = nn.Parameter(
            torch.randn(1, int(c.pooled_channels) * int(c.pooled_time), hidden) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden,
            nhead=int(c.num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(c.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(c.decoder_layers))
        self.query_embed = nn.Embedding(int(c.max_queries), hidden)

        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.visibility_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(c.n_channels)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = F.interpolate(
            self.backbone(x),
            size=self.pool_size,
            mode="bilinear",
            align_corners=False,
        )
        memory = feat.flatten(2).transpose(1, 2)
        memory = memory + self.pos_embed[:, : memory.shape[1], :]
        query = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        tgt = torch.zeros_like(query)
        hs = self.decoder(tgt=tgt, memory=memory)
        return {
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "visibility_logits": self.visibility_head(hs),
            "time": torch.sigmoid(self.time_head(hs)),
        }


def _match_single(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["time"].device
    gt_time = target["time"].to(device)
    gt_vis = target["visibility"].to(device)
    gt_dir = target["direction"].to(device)
    gt_speed = target["speed"].to(device)
    n_gt = int(gt_time.shape[0])
    if n_gt == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    with torch.no_grad():
        pred_time = outputs["time"][batch_idx]
        pred_vis = torch.sigmoid(outputs["visibility_logits"][batch_idx])
        pred_obj = torch.sigmoid(outputs["objectness_logits"][batch_idx])
        pred_dir = torch.softmax(outputs["direction_logits"][batch_idx], dim=-1)
        pred_speed = outputs["speed"][batch_idx]

        diff = torch.abs(pred_time[:, None, :] - gt_time[None, :, :])
        denom = torch.clamp(gt_vis.sum(dim=-1), min=1.0)[None, :]
        time_cost = (diff * gt_vis[None, :, :]).sum(dim=-1) / denom
        vis_cost = torch.mean(torch.abs(pred_vis[:, None, :] - gt_vis[None, :, :]), dim=-1)
        dir_cost = -pred_dir[:, gt_dir]
        speed_cost = torch.abs(pred_speed[:, None] - gt_speed[None, :])
        obj_cost = -pred_obj[:, None]
        cost = 4.0 * time_cost + 1.0 * vis_cost + 0.5 * dir_cost + 0.25 * speed_cost + 0.75 * obj_cost
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


def trajectory_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
    no_object_weight: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["time"].device
    batch_size, max_queries = outputs["objectness_logits"].shape
    loss_obj_terms: list[torch.Tensor] = []
    loss_time_terms: list[torch.Tensor] = []
    loss_vis_terms: list[torch.Tensor] = []
    loss_dir_terms: list[torch.Tensor] = []
    loss_speed_terms: list[torch.Tensor] = []
    matched_total = 0
    gt_total = 0
    max_objectness = 0.0
    mean_objectness = 0.0

    for b in range(batch_size):
        target = targets[b]
        gt_total += int(target["time"].shape[0])
        src_idx, tgt_idx = _match_single(outputs, target, b)
        matched_total += int(src_idx.numel())
        with torch.no_grad():
            obj_prob = torch.sigmoid(outputs["objectness_logits"][b])
            max_objectness = max(max_objectness, float(torch.max(obj_prob).detach().cpu()))
            mean_objectness += float(torch.mean(obj_prob).detach().cpu())

        obj_target = torch.zeros((max_queries,), dtype=torch.float32, device=device)
        obj_weight = torch.full((max_queries,), float(no_object_weight), dtype=torch.float32, device=device)
        if src_idx.numel() > 0:
            obj_target[src_idx] = 1.0
            obj_weight[src_idx] = 1.0
        loss_obj_terms.append(
            F.binary_cross_entropy_with_logits(
                outputs["objectness_logits"][b],
                obj_target,
                weight=obj_weight,
                reduction="mean",
            )
        )

        if src_idx.numel() == 0:
            continue

        gt_time = target["time"].to(device)[tgt_idx]
        gt_vis = target["visibility"].to(device)[tgt_idx]
        gt_dir = target["direction"].to(device)[tgt_idx]
        gt_speed = target["speed"].to(device)[tgt_idx]
        pred_time = outputs["time"][b, src_idx]
        pred_vis_logits = outputs["visibility_logits"][b, src_idx]
        pred_dir_logits = outputs["direction_logits"][b, src_idx]
        pred_speed = outputs["speed"][b, src_idx]

        vis_mask = gt_vis > 0.5
        if torch.any(vis_mask):
            loss_time_terms.append(F.smooth_l1_loss(pred_time[vis_mask], gt_time[vis_mask], reduction="mean"))
        loss_vis_terms.append(F.binary_cross_entropy_with_logits(pred_vis_logits, gt_vis, reduction="mean"))
        loss_dir_terms.append(F.cross_entropy(pred_dir_logits, gt_dir, reduction="mean"))
        loss_speed_terms.append(F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean"))

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss_obj = torch.stack(loss_obj_terms).mean() if loss_obj_terms else zero
    loss_time = torch.stack(loss_time_terms).mean() if loss_time_terms else zero
    loss_vis = torch.stack(loss_vis_terms).mean() if loss_vis_terms else zero
    loss_dir = torch.stack(loss_dir_terms).mean() if loss_dir_terms else zero
    loss_speed = torch.stack(loss_speed_terms).mean() if loss_speed_terms else zero
    total = loss_obj + 5.0 * loss_time + 1.0 * loss_vis + 0.5 * loss_dir + 0.5 * loss_speed
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_time": float(loss_time.detach().cpu()),
        "loss_vis": float(loss_vis.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(max_objectness),
        "mean_objectness": float(mean_objectness / max(1, batch_size)),
    }
    return total, metrics


def auto_torch_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_checkpoint(
    path: str | Path,
    model: TrajectorySetPredictor,
    optimizer: Optional[torch.optim.Optimizer],
    model_config: ModelConfig,
    dataset_config: WindowDatasetConfig,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": asdict(model_config),
        "dataset_config": asdict(dataset_config),
        "epoch": int(epoch),
        "metrics": dict(metrics),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, str(Path(path).expanduser()))


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: Optional[str] = None,
) -> tuple[TrajectorySetPredictor, dict[str, Any]]:
    resolved_device = device or auto_torch_device()
    # Always deserialize onto CPU first. Checkpoints may have been produced on
    # CUDA and later used on MPS/CPU, and direct cross-backend map_location can
    # hit backend-specific restore bugs.
    checkpoint = torch.load(str(Path(checkpoint_path).expanduser()), map_location="cpu", weights_only=False)
    model_config = ModelConfig(**checkpoint.get("model_config", {}))
    model = TrajectorySetPredictor(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _local_speed_series(points: list[TrackPoint]) -> list[float]:
    if not points:
        return []
    speeds = [float("nan")] * len(points)
    for i, point in enumerate(points):
        vals = []
        if i > 0:
            prev = points[i - 1]
            vals.append(3.6 * abs(point.offset_m - prev.offset_m) / max(1e-9, abs(point.time_s - prev.time_s)))
        if i + 1 < len(points):
            nxt = points[i + 1]
            vals.append(3.6 * abs(nxt.offset_m - point.offset_m) / max(1e-9, abs(nxt.time_s - point.time_s)))
        vals = [v for v in vals if np.isfinite(v)]
        speeds[i] = float(np.mean(vals)) if vals else float("nan")
    return speeds


def _track_stats(track_id: int, direction: str, points: list[TrackPoint]) -> Track:
    points_sorted = sorted(points, key=lambda p: p.ch_idx)
    speeds = [v for v in _local_speed_series(points_sorted) if np.isfinite(v)]
    mean_speed = float(np.mean(speeds)) if speeds else float("nan")
    total_score = float(np.sum([p.score for p in points_sorted]))
    return Track(
        track_id=int(track_id),
        direction=direction,
        points=points_sorted,
        total_score=total_score,
        mean_speed_kmh=mean_speed,
    )


def _refine_t_idx(data: np.ndarray, ch: int, t_idx: int, radius: int) -> int:
    n_samples = int(data.shape[1])
    center = int(max(0, min(n_samples - 1, t_idx)))
    rad = int(max(0, radius))
    if rad <= 0:
        return center
    left = max(0, center - rad)
    right = min(n_samples - 1, center + rad)
    local = np.abs(data[int(ch), left : right + 1])
    if local.size == 0:
        return center
    return int(left + int(np.argmax(local)))


def _track_overlap(track_a: Track, track_b: Track, tol_samples: int) -> int:
    a = {int(p.ch_idx): int(p.t_idx) for p in track_a.points}
    b = {int(p.ch_idx): int(p.t_idx) for p in track_b.points}
    overlap = 0
    for ch in set(a) & set(b):
        if abs(a[ch] - b[ch]) <= int(tol_samples):
            overlap += 1
    return overlap


def _deduplicate_tracks(tracks: list[Track], tol_samples: int) -> list[Track]:
    kept: list[Track] = []
    for track in sorted(tracks, key=lambda tr: tr.total_score, reverse=True):
        duplicate = False
        for existing in kept:
            overlap = _track_overlap(existing, track, tol_samples)
            ratio = overlap / max(1, min(len(existing.points), len(track.points)))
            if ratio >= 0.6:
                duplicate = True
                break
        if not duplicate:
            kept.append(track)
    return [_track_stats(i, tr.direction, tr.points) for i, tr in enumerate(kept)]


def predict_tracks_from_window(
    model: TrajectorySetPredictor,
    data_window: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    config: Optional[InferenceConfig] = None,
    device: Optional[str] = None,
) -> list[Track]:
    cfg = config or InferenceConfig()
    arr = np.asarray(data_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data_window must have shape [n_channel, n_sample]")
    if arr.shape[0] != int(model.config.n_channels):
        raise ValueError(
            f"Model expects {model.config.n_channels} channels, but input has {arr.shape[0]} channels"
        )
    resolved_device = device or next(model.parameters()).device
    x = prepare_window_input(arr, int(cfg.time_downsample), float(cfg.clip_ratio)).unsqueeze(0).to(resolved_device)
    with torch.inference_mode():
        outputs = model(x)
    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    vis_prob = torch.sigmoid(outputs["visibility_logits"][0]).detach().cpu().numpy()
    time_norm = outputs["time"][0].detach().cpu().numpy()
    direction_label = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    order = np.argsort(obj)[::-1]
    tracks: list[Track] = []
    n_samples = int(arr.shape[1])
    for q_idx in order[: int(max(1, cfg.max_tracks))]:
        if float(obj[q_idx]) < float(cfg.objectness_threshold):
            continue
        visible = vis_prob[q_idx] >= float(cfg.visibility_threshold)
        if int(np.sum(visible)) < int(cfg.min_visible_channels):
            continue
        points: list[TrackPoint] = []
        for ch in np.where(visible)[0].tolist():
            t_idx_raw = int(round(float(np.clip(time_norm[q_idx, ch], 0.0, 1.0)) * float(max(1, n_samples - 1))))
            t_idx = _refine_t_idx(arr, int(ch), t_idx_raw, int(cfg.refine_radius_samples))
            amp = float(abs(arr[int(ch), t_idx]))
            score = float(obj[q_idx] * vis_prob[q_idx, ch])
            points.append(
                TrackPoint(
                    ch_idx=int(ch),
                    t_idx=int(t_idx),
                    time_s=float(t_idx) / float(fs),
                    offset_m=float(x_axis_m[int(ch)]),
                    amp=amp,
                    score=score,
                )
            )
        if len(points) < int(cfg.min_visible_channels):
            continue
        direction = LABEL_TO_DIRECTION.get(int(direction_label[q_idx]), "forward")
        tracks.append(_track_stats(len(tracks), direction, points))
    return _deduplicate_tracks(tracks, int(cfg.dedup_tolerance_samples))
