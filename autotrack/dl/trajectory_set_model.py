"""Trajectory set prediction utilities for simulated DAS vehicle tracks.

This module implements the deep-learning path planned for the auto-track GUI:
it learns to predict a variable-size set of vehicle trajectories from one
`[channel, time]` DAS window. Each transformer query represents one possible
vehicle and directly regresses a polyline-like point set for that vehicle.
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

from autotrack.core.track_extractor_graph import Track, TrackPoint


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
    trajectory_points: int = 32
    denoising_queries: int = 0
    dn_point_noise: float = 0.04
    dropout: float = 0.1


@dataclass
class InferenceConfig:
    time_downsample: int = 10
    objectness_threshold: float = 0.35
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
        dn_input_dim = int(c.trajectory_points) * 3 + 3
        self.dn_query_mlp = nn.Sequential(
            nn.LayerNorm(dn_input_dim),
            nn.Linear(dn_input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.objectness_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))
        self.speed_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
        self.point_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(c.trajectory_points) * 2),
        )
        self.point_valid_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.trajectory_points)))
        # Legacy dense channel heads are kept as auxiliary outputs and for
        # compatibility with checkpoints/tools created before the polyline head.
        self.visibility_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, int(c.n_channels)))
        self.time_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, int(c.n_channels)),
        )

    def _build_denoising_queries(
        self,
        targets: Optional[Sequence[dict[str, torch.Tensor]]],
        batch_size: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[dict[str, torch.Tensor]]]:
        if targets is None or not self.training or int(self.config.denoising_queries) <= 0:
            return None, None
        max_dn = min(int(self.config.denoising_queries), max((int(t["time"].shape[0]) for t in targets), default=0))
        if max_dn <= 0:
            return None, None

        feat_dim = int(self.config.trajectory_points) * 3 + 3
        dn_features = torch.zeros((batch_size, max_dn, feat_dim), dtype=torch.float32, device=device)
        target_indices = torch.full((batch_size, max_dn), -1, dtype=torch.long, device=device)
        dn_valid = torch.zeros((batch_size, max_dn), dtype=torch.bool, device=device)
        noise_scale = float(max(0.0, self.config.dn_point_noise))

        for b, target in enumerate(targets):
            n_gt = int(target["time"].shape[0])
            if n_gt <= 0:
                continue
            count = min(max_dn, n_gt)
            perm = torch.randperm(n_gt, device=device)[:count]
            gt_points, gt_point_valid = _target_to_polyline(
                target,
                n_channels=int(self.config.n_channels),
                trajectory_points=int(self.config.trajectory_points),
                device=device,
            )
            noisy_points = gt_points[perm]
            if noise_scale > 0.0:
                noisy_points = (noisy_points + torch.randn_like(noisy_points) * noise_scale).clamp(0.0, 1.0)
            point_valid = gt_point_valid[perm]
            direction = F.one_hot(target["direction"].to(device)[perm].clamp(0, 1), num_classes=2).to(torch.float32)
            speed = target["speed"].to(device)[perm].unsqueeze(-1)
            dn_features[b, :count] = torch.cat(
                [
                    noisy_points.flatten(1),
                    point_valid,
                    direction,
                    speed,
                ],
                dim=-1,
            )
            target_indices[b, :count] = perm
            dn_valid[b, :count] = True

        if not torch.any(dn_valid):
            return None, None
        return self.dn_query_mlp(dn_features), {"target_indices": target_indices, "valid": dn_valid}

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[Sequence[dict[str, torch.Tensor]]] = None,
    ) -> dict[str, torch.Tensor]:
        feat = F.interpolate(
            self.backbone(x),
            size=self.pool_size,
            mode="bilinear",
            align_corners=False,
        )
        memory = feat.flatten(2).transpose(1, 2)
        memory = memory + self.pos_embed[:, : memory.shape[1], :]
        query = self.query_embed.weight.unsqueeze(0).expand(x.shape[0], -1, -1)
        dn_query, dn_meta = self._build_denoising_queries(targets, int(x.shape[0]), x.device)
        if dn_query is not None:
            query = torch.cat([query, dn_query], dim=1)
        hs = self.decoder(tgt=query, memory=memory)
        return {
            "num_regular_queries": int(self.config.max_queries),
            "dn_meta": dn_meta,
            "objectness_logits": self.objectness_head(hs).squeeze(-1),
            "direction_logits": self.direction_head(hs),
            "speed": self.speed_head(hs).squeeze(-1),
            "points": torch.sigmoid(self.point_head(hs)).view(
                x.shape[0],
                int(hs.shape[1]),
                int(self.config.trajectory_points),
                2,
            ),
            "point_valid_logits": self.point_valid_head(hs),
            "visibility_logits": self.visibility_head(hs),
            "time": torch.sigmoid(self.time_head(hs)),
        }


def _target_to_polyline(
    target: dict[str, torch.Tensor],
    n_channels: int,
    trajectory_points: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert dense channel labels to fixed-size MapTR-style polyline labels."""
    gt_time = target["time"].to(device)
    gt_vis = target["visibility"].to(device)
    n_gt = int(gt_time.shape[0])
    n_points = int(max(1, trajectory_points))
    points = torch.zeros((n_gt, n_points, 2), dtype=torch.float32, device=device)
    valid = torch.zeros((n_gt, n_points), dtype=torch.float32, device=device)
    if n_gt == 0:
        return points, valid

    denom_ch = float(max(1, int(n_channels) - 1))
    for i in range(n_gt):
        chs = torch.where(gt_vis[i] > 0.5)[0]
        if chs.numel() == 0:
            continue
        if int(chs.numel()) > n_points:
            sample_idx = torch.linspace(0, int(chs.numel()) - 1, n_points, device=device).round().long()
            chs = chs[sample_idx]
        count = int(min(int(chs.numel()), n_points))
        chosen = chs[:count]
        points[i, :count, 0] = chosen.to(torch.float32) / denom_ch
        points[i, :count, 1] = gt_time[i, chosen].clamp(0.0, 1.0)
        valid[i, :count] = 1.0
    return points, valid


def _regular_query_count(outputs: dict[str, Any]) -> int:
    return int(outputs.get("num_regular_queries", int(outputs["objectness_logits"].shape[1])))


def _match_single(
    outputs: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    batch_idx: int,
    matcher: str = "hungarian",
) -> tuple[torch.Tensor, torch.Tensor]:
    device = outputs["objectness_logits"].device
    gt_time = target["time"].to(device)
    gt_vis = target["visibility"].to(device)
    gt_dir = target["direction"].to(device)
    gt_speed = target["speed"].to(device)
    n_gt = int(gt_time.shape[0])
    if n_gt == 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    with torch.no_grad():
        regular_q = _regular_query_count(outputs)
        pred_points = outputs["points"][batch_idx, :regular_q]
        pred_valid = torch.sigmoid(outputs["point_valid_logits"][batch_idx, :regular_q])
        gt_points, gt_point_valid = _target_to_polyline(
            target,
            n_channels=int(outputs["time"].shape[-1]),
            trajectory_points=int(pred_points.shape[1]),
            device=device,
        )
        pred_obj = torch.sigmoid(outputs["objectness_logits"][batch_idx, :regular_q])
        pred_dir = torch.softmax(outputs["direction_logits"][batch_idx, :regular_q], dim=-1)
        pred_speed = outputs["speed"][batch_idx, :regular_q]

        diff = torch.abs(pred_points[:, None, :, :] - gt_points[None, :, :, :]).sum(dim=-1)
        denom = torch.clamp(gt_point_valid.sum(dim=-1), min=1.0)[None, :]
        point_cost = (diff * gt_point_valid[None, :, :]).sum(dim=-1) / denom
        valid_cost = torch.mean(torch.abs(pred_valid[:, None, :] - gt_point_valid[None, :, :]), dim=-1)
        dir_cost = -pred_dir[:, gt_dir]
        speed_cost = torch.abs(pred_speed[:, None] - gt_speed[None, :])
        obj_cost = -pred_obj[:, None]
        cost = 5.0 * point_cost + 1.0 * valid_cost + 0.5 * dir_cost + 0.25 * speed_cost + 0.75 * obj_cost
        if str(matcher).lower() == "greedy":
            rows_t, cols_t = _greedy_match_cost(cost)
            return rows_t.to(device=device), cols_t.to(device=device)
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(cols, dtype=torch.long, device=device),
    )


def _greedy_match_cost(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate one-to-one matching without copying the cost matrix to CPU."""
    device = cost.device
    q_count = int(cost.shape[0])
    gt_count = int(cost.shape[1])
    match_count = int(min(q_count, gt_count))
    if match_count <= 0:
        empty = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty

    work = cost.float().clone()
    large = torch.finfo(work.dtype).max
    rows: list[torch.Tensor] = []
    cols: list[torch.Tensor] = []
    for _ in range(match_count):
        flat_idx = torch.argmin(work)
        row = torch.div(flat_idx, gt_count, rounding_mode="floor").long()
        col = (flat_idx - row * gt_count).long()
        rows.append(row)
        cols.append(col)
        work[row, :] = large
        work[:, col] = large
    return torch.stack(rows).to(dtype=torch.long), torch.stack(cols).to(dtype=torch.long)


def _duplicate_query_loss(
    outputs: dict[str, torch.Tensor],
    regular_q: int,
    distance_tau: float = 0.04,
) -> torch.Tensor:
    device = outputs["objectness_logits"].device
    q_count = int(max(0, min(regular_q, int(outputs["objectness_logits"].shape[1]))))
    if q_count <= 1:
        return torch.zeros((), dtype=torch.float32, device=device)

    obj = torch.sigmoid(outputs["objectness_logits"][:, :q_count])
    points = outputs["points"][:, :q_count]
    valid = torch.sigmoid(outputs["point_valid_logits"][:, :q_count]).detach()
    tau = float(max(1e-6, distance_tau))
    losses: list[torch.Tensor] = []
    pair_mask = torch.triu(torch.ones((q_count, q_count), dtype=torch.bool, device=device), diagonal=1)

    for b in range(int(points.shape[0])):
        pair_valid = valid[b, :, None, :] * valid[b, None, :, :]
        denom = torch.clamp(pair_valid.sum(dim=-1), min=1.0)
        distance = (torch.abs(points[b, :, None, :, :] - points[b, None, :, :, :]).sum(dim=-1) * pair_valid).sum(dim=-1) / denom
        similarity = torch.exp(-distance.detach() / tau)
        pair_obj = obj[b, :, None] * obj[b, None, :]
        losses.append((pair_obj[pair_mask] * similarity[pair_mask]).mean())

    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32, device=device)


def _denoising_query_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
) -> torch.Tensor:
    device = outputs["objectness_logits"].device
    dn_meta = outputs.get("dn_meta")
    if not dn_meta:
        return torch.zeros((), dtype=torch.float32, device=device)
    target_indices = dn_meta["target_indices"].to(device)
    dn_valid = dn_meta["valid"].to(device)
    if target_indices.numel() == 0 or not torch.any(dn_valid):
        return torch.zeros((), dtype=torch.float32, device=device)

    regular_q = _regular_query_count(outputs)
    dn_count = int(target_indices.shape[1])
    dn_slice = slice(regular_q, regular_q + dn_count)
    obj_logits = outputs["objectness_logits"][:, dn_slice]
    pred_points = outputs["points"][:, dn_slice]
    pred_valid_logits = outputs["point_valid_logits"][:, dn_slice]
    pred_dir_logits = outputs["direction_logits"][:, dn_slice]
    pred_speed = outputs["speed"][:, dn_slice]

    point_terms: list[torch.Tensor] = []
    valid_terms: list[torch.Tensor] = []
    dir_terms: list[torch.Tensor] = []
    speed_terms: list[torch.Tensor] = []
    obj_terms: list[torch.Tensor] = []

    for b, target in enumerate(targets):
        keep = torch.where(dn_valid[b] & (target_indices[b] >= 0))[0]
        if keep.numel() == 0:
            continue
        gt_idx = target_indices[b, keep]
        gt_points_all, gt_point_valid_all = _target_to_polyline(
            target,
            n_channels=int(outputs["time"].shape[-1]),
            trajectory_points=int(outputs["points"].shape[2]),
            device=device,
        )
        gt_points = gt_points_all[gt_idx]
        gt_point_valid = gt_point_valid_all[gt_idx]
        gt_dir = target["direction"].to(device)[gt_idx]
        gt_speed = target["speed"].to(device)[gt_idx]

        point_mask = gt_point_valid > 0.5
        if torch.any(point_mask):
            point_terms.append(F.smooth_l1_loss(pred_points[b, keep][point_mask], gt_points[point_mask], reduction="mean"))
        valid_terms.append(F.binary_cross_entropy_with_logits(pred_valid_logits[b, keep], gt_point_valid, reduction="mean"))
        dir_terms.append(F.cross_entropy(pred_dir_logits[b, keep], gt_dir, reduction="mean"))
        speed_terms.append(F.smooth_l1_loss(pred_speed[b, keep], gt_speed, reduction="mean"))
        obj_terms.append(F.binary_cross_entropy_with_logits(obj_logits[b, keep], torch.ones_like(obj_logits[b, keep]), reduction="mean"))

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss_point = torch.stack(point_terms).mean() if point_terms else zero
    loss_valid = torch.stack(valid_terms).mean() if valid_terms else zero
    loss_dir = torch.stack(dir_terms).mean() if dir_terms else zero
    loss_speed = torch.stack(speed_terms).mean() if speed_terms else zero
    loss_obj = torch.stack(obj_terms).mean() if obj_terms else zero
    return loss_obj + 8.0 * loss_point + loss_valid + 0.5 * loss_dir + 0.5 * loss_speed


def _linearity_loss_for_points(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Softly penalize curved vehicle polylines by fitting time = a * channel + b."""
    device = points.device
    losses: list[torch.Tensor] = []
    for track_points, track_valid in zip(points, valid):
        keep = torch.where(track_valid > 0.5)[0]
        if int(keep.numel()) < 2:
            continue
        xy = track_points[keep]
        x = xy[:, 0]
        y = xy[:, 1]
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        slope = torch.sum(x_centered * y_centered) / torch.clamp(torch.sum(x_centered * x_centered), min=1e-6)
        intercept = y.mean() - slope * x.mean()
        fitted = slope * x + intercept
        losses.append(F.smooth_l1_loss(y, fitted, reduction="mean"))
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32, device=device)


def _slope_smooth_loss_for_points(points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Allow mild acceleration, but penalize sharp changes in local track slope."""
    device = points.device
    losses: list[torch.Tensor] = []
    for track_points, track_valid in zip(points, valid):
        keep = torch.where(track_valid > 0.5)[0]
        if int(keep.numel()) < 3:
            continue
        xy = track_points[keep]
        dx = torch.clamp(torch.abs(xy[1:, 0] - xy[:-1, 0]), min=0.02)
        dy = xy[1:, 1] - xy[:-1, 1]
        slopes = dy / dx
        losses.append(F.smooth_l1_loss(slopes[1:], slopes[:-1], reduction="mean"))
    return torch.stack(losses).mean() if losses else torch.zeros((), dtype=torch.float32, device=device)


def trajectory_set_loss(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
    no_object_weight: float = 0.05,
    duplicate_loss_weight: float = 0.0,
    duplicate_distance_tau: float = 0.04,
    denoising_loss_weight: float = 0.0,
    line_loss_weight: float = 0.0,
    slope_smooth_loss_weight: float = 0.0,
    matcher: str = "hungarian",
    collect_metrics: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = outputs["objectness_logits"].device
    batch_size = int(outputs["objectness_logits"].shape[0])
    max_queries = _regular_query_count(outputs)
    loss_obj_terms: list[torch.Tensor] = []
    loss_point_terms: list[torch.Tensor] = []
    loss_valid_terms: list[torch.Tensor] = []
    loss_dir_terms: list[torch.Tensor] = []
    loss_speed_terms: list[torch.Tensor] = []
    loss_line_terms: list[torch.Tensor] = []
    loss_slope_smooth_terms: list[torch.Tensor] = []
    matched_total = 0
    gt_total = 0
    max_objectness = 0.0
    mean_objectness = 0.0

    for b in range(batch_size):
        target = targets[b]
        gt_total += int(target["time"].shape[0])
        src_idx, tgt_idx = _match_single(outputs, target, b, matcher=matcher)
        matched_total += int(src_idx.numel())
        if collect_metrics:
            obj_prob = torch.sigmoid(outputs["objectness_logits"][b, :max_queries])
            max_objectness = max(max_objectness, float(torch.max(obj_prob).detach().cpu()))
            mean_objectness += float(torch.mean(obj_prob).detach().cpu())

        obj_target = torch.zeros((max_queries,), dtype=torch.float32, device=device)
        obj_weight = torch.full((max_queries,), float(no_object_weight), dtype=torch.float32, device=device)
        if src_idx.numel() > 0:
            obj_target[src_idx] = 1.0
            obj_weight[src_idx] = 1.0
        loss_obj_terms.append(
            F.binary_cross_entropy_with_logits(
                outputs["objectness_logits"][b, :max_queries],
                obj_target,
                weight=obj_weight,
                reduction="mean",
            )
        )

        if src_idx.numel() == 0:
            continue

        gt_points_all, gt_point_valid_all = _target_to_polyline(
            target,
            n_channels=int(outputs["time"].shape[-1]),
            trajectory_points=int(outputs["points"].shape[2]),
            device=device,
        )
        gt_points = gt_points_all[tgt_idx]
        gt_point_valid = gt_point_valid_all[tgt_idx]
        gt_dir = target["direction"].to(device)[tgt_idx]
        gt_speed = target["speed"].to(device)[tgt_idx]
        pred_points = outputs["points"][b, src_idx]
        pred_valid_logits = outputs["point_valid_logits"][b, src_idx]
        pred_dir_logits = outputs["direction_logits"][b, src_idx]
        pred_speed = outputs["speed"][b, src_idx]

        point_mask = gt_point_valid > 0.5
        if torch.any(point_mask):
            loss_point_terms.append(
                F.smooth_l1_loss(pred_points[point_mask], gt_points[point_mask], reduction="mean")
            )
        if float(line_loss_weight) > 0.0:
            loss_line_terms.append(_linearity_loss_for_points(pred_points, gt_point_valid))
        if float(slope_smooth_loss_weight) > 0.0:
            loss_slope_smooth_terms.append(_slope_smooth_loss_for_points(pred_points, gt_point_valid))
        loss_valid_terms.append(F.binary_cross_entropy_with_logits(pred_valid_logits, gt_point_valid, reduction="mean"))
        loss_dir_terms.append(F.cross_entropy(pred_dir_logits, gt_dir, reduction="mean"))
        loss_speed_terms.append(F.smooth_l1_loss(pred_speed, gt_speed, reduction="mean"))

    zero = torch.zeros((), dtype=torch.float32, device=device)
    loss_obj = torch.stack(loss_obj_terms).mean() if loss_obj_terms else zero
    loss_point = torch.stack(loss_point_terms).mean() if loss_point_terms else zero
    loss_valid = torch.stack(loss_valid_terms).mean() if loss_valid_terms else zero
    loss_dir = torch.stack(loss_dir_terms).mean() if loss_dir_terms else zero
    loss_speed = torch.stack(loss_speed_terms).mean() if loss_speed_terms else zero
    loss_line = torch.stack(loss_line_terms).mean() if loss_line_terms else zero
    loss_slope_smooth = torch.stack(loss_slope_smooth_terms).mean() if loss_slope_smooth_terms else zero
    loss_duplicate = _duplicate_query_loss(
        outputs,
        regular_q=max_queries,
        distance_tau=float(duplicate_distance_tau),
    ) if float(duplicate_loss_weight) > 0.0 else zero
    loss_dn = _denoising_query_loss(outputs, targets) if float(denoising_loss_weight) > 0.0 else zero
    total = (
        loss_obj
        + 8.0 * loss_point
        + 1.0 * loss_valid
        + 0.5 * loss_dir
        + 0.5 * loss_speed
        + float(duplicate_loss_weight) * loss_duplicate
        + float(denoising_loss_weight) * loss_dn
        + float(line_loss_weight) * loss_line
        + float(slope_smooth_loss_weight) * loss_slope_smooth
    )
    if not collect_metrics:
        return total, {}
    metrics = {
        "loss": float(total.detach().cpu()),
        "loss_obj": float(loss_obj.detach().cpu()),
        "loss_time": float(loss_point.detach().cpu()),
        "loss_vis": float(loss_valid.detach().cpu()),
        "loss_point": float(loss_point.detach().cpu()),
        "loss_valid": float(loss_valid.detach().cpu()),
        "loss_dir": float(loss_dir.detach().cpu()),
        "loss_speed": float(loss_speed.detach().cpu()),
        "loss_duplicate": float(loss_duplicate.detach().cpu()),
        "loss_dn": float(loss_dn.detach().cpu()),
        "loss_line": float(loss_line.detach().cpu()),
        "loss_slope_smooth": float(loss_slope_smooth.detach().cpu()),
        "matched": float(matched_total),
        "gt": float(gt_total),
        "max_objectness": float(max_objectness),
        "mean_objectness": float(mean_objectness / max(1, batch_size)),
    }
    return total, metrics


def trajectory_detection_metrics(
    outputs: dict[str, torch.Tensor],
    targets: Sequence[dict[str, torch.Tensor]],
    objectness_threshold: float = 0.5,
    point_threshold: float = 0.05,
    matcher: str = "hungarian",
) -> dict[str, float]:
    """Vehicle-level detection metrics for trajectory set prediction.

    A predicted query is a vehicle detection when its objectness is above
    `objectness_threshold`. A detected vehicle is counted as correct when its
    matched normalized polyline point error is below `point_threshold`.
    """
    device = outputs["objectness_logits"].device
    max_queries = _regular_query_count(outputs)
    threshold = float(objectness_threshold)
    point_threshold = float(point_threshold)

    good_total = 0
    pred_total = 0
    gt_total = 0
    count_abs_error = 0.0
    count_exact = 0
    point_errors: list[torch.Tensor] = []
    time_errors: list[torch.Tensor] = []

    with torch.no_grad():
        obj_prob = torch.sigmoid(outputs["objectness_logits"][:, :max_queries])
        for b, target in enumerate(targets):
            n_gt = int(target["time"].shape[0])
            active = obj_prob[b] >= threshold
            pred_count = int(active.sum().detach().cpu())
            pred_total += pred_count
            gt_total += n_gt
            count_abs_error += abs(pred_count - n_gt)
            count_exact += int(pred_count == n_gt)
            if n_gt == 0:
                continue

            src_idx, tgt_idx = _match_single(outputs, target, b, matcher=matcher)
            if src_idx.numel() == 0:
                continue
            gt_points_all, gt_point_valid_all = _target_to_polyline(
                target,
                n_channels=int(outputs["time"].shape[-1]),
                trajectory_points=int(outputs["points"].shape[2]),
                device=device,
            )
            pred_points = outputs["points"][b, src_idx]
            gt_points = gt_points_all[tgt_idx]
            gt_valid = gt_point_valid_all[tgt_idx]
            matched_active = active[src_idx]
            if not torch.any(matched_active):
                continue
            pair_valid = gt_valid[matched_active] > 0.5
            pair_pred = pred_points[matched_active]
            pair_gt = gt_points[matched_active]
            denom = torch.clamp(pair_valid.float().sum(dim=-1), min=1.0)
            point_err = (torch.abs(pair_pred - pair_gt).sum(dim=-1) * pair_valid.float()).sum(dim=-1) / denom
            time_err = (torch.abs(pair_pred[..., 1] - pair_gt[..., 1]) * pair_valid.float()).sum(dim=-1) / denom
            good_total += int((point_err <= point_threshold).sum().detach().cpu())
            point_errors.append(point_err)
            time_errors.append(time_err)

    precision = float(good_total / max(1, pred_total))
    recall = float(good_total / max(1, gt_total))
    f1 = float(2.0 * precision * recall / max(1e-12, precision + recall))
    point_mae = float(torch.cat(point_errors).mean().detach().cpu()) if point_errors else float("nan")
    time_mae = float(torch.cat(time_errors).mean().detach().cpu()) if time_errors else float("nan")
    batch_size = int(outputs["objectness_logits"].shape[0])
    return {
        "track_precision": precision,
        "track_recall": recall,
        "track_f1": f1,
        "track_tp": float(good_total),
        "pred_count": float(pred_total),
        "gt_count": float(gt_total),
        "count_mae": float(count_abs_error / max(1, batch_size)),
        "count_acc": float(count_exact / max(1, batch_size)),
        "point_mae_norm": point_mae,
        "time_mae_norm": time_mae,
    }


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
    raw_model_config = dict(checkpoint.get("model_config", {}))
    has_polyline_head = "trajectory_points" in raw_model_config
    model_config = ModelConfig(**raw_model_config)
    model = TrajectorySetPredictor(model_config).to(resolved_device)
    missing, _unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    model.prefer_dense_output = bool(not has_polyline_head or any(key.startswith("point_") for key in missing))
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


def _predict_tracks_from_polyline_outputs(
    outputs: dict[str, torch.Tensor],
    arr: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    cfg: InferenceConfig,
) -> list[Track]:
    obj = torch.sigmoid(outputs["objectness_logits"][0]).detach().cpu().numpy()
    point_valid = torch.sigmoid(outputs["point_valid_logits"][0]).detach().cpu().numpy()
    point_coords = outputs["points"][0].detach().cpu().numpy()
    direction_label = torch.argmax(outputs["direction_logits"][0], dim=-1).detach().cpu().numpy()

    order = np.argsort(obj)[::-1]
    tracks: list[Track] = []
    n_channels = int(arr.shape[0])
    n_samples = int(arr.shape[1])
    for q_idx in order[: int(max(1, cfg.max_tracks))]:
        if float(obj[q_idx]) < float(cfg.objectness_threshold):
            continue
        valid = point_valid[q_idx] >= float(cfg.visibility_threshold)
        if int(np.sum(valid)) < int(cfg.min_visible_channels):
            continue

        point_by_channel: dict[int, TrackPoint] = {}
        for p_idx in np.where(valid)[0].tolist():
            ch_float = float(np.clip(point_coords[q_idx, p_idx, 0], 0.0, 1.0))
            t_float = float(np.clip(point_coords[q_idx, p_idx, 1], 0.0, 1.0))
            ch = int(round(ch_float * float(max(1, n_channels - 1))))
            ch = int(max(0, min(n_channels - 1, ch)))
            t_idx_raw = int(round(t_float * float(max(1, n_samples - 1))))
            t_idx = _refine_t_idx(arr, ch, t_idx_raw, int(cfg.refine_radius_samples))
            amp = float(abs(arr[ch, t_idx]))
            score = float(obj[q_idx] * point_valid[q_idx, p_idx])
            point = TrackPoint(
                ch_idx=ch,
                t_idx=int(t_idx),
                time_s=float(t_idx) / float(fs),
                offset_m=float(x_axis_m[ch]),
                amp=amp,
                score=score,
            )
            old = point_by_channel.get(ch)
            if old is None or point.score > old.score:
                point_by_channel[ch] = point

        points = sorted(point_by_channel.values(), key=lambda p: p.ch_idx)
        if len(points) < int(cfg.min_visible_channels):
            continue
        direction = LABEL_TO_DIRECTION.get(int(direction_label[q_idx]), "forward")
        tracks.append(_track_stats(len(tracks), direction, points))
    return tracks


def _predict_tracks_from_dense_outputs(
    outputs: dict[str, torch.Tensor],
    arr: np.ndarray,
    fs: float,
    x_axis_m: np.ndarray,
    cfg: InferenceConfig,
) -> list[Track]:
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
    return tracks


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
    if "points" in outputs and "point_valid_logits" in outputs and not bool(getattr(model, "prefer_dense_output", False)):
        tracks = _predict_tracks_from_polyline_outputs(outputs, arr, float(fs), x_axis_m, cfg)
    else:
        tracks = _predict_tracks_from_dense_outputs(outputs, arr, float(fs), x_axis_m, cfg)
    return _deduplicate_tracks(tracks, int(cfg.dedup_tolerance_samples))
