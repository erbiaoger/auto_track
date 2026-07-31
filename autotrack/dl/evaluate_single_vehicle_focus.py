"""Evaluate the single-vehicle focus network on synthetic or benchmark windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.single_vehicle_focus_net import (
    FocusInferenceConfig,
    SingleVehicleFocusNet,
    load_checkpoint_model,
    predict_single_vehicle_focus_track,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the single-vehicle focus model.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--benchmark-file", type=Path, default=None, help="Optional benchmark .pt file.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--samples", type=int, default=256, help="Number of synthetic samples.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window duration.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample.")
    parser.add_argument("--vehicles-min", type=int, default=1, help="Minimum vehicles per sample.")
    parser.add_argument("--vehicles-max", type=int, default=1, help="Maximum vehicles per sample.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=100.0, help="Maximum vehicle speed.")
    parser.add_argument("--noise-std", type=float, default=0.12, help="Background noise std.")
    parser.add_argument("--amp-min", type=float, default=0.8, help="Minimum pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=2.0, help="Maximum pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.03, help="Minimum pulse width.")
    parser.add_argument("--sigma-max-s", type=float, default=0.08, help="Maximum pulse width.")
    parser.add_argument("--primary-ratio", type=float, default=1.0, help="Forward-direction ratio.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Background layout.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="Background channel start.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Background scale.")
    parser.add_argument("--artifact-dropout-ratio", type=float, default=0.0)
    parser.add_argument("--artifact-dropout-min-channels", type=int, default=2)
    parser.add_argument("--artifact-dropout-max-channels", type=int, default=6)
    parser.add_argument("--artifact-decoy-ratio", type=float, default=0.0)
    parser.add_argument("--artifact-decoy-min-points", type=int, default=1)
    parser.add_argument("--artifact-decoy-max-points", type=int, default=3)
    parser.add_argument("--artifact-decoy-amp-scale-min", type=float, default=1.1)
    parser.add_argument("--artifact-decoy-amp-scale-max", type=float, default=2.2)
    parser.add_argument("--artifact-decoy-time-jitter-s", type=float, default=0.18)
    parser.add_argument("--artifact-competing-ratio", type=float, default=0.4)
    parser.add_argument("--artifact-competing-time-jitter-s", type=float, default=0.8)
    parser.add_argument("--artifact-competing-amp-scale-min", type=float, default=0.8)
    parser.add_argument("--artifact-competing-amp-scale-max", type=float, default=1.6)
    parser.add_argument("--artifact-competing-speed-ratio-min", type=float, default=0.88)
    parser.add_argument("--artifact-competing-speed-ratio-max", type=float, default=1.12)
    parser.add_argument("--artifact-competing-channel-offset-max", type=int, default=5)
    parser.add_argument("--artifact-competing-opposite-direction-ratio", type=float, default=0.25)
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Prior heatmap weight passed to the tracker.")
    parser.add_argument("--competitor-weight", type=float, default=0.9, help="Competitor suppression weight passed to the tracker.")
    parser.add_argument("--plot-samples", type=int, default=4, help="How many overlay plots to render.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="Overlay DPI.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    return parser.parse_args(argv)


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _build_dataset(args: argparse.Namespace, *, length: int, seed: int) -> OnlineSyntheticTrajectoryDataset:
    return OnlineSyntheticTrajectoryDataset(
        length=int(max(1, length)),
        n_channels=int(args.n_channels),
        fs=float(args.fs),
        window_seconds=float(args.window_seconds),
        time_downsample=int(args.time_downsample),
        dx_m=float(args.dx_m),
        vehicles_min=int(args.vehicles_min),
        vehicles_max=int(args.vehicles_max),
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        speed_outlier_ratio=0.0,
        slow_speed_min_kmh=float(args.speed_min_kmh),
        slow_speed_max_kmh=float(args.speed_max_kmh),
        fast_speed_min_kmh=float(args.speed_min_kmh),
        fast_speed_max_kmh=float(args.speed_max_kmh),
        noise_std=float(args.noise_std),
        amp_min=float(args.amp_min),
        amp_max=float(args.amp_max),
        sigma_min_s=float(args.sigma_min_s),
        sigma_max_s=float(args.sigma_max_s),
        primary_ratio=float(args.primary_ratio),
        min_visible_channels=int(args.min_visible_channels),
        speed_norm_kmh=float(args.speed_norm_kmh),
        clip_ratio=float(args.clip_ratio),
        input_mode="raw",
        seed=int(seed),
        mask_sigma_ch=0.8,
        mask_sigma_t=2.0,
        cache_dataset=False,
        return_raw_window=True,
        background_npy=args.background_npy,
        background_layout=str(args.background_layout),
        background_channel_start=int(args.background_channel_start),
        background_scale=float(args.background_scale),
        artifact_dropout_ratio=float(args.artifact_dropout_ratio),
        artifact_dropout_min_channels=int(args.artifact_dropout_min_channels),
        artifact_dropout_max_channels=int(args.artifact_dropout_max_channels),
        artifact_decoy_ratio=float(args.artifact_decoy_ratio),
        artifact_decoy_min_points=int(args.artifact_decoy_min_points),
        artifact_decoy_max_points=int(args.artifact_decoy_max_points),
        artifact_decoy_amp_scale_min=float(args.artifact_decoy_amp_scale_min),
        artifact_decoy_amp_scale_max=float(args.artifact_decoy_amp_scale_max),
        artifact_decoy_time_jitter_s=float(args.artifact_decoy_time_jitter_s),
        artifact_competing_ratio=float(args.artifact_competing_ratio),
        artifact_competing_time_jitter_s=float(args.artifact_competing_time_jitter_s),
        artifact_competing_amp_scale_min=float(args.artifact_competing_amp_scale_min),
        artifact_competing_amp_scale_max=float(args.artifact_competing_amp_scale_max),
        artifact_competing_speed_ratio_min=float(args.artifact_competing_speed_ratio_min),
        artifact_competing_speed_ratio_max=float(args.artifact_competing_speed_ratio_max),
        artifact_competing_channel_offset_max=int(args.artifact_competing_channel_offset_max),
        artifact_competing_opposite_direction_ratio=float(args.artifact_competing_opposite_direction_ratio),
    )


class _BenchmarkDataset(Dataset):
    def __init__(self, payload: dict[str, Any]):
        self.samples = list(payload.get("samples", []))
        if not self.samples:
            raise ValueError("benchmark contains no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[int(index)]
        return {"x": sample["x"].to(torch.float32), "target": {k: v.clone() for k, v in sample["target"].items()}}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    xs_list: list[torch.Tensor] = []
    targets: list[dict[str, torch.Tensor]] = []
    for item in batch:
        if isinstance(item, dict):
            xs_list.append(item["x"])
            targets.append(item["target"])
        else:
            x, target = item
            xs_list.append(x)
            targets.append(target)
    xs = torch.stack(xs_list, dim=0).contiguous()
    return {"x": xs, "targets": targets}


def _plot_sample(out_path: Path, *, raw_window: np.ndarray, gt_time: np.ndarray, pred_tracks: list[Any], dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.5), constrained_layout=True, dpi=int(dpi))
    ax.imshow(raw_window, aspect="auto", origin="lower", cmap="magma")
    ax.plot(gt_time * max(1, raw_window.shape[1] - 1), np.arange(len(gt_time)), color="lime", lw=1.6, label="gt")
    for idx, tr in enumerate(pred_tracks):
        pts = sorted(tr.points, key=lambda p: int(p.ch_idx))
        ax.plot([float(p.t_idx) for p in pts], [int(p.ch_idx) for p in pts], lw=2.0, label=f"pred {idx}")
    ax.set_title(f"single focus overlay | pred={len(pred_tracks)}")
    ax.set_xlabel("time [sample idx]")
    ax.set_ylabel("channel")
    if pred_tracks:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.75)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_file is not None:
        payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
        dataset = _BenchmarkDataset(payload)
        meta = dict(payload.get("meta", {}))
        fs = float(meta.get("fs", args.fs))
        dx_m = float(meta.get("dx_m", args.dx_m))
    else:
        dataset = _build_dataset(args, length=int(max(1, args.samples)), seed=int(args.seed))
        fs = float(args.fs)
        dx_m = float(args.dx_m)

    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    model, checkpoint = load_checkpoint_model(args.model, device=device)

    rows: list[dict[str, Any]] = []
    total_found = 0
    total_dir_correct = 0
    speed_errors: list[float] = []
    time_errors: list[float] = []
    count_pred = 0
    count_gt = 0

    for idx, batch in enumerate(loader):
        x = batch["x"].to(device)
        targets = batch["targets"]
        for b in range(int(x.shape[0])):
            sample = targets[b]
            raw_window = sample["raw_window"].detach().cpu().numpy()
            gt_time = sample["time"][0].detach().cpu().numpy()
            gt_vis = sample["visibility"][0].detach().cpu().numpy()
            gt_dir = int(sample["direction"][0].detach().cpu())
            gt_speed = float(sample["speed"][0].detach().cpu())
            tracks = predict_single_vehicle_focus_track(
                model,
                raw_window,
                float(fs),
                float(dx_m),
                "auto",
                60.0,
                100.0,
                FocusInferenceConfig(
                    time_downsample=int(args.time_downsample),
                    min_visible_channels=int(args.min_visible_channels),
                    objectness_threshold=0.35,
                    visibility_threshold=0.5,
                    prior_weight=float(args.prior_weight),
                    competitor_weight=float(args.competitor_weight),
                ),
                device=device,
            )
            count_gt += 1
            count_pred += len(tracks)
            if tracks:
                total_found += 1
                tr = tracks[0]
                total_dir_correct += int((0 if str(tr.direction).lower() == "forward" else 1) == gt_dir)
                speed_errors.append(abs(float(tr.mean_speed_kmh) - gt_speed * float(args.speed_norm_kmh)))
                pred_map = {int(p.ch_idx): float(p.time_s) for p in tr.points}
                common = [ch for ch, v in enumerate(gt_vis) if v > 0.5 and ch in pred_map]
                if common:
                    scale = float(raw_window.shape[1] - 1) / float(fs)
                    errs = [abs(float(gt_time[ch]) * scale - pred_map[ch]) for ch in common]
                    time_errors.append(float(np.mean(errs)))
            rows.append(
                {
                    "sample_index": int(idx * int(x.shape[0]) + b),
                    "gt_speed_norm": float(gt_speed),
                    "pred_track_count": int(len(tracks)),
                    "track_found": int(bool(tracks)),
                }
            )
            if len(rows) <= int(args.plot_samples):
                _plot_sample(out_dir / f"sample_{len(rows)-1:02d}.png", raw_window=raw_window, gt_time=gt_time, pred_tracks=tracks, dpi=int(args.plot_dpi))

    summary = {
        "model": str(args.model),
        "device": device,
        "sample_count": int(count_gt),
        "track_found_rate": float(total_found / max(1, count_gt)),
        "direction_acc": float(total_dir_correct / max(1, count_gt)),
        "mean_speed_abs_error_kmh": float(np.mean(speed_errors)) if speed_errors else None,
        "mean_time_abs_error_s": float(np.mean(time_errors)) if time_errors else None,
        "avg_pred_tracks_per_sample": float(count_pred / max(1, count_gt)),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "sample_summary.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["sample_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
