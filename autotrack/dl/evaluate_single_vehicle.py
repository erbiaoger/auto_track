"""Evaluate the single-vehicle model end to end on synthetic windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from autotrack.dl.online_synth_dataset import OnlineSyntheticTrajectoryDataset
from autotrack.dl.single_vehicle_net import (
    InferenceConfig,
    SingleVehicleTrackerConfig,
    estimate_direction_index_from_outputs,
    load_checkpoint_model,
    predict_single_vehicle_track,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single-vehicle checkpoint on synthetic windows.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary and CSV.")
    parser.add_argument("--benchmark-file", type=Path, default=None, help="Optional fixed benchmark .pt file.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--samples", type=int, default=256, help="Number of synthetic evaluation windows.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for loading windows.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of DAS channels.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=100.0, help="Channel spacing in meters.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Window duration in seconds.")
    parser.add_argument("--time-downsample", type=int, default=10, help="Temporal downsample factor.")
    parser.add_argument("--vehicles-min", type=int, default=1, help="Minimum vehicles per sample.")
    parser.add_argument("--vehicles-max", type=int, default=1, help="Maximum vehicles per sample.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=100.0, help="Maximum vehicle speed.")
    parser.add_argument("--noise-std", type=float, default=0.12, help="Background noise standard deviation.")
    parser.add_argument("--amp-min", type=float, default=0.8, help="Minimum pulse amplitude.")
    parser.add_argument("--amp-max", type=float, default=2.0, help="Maximum pulse amplitude.")
    parser.add_argument("--sigma-min-s", type=float, default=0.03, help="Minimum pulse width in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.08, help="Maximum pulse width in seconds.")
    parser.add_argument("--primary-ratio", type=float, default=1.0, help="Probability of forward-direction samples.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels per sample.")
    parser.add_argument("--speed-norm-kmh", type=float, default=150.0, help="Speed normalization used by synthetic labels.")
    parser.add_argument("--clip-ratio", type=float, default=1.35, help="Input clipping ratio.")
    parser.add_argument("--input-mode", default="raw", choices=["raw"], help="Synthetic input mode.")
    parser.add_argument("--background-npy", type=Path, default=None, help="Optional real background .npy to sample windows from.")
    parser.add_argument("--background-layout", default="time_channel", choices=["time_channel", "channel_time"], help="Layout of the real background .npy.")
    parser.add_argument("--background-channel-start", type=int, default=0, help="First channel index to slice from the real background.")
    parser.add_argument("--background-scale", type=float, default=1.0, help="Scale factor applied to the real background window.")
    parser.add_argument("--artifact-dropout-ratio", type=float, default=0.0, help="Chance to remove a contiguous block of visible channels.")
    parser.add_argument("--artifact-dropout-min-channels", type=int, default=2, help="Minimum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-dropout-max-channels", type=int, default=6, help="Maximum dropped channels when dropout is applied.")
    parser.add_argument("--artifact-decoy-ratio", type=float, default=0.0, help="Chance to inject a decoy branch or spike cluster.")
    parser.add_argument("--artifact-decoy-min-points", type=int, default=1, help="Minimum decoy points per sample.")
    parser.add_argument("--artifact-decoy-max-points", type=int, default=3, help="Maximum decoy points per sample.")
    parser.add_argument("--artifact-decoy-amp-scale-min", type=float, default=1.1, help="Minimum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-amp-scale-max", type=float, default=2.2, help="Maximum decoy amplitude scale relative to the true track.")
    parser.add_argument("--artifact-decoy-time-jitter-s", type=float, default=0.18, help="Random decoy time jitter in seconds.")
    parser.add_argument("--artifact-competing-ratio", type=float, default=0.0, help="Chance to inject an unlabeled competing vehicle track.")
    parser.add_argument("--artifact-competing-time-jitter-s", type=float, default=0.8, help="Random time jitter for the competing vehicle anchor.")
    parser.add_argument("--artifact-competing-amp-scale-min", type=float, default=0.8, help="Minimum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-amp-scale-max", type=float, default=1.6, help="Maximum competing-vehicle amplitude scale relative to the true track.")
    parser.add_argument("--artifact-competing-speed-ratio-min", type=float, default=0.88, help="Minimum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-speed-ratio-max", type=float, default=1.12, help="Maximum competing-vehicle speed ratio relative to the target track.")
    parser.add_argument("--artifact-competing-channel-offset-max", type=int, default=5, help="Maximum channel offset for the competing vehicle anchor relative to the target anchor.")
    parser.add_argument("--artifact-competing-opposite-direction-ratio", type=float, default=0.0, help="Probability of assigning the competing vehicle the opposite direction.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument("--objectness-threshold", type=float, default=0.35, help="Not used by the tracker yet; kept for symmetry.")
    parser.add_argument("--peak-threshold", type=float, default=0.25, help="Not used by the tracker yet; kept for symmetry.")
    parser.add_argument("--prior-weight", type=float, default=1.0, help="Weight applied to the model prior heatmap.")
    parser.add_argument("--candidate-prominence", type=float, default=0.22, help="Tracker peak prominence.")
    parser.add_argument("--candidate-min-distance", type=int, default=180, help="Tracker minimum peak distance.")
    parser.add_argument("--candidate-max-peaks-per-channel", type=int, default=32, help="Tracker per-channel peak cap.")
    parser.add_argument("--max-skip-channels", type=int, default=8, help="Tracker graph skip limit.")
    parser.add_argument("--min-track-channels", type=int, default=8, help="Minimum track length.")
    parser.add_argument("--min-track-score", type=float, default=8.0, help="Minimum track score.")
    parser.add_argument("--kalman-bridge-gap-channels", type=int, default=12, help="Maximum gap to bridge with Kalman smoothing.")
    parser.add_argument("--kalman-fill-missing", action=argparse.BooleanOptionalAction, default=True, help="Fill missing channels after smoothing.")
    parser.add_argument("--kalman-gate-seconds", type=float, default=0.35, help="Gate for Hungarian gap reassignment.")
    parser.add_argument("--kalman-speed-gate-kmh", type=float, default=30.0, help="Not used directly; kept for config parity.")
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


def _collate(items: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    xs = torch.stack([item[0] for item in items], dim=0).contiguous()
    common_keys = set(items[0][1].keys())
    for _, target in items[1:]:
        common_keys &= set(target.keys())
    keys = sorted(common_keys)
    targets = {key: torch.stack([item[1][key] for item in items], dim=0).contiguous() for key in keys}
    return xs, targets


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
        input_mode=str(args.input_mode),
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


class _BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, payload: dict[str, Any]):
        self.samples = list(payload.get("samples", []))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[int(index)]
        x = sample["x"].to(torch.float32)
        target = {key: value.clone() for key, value in sample["target"].items()}
        return x, target


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _first_track(targets: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in targets.items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = _resolve_device(args.device)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.benchmark_file is not None:
        payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
        dataset = _BenchmarkDataset(payload)
        bench_meta = dict(payload.get("meta", {}))
        for key in ("fs", "dx_m", "window_seconds", "time_downsample", "speed_norm_kmh", "min_visible_channels"):
            if key in bench_meta:
                setattr(args, key if key != "min_visible_channels" else "min_visible_channels", bench_meta[key])
        benchmark_window_seconds = float(bench_meta.get("window_seconds", args.window_seconds))
    else:
        dataset = _build_dataset(args, length=int(max(1, args.samples)), seed=int(args.seed))
        benchmark_window_seconds = float(args.window_seconds)
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=0, collate_fn=_collate)
    model, checkpoint = load_checkpoint_model(args.model, device=device)

    rows: list[dict[str, Any]] = []
    total_pred = 0
    total_gt = 0
    track_found = 0
    speed_abs_err: list[float] = []
    time_abs_err: list[float] = []
    direction_correct = 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (x, targets) in enumerate(loader):
            x = x.to(device)
            targets = {key: value.to(device) for key, value in targets.items()}
            for idx in range(int(x.shape[0])):
                sample_count += 1
                sample_targets = _first_track(targets, idx)
                gt_time = sample_targets["time"][0].detach().cpu().numpy()
                gt_vis = sample_targets["visibility"][0].detach().cpu().numpy()
                raw_window = sample_targets["raw_window"].detach().to(torch.float32).cpu().numpy()
                gt_speed = float(sample_targets["speed"][0].detach().cpu()) * float(args.speed_norm_kmh)
                gt_dir = int(sample_targets["direction"][0].detach().cpu())

                outputs = model(x[idx : idx + 1])
                pred_speed = float(outputs["speed"].item()) * float(args.speed_norm_kmh)
                if not math.isfinite(pred_speed) or pred_speed <= 0:
                    pred_speed = float(args.speed_min_kmh)
                speed_margin = max(15.0, 0.2 * pred_speed)
                tracks = predict_single_vehicle_track(
                    model,
                    raw_window,
                    float(args.fs),
                    float(args.dx_m),
                    "auto",
                    max(1.0, pred_speed - speed_margin),
                    pred_speed + speed_margin,
                    InferenceConfig(
                        time_downsample=int(args.time_downsample),
                        min_visible_channels=int(args.min_visible_channels),
                        objectness_threshold=float(args.objectness_threshold),
                        peak_threshold=float(args.peak_threshold),
                        prior_weight=float(args.prior_weight),
                        single_vehicle_tracker=SingleVehicleTrackerConfig(
                            candidate_prominence=float(args.candidate_prominence),
                            candidate_min_distance=int(args.candidate_min_distance),
                            candidate_max_peaks_per_channel=int(args.candidate_max_peaks_per_channel),
                            max_skip_channels=int(args.max_skip_channels),
                            min_track_channels=int(args.min_track_channels),
                            min_track_score=float(args.min_track_score),
                            kalman_bridge_gap_channels=int(args.kalman_bridge_gap_channels),
                            kalman_fill_missing=bool(args.kalman_fill_missing),
                            kalman_gate_seconds=float(args.kalman_gate_seconds),
                            kalman_speed_gate_kmh=float(args.kalman_speed_gate_kmh),
                        ),
                    ),
                    device=device,
                )

                total_gt += 1
                total_pred += len(tracks)
                if tracks:
                    track_found += 1
                    track = tracks[0]
                    pred_dir = 0 if str(track.direction).lower() == "forward" else 1
                    direction_correct += int(pred_dir == gt_dir)
                    pred_map = {int(point.ch_idx): float(point.time_s) for point in track.points}
                    common = [ch for ch, vis in enumerate(gt_vis) if vis > 0.5 and ch in pred_map]
                    if common:
                        if args.benchmark_file is not None:
                            scale = float(benchmark_window_seconds)
                        else:
                            scale = float(max(1, raw_window.shape[-1] - 1)) / float(args.fs)
                        gt_time_s = [float(gt_time[ch]) * scale for ch in common]
                        errors = [abs(gt_t - pred_map[ch]) for gt_t, ch in zip(gt_time_s, common)]
                        time_abs_err.append(float(np.mean(errors)))
                    speed_abs_err.append(abs(float(track.mean_speed_kmh) - gt_speed))
                else:
                    pred_dir = int(estimate_direction_index_from_outputs(outputs))
                    direction_correct += int(pred_dir == gt_dir)
                rows.append(
                    {
                        "sample_index": int(sample_count - 1),
                        "gt_direction": int(gt_dir),
                        "pred_direction": int(pred_dir),
                        "gt_speed_kmh": float(gt_speed),
                        "pred_speed_kmh": float(pred_speed),
                        "pred_track_count": int(len(tracks)),
                        "track_found": int(bool(tracks)),
                    }
                )

    summary = {
        "model": str(args.model),
        "device": device,
        "sample_count": int(sample_count),
        "track_found_rate": float(track_found / max(1, sample_count)),
        "direction_acc": float(direction_correct / max(1, sample_count)),
        "mean_speed_abs_error_kmh": float(np.mean(speed_abs_err)) if speed_abs_err else None,
        "mean_time_abs_error_s": float(np.mean(time_abs_err)) if time_abs_err else None,
        "mean_time_abs_error_norm": float(np.mean(time_abs_err) / max(1e-6, benchmark_window_seconds)) if time_abs_err else None,
        "avg_pred_tracks_per_sample": float(total_pred / max(1, sample_count)),
        "avg_gt_tracks_per_sample": float(total_gt / max(1, sample_count)),
        "checkpoint_metrics": checkpoint.get("metrics", {}),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "sample_summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()) if rows else ["sample_index"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
