from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from autotrack.dl.simple_vehicle_peak_dataset import SimpleLinearVehiclePeakDataset, SimplePeakSetDatasetConfig


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render profile-like synthetic dataset samples.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for the figure and summary.")
    parser.add_argument("--sample-index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of consecutive samples to render.")
    parser.add_argument("--plot-dpi", type=int, default=170, help="Figure DPI.")
    parser.add_argument("--seed", type=int, default=42, help="Dataset seed.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Synthetic window length in seconds.")
    parser.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    parser.add_argument("--dx-m", type=float, default=20.0, help="Channel spacing in meters.")
    parser.add_argument("--n-channels", type=int, default=50, help="Number of channels.")
    parser.add_argument("--vehicles-min", type=int, default=4, help="Minimum vehicles per scene.")
    parser.add_argument("--vehicles-max", type=int, default=10, help="Maximum vehicles per scene.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Minimum vehicle speed.")
    parser.add_argument("--speed-max-kmh", type=float, default=90.0, help="Maximum vehicle speed.")
    parser.add_argument("--dead-channel-indices", default="2,5,8,13,17,22,27,31,36,41,46", help="Fixed dead channels.")
    parser.add_argument("--intermittent-dead-channel-rates", default="", help="Per-window intermittent dead channels, e.g. 11:0.69,17:0.37,42:0.75.")
    parser.add_argument("--amp-min", type=float, default=3.0, help="Vehicle amplitude minimum.")
    parser.add_argument("--amp-max", type=float, default=8.0, help="Vehicle amplitude maximum.")
    parser.add_argument("--sigma-min-s", type=float, default=0.42, help="Vehicle sigma minimum in seconds.")
    parser.add_argument("--sigma-max-s", type=float, default=0.60, help="Vehicle sigma maximum in seconds.")
    parser.add_argument("--isolated-noise-ratio", type=float, default=0.30, help="Probability of isolated background peaks.")
    parser.add_argument("--isolated-noise-rate", type=float, default=14.0, help="Isolated background peak count.")
    parser.add_argument("--isolated-noise-amp-min", type=float, default=0.6, help="Isolated peak amplitude minimum.")
    parser.add_argument("--isolated-noise-amp-max", type=float, default=4.0, help="Isolated peak amplitude maximum.")
    parser.add_argument("--isolated-noise-sigma-min-s", type=float, default=0.42, help="Isolated peak sigma minimum.")
    parser.add_argument("--isolated-noise-sigma-max-s", type=float, default=0.60, help="Isolated peak sigma maximum.")
    parser.add_argument("--output-name", default="synthetic_profile_like_dataset.png", help="Figure filename.")
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _draw_tracks(ax: Any, target: dict[str, torch.Tensor], *, window_seconds: float) -> None:
    colors = ["cyan", "lime", "yellow", "orange", "red", "white", "deepskyblue", "magenta"]
    full_time = target["full_time"].cpu().numpy()
    full_valid = target["full_valid"].cpu().numpy() > 0.5
    observed = target["observed_visibility"].cpu().numpy() > 0.5
    for idx in range(full_time.shape[0]):
        ch_idx = np.where(full_valid[idx])[0]
        if ch_idx.size == 0:
            continue
        xs = ch_idx.astype(np.float32)
        ys = full_time[idx, ch_idx] * float(window_seconds)
        color = colors[idx % len(colors)]
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85)
        obs_ch = ch_idx[observed[idx, ch_idx]]
        miss_ch = ch_idx[~observed[idx, ch_idx]]
        if obs_ch.size > 0:
            ax.scatter(
                obs_ch,
                full_time[idx, obs_ch] * float(window_seconds),
                s=14,
                color=color,
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
            )
        if miss_ch.size > 0:
            ax.scatter(
                miss_ch,
                full_time[idx, miss_ch] * float(window_seconds),
                s=22,
                marker="x",
                color=color,
                linewidths=1.0,
                zorder=4,
            )


def _dead_channels_from_target(target: dict[str, torch.Tensor], n_channels: int) -> list[int]:
    mask = target.get("dead_channel_mask")
    if torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy().reshape(-1)
        if arr.size >= n_channels:
            return [int(i) for i, v in enumerate(arr[:n_channels]) if float(v) > 0.5]
    return []


def _shade_dead_channels(ax: Any, dead_channels: list[int], *, window_seconds: float, n_channels: int) -> None:
    if not dead_channels:
        return
    half_band = 0.38
    for ch in dead_channels:
        if 0 <= ch < n_channels:
            ax.axvspan(
                float(ch) - half_band,
                float(ch) + half_band,
                facecolor="#7f7f7f",
                edgecolor="#606060",
                linewidth=0.3,
                alpha=0.18,
                zorder=0,
            )
    ax.text(
        0.01,
        0.02,
        f"dead channels: {len(dead_channels)}",
        transform=ax.transAxes,
        color="#666666",
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.4},
        zorder=7,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = SimpleLinearVehiclePeakDataset(
        config=SimplePeakSetDatasetConfig(
            length=max(1, int(args.sample_index) + int(args.num_samples)),
            n_channels=int(args.n_channels),
            fs=float(args.fs),
            window_seconds=float(args.window_seconds),
            time_downsample=10,
            dx_m=float(args.dx_m),
            vehicles_min=int(args.vehicles_min),
            vehicles_max=int(args.vehicles_max),
            speed_min_kmh=float(args.speed_min_kmh),
            speed_max_kmh=float(args.speed_max_kmh),
            amp_min=float(args.amp_min),
            amp_max=float(args.amp_max),
            sigma_min_s=float(args.sigma_min_s),
            sigma_max_s=float(args.sigma_max_s),
            primary_ratio=0.82,
            same_direction_ratio=0.82,
            crossing_ratio=0.18,
            scene_cluster_ratio=0.35,
            motion_mix="constant_sparse,smooth_random,stop_go",
            motion_weights="0.84,0.15,0.01",
            constant_perturb_prob=0.0,
            constant_perturb_max_frac=0.0,
            smooth_speed_max_frac=0.0,
            track_time_jitter_max_s=0.0,
            stop_duration_min_s=1.0,
            stop_duration_max_s=1.0,
            dead_channel_indices=str(args.dead_channel_indices),
            intermittent_dead_channel_rates=str(args.intermittent_dead_channel_rates),
            random_dead_channel_ratio=0.0,
            zero_background_ratio=0.0,
            zero_background_rate=0.0,
            per_vehicle_drop_channel_ratio=0.0,
            missing_random_ratio_min=0.0,
            missing_random_ratio_max=0.0,
            missing_segment_count_max=0,
            interaction_ratio=0.0,
            isolated_noise_ratio=float(args.isolated_noise_ratio),
            isolated_noise_rate=float(args.isolated_noise_rate),
            isolated_noise_amp_min=float(args.isolated_noise_amp_min),
            isolated_noise_amp_max=float(args.isolated_noise_amp_max),
            isolated_noise_sigma_min_s=float(args.isolated_noise_sigma_min_s),
            isolated_noise_sigma_max_s=float(args.isolated_noise_sigma_max_s),
            return_raw_window=True,
            seed=int(args.seed),
        )
    )

    matplotlib.use("Agg")
    fig, axes = plt.subplots(int(args.num_samples), 2, figsize=(16, 5 * int(args.num_samples)), dpi=int(args.plot_dpi), constrained_layout=True)
    if int(args.num_samples) == 1:
        axes = np.asarray([axes])

    summary_rows: list[dict[str, Any]] = []
    for row, sample_idx in enumerate(range(int(args.sample_index), int(args.sample_index) + int(args.num_samples))):
        x, target = dataset[sample_idx]
        raw = target["raw_window"].cpu().numpy()
        raw_ds = raw[:, :: int(dataset.time_downsample)]
        window_seconds = float(args.window_seconds)
        dead_channels = _dead_channels_from_target(target, int(raw.shape[0]))
        ax0, ax1 = axes[row]

        ax0.imshow(raw_ds.T, aspect="auto", origin="lower", cmap="magma", extent=(-0.5, float(raw.shape[0]) - 0.5, 0.0, window_seconds))
        _shade_dead_channels(ax0, dead_channels, window_seconds=window_seconds, n_channels=int(raw.shape[0]))
        _draw_tracks(ax0, target, window_seconds=window_seconds)
        ax0.set_title(f"sample={sample_idx} | vehicles={int(target['track_id'].numel())}")
        ax0.set_xlabel("channel")
        ax0.set_ylabel("time [s]")

        observed = target["observed_visibility"].cpu().numpy() > 0.5
        full_valid = target["full_valid"].cpu().numpy() > 0.5
        mask = np.zeros_like(raw_ds, dtype=np.float32)
        for vidx in range(full_valid.shape[0]):
            ch_idx = np.where(full_valid[vidx])[0]
            if ch_idx.size == 0:
                continue
            vis = observed[vidx, ch_idx]
            mask[ch_idx, np.clip(np.round(target["full_time"][vidx, ch_idx].cpu().numpy() * (raw_ds.shape[1] - 1)).astype(int), 0, raw_ds.shape[1] - 1)] = vis.astype(np.float32) + 0.5 * (~vis).astype(np.float32)
        ax1.imshow(mask.T, aspect="auto", origin="lower", cmap="viridis", extent=(-0.5, float(raw.shape[0]) - 0.5, 0.0, window_seconds), vmin=0.0, vmax=1.0)
        _shade_dead_channels(ax1, dead_channels, window_seconds=window_seconds, n_channels=int(raw.shape[0]))
        _draw_tracks(ax1, target, window_seconds=window_seconds)
        ax1.set_title(f"observed mask | visible={int(observed.sum())} / full={int(full_valid.sum())}")
        ax1.set_xlabel("channel")
        ax1.set_ylabel("time [s]")

        summary_rows.append(
            {
                "sample_index": int(sample_idx),
                "vehicles": int(target["track_id"].numel()),
                "visible_points": int(observed.sum()),
                "full_points": int(full_valid.sum()),
                "raw_window_shape": list(raw.shape),
            }
        )

    overlay = out_dir / str(args.output_name)
    fig.savefig(str(overlay))
    plt.close(fig)

    summary = {"overlay": str(overlay), "samples": summary_rows, "seed": int(args.seed)}
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
