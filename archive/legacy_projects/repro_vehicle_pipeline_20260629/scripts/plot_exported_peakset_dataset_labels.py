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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exported peak-set shard samples with GT labels.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Exported shard dataset directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--sample-index", type=int, default=0, help="First sample index.")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of consecutive samples.")
    parser.add_argument("--plot-dpi", type=int, default=170, help="Figure DPI.")
    return parser.parse_args(argv)


class ExportedDataset:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.meta = json.loads((self.dataset_dir / "meta.json").read_text(encoding="utf-8"))
        self.shard_paths = [self.dataset_dir / str(name) for name in self.meta.get("shards", [])]
        self.shard_sizes = [int(size) for size in self.meta.get("shard_sizes", [])]
        self.total = int(sum(self.shard_sizes))
        self._cache: dict[int, dict[str, Any]] = {}

    def _resolve(self, index: int) -> tuple[int, int]:
        acc = 0
        for shard_idx, size in enumerate(self.shard_sizes):
            nxt = acc + int(size)
            if int(index) < nxt:
                return shard_idx, int(index) - acc
            acc = nxt
        raise IndexError(index)

    def _load(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(str(self.shard_paths[shard_idx]), map_location="cpu", weights_only=False)
        return self._cache[shard_idx]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        shard_idx, local_idx = self._resolve(index)
        payload = self._load(shard_idx)
        x = payload["x"][local_idx].to(torch.float32)
        target = {key: value[local_idx].clone() for key, value in payload["targets"].items() if torch.is_tensor(value)}
        return x, target


def _window_seconds(meta: dict[str, Any], target: dict[str, torch.Tensor], x: torch.Tensor) -> float:
    cfg = meta.get("dataset_config", {})
    if isinstance(cfg, dict) and isinstance(cfg.get("window_seconds"), (float, int)):
        return float(cfg["window_seconds"])
    for comp in meta.get("components", []):
        ccfg = comp.get("dataset_config", {}) if isinstance(comp, dict) else {}
        if isinstance(ccfg, dict) and isinstance(ccfg.get("window_seconds"), (float, int)):
            return float(ccfg["window_seconds"])
    return 120.0


def _draw_gt(ax: Any, target: dict[str, torch.Tensor], *, window_seconds: float) -> None:
    full_time = target["full_time"].to(torch.float32).cpu().numpy() * float(window_seconds)
    full_valid = target["full_valid"].to(torch.bool).cpu().numpy()
    observed = target.get("observed_visibility", target["full_valid"]).to(torch.bool).cpu().numpy()
    gt_valid = target.get("gt_valid", torch.ones((full_time.shape[0],), dtype=torch.bool)).to(torch.bool).cpu().numpy()
    colors = ["#2f80ed", "#27ae60", "#d35400", "#8e44ad", "#c0392b", "#16a085", "#f1c40f", "#34495e"]
    for idx in range(int(full_time.shape[0])):
        if not bool(gt_valid[idx]):
            continue
        channels = np.where(full_valid[idx])[0]
        if channels.size == 0:
            continue
        color = colors[idx % len(colors)]
        times = full_time[idx, channels]
        ax.plot(channels, times, color=color, linewidth=1.2, alpha=0.8)
        obs_channels = channels[observed[idx, channels]]
        miss_channels = channels[~observed[idx, channels]]
        if obs_channels.size:
            ax.scatter(obs_channels, full_time[idx, obs_channels], s=8, marker="o", color=color, alpha=0.9, zorder=4)
        if miss_channels.size:
            ax.scatter(miss_channels, full_time[idx, miss_channels], s=12, marker="x", color=color, alpha=0.9, zorder=5)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = ExportedDataset(Path(args.dataset_dir))
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = int(max(1, args.num_samples))
    fig, axes = plt.subplots(rows, 1, figsize=(13, 4.5 * rows), dpi=int(args.plot_dpi), constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])

    outputs: list[str] = []
    summaries: list[dict[str, Any]] = []
    for row, sample_index in enumerate(range(int(args.sample_index), int(args.sample_index) + rows)):
        x, target = dataset[sample_index]
        base = x[0].to(torch.float32).cpu().numpy()
        window_seconds = _window_seconds(dataset.meta, target, x)
        ax = axes[row]
        ax.imshow(
            base.T,
            aspect="auto",
            origin="lower",
            cmap="gray_r",
            extent=(-0.5, float(base.shape[0]) - 0.5, 0.0, float(window_seconds)),
            vmin=-1.0,
            vmax=1.0,
        )
        _draw_gt(ax, target, window_seconds=window_seconds)
        gt_count = int(target.get("gt_valid", torch.ones((target["full_time"].shape[0],), dtype=torch.bool)).to(torch.bool).sum().item())
        missing_count = int(target.get("missing_channel_mask", torch.zeros_like(target["full_valid"])).to(torch.bool).sum().item())
        ax.set_title(f"sample={sample_index} gt={gt_count} missing_points={missing_count}")
        ax.set_xlabel("channel")
        ax.set_ylabel("time [s]")
        single_path = out_dir / f"exported_peakset_label_{sample_index:06d}.png"
        single_fig, single_ax = plt.subplots(1, 1, figsize=(13, 5), dpi=int(args.plot_dpi), constrained_layout=True)
        single_ax.imshow(base.T, aspect="auto", origin="lower", cmap="gray_r", extent=(-0.5, float(base.shape[0]) - 0.5, 0.0, float(window_seconds)), vmin=-1.0, vmax=1.0)
        _draw_gt(single_ax, target, window_seconds=window_seconds)
        single_ax.set_title(f"sample={sample_index} gt={gt_count} missing_points={missing_count}")
        single_ax.set_xlabel("channel")
        single_ax.set_ylabel("time [s]")
        single_fig.savefig(str(single_path))
        plt.close(single_fig)
        outputs.append(str(single_path))
        summaries.append({"sample_index": int(sample_index), "gt_count": gt_count, "missing_points": missing_count, "image": str(single_path)})

    sheet_path = out_dir / "exported_peakset_label_sheet.png"
    fig.savefig(str(sheet_path))
    plt.close(fig)
    summary = {"dataset_dir": str(Path(args.dataset_dir)), "sheet": str(sheet_path), "images": outputs, "samples": summaries}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
