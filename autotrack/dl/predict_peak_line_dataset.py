"""Predict and visualize PeakLineNet checkpoints on tensor shards.

Purpose:
    Evaluate a trained `peak_line` checkpoint directly on sparse point-image
    shards and write quick visual diagnostics. This script shows whether the
    model connects sparse peaks into continuous line probability maps while
    suppressing false input peaks.

Example:
    uv run python -m autotrack.dl.predict_peak_line_dataset \
        --data-dir datasets/peak_line/train \
        --model models/peak_line_cuda/checkpoint_best.pt \
        --out-dir /tmp/peak_line_prediction_check \
        --device cuda \
        --plot-samples 16

Outputs:
    <out-dir>/summary.json
    <out-dir>/plots/sample_000000.png, ...
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from autotrack.dl.peak_line_model import load_checkpoint_model, peak_line_loss, peak_line_metrics
from autotrack.dl.trajectory_set_model import auto_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PeakLineNet outputs from .pt tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Peak-line dataset directory.")
    parser.add_argument("--model", required=True, type=Path, help="PeakLineNet checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary and plots.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum evaluated samples; 0 evaluates all.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of PNG figures to write.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for PNG figures.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for metrics and overlay.")
    parser.add_argument("--pos-weight", type=float, default=20.0, help="Positive class BCE weight for loss reporting.")
    parser.add_argument("--dice-weight", type=float, default=1.0, help="Dice loss weight for loss reporting.")
    parser.add_argument("--focal-weight", type=float, default=0.25, help="Focal loss weight for loss reporting.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma for loss reporting.")
    return parser.parse_args()


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _iter_batches(
    data_dir: Path,
    shards: list[str],
    *,
    batch_size: int,
    max_samples: int,
) -> Iterator[tuple[list[int], torch.Tensor, dict[str, torch.Tensor]]]:
    emitted = 0
    global_start = 0
    for shard in shards:
        payload = torch.load(str(data_dir / shard), map_location="cpu", weights_only=False)
        n = int(payload["x"].shape[0])
        for start in range(0, n, int(batch_size)):
            if int(max_samples) > 0 and emitted >= int(max_samples):
                return
            take = min(int(batch_size), n - start)
            if int(max_samples) > 0:
                take = min(take, int(max_samples) - emitted)
            if take <= 0:
                return
            idx = torch.arange(start, start + take)
            sample_indices = list(range(global_start + start, global_start + start + take))
            emitted += take
            targets = {
                "line_mask": payload["line_mask"][idx].to(torch.float32),
                "point_target": payload["point_target"][idx].to(torch.float32),
            }
            yield sample_indices, payload["x"][idx].to(torch.float32), targets
        global_start += n


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _weighted_add(sums: defaultdict[str, float], weights: defaultdict[str, float], metrics: dict[str, float], weight: int) -> None:
    for key, value in metrics.items():
        value_f = float(value)
        if math.isfinite(value_f):
            sums[str(key)] += value_f * float(weight)
            weights[str(key)] += float(weight)


def _weighted_mean(sums: defaultdict[str, float], weights: defaultdict[str, float]) -> dict[str, float]:
    return {key: float(sums[key] / max(1e-12, weights[key])) for key in sorted(sums) if weights[key] > 0.0}


def _plot_sample(
    out_path: Path,
    *,
    sample_index: int,
    x: torch.Tensor,
    target: torch.Tensor,
    prob: torch.Tensor,
    meta: dict[str, Any],
    threshold: float,
    dpi: int,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    x_np = x.detach().cpu().squeeze(0).numpy()
    target_np = target.detach().cpu().squeeze(0).numpy()
    prob_np = prob.detach().cpu().squeeze(0).numpy()
    window_seconds = float(meta.get("window_seconds", x_np.shape[-1]))
    extent = [0.0, window_seconds, 0.0, float(x_np.shape[0] - 1)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [
        (axes[0, 0], x_np, "Input sparse peak image", "magma", 0.0, 1.0),
        (axes[0, 1], target_np, "GT continuous line mask", "viridis", 0.0, 1.0),
        (axes[1, 0], prob_np, "Predicted line probability", "viridis", 0.0, 1.0),
    ]
    for ax, image, title, cmap, vmin, vmax in panels:
        im = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel index")
        fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 1]
    ax.imshow(x_np, origin="lower", aspect="auto", extent=extent, cmap="gray", vmin=0.0, vmax=max(1e-6, float(np.max(x_np))))
    gt_mask = np.ma.masked_where(target_np < 0.5, target_np)
    pred_mask = np.ma.masked_where(prob_np < float(threshold), prob_np)
    ax.imshow(gt_mask, origin="lower", aspect="auto", extent=extent, cmap="Greens", alpha=0.45, vmin=0.0, vmax=1.0)
    ax.imshow(pred_mask, origin="lower", aspect="auto", extent=extent, cmap="Reds", alpha=0.55, vmin=0.0, vmax=1.0)
    ax.set_title(f"Threshold overlay, sample {sample_index}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel index")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    device = _resolve_device(str(args.device))
    model, checkpoint = load_checkpoint_model(args.model, device=device)
    model.eval()
    sums: defaultdict[str, float] = defaultdict(float)
    weights: defaultdict[str, float] = defaultdict(float)
    sample_count = 0
    plotted = 0
    with torch.no_grad():
        for sample_indices, x_cpu, targets_cpu in _iter_batches(
            data_dir,
            shards,
            batch_size=int(args.batch_size),
            max_samples=int(args.max_samples),
        ):
            x = x_cpu.to(device)
            targets = {key: value.to(device) for key, value in targets_cpu.items()}
            outputs = model(x)
            _, metrics = peak_line_loss(
                outputs,
                targets,
                pos_weight=float(args.pos_weight),
                dice_weight=float(args.dice_weight),
                focal_weight=float(args.focal_weight),
                focal_gamma=float(args.focal_gamma),
                threshold=float(args.threshold),
            )
            metrics.update(peak_line_metrics(outputs, targets, threshold=float(args.threshold)))
            _weighted_add(sums, weights, metrics, int(x_cpu.shape[0]))
            sample_count += int(x_cpu.shape[0])
            prob_cpu = torch.sigmoid(outputs["line_logits"]).detach().cpu()
            for batch_idx, sample_index in enumerate(sample_indices):
                if plotted >= int(args.plot_samples):
                    break
                _plot_sample(
                    out_dir / "plots" / f"sample_{int(sample_index):06d}.png",
                    sample_index=int(sample_index),
                    x=x_cpu[batch_idx],
                    target=targets_cpu["line_mask"][batch_idx],
                    prob=prob_cpu[batch_idx],
                    meta=meta,
                    threshold=float(args.threshold),
                    dpi=int(args.plot_dpi),
                )
                plotted += 1

    summary = {
        "model_family": "peak_line",
        "model": str(Path(args.model).expanduser()),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "device": device,
        "threshold": float(args.threshold),
        "sample_count": int(sample_count),
        "plotted": int(plotted),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "checkpoint_metrics": _json_ready(checkpoint.get("metrics", {})),
        "metrics": _weighted_mean(sums, weights),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
