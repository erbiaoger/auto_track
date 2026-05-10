"""Predict and evaluate PeakSlotNet checkpoints on peak-slot tensor shards.

Purpose:
    Inspect a trained `peak_slot` checkpoint directly on converted tensor
    shards. Predictions are exported as selected peak candidates, so plotted
    points always lie on detected heatmap peaks.

Example:
    uv run python -m autotrack.dl.predict_peak_slot_dataset \
        --data-dir datasets/peak_slot/train \
        --model models/peak_slot_cuda/checkpoint_best.pt \
        --out-dir /tmp/peak_slot_prediction_check \
        --device cuda \
        --plot-samples 16

Outputs:
    <out-dir>/summary.json
    <out-dir>/sample_summary.csv
    <out-dir>/predicted_tracks.csv
    <out-dir>/ground_truth_tracks.csv
    <out-dir>/plots/sample_000000.png, ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from autotrack.dl.peak_slot_model import (
    InferenceConfig,
    argmax_peak_slot_path,
    decode_peak_slot_path,
    decode_peak_slot_paths,
    load_checkpoint_model,
    peak_slot_detection_metrics,
    peak_slot_physics_metrics,
    peak_slot_set_loss,
)
from autotrack.dl.trajectory_set_model import LABEL_TO_DIRECTION, auto_torch_device, move_targets_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PeakSlotNet outputs from converted .pt tensor shards.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Peak-slot dataset directory.")
    parser.add_argument("--model", required=True, type=Path, help="PeakSlotNet checkpoint path.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for summary, CSV, and plots.")
    parser.add_argument("--device", default="auto", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size.")
    parser.add_argument("--max-samples", type=int, default=256, help="Maximum evaluated samples; 0 evaluates all.")
    parser.add_argument("--max-csv-samples", type=int, default=32, help="Detailed CSV sample limit; 0 disables detailed CSV.")
    parser.add_argument("--plot-samples", type=int, default=16, help="Number of overlay figures to write.")
    parser.add_argument("--plot-dpi", type=int, default=160, help="DPI for overlay PNG figures.")
    parser.add_argument("--plot-style", default="waveform", choices=["waveform", "heatmap"], help="Overlay plot style: GUI-like waveform or heatmap.")
    parser.add_argument("--objectness-threshold", type=float, default=0.5, help="Predicted slot objectness threshold.")
    parser.add_argument("--peak-threshold", type=float, default=0.4, help="Minimum selected peak probability.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum selected peaks for a predicted track.")
    parser.add_argument("--max-predicted-tracks", type=int, default=64, help="Maximum slots kept per sample.")
    parser.add_argument("--decoder-mode", default="beam_global", choices=["argmax", "viterbi", "beam_global"], help="Peak decoding strategy.")
    parser.add_argument("--no-viterbi-decoder", action="store_true", help="Use legacy per-channel argmax decoding instead of Viterbi.")
    parser.add_argument("--viterbi-beam-size", type=int, default=4, help="Number of candidate paths retained per slot in beam_global decoding.")
    parser.add_argument("--time-prior-weight", type=float, default=2.0, help="Penalty weight for peak distance from time_prior.")
    parser.add_argument("--global-conflict-penalty", type=float, default=2.0, help="Penalty for overlapping decoded paths in beam_global mode.")
    parser.add_argument("--viterbi-topk", type=int, default=16, help="Top peak candidates per channel considered by Viterbi.")
    parser.add_argument("--viterbi-candidate-threshold", type=float, default=0.01, help="Low probability floor for candidates entering Viterbi.")
    parser.add_argument("--viterbi-speed-min-kmh", type=float, default=60.0, help="Minimum hard transition speed for Viterbi.")
    parser.add_argument("--viterbi-speed-max-kmh", type=float, default=100.0, help="Maximum hard transition speed for Viterbi.")
    parser.add_argument("--viterbi-max-skip-channels", type=int, default=4, help="Maximum channel gap for one Viterbi transition.")
    parser.add_argument("--viterbi-point-bonus", type=float, default=3.0, help="Per-point reward that lets Viterbi prefer long paths.")
    parser.add_argument("--viterbi-skip-penalty", type=float, default=2.0, help="Penalty per skipped channel in Viterbi.")
    parser.add_argument("--viterbi-speed-penalty", type=float, default=1.0, help="Soft penalty for deviation from slot speed.")
    parser.add_argument("--viterbi-smoothness-penalty", type=float, default=0.6, help="Soft penalty for slope changes.")
    parser.add_argument("--viterbi-inertia-penalty", type=float, default=2.5, help="Penalty for deviating from the previous speed prediction.")
    parser.add_argument("--viterbi-slope-memory", type=float, default=0.75, help="Exponential memory for the slot's running slope.")
    parser.add_argument("--matcher", default="hungarian", choices=["hungarian", "greedy", "auction"], help="Metric matching strategy.")
    parser.add_argument("--none-weight", type=float, default=0.35, help="GT-none weight for loss reporting.")
    parser.add_argument("--no-object-weight", type=float, default=0.15, help="Unmatched slot weight for loss reporting.")
    parser.add_argument("--metric-point-threshold", type=float, default=0.05, help="Normalized time-error threshold for TP.")
    parser.add_argument("--no-ground-truth-csv", action="store_true", help="Do not write ground_truth_tracks.csv.")
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


def _targets_from_payload(payload: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    targets = {
        "peak_time": payload["peak_time"][idx].to(torch.float32),
        "peak_amp": payload["peak_amp"][idx].to(torch.float32),
        "peak_valid": payload["peak_valid"][idx].to(torch.bool),
        "peak_index": payload["peak_index"][idx].to(torch.long),
        "gt_peak_index": payload["gt_peak_index"][idx].to(torch.long),
        "visibility": payload["visibility"][idx].to(torch.float32),
        "direction": payload["direction"][idx].to(torch.long),
        "speed": payload["speed"][idx].to(torch.float32),
        "gt_valid": payload["gt_valid"][idx].to(torch.bool),
    }
    targets["gt_count"] = targets["gt_valid"].sum(dim=1).to(torch.long)
    return targets


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
            yield sample_indices, payload["x"][idx].to(torch.float32), _targets_from_payload(payload, idx)
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


def _active_predictions(
    outputs_cpu: dict[str, torch.Tensor],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    objectness_threshold: float,
    peak_threshold: float,
    min_visible_channels: int,
    max_predicted_tracks: int,
    speed_norm_kmh: float,
    fs: float,
    dx_m: float,
    time_downsample: int,
    inference_config: InferenceConfig,
) -> list[dict[str, Any]]:
    obj = torch.sigmoid(outputs_cpu["objectness_logits"][batch_index])
    peak_prob = torch.softmax(outputs_cpu["peak_logits"][batch_index], dim=-1)
    direction = torch.argmax(outputs_cpu["direction_logits"][batch_index], dim=-1)
    speed = outputs_cpu["speed"][batch_index]
    time_prior = outputs_cpu.get("time_prior")
    order = torch.argsort(obj, descending=True)[: min(int(max_predicted_tracks), int(obj.shape[0]))]
    predictions: list[dict[str, Any]] = []
    for rank, slot_tensor in enumerate(order.tolist()):
        slot = int(slot_tensor)
        score = float(obj[slot].item())
        if score < float(objectness_threshold):
            continue
        direction_label = int(direction[slot].item())
        decoder_mode = str(inference_config.decoder_mode).lower()
        path_options: list[tuple[list[dict[str, float | int]], float]] = []
        if bool(inference_config.use_viterbi_decoder) and decoder_mode == "beam_global":
            x_axis_m = np.arange(int(peak_prob.shape[1]), dtype=np.float64) * float(dx_m)
            decoded_options = decode_peak_slot_paths(
                peak_prob[slot].numpy(),
                targets_cpu["peak_time"][batch_index],
                targets_cpu["peak_valid"][batch_index],
                targets_cpu["peak_index"][batch_index],
                direction_label=direction_label,
                predicted_speed_kmh=float(speed[slot].item()) * float(speed_norm_kmh),
                x_axis_m=x_axis_m,
                time_downsample=int(time_downsample),
                fs=float(fs),
                config=inference_config,
                time_prior=None if not torch.is_tensor(time_prior) else time_prior[batch_index, slot].numpy(),
            )
            path_options = [(list(item["path"]), float(item["score"])) for item in decoded_options]
        elif bool(inference_config.use_viterbi_decoder) and decoder_mode != "argmax":
            x_axis_m = np.arange(int(peak_prob.shape[1]), dtype=np.float64) * float(dx_m)
            decoded = decode_peak_slot_path(
                peak_prob[slot].numpy(),
                targets_cpu["peak_time"][batch_index],
                targets_cpu["peak_valid"][batch_index],
                targets_cpu["peak_index"][batch_index],
                direction_label=direction_label,
                predicted_speed_kmh=float(speed[slot].item()) * float(speed_norm_kmh),
                x_axis_m=x_axis_m,
                time_downsample=int(time_downsample),
                fs=float(fs),
                config=inference_config,
                time_prior=None if not torch.is_tensor(time_prior) else time_prior[batch_index, slot].numpy(),
            )
            path_options = [(decoded, 0.0)]
        else:
            decoded = argmax_peak_slot_path(
                peak_prob[slot].numpy(),
                targets_cpu["peak_valid"][batch_index],
                peak_threshold=float(peak_threshold),
            )
            path_options = [(decoded, 0.0)]
        for option_idx, (decoded, path_score) in enumerate(path_options):
            channels = [int(item["ch"]) for item in decoded]
            peak_indices = [int(item["peak_idx"]) for item in decoded]
            probs = [float(item["prob"]) for item in decoded]
            if len(channels) < int(min_visible_channels):
                continue
            norm_path_score = float(path_score) / float(max(1, len(channels)))
            predictions.append(
                {
                    "rank": int(rank),
                    "slot": slot,
                    "beam_index": int(option_idx),
                    "score": score,
                    "selection_score": float(score + 0.02 * norm_path_score),
                    "path_score": float(path_score),
                    "direction_label": direction_label,
                    "direction": LABEL_TO_DIRECTION.get(direction_label, str(direction_label)),
                    "speed": float(speed[slot].item()),
                    "speed_kmh": float(speed[slot].item() * float(speed_norm_kmh)),
                    "channels": channels,
                    "peak_indices": peak_indices,
                    "peak_probs": probs,
                }
            )
    if str(inference_config.decoder_mode).lower() == "beam_global" and float(inference_config.global_conflict_penalty) > 0.0:
        kept: list[dict[str, Any]] = []
        used_slots: set[int] = set()
        for pred in sorted(predictions, key=lambda item: float(item.get("selection_score", item["score"])), reverse=True):
            if int(pred["slot"]) in used_slots:
                continue
            peak_set = {(int(ch), int(pk)) for ch, pk in zip(pred["channels"], pred["peak_indices"])}
            conflict = False
            for existing in kept:
                existing_set = {(int(ch), int(pk)) for ch, pk in zip(existing["channels"], existing["peak_indices"])}
                common = len(peak_set & existing_set)
                min_len = max(1, min(len(peak_set), len(existing_set)))
                if common >= 3 or common / min_len >= 0.35:
                    conflict = True
                    break
            if not conflict:
                kept.append(pred)
                used_slots.add(int(pred["slot"]))
        predictions = kept
    return predictions


def _write_prediction_rows(writer: csv.DictWriter, *, sample_index: int, predictions: list[dict[str, Any]], targets_cpu: dict[str, torch.Tensor], batch_index: int) -> None:
    for pred_id, pred in enumerate(predictions):
        for ch, peak_idx, prob in zip(pred["channels"], pred["peak_indices"], pred["peak_probs"]):
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "pred_track_id": int(pred_id),
                    "slot": int(pred["slot"]),
                    "rank": int(pred["rank"]),
                    "score": f"{float(pred['score']):.6f}",
                    "direction": str(pred["direction"]),
                    "speed_kmh": f"{float(pred['speed_kmh']):.6f}",
                    "channel": int(ch),
                    "peak_index": int(peak_idx),
                    "time_norm": f"{float(targets_cpu['peak_time'][batch_index, ch, peak_idx].item()):.8f}",
                    "peak_amp": f"{float(targets_cpu['peak_amp'][batch_index, ch, peak_idx].item()):.6f}",
                    "peak_prob": f"{float(prob):.6f}",
                }
            )


def _write_ground_truth_rows(writer: csv.DictWriter, *, sample_index: int, targets_cpu: dict[str, torch.Tensor], batch_index: int) -> None:
    none_idx = int(targets_cpu["peak_time"].shape[-1])
    valid_indices = torch.where(targets_cpu["gt_valid"][batch_index])[0].tolist()
    for gt_id, gt_idx in enumerate(valid_indices):
        direction_label = int(targets_cpu["direction"][batch_index, gt_idx].item())
        for ch in torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist():
            peak_idx = int(targets_cpu["gt_peak_index"][batch_index, gt_idx, ch].item())
            if peak_idx >= none_idx:
                continue
            writer.writerow(
                {
                    "sample_index": int(sample_index),
                    "gt_track_id": int(gt_id),
                    "direction": LABEL_TO_DIRECTION.get(direction_label, str(direction_label)),
                    "channel": int(ch),
                    "peak_index": int(peak_idx),
                    "time_norm": f"{float(targets_cpu['peak_time'][batch_index, ch, peak_idx].item()):.8f}",
                }
            )


def _decoded_physics_counts(
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    *,
    window_seconds: float,
    dx_m: float,
    speed_min_kmh: float,
    speed_max_kmh: float,
    smooth_tolerance_s: float = 2.0,
) -> dict[str, int]:
    pair_count = 0
    direction_bad = 0
    speed_bad = 0
    triple_count = 0
    smooth_bad = 0
    for pred in predictions:
        chs = [int(ch) for ch in pred["channels"]]
        times = [
            float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds)
            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"])
        ]
        sign = 1.0 if str(pred.get("direction", "forward")) == "forward" else -1.0
        for (ch0, t0), (ch1, t1) in zip(zip(chs[:-1], times[:-1]), zip(chs[1:], times[1:])):
            pair_count += 1
            dt = float(t1 - t0)
            if sign * dt < 0.0:
                direction_bad += 1
            dx = abs(float(ch1 - ch0)) * float(dx_m)
            speed = 3.6 * dx / max(1e-9, abs(dt))
            if speed < float(speed_min_kmh) or speed > float(speed_max_kmh):
                speed_bad += 1
        for t0, t1, t2 in zip(times[:-2], times[1:-1], times[2:]):
            triple_count += 1
            if abs(float(t2 - 2.0 * t1 + t0)) > float(smooth_tolerance_s):
                smooth_bad += 1
    return {
        "pair_count": int(pair_count),
        "direction_bad": int(direction_bad),
        "speed_bad": int(speed_bad),
        "triple_count": int(triple_count),
        "smooth_bad": int(smooth_bad),
    }


def _plot_sample_overlay(
    out_path: Path,
    *,
    heatmap: torch.Tensor,
    sample_index: int,
    predictions: list[dict[str, Any]],
    targets_cpu: dict[str, torch.Tensor],
    batch_index: int,
    window_seconds: float,
    dx_m: float,
    plot_style: str,
    dpi: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
    arr = heatmap.detach().cpu().to(torch.float32).numpy()
    if arr.ndim != 2:
        raise ValueError("heatmap must have shape [channel, time]")
    n_ch, n_t = int(arr.shape[0]), int(arr.shape[1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    finite = arr[np.isfinite(arr)]
    vmax = max(float(np.quantile(np.abs(finite), 0.995)), 1e-6) if finite.size else 1.0
    fig, ax = plt.subplots(figsize=(12.0, 7.0))
    plot_style = str(plot_style).lower()
    if plot_style == "waveform":
        if float(dx_m) > 0.0:
            x_axis = np.arange(n_ch, dtype=np.float64) * float(dx_m) * 1e-3
            x_label = "Offset [km]"
        else:
            x_axis = np.arange(n_ch, dtype=np.float64)
            x_label = "Channel index"
        t_axis = np.linspace(0.0, float(window_seconds), n_t, dtype=np.float64)
        spacing = float(np.median(np.diff(x_axis))) if x_axis.size >= 2 else 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        wiggle_amp = 0.27 * spacing
        clip_ratio = 1.35
        for ch in range(n_ch):
            ratio = np.clip(arr[ch].astype(np.float64) / max(vmax, 1e-12), -clip_ratio, clip_ratio)
            ax.plot(x_axis[ch] + ratio * wiggle_amp, t_axis, color="0.45", linewidth=0.8, alpha=0.9)
        pad = 0.4 * spacing
        x_min_plot = float(x_axis[0] - pad)
        x_max_plot = float(x_axis[-1] + pad)
        x_span_full = max(1e-6, x_max_plot - x_min_plot)
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(0.0, float(window_seconds))
        ax.invert_yaxis()
        ax.set_xlabel(x_label)
        ax.set_ylabel("Time (s)")
        im = None
    else:
        im = ax.imshow(
            arr,
            origin="lower",
            aspect="auto",
            cmap="gray_r",
            vmin=-vmax,
            vmax=vmax,
            extent=(0.0, float(window_seconds), -0.5, float(n_ch) - 0.5),
            interpolation="nearest",
        )
        x_axis = np.arange(n_ch, dtype=np.float64)
        x_max_plot = float(window_seconds)
        x_span_full = max(1e-6, float(window_seconds))
    gt_label_added = False
    for gt_idx in torch.where(targets_cpu["gt_valid"][batch_index])[0].tolist():
        chs = []
        times = []
        for ch in torch.where(targets_cpu["visibility"][batch_index, gt_idx] > 0.5)[0].tolist():
            peak_idx = int(targets_cpu["gt_peak_index"][batch_index, gt_idx, ch].item())
            if peak_idx >= int(targets_cpu["peak_time"].shape[-1]):
                continue
            chs.append(int(ch))
            times.append(float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds))
        if len(chs) >= 2:
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) for ch in chs]
                ys = times
            else:
                xs = times
                ys = chs
            ax.plot(xs, ys, color="black", linewidth=0.6, alpha=0.9, label="GT" if not gt_label_added else None)
            gt_label_added = True
    cmap = plt.get_cmap("tab20", max(1, len(predictions)))
    pred_label_added = False
    for pred_id, pred in enumerate(predictions):
        chs = list(pred["channels"])
        times = [
            float(targets_cpu["peak_time"][batch_index, ch, peak_idx].item()) * float(window_seconds)
            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"])
        ]
        if len(chs) >= 2:
            color = cmap(pred_id % max(1, cmap.N))
            if plot_style == "waveform":
                xs = [float(x_axis[int(ch)]) for ch in chs]
                ys = times
            else:
                xs = times
                ys = chs
            ax.plot(xs, ys, color=color, linewidth=1.6, alpha=0.9, label="Prediction" if not pred_label_added else None)
            ax.scatter(xs, ys, s=9, color=[color], edgecolors="black", linewidths=0.2, alpha=0.95)
            speed_kmh = float(pred.get("speed_kmh", float("nan")))
            if math.isfinite(speed_kmh):
                mid = len(xs) // 2
                x_text = min(x_max_plot - 0.02 * x_span_full, float(xs[mid]) + 0.01 * x_span_full)
                ax.text(
                    x_text,
                    float(ys[mid]),
                    f"{speed_kmh:.1f} km/h",
                    color=color,
                    fontsize=8,
                    ha="left",
                    va="center",
                    alpha=0.95,
                    bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none", "pad": 0.8},
                )
            pred_label_added = True
    gt_count = int(targets_cpu["gt_valid"][batch_index].sum().item())
    ax.set_title(f"PeakSlotNet prediction overlay, sample {sample_index}  GT={gt_count}  Pred={len(predictions)}")
    if plot_style != "waveform":
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel index")
        ax.set_xlim(0.0, float(window_seconds))
        ax.set_ylim(-0.5, float(n_ch) - 0.5)
    if gt_label_added or pred_label_added:
        ax.legend(loc="upper right", frameon=True)
    if im is not None:
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cbar.set_label("Normalized heatmap amplitude")
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {data_dir / 'meta.json'}")
    device = _resolve_device(str(args.device))
    model, checkpoint = load_checkpoint_model(args.model, device=device)
    if str(checkpoint.get("model_family", "peak_slot")) != "peak_slot":
        raise ValueError(f"Checkpoint is not peak_slot: model_family={checkpoint.get('model_family')}")
    speed_norm_kmh = float(meta.get("speed_norm_kmh", 150.0))
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    time_downsample = int(meta.get("time_downsample", 10))
    time_prior_weight = float(args.time_prior_weight) if bool(checkpoint.get("has_time_prior", True)) else 0.0
    inference_config = InferenceConfig(
        time_downsample=time_downsample,
        objectness_threshold=float(args.objectness_threshold),
        peak_threshold=float(args.peak_threshold),
        min_visible_channels=int(args.min_visible_channels),
        max_tracks=int(args.max_predicted_tracks),
        speed_norm_kmh=float(speed_norm_kmh),
        use_viterbi_decoder=not bool(args.no_viterbi_decoder),
        decoder_mode="argmax" if bool(args.no_viterbi_decoder) else str(args.decoder_mode),
        viterbi_beam_size=int(args.viterbi_beam_size),
        time_prior_weight=float(time_prior_weight),
        global_conflict_penalty=float(args.global_conflict_penalty),
        viterbi_topk=int(args.viterbi_topk),
        viterbi_candidate_threshold=float(args.viterbi_candidate_threshold),
        viterbi_speed_min_kmh=float(args.viterbi_speed_min_kmh),
        viterbi_speed_max_kmh=float(args.viterbi_speed_max_kmh),
        viterbi_max_skip_channels=int(args.viterbi_max_skip_channels),
        viterbi_point_bonus=float(args.viterbi_point_bonus),
        viterbi_skip_penalty=float(args.viterbi_skip_penalty),
        viterbi_speed_penalty=float(args.viterbi_speed_penalty),
        viterbi_smoothness_penalty=float(args.viterbi_smoothness_penalty),
        viterbi_inertia_penalty=float(args.viterbi_inertia_penalty),
        viterbi_slope_memory=float(args.viterbi_slope_memory),
    )
    summary_path = out_dir / "summary.json"
    sample_summary_path = out_dir / "sample_summary.csv"
    pred_csv_path = out_dir / "predicted_tracks.csv"
    gt_csv_path = out_dir / "ground_truth_tracks.csv"
    plots_dir = out_dir / "plots"
    metric_sums: defaultdict[str, float] = defaultdict(float)
    metric_weights: defaultdict[str, float] = defaultdict(float)
    loss_sums: defaultdict[str, float] = defaultdict(float)
    loss_weights: defaultdict[str, float] = defaultdict(float)
    filtered_pred_total = 0
    filtered_gt_total = 0
    filtered_count_abs_error = 0.0
    filtered_count_exact = 0
    decoded_pair_total = 0
    decoded_direction_bad = 0
    decoded_speed_bad = 0
    decoded_triple_total = 0
    decoded_smooth_bad = 0
    decoded_peak_use_total = 0
    decoded_peak_conflict_total = 0
    sample_count = 0
    csv_sample_count = 0
    plot_sample_count = 0
    batch_count = 0
    t0 = time.perf_counter()
    pred_fields = ["sample_index", "pred_track_id", "slot", "rank", "score", "direction", "speed_kmh", "channel", "peak_index", "time_norm", "peak_amp", "peak_prob"]
    gt_fields = ["sample_index", "gt_track_id", "direction", "channel", "peak_index", "time_norm"]
    sample_fields = ["sample_index", "gt_count", "pred_count", "max_objectness", "mean_objectness", "top_score"]
    with sample_summary_path.open("w", newline="", encoding="utf-8") as sample_fp, pred_csv_path.open("w", newline="", encoding="utf-8") as pred_fp:
        sample_writer = csv.DictWriter(sample_fp, fieldnames=sample_fields)
        pred_writer = csv.DictWriter(pred_fp, fieldnames=pred_fields)
        sample_writer.writeheader()
        pred_writer.writeheader()
        gt_fp = None
        gt_writer: Optional[csv.DictWriter] = None
        if not bool(args.no_ground_truth_csv):
            gt_fp = gt_csv_path.open("w", newline="", encoding="utf-8")
            gt_writer = csv.DictWriter(gt_fp, fieldnames=gt_fields)
            gt_writer.writeheader()
        try:
            model.eval()
            with torch.no_grad():
                for sample_indices, x_cpu, targets_cpu in _iter_batches(data_dir, shards, batch_size=int(args.batch_size), max_samples=int(args.max_samples)):
                    batch_n = int(x_cpu.shape[0])
                    non_blocking = str(device).startswith("cuda")
                    x = x_cpu.to(device=device, non_blocking=non_blocking)
                    targets = move_targets_to_device(targets_cpu, device, non_blocking=non_blocking)
                    outputs = model(x, targets["peak_time"], targets["peak_amp"], targets["peak_valid"])
                    _, loss_metrics = peak_slot_set_loss(
                        outputs,
                        targets,
                        no_object_weight=float(args.no_object_weight),
                        none_weight=float(args.none_weight),
                        matcher=str(args.matcher),
                        collect_metrics=True,
                    )
                    det_metrics = peak_slot_detection_metrics(
                        outputs,
                        targets,
                        objectness_threshold=float(args.objectness_threshold),
                        point_threshold=float(args.metric_point_threshold),
                        matcher=str(args.matcher),
                    )
                    det_metrics.update(
                        peak_slot_physics_metrics(
                            outputs,
                            targets,
                            objectness_threshold=float(args.objectness_threshold),
                            peak_threshold=float(args.peak_threshold),
                            speed_min_kmh=float(args.viterbi_speed_min_kmh),
                            speed_max_kmh=float(args.viterbi_speed_max_kmh),
                            time_downsample=time_downsample,
                            fs=fs,
                            dx_m=dx_m,
                        )
                    )
                    _weighted_add(loss_sums, loss_weights, loss_metrics, batch_n)
                    _weighted_add(metric_sums, metric_weights, det_metrics, batch_n)
                    outputs_cpu = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}
                    obj_cpu = torch.sigmoid(outputs_cpu["objectness_logits"])
                    for b, sample_index in enumerate(sample_indices):
                        predictions = _active_predictions(
                            outputs_cpu,
                            targets_cpu,
                            b,
                            objectness_threshold=float(args.objectness_threshold),
                            peak_threshold=float(args.peak_threshold),
                            min_visible_channels=int(args.min_visible_channels),
                            max_predicted_tracks=int(args.max_predicted_tracks),
                            speed_norm_kmh=float(speed_norm_kmh),
                            fs=fs,
                            dx_m=dx_m,
                            time_downsample=time_downsample,
                            inference_config=inference_config,
                        )
                        gt_count = int(targets_cpu["gt_valid"][b].sum().item())
                        pred_count = int(len(predictions))
                        decoded_counts = _decoded_physics_counts(
                            predictions,
                            targets_cpu,
                            b,
                            window_seconds=float(meta.get("window_seconds", 1.0)),
                            dx_m=dx_m,
                            speed_min_kmh=float(args.viterbi_speed_min_kmh),
                            speed_max_kmh=float(args.viterbi_speed_max_kmh),
                        )
                        decoded_pair_total += int(decoded_counts["pair_count"])
                        decoded_direction_bad += int(decoded_counts["direction_bad"])
                        decoded_speed_bad += int(decoded_counts["speed_bad"])
                        decoded_triple_total += int(decoded_counts["triple_count"])
                        decoded_smooth_bad += int(decoded_counts["smooth_bad"])
                        peak_use_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
                        for pred in predictions:
                            for ch, peak_idx in zip(pred["channels"], pred["peak_indices"]):
                                peak_use_counts[(int(ch), int(peak_idx))] += 1
                        decoded_peak_use_total += sum(peak_use_counts.values())
                        decoded_peak_conflict_total += sum(max(0, count - 1) for count in peak_use_counts.values())
                        filtered_gt_total += gt_count
                        filtered_pred_total += pred_count
                        filtered_count_abs_error += abs(pred_count - gt_count)
                        filtered_count_exact += int(pred_count == gt_count)
                        top_score = float(predictions[0]["score"]) if predictions else float("nan")
                        sample_writer.writerow(
                            {
                                "sample_index": int(sample_index),
                                "gt_count": gt_count,
                                "pred_count": pred_count,
                                "max_objectness": f"{float(obj_cpu[b].max().item()):.6f}",
                                "mean_objectness": f"{float(obj_cpu[b].mean().item()):.6f}",
                                "top_score": "" if not math.isfinite(top_score) else f"{top_score:.6f}",
                            }
                        )
                        if int(args.max_csv_samples) > 0 and csv_sample_count < int(args.max_csv_samples):
                            _write_prediction_rows(pred_writer, sample_index=int(sample_index), predictions=predictions, targets_cpu=targets_cpu, batch_index=b)
                            if gt_writer is not None:
                                _write_ground_truth_rows(gt_writer, sample_index=int(sample_index), targets_cpu=targets_cpu, batch_index=b)
                            csv_sample_count += 1
                        if int(args.plot_samples) > 0 and plot_sample_count < int(args.plot_samples):
                            _plot_sample_overlay(
                                plots_dir / f"sample_{int(sample_index):06d}.png",
                                heatmap=x_cpu[b, 0],
                                sample_index=int(sample_index),
                                predictions=predictions,
                                targets_cpu=targets_cpu,
                                batch_index=b,
                                window_seconds=float(meta.get("window_seconds", 1.0)),
                                dx_m=float(meta.get("dx_m", 0.0)),
                                plot_style=str(args.plot_style),
                                dpi=int(args.plot_dpi),
                            )
                            plot_sample_count += 1
                    sample_count += batch_n
                    batch_count += 1
        finally:
            if gt_fp is not None:
                gt_fp.close()
    filtered_count_mae = float(filtered_count_abs_error / max(1, sample_count))
    filtered_count_acc = float(filtered_count_exact / max(1, sample_count))
    summary = {
        "mode": "peak_slot_tensor_shard_prediction",
        "data_dir": str(data_dir),
        "model": str(Path(args.model).expanduser()),
        "model_family": str(checkpoint.get("model_family", "peak_slot")),
        "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
        "device": device,
        "speed_norm_kmh": float(speed_norm_kmh),
        "sample_count": int(sample_count),
        "batch_count": int(batch_count),
        "elapsed_seconds": float(time.perf_counter() - t0),
        "thresholds": {
            "objectness_threshold": float(args.objectness_threshold),
            "peak_threshold": float(args.peak_threshold),
            "min_visible_channels": int(args.min_visible_channels),
            "metric_point_threshold": float(args.metric_point_threshold),
        },
        "decoder": {
            "use_viterbi_decoder": bool(inference_config.use_viterbi_decoder),
            "decoder_mode": str(inference_config.decoder_mode),
            "viterbi_topk": int(inference_config.viterbi_topk),
            "viterbi_beam_size": int(inference_config.viterbi_beam_size),
            "time_prior_weight": float(inference_config.time_prior_weight),
            "global_conflict_penalty": float(inference_config.global_conflict_penalty),
            "viterbi_candidate_threshold": float(inference_config.viterbi_candidate_threshold),
            "viterbi_speed_min_kmh": float(inference_config.viterbi_speed_min_kmh),
            "viterbi_speed_max_kmh": float(inference_config.viterbi_speed_max_kmh),
            "viterbi_max_skip_channels": int(inference_config.viterbi_max_skip_channels),
            "viterbi_inertia_penalty": float(inference_config.viterbi_inertia_penalty),
            "viterbi_slope_memory": float(inference_config.viterbi_slope_memory),
        },
        "loss_metrics": _weighted_mean(loss_sums, loss_weights),
        "batch_mean_detection_metrics": _weighted_mean(metric_sums, metric_weights),
        "filtered_count_metrics": {
            "pred_count": int(filtered_pred_total),
            "gt_count": int(filtered_gt_total),
            "count_mae": filtered_count_mae,
            "count_acc": filtered_count_acc,
        },
        "filtered_physics_metrics": {
            "pair_count": int(decoded_pair_total),
            "direction_violation_rate": float(decoded_direction_bad / max(1, decoded_pair_total)),
            "speed_window_violation_rate": float(decoded_speed_bad / max(1, decoded_pair_total)),
            "smoothness_violation_rate": float(decoded_smooth_bad / max(1, decoded_triple_total)),
            "peak_conflict_rate": float(decoded_peak_conflict_total / max(1, decoded_peak_use_total)),
        },
        "outputs": {
            "summary_json": str(summary_path),
            "sample_summary_csv": str(sample_summary_path),
            "predicted_tracks_csv": str(pred_csv_path),
            "ground_truth_tracks_csv": None if bool(args.no_ground_truth_csv) else str(gt_csv_path),
            "plots_dir": str(plots_dir) if int(args.plot_samples) > 0 else None,
        },
    }
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: samples={sample_count}, filtered_pred={filtered_pred_total}, gt={filtered_gt_total}, "
        f"count_mae={filtered_count_mae:.2f}, out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
