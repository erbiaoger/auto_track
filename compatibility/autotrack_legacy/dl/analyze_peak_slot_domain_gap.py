"""Analyze PeakSlotNet domain gap between two peak-slot datasets.

Purpose:
    Compare a reference peak-slot dataset and a target peak-slot dataset from
    three views:
    1. Input tensor statistics (`x` distribution, saturation, positive ratio).
    2. Peak-candidate statistics (`peak_valid` density and per-sample totals).
    3. Optional model-output statistics from a PeakSlotNet checkpoint
       (objectness distribution, soft count, and active slot counts).

    This tool is intended for diagnosing why a model that performs well on
    synthetic data may over-predict or under-perform on real data.

How to run:
    Use the project environment:

    uv run python -m autotrack.dl.analyze_peak_slot_domain_gap \
        --reference-dir datasets/peak_slot_v3_120s_realistic/test \
        --target-dir datasets/peak_slot/xi_gauss_50_120s_stride60_saved_arrays04 \
        --model models/peak_slot_v4_120s_noisy_badch_cuda/checkpoint_best.pt \
        --out-dir /tmp/peak_slot_domain_gap \
        --max-samples 64 \
        --device cpu

Example output:
    <out-dir>/summary.json
    <out-dir>/report.md

Outputs:
    - `summary.json`: machine-readable statistics and heuristic flags.
    - `report.md`: short human-readable diagnosis with recommendations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

from autotrack.dl.peak_slot_model import load_checkpoint_model
from autotrack.dl.trajectory_set_model import auto_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze domain gap between two PeakSlotNet datasets.")
    parser.add_argument("--reference-dir", required=True, type=Path, help="Reference peak_slot dataset directory.")
    parser.add_argument("--target-dir", required=True, type=Path, help="Target peak_slot dataset directory to compare.")
    parser.add_argument("--model", type=Path, default=None, help="Optional PeakSlotNet checkpoint used to compare model outputs.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory that will receive summary.json and report.md.")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum samples to inspect per dataset; 0 uses all samples.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for optional model-forward statistics.")
    parser.add_argument("--device", default="auto", help="Torch device for optional model forward: auto, cpu, cuda, mps.")
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    raw = str(device_arg).strip()
    if raw in {"", "auto", "None"}:
        return auto_torch_device()
    return raw


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _iter_shards(data_dir: Path, max_samples: int) -> Iterator[dict[str, Any]]:
    meta = _load_meta(data_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {data_dir / 'meta.json'}")
    emitted = 0
    for shard_name in shards:
        payload = torch.load(str(data_dir / shard_name), map_location="cpu", weights_only=False)
        count = int(payload["x"].shape[0])
        if int(max_samples) > 0:
            remaining = int(max_samples) - emitted
            if remaining <= 0:
                break
            count = min(count, remaining)
        if count <= 0:
            break
        yield {
            "shard_name": shard_name,
            "x": payload["x"][:count].to(torch.float32),
            "peak_valid": payload["peak_valid"][:count].to(torch.bool),
            "peak_time": payload["peak_time"][:count].to(torch.float32),
            "peak_amp": payload["peak_amp"][:count].to(torch.float32),
            "gt_valid": payload.get("gt_valid", torch.empty((count, 0), dtype=torch.bool))[:count].to(torch.bool),
        }
        emitted += count


def _safe_stats(values: list[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "q10": float("nan"), "q50": float("nan"), "q90": float("nan"), "max": float("nan")}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _dataset_stats(data_dir: Path, max_samples: int) -> dict[str, Any]:
    meta = _load_meta(data_dir)
    x_min: list[float] = []
    x_max: list[float] = []
    x_mean: list[float] = []
    x_std: list[float] = []
    x_pos_ratio: list[float] = []
    x_abs_gt_098_ratio: list[float] = []
    peaks_per_channel: list[float] = []
    total_peaks_per_sample: list[float] = []
    gt_count_per_sample: list[float] = []
    sample_count = 0

    for payload in _iter_shards(data_dir, max_samples=max_samples):
        x = payload["x"]
        peak_valid = payload["peak_valid"]
        gt_valid = payload["gt_valid"]
        sample_count += int(x.shape[0])

        x_min.extend([float(v) for v in x.amin(dim=(1, 2, 3)).tolist()])
        x_max.extend([float(v) for v in x.amax(dim=(1, 2, 3)).tolist()])
        x_mean.extend([float(v) for v in x.mean(dim=(1, 2, 3)).tolist()])
        x_std.extend([float(v) for v in x.std(dim=(1, 2, 3), unbiased=False).tolist()])
        x_pos_ratio.extend([float(v) for v in (x > 0).to(torch.float32).mean(dim=(1, 2, 3)).tolist()])
        x_abs_gt_098_ratio.extend([float(v) for v in (x.abs() > 0.98).to(torch.float32).mean(dim=(1, 2, 3)).tolist()])

        peak_per_ch = peak_valid.sum(dim=2).to(torch.float32)
        peaks_per_channel.extend([float(v) for v in peak_per_ch.reshape(-1).tolist()])
        total_peaks_per_sample.extend([float(v) for v in peak_valid.sum(dim=(1, 2)).to(torch.float32).tolist()])
        if gt_valid.numel() > 0:
            gt_count_per_sample.extend([float(v) for v in gt_valid.sum(dim=1).to(torch.float32).tolist()])

    gt_stats = _safe_stats(gt_count_per_sample) if gt_count_per_sample else None
    gt_zero_fraction = None
    if gt_count_per_sample:
        gt_zero_fraction = float(sum(1 for value in gt_count_per_sample if float(value) <= 0.0) / len(gt_count_per_sample))

    return {
        "data_dir": str(data_dir),
        "sample_count": int(sample_count),
        "meta": {
            "format": meta.get("format"),
            "has_ground_truth": bool(meta.get("has_ground_truth", False)),
            "n_channels": int(meta.get("n_channels", 0)),
            "in_channels": int(meta.get("in_channels", 0)),
            "window_seconds": float(meta.get("window_seconds", 0.0)),
            "time_downsample": int(meta.get("time_downsample", 0)),
            "dx_m": float(meta.get("dx_m", 0.0)),
            "clip_ratio": float(meta.get("clip_ratio", 0.0)),
            "input_mode": str(meta.get("input_mode", "")),
            "peak_detection": meta.get("peak_detection", {}),
        },
        "x_stats": {
            "min": _safe_stats(x_min),
            "max": _safe_stats(x_max),
            "mean": _safe_stats(x_mean),
            "std": _safe_stats(x_std),
            "positive_ratio": _safe_stats(x_pos_ratio),
            "abs_gt_0p98_ratio": _safe_stats(x_abs_gt_098_ratio),
        },
        "peak_stats": {
            "valid_peaks_per_channel": _safe_stats(peaks_per_channel),
            "valid_peaks_per_sample": _safe_stats(total_peaks_per_sample),
        },
        "gt_count_stats": gt_stats,
        "gt_zero_fraction": gt_zero_fraction,
    }


def _iter_model_batches(data_dir: Path, max_samples: int, batch_size: int) -> Iterator[dict[str, torch.Tensor]]:
    for payload in _iter_shards(data_dir, max_samples=max_samples):
        x = payload["x"]
        peak_time = payload["peak_time"]
        peak_amp = payload["peak_amp"]
        peak_valid = payload["peak_valid"]
        for start in range(0, int(x.shape[0]), int(batch_size)):
            end = min(int(x.shape[0]), start + int(batch_size))
            yield {
                "x": x[start:end],
                "peak_time": peak_time[start:end],
                "peak_amp": peak_amp[start:end],
                "peak_valid": peak_valid[start:end],
            }


@torch.no_grad()
def _model_stats(
    data_dir: Path,
    *,
    model_path: Path,
    max_samples: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint_model(model_path, device=device)
    model.eval()

    objectness_mean_per_sample: list[float] = []
    objectness_max_per_sample: list[float] = []
    soft_count_per_sample: list[float] = []
    active_035_per_sample: list[float] = []
    active_045_per_sample: list[float] = []
    active_060_per_sample: list[float] = []

    for batch in _iter_model_batches(data_dir, max_samples=max_samples, batch_size=batch_size):
        x = batch["x"].to(device)
        peak_time = batch["peak_time"].to(device)
        peak_amp = batch["peak_amp"].to(device)
        peak_valid = batch["peak_valid"].to(device)
        outputs = model(x, peak_time, peak_amp, peak_valid)
        obj = torch.sigmoid(outputs["objectness_logits"][:, : int(outputs["num_regular_queries"])])
        objectness_mean_per_sample.extend([float(v) for v in obj.mean(dim=1).tolist()])
        objectness_max_per_sample.extend([float(v) for v in obj.max(dim=1).values.tolist()])
        soft_count_per_sample.extend([float(v) for v in obj.sum(dim=1).tolist()])
        active_035_per_sample.extend([float(v) for v in (obj >= 0.35).sum(dim=1).tolist()])
        active_045_per_sample.extend([float(v) for v in (obj >= 0.45).sum(dim=1).tolist()])
        active_060_per_sample.extend([float(v) for v in (obj >= 0.60).sum(dim=1).tolist()])

    return {
        "model": str(model_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "objectness_mean_per_sample": _safe_stats(objectness_mean_per_sample),
        "objectness_max_per_sample": _safe_stats(objectness_max_per_sample),
        "soft_count_per_sample": _safe_stats(soft_count_per_sample),
        "active_slots_threshold_0p35": _safe_stats(active_035_per_sample),
        "active_slots_threshold_0p45": _safe_stats(active_045_per_sample),
        "active_slots_threshold_0p60": _safe_stats(active_060_per_sample),
    }


def _compare(reference: dict[str, Any], target: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    ref_gt = reference.get("gt_count_stats")
    if ref_gt and float(ref_gt.get("min", 0.0)) >= 1.0 and float(reference.get("gt_zero_fraction", 0.0) or 0.0) <= 0.01:
        findings.append(
            {
                "severity": "high",
                "title": "Reference training/test distribution lacks empty or low-count windows",
                "detail": (
                    f"Reference GT count min={ref_gt['min']:.1f}, q10={ref_gt['q10']:.1f}, "
                    f"mean={ref_gt['mean']:.1f}, zero_fraction={float(reference.get('gt_zero_fraction', 0.0) or 0.0):.3f}. "
                    "This teaches the model that each 120 s window should contain many vehicles."
                ),
            }
        )

    ref_model = reference.get("model_stats")
    tgt_model = target.get("model_stats")
    if ref_model and tgt_model:
        ref_soft = float(ref_model["soft_count_per_sample"]["mean"])
        tgt_soft = float(tgt_model["soft_count_per_sample"]["mean"])
        ref_active = float(ref_model["active_slots_threshold_0p45"]["mean"])
        tgt_active = float(tgt_model["active_slots_threshold_0p45"]["mean"])
        if math.isfinite(ref_soft) and math.isfinite(tgt_soft) and tgt_soft > ref_soft * 1.25:
            findings.append(
                {
                    "severity": "high",
                    "title": "Model objectness prior is much higher on target data",
                    "detail": (
                        f"Soft count mean rises from {ref_soft:.2f} on reference to {tgt_soft:.2f} on target. "
                        f"Active slots @0.45 rise from {ref_active:.2f} to {tgt_active:.2f}. "
                        "This indicates the model is treating target background structure as vehicle evidence."
                    ),
                }
            )

    ref_peaks = float(reference["peak_stats"]["valid_peaks_per_sample"]["mean"])
    tgt_peaks = float(target["peak_stats"]["valid_peaks_per_sample"]["mean"])
    if math.isfinite(ref_peaks) and math.isfinite(tgt_peaks):
        ratio = tgt_peaks / max(1e-6, ref_peaks)
        if ratio > 1.10:
            findings.append(
                {
                    "severity": "medium",
                    "title": "Target data has denser peak candidates",
                    "detail": (
                        f"Valid peak candidates per sample mean rises from {ref_peaks:.1f} to {tgt_peaks:.1f}. "
                        "More candidate clutter increases the chance that Viterbi can assemble false tracks."
                    ),
                }
            )
        elif ratio < 0.90:
            findings.append(
                {
                    "severity": "medium",
                    "title": "Target data has sparser peak candidates",
                    "detail": (
                        f"Valid peak candidates per sample mean drops from {ref_peaks:.1f} to {tgt_peaks:.1f}. "
                        "If real vehicles are present, missing candidates can hurt recall before the network stage."
                    ),
                }
            )

    ref_x_mean = float(reference["x_stats"]["mean"]["mean"])
    tgt_x_mean = float(target["x_stats"]["mean"]["mean"])
    ref_x_std = float(reference["x_stats"]["std"]["mean"])
    tgt_x_std = float(target["x_stats"]["std"]["mean"])
    if math.isfinite(ref_x_mean) and math.isfinite(tgt_x_mean) and abs(tgt_x_mean - ref_x_mean) > 0.03:
        findings.append(
            {
                "severity": "medium",
                "title": "Input normalization statistics shifted",
                "detail": (
                    f"Mean input value changes from {ref_x_mean:.4f} to {tgt_x_mean:.4f}; "
                    f"mean sample std changes from {ref_x_std:.4f} to {tgt_x_std:.4f}. "
                    "The network is not seeing the same normalized background texture."
                ),
            }
        )

    return findings


def _recommendations(summary: dict[str, Any]) -> list[str]:
    recs: list[str] = []
    findings = summary.get("findings", [])
    titles = {item.get("title", "") for item in findings}

    if "Reference training/test distribution lacks empty or low-count windows" in titles:
        recs.append(
            "Add explicit low-traffic and empty-window samples to training. For 120 s windows, include a broad GT-count distribution such as 0-36 instead of always 24-36."
        )
    recs.append(
        "Build a mixed training set with real background windows plus injected synthetic trajectories, so the model learns to reject real structured background while still receiving labels."
    )
    recs.append(
        "Create a small real labeled validation set and tune inference on that set. Do not tune real-data thresholds only from qualitative plots."
    )
    recs.append(
        "Evaluate candidate-peak recall separately from network accuracy. If real vehicles are not entering the candidate table, improve peak detection before retraining the network."
    )
    if any("objectness prior" in item.get("title", "") for item in findings):
        recs.append(
            "Reduce false positives by retraining with stronger background negatives, then recalibrate objectness thresholds on labeled real data instead of only raising inference thresholds."
        )
    recs.append(
        "Add structured hard negatives that form long diagonal patterns without corresponding vehicles. White noise alone is not enough."
    )
    return recs


def _markdown_report(summary: dict[str, Any]) -> str:
    ref = summary["reference"]
    tgt = summary["target"]
    lines = [
        "# PeakSlotNet Domain Gap Report",
        "",
        f"- Reference: `{ref['data_dir']}`",
        f"- Target: `{tgt['data_dir']}`",
        f"- Model: `{summary.get('model') or 'not provided'}`",
        "",
        "## Key Statistics",
        "",
        "| Metric | Reference | Target |",
        "| --- | ---: | ---: |",
        f"| Samples inspected | {ref['sample_count']} | {tgt['sample_count']} |",
        f"| Mean valid peaks / sample | {ref['peak_stats']['valid_peaks_per_sample']['mean']:.2f} | {tgt['peak_stats']['valid_peaks_per_sample']['mean']:.2f} |",
        f"| Mean input value | {ref['x_stats']['mean']['mean']:.4f} | {tgt['x_stats']['mean']['mean']:.4f} |",
        f"| Mean input std | {ref['x_stats']['std']['mean']:.4f} | {tgt['x_stats']['std']['mean']:.4f} |",
    ]
    if ref.get("gt_count_stats") is not None:
        lines.append(f"| GT count mean | {ref['gt_count_stats']['mean']:.2f} | n/a |")
        lines.append(f"| GT count min / q10 / max | {ref['gt_count_stats']['min']:.0f} / {ref['gt_count_stats']['q10']:.0f} / {ref['gt_count_stats']['max']:.0f} | n/a |")
    if ref.get("model_stats") and tgt.get("model_stats"):
        lines.extend(
            [
                f"| Model soft count mean | {ref['model_stats']['soft_count_per_sample']['mean']:.2f} | {tgt['model_stats']['soft_count_per_sample']['mean']:.2f} |",
                f"| Active slots @0.45 mean | {ref['model_stats']['active_slots_threshold_0p45']['mean']:.2f} | {tgt['model_stats']['active_slots_threshold_0p45']['mean']:.2f} |",
                f"| Mean objectness | {ref['model_stats']['objectness_mean_per_sample']['mean']:.4f} | {tgt['model_stats']['objectness_mean_per_sample']['mean']:.4f} |",
            ]
        )

    lines.extend(["", "## Findings", ""])
    if summary["findings"]:
        for item in summary["findings"]:
            lines.append(f"- [{item['severity']}] {item['title']}: {item['detail']}")
    else:
        lines.append("- No strong heuristic finding was triggered.")

    lines.extend(["", "## Recommendations", ""])
    for item in summary["recommendations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = _dataset_stats(Path(args.reference_dir).expanduser(), max_samples=int(args.max_samples))
    target = _dataset_stats(Path(args.target_dir).expanduser(), max_samples=int(args.max_samples))

    model_path: Optional[Path] = None
    if args.model is not None:
        model_path = Path(args.model).expanduser()
        device = _resolve_device(str(args.device))
        reference["model_stats"] = _model_stats(
            Path(args.reference_dir).expanduser(),
            model_path=model_path,
            max_samples=int(args.max_samples),
            batch_size=int(args.batch_size),
            device=device,
        )
        target["model_stats"] = _model_stats(
            Path(args.target_dir).expanduser(),
            model_path=model_path,
            max_samples=int(args.max_samples),
            batch_size=int(args.batch_size),
            device=device,
        )

    summary = {
        "reference": reference,
        "target": target,
        "model": str(model_path) if model_path is not None else None,
    }
    summary["findings"] = _compare(reference, target)
    summary["recommendations"] = _recommendations(summary)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(_markdown_report(summary), encoding="utf-8")
    print(f"wrote report: {out_dir / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
