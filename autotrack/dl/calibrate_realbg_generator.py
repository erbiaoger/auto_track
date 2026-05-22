"""Compare generated real-background datasets against a realism profile.

Purpose:
    Score one or more generated `track_slot` / `peak_slot` datasets against a
    reference `realism_profile.json`, then write a calibration report that
    highlights the largest remaining domain mismatches and recommends the best
    available generator setting.

    This tool supports two practical workflows:

    1. Evaluate existing datasets:
       uv run python -m autotrack.dl.calibrate_realbg_generator \
           --profile /tmp/real_profile/realism_profile.json \
           --data-dir datasets/track_slot_realbg_120s_heavy/train \
           --data-dir datasets/track_slot_realbg_120s_profile/train \
           --out-dir /tmp/realbg_calibration

    2. Evaluate a small candidate list described in JSON:
       uv run python -m autotrack.dl.calibrate_realbg_generator \
           --profile /tmp/real_profile/realism_profile.json \
           --candidate-json docs/realbg_candidates.json \
           --out-dir /tmp/realbg_calibration

JSON candidate format:
    [
      {"name": "heavy", "data_dir": "datasets/track_slot_realbg_120s_heavy/train"},
      {"name": "profile", "data_dir": "datasets/track_slot_realbg_120s_profile/train"}
    ]

Outputs:
    - `calibration_summary.json`
    - `calibration_report.md`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate generated real-background datasets against a realism profile.")
    parser.add_argument("--profile", required=True, type=Path, help="realism_profile.json produced by profile_real_npy_background.py.")
    parser.add_argument("--data-dir", action="append", default=[], help="Generated track_slot or peak_slot dataset directory. Repeatable.")
    parser.add_argument("--candidate-json", type=Path, default=None, help="Optional JSON candidate list with {name, data_dir, params}.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for calibration_summary.json and calibration_report.md.")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum samples inspected per dataset.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates used when scoring track_slot datasets.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum peak spacing for track_slot scoring.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Minimum normalized peak height for track_slot scoring.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum normalized peak prominence for track_slot scoring.")
    return parser.parse_args()


def _safe_stats(values: Iterable[float]) -> dict[str, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q10": float("nan"),
            "q50": float("nan"),
            "q90": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q10": float(np.quantile(arr, 0.10)),
        "q50": float(np.quantile(arr, 0.50)),
        "q90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _iter_shards(data_dir: Path, max_samples: int) -> Iterable[dict[str, Any]]:
    meta = _load_meta(data_dir)
    emitted = 0
    for shard_name in meta.get("shards", []):
        payload = torch.load(str(data_dir / str(shard_name)), map_location="cpu", weights_only=False)
        count = int(payload["x"].shape[0])
        if int(max_samples) > 0:
            remaining = int(max_samples) - emitted
            if remaining <= 0:
                break
            count = min(count, remaining)
        if count <= 0:
            break
        yield {"meta": meta, "payload": {key: value[:count] for key, value in payload.items() if hasattr(value, "__getitem__")}}
        emitted += count


def _profile_zero_components_from_tensor(x_ct: np.ndarray) -> tuple[int, list[float], list[float]]:
    mask = np.abs(x_ct) <= 1e-8
    dead_channels = int(np.sum(mask.mean(axis=1) >= 0.95))
    widths: list[float] = []
    durations: list[float] = []
    for ch in range(mask.shape[0]):
        row = mask[ch]
        start = None
        for idx, flag in enumerate(row.tolist() + [False]):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                length = idx - start
                if 0 < length < row.size:
                    widths.append(1.0)
                    durations.append(float(length))
                start = None
    return dead_channels, widths, durations


def _summarize_track_slot_dataset(data_dir: Path, max_samples: int, peak_cfg: PeakDetectionConfig) -> dict[str, Any]:
    meta = _load_meta(data_dir)
    x_mean: list[float] = []
    x_std: list[float] = []
    x_pos_ratio: list[float] = []
    x_abs_q95: list[float] = []
    peaks_per_sample: list[float] = []
    peaks_per_channel: list[float] = []
    peak_amp_values: list[float] = []
    dead_channel_counts: list[float] = []
    drop_widths: list[float] = []
    drop_durations_s: list[float] = []

    for shard in _iter_shards(data_dir, max_samples=max_samples):
        payload = shard["payload"]
        x = payload["x"].to(torch.float32)
        for sample_idx in range(int(x.shape[0])):
            heatmap = x[sample_idx, 0]
            x_mean.append(float(torch.mean(heatmap).item()))
            x_std.append(float(torch.std(heatmap, unbiased=False).item()))
            x_pos_ratio.append(float(torch.mean((heatmap > 0).to(torch.float32)).item()))
            x_abs_q95.append(float(torch.quantile(torch.abs(heatmap).reshape(-1), 0.95).item()))
            peak_time, peak_amp, peak_valid, _ = detect_peak_candidates_from_tensor(
                heatmap,
                fs=float(meta["fs"]),
                time_downsample=int(meta["time_downsample"]),
                window_samples=int(meta["window_samples"]),
                config=peak_cfg,
            )
            counts = peak_valid.sum(dim=1).to(torch.float32)
            peaks_per_sample.append(float(torch.sum(counts).item()))
            peaks_per_channel.extend(float(v) for v in counts.tolist())
            peak_amp_values.extend(float(v) for v in peak_amp[peak_valid].tolist())
            dead_count, widths, durations = _profile_zero_components_from_tensor(heatmap.numpy())
            dead_channel_counts.append(float(dead_count))
            drop_widths.extend(widths)
            drop_durations_s.extend(
                float(duration) * float(meta["time_downsample"]) / float(meta["fs"]) for duration in durations
            )

    return {
        "format": str(meta.get("format", "")),
        "data_dir": str(data_dir),
        "meta": meta,
        "normalized_window_stats": {
            "mean": _safe_stats(x_mean),
            "std": _safe_stats(x_std),
            "positive_ratio": _safe_stats(x_pos_ratio),
            "abs_q95": _safe_stats(x_abs_q95),
        },
        "peak_density": {
            "total_peaks_per_window": _safe_stats(peaks_per_sample),
            "peaks_per_channel_per_window": _safe_stats(peaks_per_channel),
            "peak_value": _safe_stats(peak_amp_values),
        },
        "zero_components": {
            "dead_channel_count_per_window": _safe_stats(dead_channel_counts),
            "drop_block_channel_width": _safe_stats(drop_widths),
            "drop_block_duration_s": _safe_stats(drop_durations_s),
        },
    }


def _summarize_peak_slot_dataset(data_dir: Path, max_samples: int) -> dict[str, Any]:
    meta = _load_meta(data_dir)
    x_mean: list[float] = []
    x_std: list[float] = []
    x_pos_ratio: list[float] = []
    x_abs_q95: list[float] = []
    peaks_per_sample: list[float] = []
    peaks_per_channel: list[float] = []
    peak_amp_values: list[float] = []
    dead_channel_counts: list[float] = []
    drop_widths: list[float] = []
    drop_durations_s: list[float] = []

    for shard in _iter_shards(data_dir, max_samples=max_samples):
        payload = shard["payload"]
        x = payload["x"].to(torch.float32)
        peak_valid = payload["peak_valid"].to(torch.bool)
        peak_amp = payload["peak_amp"].to(torch.float32)
        for sample_idx in range(int(x.shape[0])):
            heatmap = x[sample_idx, 0]
            x_mean.append(float(torch.mean(heatmap).item()))
            x_std.append(float(torch.std(heatmap, unbiased=False).item()))
            x_pos_ratio.append(float(torch.mean((heatmap > 0).to(torch.float32)).item()))
            x_abs_q95.append(float(torch.quantile(torch.abs(heatmap).reshape(-1), 0.95).item()))
            counts = peak_valid[sample_idx].sum(dim=1).to(torch.float32)
            peaks_per_sample.append(float(torch.sum(counts).item()))
            peaks_per_channel.extend(float(v) for v in counts.tolist())
            peak_amp_values.extend(float(v) for v in peak_amp[sample_idx][peak_valid[sample_idx]].tolist())
            dead_count, widths, durations = _profile_zero_components_from_tensor(heatmap.numpy())
            dead_channel_counts.append(float(dead_count))
            drop_widths.extend(widths)
            drop_durations_s.extend(
                float(duration) * float(meta["time_downsample"]) / float(meta["fs"]) for duration in durations
            )

    return {
        "format": str(meta.get("format", "")),
        "data_dir": str(data_dir),
        "meta": meta,
        "normalized_window_stats": {
            "mean": _safe_stats(x_mean),
            "std": _safe_stats(x_std),
            "positive_ratio": _safe_stats(x_pos_ratio),
            "abs_q95": _safe_stats(x_abs_q95),
        },
        "peak_density": {
            "total_peaks_per_window": _safe_stats(peaks_per_sample),
            "peaks_per_channel_per_window": _safe_stats(peaks_per_channel),
            "peak_value": _safe_stats(peak_amp_values),
        },
        "zero_components": {
            "dead_channel_count_per_window": _safe_stats(dead_channel_counts),
            "drop_block_channel_width": _safe_stats(drop_widths),
            "drop_block_duration_s": _safe_stats(drop_durations_s),
        },
    }


def _score_stat(reference: float, observed: float) -> float:
    if not (math.isfinite(reference) and math.isfinite(observed)):
        return 1.0
    if abs(reference) <= 1e-6:
        return abs(observed - reference)
    denom = max(1e-6, abs(reference))
    return abs(observed - reference) / denom


def _parse_dead_index_text(text: str) -> set[int]:
    values: set[int] = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.add(int(item))
        except ValueError:
            continue
    return values


def _dataset_fixed_dead_channels(summary: dict[str, Any]) -> set[int]:
    meta = summary.get("meta", {})
    profile_mode = meta.get("profile_mode", {})
    stable = profile_mode.get("stable_dead_channels")
    if isinstance(stable, list):
        return {int(v) for v in stable}
    generator_args = meta.get("generator_args", {})
    return _parse_dead_index_text(generator_args.get("dead_channel_indices", ""))


def _score_dataset(profile: dict[str, Any], summary: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    reference_fixed_dead = {
        int(v) for v in profile.get("zero_components", {}).get("stable_dead_channel_indices", profile.get("zero_components", {}).get("dead_channel_indices", []))
    }
    observed_fixed_dead = _dataset_fixed_dead_channels(summary)
    if reference_fixed_dead or observed_fixed_dead:
        fixed_dead_score = 1.0 - (
            float(len(reference_fixed_dead & observed_fixed_dead))
            / float(max(1, len(reference_fixed_dead | observed_fixed_dead)))
        )
    else:
        fixed_dead_score = 0.0
    comparisons = [
        ("stable_dead_channel_match", 0.0, fixed_dead_score, 2.5),
        ("peak_total_q50", profile["peak_density"]["total_peaks_per_window"]["q50"], summary["peak_density"]["total_peaks_per_window"]["q50"], 2.8),
        ("peaks_per_channel_q50", profile["peak_density"]["peaks_per_channel_per_window"]["q50"], summary["peak_density"]["peaks_per_channel_per_window"]["q50"], 2.4),
        ("peak_amp_q90", profile["peak_density"]["peak_value"]["q90"], summary["peak_density"]["peak_value"]["q90"], 0.8),
        ("peak_sigma_q50", profile["peak_density"]["peak_sigma_t_s"]["q50"], math.nan, 0.0),
        ("x_positive_ratio_q50", profile["normalized_window_stats"]["positive_ratio"]["q50"], summary["normalized_window_stats"]["positive_ratio"]["q50"], 0.8),
        ("x_abs_q95_q50", profile["normalized_window_stats"]["abs_q95"]["q50"], summary["normalized_window_stats"]["abs_q95"]["q50"], 0.8),
        ("dead_channel_q50", profile["zero_components"]["dead_channel_count_per_window"]["q50"], summary["zero_components"]["dead_channel_count_per_window"]["q50"], 1.8),
        ("drop_width_q50", profile["zero_components"]["drop_block_channel_width"]["q50"], summary["zero_components"]["drop_block_channel_width"]["q50"], 0.4),
        ("drop_duration_q50", profile["zero_components"]["drop_block_duration_s"]["q50"], summary["zero_components"]["drop_block_duration_s"]["q50"], 0.2),
    ]
    findings: list[dict[str, Any]] = []
    weighted_total = 0.0
    weight_sum = 0.0
    for name, reference, observed, weight in comparisons:
        if weight <= 0.0:
            continue
        score = _score_stat(float(reference), float(observed))
        findings.append(
            {
                "metric": name,
                "reference": float(reference) if math.isfinite(float(reference)) else None,
                "observed": float(observed) if math.isfinite(float(observed)) else None,
                "relative_error": float(score),
                "weight": float(weight),
                "weighted_error": float(score * weight),
            }
        )
        weighted_total += score * weight
        weight_sum += weight
    findings.sort(key=lambda item: float(item["weighted_error"]), reverse=True)
    return (weighted_total / max(1e-6, weight_sum), findings)


def _render_report(profile_path: Path, ranked: list[dict[str, Any]]) -> str:
    lines = [
        "# Real-Background Calibration Report",
        "",
        f"- Reference profile: `{profile_path}`",
        f"- Evaluated datasets: `{len(ranked)}`",
        "",
    ]
    if ranked:
        best = ranked[0]
        lines.extend(
            [
                "## Recommended Dataset",
                "",
                f"- Name: `{best['name']}`",
                f"- Data dir: `{best['data_dir']}`",
                f"- Score: `{best['score']:.4f}` (lower is better)",
                "",
            ]
        )
        params = best.get("recommended_params")
        if params:
            lines.append("- Recommended generator params from best dataset meta:")
            for key in sorted(params):
                lines.append(f"  - `{key}` = `{params[key]}`")
            lines.append("")
    lines.append("## Ranking")
    lines.append("")
    for item in ranked:
        lines.append(f"- `{item['name']}`: score=`{item['score']:.4f}` data_dir=`{item['data_dir']}`")
        for finding in item["top_findings"][:5]:
            lines.append(
                f"  - `{finding['metric']}` relative_error=`{finding['relative_error']:.3f}` "
                f"ref=`{finding['reference']}` obs=`{finding['observed']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = _load_json(Path(args.profile).expanduser())
    if str(profile.get("format", "")) != "realism_profile_v1":
        raise ValueError(f"Unsupported realism profile format in {args.profile}")

    candidates: list[dict[str, Any]] = []
    for data_dir in args.data_dir:
        path = Path(data_dir).expanduser()
        candidates.append({"name": path.name, "data_dir": str(path), "params": None})
    if args.candidate_json is not None:
        payload = _load_json(Path(args.candidate_json).expanduser())
        if not isinstance(payload, list):
            raise ValueError("--candidate-json must contain a JSON list")
        for item in payload:
            if not isinstance(item, dict) or "data_dir" not in item:
                raise ValueError("Each candidate entry must contain at least {data_dir}")
            candidates.append(
                {
                    "name": str(item.get("name", Path(str(item["data_dir"])).name)),
                    "data_dir": str(Path(str(item["data_dir"])).expanduser()),
                    "params": item.get("params"),
                }
            )
    if not candidates:
        raise ValueError("At least one --data-dir or --candidate-json entry is required")

    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(args.peak_candidates_per_channel),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=0.25,
    )
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        data_dir = Path(candidate["data_dir"]).expanduser()
        meta = _load_meta(data_dir)
        fmt = str(meta.get("format", ""))
        if fmt == "track_slot_shards_v1":
            summary = _summarize_track_slot_dataset(data_dir, max_samples=int(args.max_samples), peak_cfg=peak_cfg)
        elif fmt == "peak_slot_shards_v1":
            summary = _summarize_peak_slot_dataset(data_dir, max_samples=int(args.max_samples))
        else:
            raise ValueError(f"Unsupported dataset format in {data_dir}: {fmt}")
        score, findings = _score_dataset(profile, summary)
        recommended_params = candidate.get("params")
        if recommended_params is None:
            recommended_params = meta.get("generator_args")
        ranked.append(
            {
                "name": str(candidate["name"]),
                "data_dir": str(data_dir),
                "score": float(score),
                "summary": summary,
                "top_findings": findings,
                "recommended_params": recommended_params,
            }
        )
    ranked.sort(key=lambda item: float(item["score"]))

    summary = {
        "format": "realbg_calibration_summary_v1",
        "profile": str(Path(args.profile).expanduser()),
        "best_dataset": ranked[0]["data_dir"] if ranked else None,
        "best_score": ranked[0]["score"] if ranked else None,
        "datasets": ranked,
    }
    (out_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "calibration_report.md").write_text(_render_report(Path(args.profile).expanduser(), ranked), encoding="utf-8")
    print(f"wrote summary: {out_dir / 'calibration_summary.json'}", flush=True)
    print(f"wrote report: {out_dir / 'calibration_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
