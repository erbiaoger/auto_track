"""Calibrate inference thresholds for the compact slot model.

This sweeps objectness/visibility/extra-candidate parameters on a shard
dataset and reports the best settings by track count MAE and F1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from autotrack.dl.compact_slot_model import load_checkpoint_model
from autotrack.dl.predict_compact_slot_dataset import (
    _gt_tracks_from_targets,
    _iter_batches,
    _pred_tracks_from_outputs,
    _resolve_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate compact slot inference thresholds.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Shard dataset directory.")
    parser.add_argument("--model", required=True, type=Path, help="Checkpoint path.")
    parser.add_argument("--out-json", required=True, type=Path, help="Calibration report path.")
    parser.add_argument("--device", default="auto", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-samples", type=int, default=64, help="Maximum samples to scan.")
    parser.add_argument("--objectness-thresholds", default="0.15,0.25,0.35,0.45,0.55", help="Comma-separated objectness thresholds.")
    parser.add_argument("--visibility-thresholds", default="0.25,0.35,0.45,0.55", help="Comma-separated visibility thresholds.")
    parser.add_argument("--candidate-objectness-floors", default="0.05,0.10,0.15", help="Comma-separated candidate floors.")
    parser.add_argument("--objectness-count-scales", default="0.45,0.55,0.65,0.75", help="Comma-separated soft count scales.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible channels for a predicted track.")
    return parser.parse_args()


def _parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_csv_ints(text: str) -> list[int]:
    return [int(float(item.strip())) for item in str(text).split(",") if item.strip()]


def main() -> int:
    args = parse_args()
    device = _resolve_device(str(args.device))
    model, _ = load_checkpoint_model(args.model, device=device)
    data_dir = Path(args.data_dir).expanduser()
    out_path = Path(args.out_json).expanduser()

    obj_thresholds = _parse_csv_floats(args.objectness_thresholds)
    vis_thresholds = _parse_csv_floats(args.visibility_thresholds)
    floors = _parse_csv_floats(args.candidate_objectness_floors)
    count_scales = _parse_csv_floats(args.objectness_count_scales)

    grid: list[dict[str, Any]] = []
    best = None
    best_key = None
    for obj_t in obj_thresholds:
        for vis_t in vis_thresholds:
            for scale in count_scales:
                for floor in floors:
                    sample_rows: list[dict[str, Any]] = []
                    gt_total = pred_total = tp_total = 0
                    count_abs_error = 0.0
                    with torch.no_grad():
                        for sample_indices, x, targets, raw_x in _iter_batches(
                            data_dir,
                            [str(item) for item in (json.loads((data_dir / "meta.json").read_text(encoding="utf-8")).get("shards", []))],
                            batch_size=int(args.batch_size),
                            max_samples=int(args.max_samples),
                        ):
                            x_dev = x.to(device)
                            outputs = model(x_dev)
                            for local_i, sample_idx in enumerate(sample_indices):
                                pred_tracks = _pred_tracks_from_outputs(
                                    outputs,
                                    x_dev,
                                    local_i,
                                    objectness_threshold=float(obj_t),
                                    candidate_objectness_floor=float(floor),
                                    objectness_count_scale=float(scale),
                                    visibility_threshold=float(vis_t),
                                    min_visible_channels=int(args.min_visible_channels),
                                )
                                gt_tracks = _gt_tracks_from_targets(targets, local_i)
                                pred_total += len(pred_tracks)
                                gt_total += len(gt_tracks)
                                count_abs_error += abs(len(pred_tracks) - len(gt_tracks))
                                sample_rows.append({"sample": int(sample_idx), "pred": len(pred_tracks), "gt": len(gt_tracks)})
                    count_mae = float(count_abs_error / max(1, len(sample_rows)))
                    summary = {
                        "objectness_threshold": float(obj_t),
                        "visibility_threshold": float(vis_t),
                        "candidate_objectness_floor": float(floor),
                        "objectness_count_scale": float(scale),
                        "sample_count": int(len(sample_rows)),
                        "pred_count": int(pred_total),
                        "gt_count": int(gt_total),
                        "count_mae": float(count_mae),
                    }
                    grid.append(summary)
                    key = (count_mae, abs(pred_total - gt_total), -scale, obj_t, vis_t, floor)
                    if best is None or key < best_key:
                        best = summary
                        best_key = key

    report = {"best": best, "grid": grid}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(best, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
