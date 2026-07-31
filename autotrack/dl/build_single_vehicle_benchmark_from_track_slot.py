"""Build a single-vehicle benchmark from TrackSlotNet shards.

This converts a multi-vehicle shard dataset into a one-trajectory benchmark by
selecting one valid slot per sample. It is useful when you already have
realism-profile-driven multi-vehicle data and want to finetune the single-
vehicle model on the same background / motion distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-vehicle benchmark from TrackSlotNet shards.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output .pt benchmark file.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Directory containing TrackSlotNet shard_*.pt files.")
    parser.add_argument("--slot-policy", default="round_robin", choices=["round_robin", "first", "most_visible"], help="How to choose one valid slot per sample.")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum benchmark samples; 0 means all samples across all shards.")
    parser.add_argument("--raw-window-dtype", default="float32", choices=["float32", "float16", "bfloat16"], help="Storage dtype for raw_window.")
    return parser.parse_args(argv)


def _iter_shard_paths(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.expanduser().glob("shard_*.pt") if p.is_file())


def _choose_slot(valid_slots: list[int], *, slot_policy: str, sample_index: int, visibility: torch.Tensor) -> int:
    if not valid_slots:
        raise ValueError("valid_slots cannot be empty")
    if slot_policy == "first":
        return int(valid_slots[0])
    if slot_policy == "most_visible":
        counts = []
        for slot in valid_slots:
            counts.append((int(visibility[int(slot)].sum().item()), int(slot)))
        counts.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return int(counts[0][1])
    pos = int(sample_index) % int(len(valid_slots))
    return int(valid_slots[pos])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    shard_paths = _iter_shard_paths(args.data_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {args.data_dir}")

    samples: list[dict[str, Any]] = []
    selected_slot_hist: dict[int, int] = {}
    window_seconds = None
    fs = None
    dx_m = None
    time_downsample = None
    speed_norm_kmh = None

    for shard_path in shard_paths:
        payload = torch.load(str(shard_path), map_location="cpu", weights_only=False)
        x = payload["x"]
        time = payload["time"]
        visibility = payload["visibility"]
        direction = payload["direction"]
        speed = payload["speed"]
        gt_valid = payload["gt_valid"]
        if window_seconds is None:
            meta_path = shard_path.with_name("meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            window_seconds = float(meta.get("window_seconds", 120.0))
            # The benchmark uses the shard resolution as its own pseudo-raw
            # timeline, so fs is inferred from the stored tensor length.
            fs = float(x.shape[-1]) / float(window_seconds)
            dx_m = float(meta.get("dx_m", 100.0))
            time_downsample = 1
            speed_norm_kmh = float(meta.get("speed_norm_kmh", 150.0))
        for i in range(int(x.shape[0])):
            valid_slots = [int(slot) for slot in torch.where(gt_valid[i])[0].tolist()]
            if not valid_slots:
                continue
            slot = _choose_slot(valid_slots, slot_policy=str(args.slot_policy), sample_index=len(samples), visibility=visibility[i])
            selected_slot_hist[slot] = int(selected_slot_hist.get(slot, 0)) + 1
            raw_window = x[i, 0].to(torch.float32).contiguous()
            sample_time = time[i, slot].to(torch.float32).contiguous()
            sample_visibility = visibility[i, slot].to(torch.float32).contiguous()
            x_item = x[i].to(torch.float32).contiguous()
            target = {
                "time": sample_time.unsqueeze(0),
                "visibility": sample_visibility.unsqueeze(0),
                "direction": direction[i, slot : slot + 1].to(torch.long).contiguous(),
                "speed": speed[i, slot : slot + 1].to(torch.float32).contiguous(),
                "raw_window": raw_window.to(torch.float32).contiguous() if str(args.raw_window_dtype) == "float32" else raw_window.to(
                    torch.float16 if str(args.raw_window_dtype) == "float16" else torch.bfloat16
                ).contiguous(),
                "sample_weight": torch.tensor([1.0], dtype=torch.float32),
            }
            samples.append({"x": x_item, "target": target})
            if int(args.max_samples) > 0 and len(samples) >= int(args.max_samples):
                break
        if int(args.max_samples) > 0 and len(samples) >= int(args.max_samples):
            break

    if not samples:
        raise RuntimeError(f"No valid slots found in shards under {args.data_dir}")

    payload = {
        "format": "single_vehicle_benchmark_v1",
        "source_format": "track_slot_shards_v1",
        "meta": _json_ready(
            {
                "data_dir": args.data_dir,
                "slot_policy": str(args.slot_policy),
                "fs": float(fs if fs is not None else 100.0),
                "dx_m": float(dx_m if dx_m is not None else 100.0),
                "window_seconds": float(window_seconds if window_seconds is not None else 120.0),
                "time_downsample": int(time_downsample if time_downsample is not None else 1),
                "speed_norm_kmh": float(speed_norm_kmh if speed_norm_kmh is not None else 150.0),
                "selected_slot_hist": dict(sorted(selected_slot_hist.items())),
            }
        ),
        "length": int(len(samples)),
        "samples": samples,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out_file))
    args.out_file.with_suffix(".json").write_text(json.dumps(_json_ready(payload["meta"]), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
