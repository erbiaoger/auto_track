"""Build a single-vehicle benchmark from a multi-vehicle benchmark file.

Each output sample keeps the original multi-vehicle raw window as background,
but supervision is reduced to one selected GT trajectory.
This is useful for training the single-vehicle network to focus on one vehicle
inside cluttered windows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-vehicle benchmark from a multi-vehicle benchmark.")
    parser.add_argument("--benchmark-file", required=True, type=Path, help="Input multi_vehicle_benchmark_v1 .pt file.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output single_vehicle_benchmark_v1 .pt file.")
    parser.add_argument(
        "--track-policy",
        default="round_robin",
        choices=["round_robin", "first", "most_visible", "random"],
        help="How to choose one GT track per sample.",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum output samples; 0 means all tracks.")
    parser.add_argument("--raw-window-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --track-policy=random.")
    return parser.parse_args(argv)


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


def _choose_track(valid_tracks: list[int], *, policy: str, sample_index: int, visibility: torch.Tensor, rng: np.random.Generator) -> int:
    if not valid_tracks:
        raise ValueError("valid_tracks cannot be empty")
    if policy == "first":
        return int(valid_tracks[0])
    if policy == "most_visible":
        scored = [(int(visibility[int(track)].sum().item()), int(track)) for track in valid_tracks]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        return int(scored[0][1])
    if policy == "random":
        return int(rng.choice(valid_tracks))
    return int(valid_tracks[int(sample_index) % len(valid_tracks)])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rng = np.random.default_rng(int(args.seed))
    payload = torch.load(str(Path(args.benchmark_file).expanduser()), map_location="cpu", weights_only=False)
    samples_in = list(payload.get("samples", []))
    if not samples_in:
        raise ValueError(f"benchmark file has no samples: {args.benchmark_file}")

    out_samples: list[dict[str, Any]] = []
    track_hist: dict[int, int] = {}
    for sample in samples_in:
        raw_window = sample["target"].get("raw_window")
        if raw_window is None:
            raw_window = sample["x"][0]
        target = sample["target"]
        gt_valid = target.get("gt_valid", torch.ones((int(target["time"].shape[0]),), dtype=torch.bool))
        valid_tracks = [int(idx) for idx in torch.where(gt_valid)[0].tolist()]
        for repeat_idx, track_idx in enumerate(valid_tracks):
            chosen = track_idx
            if str(args.track_policy) != "first":
                chosen = _choose_track(valid_tracks, policy=str(args.track_policy), sample_index=len(out_samples), visibility=target["visibility"], rng=rng)
            track_hist[chosen] = int(track_hist.get(chosen, 0)) + 1
            out_target = {
                "time": target["time"][chosen : chosen + 1].to(torch.float32).contiguous(),
                "visibility": target["visibility"][chosen : chosen + 1].to(torch.float32).contiguous(),
                "direction": target["direction"][chosen : chosen + 1].to(torch.long).contiguous(),
                "speed": target["speed"][chosen : chosen + 1].to(torch.float32).contiguous(),
                "raw_window": raw_window.to(torch.float32).contiguous()
                if str(args.raw_window_dtype) == "float32"
                else raw_window.to(torch.float16 if str(args.raw_window_dtype) == "float16" else torch.bfloat16).contiguous(),
            }
            out_samples.append({"x": sample["x"].to(torch.float32).cpu(), "target": out_target})
            if int(args.max_samples) > 0 and len(out_samples) >= int(args.max_samples):
                break
        if int(args.max_samples) > 0 and len(out_samples) >= int(args.max_samples):
            break

    meta_in = dict(payload.get("meta", {}))
    out_payload = {
        "format": "single_vehicle_benchmark_v1",
        "source_format": "multi_vehicle_benchmark_v1",
        "meta": _json_ready(
            {
                "source_benchmark": str(Path(args.benchmark_file).expanduser()),
                "track_policy": str(args.track_policy),
                "seed": int(args.seed),
                "fs": float(meta_in.get("fs", 1000.0)),
                "dx_m": float(meta_in.get("dx_m", 100.0)),
                "window_seconds": float(meta_in.get("window_seconds", 120.0)),
                "time_downsample": int(meta_in.get("time_downsample", 10)),
                "speed_norm_kmh": float(meta_in.get("speed_norm_kmh", 150.0)),
                "selected_track_hist": dict(sorted(track_hist.items())),
            }
        ),
        "length": int(len(out_samples)),
        "samples": out_samples,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, str(args.out_file))
    args.out_file.with_suffix(".json").write_text(
        json.dumps(_json_ready(out_payload["meta"]), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
