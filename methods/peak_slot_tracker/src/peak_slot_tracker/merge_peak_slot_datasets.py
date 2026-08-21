"""Merge PeakSlot shard directories by copying shards with new names."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge compatible peak_slot_shards_v1 datasets.")
    parser.add_argument("--in-dir", required=True, action="append", type=Path, help="Input dataset directory. Repeat for multiple sources.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Merged output dataset directory.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_dirs = [Path(path).expanduser() for path in args.in_dir]
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not bool(args.overwrite):
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metas = [_load_meta(path) for path in in_dirs]
    _validate_compatible(metas, in_dirs)
    out_shards: list[str] = []
    total_samples = 0
    source_entries: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for source_idx, (in_dir, meta) in enumerate(zip(in_dirs, metas)):
        source_samples = 0
        for shard_name in [str(item) for item in meta.get("shards", [])]:
            payload = torch.load(str(in_dir / shard_name), map_location="cpu", weights_only=False)
            sample_count = int(payload["x"].shape[0])
            out_name = f"shard_{len(out_shards):06d}.pt"
            shutil.copy2(str(in_dir / shard_name), str(out_dir / out_name))
            out_shards.append(out_name)
            total_samples += sample_count
            source_samples += sample_count
        source_entries.append(
            {
                "source_index": int(source_idx),
                "source_dir": str(in_dir),
                "source_shards": int(len(meta.get("shards", []))),
                "samples": int(source_samples),
            }
        )
        print(f"merged source {source_idx}: samples={source_samples}, total={total_samples}", flush=True)

    out_meta = dict(metas[0])
    out_meta.update(
        {
            "format": "peak_slot_shards_v1",
            "mode": "merged_peak_slot_shards_v1",
            "source_dirs": [str(path) for path in in_dirs],
            "merge_sources": source_entries,
            "created_at_unix": time.time(),
            "num_samples": int(total_samples),
            "shards": out_shards,
            "peak_conversion_stats": _merge_conversion_stats(metas, total_samples),
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={total_samples}, shards={len(out_shards)}, out_dir={out_dir}", flush=True)
    return 0


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _validate_compatible(metas: list[dict[str, Any]], in_dirs: list[Path]) -> None:
    if not metas:
        raise ValueError("At least one --in-dir is required")
    keys = ("format", "n_channels", "in_channels", "window_samples", "time_downsample", "peak_candidates_per_channel")
    base = metas[0]
    if str(base.get("format", "")) != "peak_slot_shards_v1":
        raise ValueError(f"Input is not peak_slot_shards_v1: {in_dirs[0]}")
    for idx, meta in enumerate(metas[1:], start=1):
        if str(meta.get("format", "")) != "peak_slot_shards_v1":
            raise ValueError(f"Input is not peak_slot_shards_v1: {in_dirs[idx]}")
        for key in keys[1:]:
            if meta.get(key) != base.get(key):
                raise ValueError(f"Incompatible {key}: {in_dirs[0]}={base.get(key)!r}, {in_dirs[idx]}={meta.get(key)!r}")


def _merge_conversion_stats(metas: list[dict[str, Any]], total_samples: int) -> dict[str, int]:
    keys = (
        "matched_gt_points",
        "injected_gt_points",
        "raw_candidate_count",
        "prior_candidate_count",
        "merged_candidate_count",
        "skipped_existing_shards",
    )
    merged = {"samples": int(total_samples)}
    for key in keys:
        merged[key] = int(sum(int(dict(meta.get("peak_conversion_stats", {})).get(key, 0) or 0) for meta in metas))
    return merged


if __name__ == "__main__":
    raise SystemExit(main())
