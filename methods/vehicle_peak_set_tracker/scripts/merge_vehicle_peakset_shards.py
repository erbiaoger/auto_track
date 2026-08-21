from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge exported vehicle peak-set shard datasets.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Merged dataset directory.")
    parser.add_argument(
        "--component",
        action="append",
        required=True,
        help="Component spec as name:path. Order is preserved in the merged shard list.",
    )
    parser.add_argument("--shuffle-shards", action="store_true", help="Shuffle merged shards so train/val splits see all profiles.")
    parser.add_argument("--shuffle-seed", type=int, default=20260703, help="Shard shuffle seed.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing merged directory.")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _parse_component(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(f"component must be name:path, got {spec!r}")
    name, raw_path = spec.split(":", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"component name is empty in {spec!r}")
    path = Path(raw_path).expanduser()
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"missing component meta: {path / 'meta.json'}")
    return name, path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    if out_dir.exists() and any(out_dir.iterdir()):
        if not bool(args.overwrite):
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_records: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    first_meta: dict[str, Any] | None = None
    total_samples = 0

    for spec in args.component:
        name, component_dir = _parse_component(str(spec))
        meta = _read_json(component_dir / "meta.json")
        if first_meta is None:
            first_meta = meta
        component_shards = [str(item) for item in meta.get("shards", [])]
        component_sizes = [int(item) for item in meta.get("shard_sizes", [])]
        if len(component_sizes) != len(component_shards):
            raise ValueError(f"component {name} has inconsistent shard metadata")

        start_sample = total_samples
        for src_name, shard_n in zip(component_shards, component_sizes):
            shard_records.append(
                {
                    "profile": name,
                    "src": component_dir / src_name,
                    "size": int(shard_n),
                }
            )
            total_samples += int(shard_n)

        components.append(
            {
                "name": name,
                "source_dir": str(component_dir),
                "num_samples": int(sum(component_sizes)),
                "sample_range": [int(start_sample), int(total_samples)],
                "dataset_config": meta.get("dataset_config", {}),
                "dataset_stats": _read_json(component_dir / "dataset_stats.json")
                if (component_dir / "dataset_stats.json").is_file()
                else {},
            }
        )

    if first_meta is None:
        raise ValueError("no components provided")
    if bool(args.shuffle_shards):
        rng = random.Random(int(args.shuffle_seed))
        rng.shuffle(shard_records)

    shards: list[str] = []
    shard_sizes: list[int] = []
    profile_by_shard: dict[str, str] = {}
    profile_sample_ranges: list[dict[str, Any]] = []
    sample_cursor = 0
    for shard_idx, record in enumerate(shard_records):
        dst_name = f"shard_{shard_idx:06d}.pt"
        shutil.copy2(Path(record["src"]), out_dir / dst_name)
        shards.append(dst_name)
        shard_sizes.append(int(record["size"]))
        profile_by_shard[dst_name] = str(record["profile"])
        profile_sample_ranges.append(
            {
                "shard": dst_name,
                "profile": str(record["profile"]),
                "sample_range": [int(sample_cursor), int(sample_cursor + int(record["size"]))],
            }
        )
        sample_cursor += int(record["size"])

    meta = {
        "format": "vehicle_peakset_shards_realshape_curriculum_v1",
        "num_samples": int(total_samples),
        "shard_size": int(first_meta.get("shard_size", 0)),
        "shards": shards,
        "shard_sizes": shard_sizes,
        "dataset_config": {
            "curriculum": "easy_completion",
            "components": [
                {
                    "name": item["name"],
                    "num_samples": item["num_samples"],
                    "sample_range": item["sample_range"],
                }
                for item in components
            ],
        },
        "components": components,
        "profile_by_shard": profile_by_shard,
        "profile_sample_ranges": profile_sample_ranges,
        "shuffle_shards": bool(args.shuffle_shards),
        "shuffle_seed": int(args.shuffle_seed),
    }
    stats = {
        "num_samples": int(total_samples),
        "curriculum": "easy_completion",
        "components": [
            {
                "name": item["name"],
                "num_samples": item["num_samples"],
                "sample_range": item["sample_range"],
                "dataset_stats": item["dataset_stats"],
            }
            for item in components
        ],
    }
    _write_json(out_dir / "meta.json", meta)
    _write_json(out_dir / "dataset_stats.json", stats)
    print(json.dumps({"out_dir": str(out_dir), "num_samples": total_samples, "components": meta["dataset_config"]["components"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
