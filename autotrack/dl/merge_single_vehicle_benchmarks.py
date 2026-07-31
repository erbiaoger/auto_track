"""Merge compatible single-vehicle benchmark files into one fixed benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge single-vehicle benchmark .pt files.")
    parser.add_argument("--out-file", required=True, type=Path, help="Output merged benchmark .pt file.")
    parser.add_argument("--in-file", required=True, type=Path, nargs="+", help="Input benchmark .pt files.")
    parser.add_argument(
        "--input-weight",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-input sample weights; defaults to 1.0 for all inputs.",
    )
    return parser.parse_args(argv)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_weights = list(args.input_weight) if args.input_weight is not None else [1.0] * len(args.in_file)
    if len(input_weights) != len(args.in_file):
        raise ValueError("--input-weight must have the same number of values as --in-file")
    merged_samples: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for weight, path in zip(input_weights, args.in_file):
        payload = torch.load(str(path.expanduser()), map_location="cpu", weights_only=False)
        if str(payload.get("format", "")) != "single_vehicle_benchmark_v1":
            raise ValueError(f"unsupported benchmark format in {path}")
        for sample in payload.get("samples", []):
            sample = dict(sample)
            target = dict(sample.get("target", {}))
            target["sample_weight"] = torch.tensor([float(weight)], dtype=torch.float32)
            sample["target"] = target
            merged_samples.append(sample)
        metas.append(dict(payload.get("meta", {})))

    payload = {
        "format": "single_vehicle_benchmark_v1",
        "meta": _json_ready(
            {
                "inputs": [str(Path(path).expanduser()) for path in args.in_file],
                "input_weights": [float(w) for w in input_weights],
                "merged_count": len(args.in_file),
                "source_meta": metas,
            }
        ),
        "length": int(len(merged_samples)),
        "samples": merged_samples,
    }
    args.out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(args.out_file))
    args.out_file.with_suffix(".json").write_text(json.dumps(_json_ready(payload["meta"]), indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
