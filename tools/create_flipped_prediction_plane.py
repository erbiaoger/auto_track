"""Create the high-is-event prediction plane expected by the separation script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert(*, manifest_path: Path, force: bool) -> None:
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pre_path = manifest_path.parent / str(manifest["pre_path"])
    if not pre_path.exists():
        raise FileNotFoundError(pre_path)
    output_path = manifest_path.parent.parent / "source" / f"prediction_{manifest['dataset_id']}.npy"
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists: {output_path}; use --force to replace it")

    source = np.load(pre_path, mmap_mode="r")
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=source.shape)
    # PRE/pRE are raw logits whose event response is a negative valley.
    # sigmoid(-logit) matches the original script's pred_flipped semantics.
    for start in range(0, source.shape[0], 200_000):
        end = min(source.shape[0], start + 200_000)
        values = np.asarray(source[start:end], dtype=np.float32)
        output[start:end] = 1.0 / (1.0 + np.exp(np.clip(values, -30.0, 30.0)))
    output.flush()
    del output
    manifest["prediction_path"] = "../source/" + output_path.name
    manifest["prediction_semantics"] = "sigmoid(-PRE/pRE logits), high values are vehicle candidates; equivalent to pred_flipped"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {manifest_path} prediction_shape={source.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert(manifest_path=args.manifest, force=args.force)


if __name__ == "__main__":
    main()
