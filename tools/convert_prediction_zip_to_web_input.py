"""Add a per-station prediction ZIP as the optional Pre plane of a web cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def station_id(member: str) -> str | None:
    parts = Path(member).stem.split("_")
    return parts[1].strip().upper() if len(parts) >= 2 else None


def convert(*, archive: Path, manifest_path: Path, force: bool) -> None:
    archive = archive.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    station_order = [str(value).upper() for value in manifest["station_ids_in_channel_order"]]
    output_path = manifest_path.parent.parent / "source" / f"pre_{manifest['dataset_id']}.npy"
    prediction_path = manifest_path.parent.parent / "source" / f"prediction_{manifest['dataset_id']}.npy"
    if (output_path.exists() or prediction_path.exists()) and not force:
        raise FileExistsError(f"output exists: {output_path} or {prediction_path}; use --force to replace it")

    with ZipFile(archive) as zf:
        by_station = {
            value: member
            for member in zf.namelist()
            if member.lower().endswith(".npy")
            for value in [station_id(member)]
            if value
        }
        missing = [value for value in station_order if value not in by_station]
        if missing:
            raise ValueError(f"{archive.name}: missing cached stations: {missing[:8]}")
        first = np.load(zf.open(by_station[station_order[0]]), mmap_mode=None)
        if first.ndim != 1:
            raise ValueError(f"expected 1-D prediction array, got {first.shape}")
        sample_count = int(first.shape[0])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(sample_count, len(station_order)))
        combined[:, 0] = np.asarray(first, dtype=np.float32)
        for index, value in enumerate(station_order[1:], start=1):
            array = np.load(zf.open(by_station[value]), mmap_mode=None)
            if array.ndim != 1 or int(array.shape[0]) != sample_count:
                raise ValueError(f"inconsistent array shape for {value}: {array.shape}; expected ({sample_count},)")
            combined[:, index] = np.asarray(array, dtype=np.float32)
            print(f"{archive.name}: assembled {index + 1}/{len(station_order)} channels", flush=True)
        combined.flush()
        del combined

    expected = int(manifest["sample_count"])
    if sample_count != expected:
        raise ValueError(f"prediction samples {sample_count} != manifest samples {expected}")
    source = np.load(output_path, mmap_mode="r")
    prediction = np.lib.format.open_memmap(prediction_path, mode="w+", dtype=np.float32, shape=source.shape)
    for start in range(0, source.shape[0], 200_000):
        end = min(source.shape[0], start + 200_000)
        prediction[start:end] = 1.0 / (1.0 + np.exp(np.clip(np.asarray(source[start:end], dtype=np.float32), -30.0, 30.0)))
    prediction.flush()
    del prediction
    manifest["pre_path"] = "../source/" + output_path.name
    manifest["prediction_path"] = "../source/" + prediction_path.name
    manifest["prediction_archive"] = str(archive)
    manifest["dataset_label"] = str(manifest.get("dataset_id", "dataset")) + " · Gauss + Pre"
    manifest["input_semantics"] = "Gauss, Pre logits, and flipped prediction station arrays assembled in workbook spatial order"
    manifest["prediction_semantics"] = "sigmoid(-PRE/pRE logits), high values are vehicle candidates; equivalent to pred_flipped"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {manifest_path} pre_shape=({sample_count}, {len(station_order)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert(archive=args.archive, manifest_path=args.manifest, force=args.force)


if __name__ == "__main__":
    main()
