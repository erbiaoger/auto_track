"""Convert the 2026-05-07 SAC archive into Raw planes for a web cache."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from obspy import read

from convert_gauss_zip_to_web_input import read_station_order


def _station_id(member: str) -> str | None:
    parts = Path(member).stem.split("_")
    return parts[1].strip().upper() if len(parts) >= 2 else None


def _web_raw(values: np.ndarray) -> np.ndarray:
    """Match the normalized Raw scale used by the Hybrid feature pipeline."""
    values = np.asarray(values, dtype=np.float32)
    centered = values - np.median(values)
    scale = max(float(np.quantile(np.abs(centered), 0.995)), 1e-6)
    return np.clip(centered / scale, -1.0, 1.0).astype(np.float32)


def convert(*, archive: Path, manifest_path: Path, workbook: Path, force: bool) -> None:
    archive = archive.expanduser().resolve()
    manifest_path = manifest_path.expanduser().resolve()
    workbook = workbook.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = str(manifest["dataset_id"])
    day_id = dataset_id[-2:]
    station_order = [str(value).upper() for value in manifest["station_ids_in_channel_order"]]
    workbook_order = [str(row["station_id"]).upper() for row in read_station_order(workbook)]
    workbook_index = {station: index for index, station in enumerate(workbook_order)}
    if any(station not in workbook_index for station in station_order):
        missing = [station for station in station_order if station not in workbook_index]
        raise ValueError(f"manifest stations missing from workbook: {missing}")
    if [workbook_index[station] for station in station_order] != sorted(workbook_index[station] for station in station_order):
        raise ValueError("manifest station order is not consistent with the workbook")

    output_path = manifest_path.parent.parent / "source" / f"raw_{dataset_id}.npy"
    if output_path.exists() and not force:
        raise FileExistsError(f"output exists: {output_path}; use --force to replace it")

    with ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".sac") and f"/{day_id}/" in f"/{name}"]
        by_station = {
            station: member
            for member in members
            for station in [_station_id(member)]
            if station
        }
        missing = [station for station in station_order if station not in by_station]
        if missing:
            raise ValueError(f"{archive.name} day {day_id}: missing stations: {missing[:8]}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zf.open(by_station[station_order[0]]) as handle:
                first_trace = read(handle, format="SAC")[0]
        sample_count = int(first_trace.stats.npts)
        sample_rate_hz = float(first_trace.stats.sampling_rate)
        expected_samples = int(manifest["sample_count"])
        expected_rate = float(manifest["sample_rate_hz"])
        if sample_count != expected_samples or not np.isclose(sample_rate_hz, expected_rate):
            raise ValueError(
                f"SAC shape/rate ({sample_count}, {sample_rate_hz}) does not match "
                f"manifest ({expected_samples}, {expected_rate})"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float32, shape=(sample_count, len(station_order))
        )
        combined[:, 0] = _web_raw(first_trace.data)
        for index, station in enumerate(station_order[1:], start=1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with zf.open(by_station[station]) as handle:
                    trace = read(handle, format="SAC")[0]
            if int(trace.stats.npts) != sample_count or not np.isclose(float(trace.stats.sampling_rate), sample_rate_hz):
                raise ValueError(f"inconsistent SAC shape/rate for {station}")
            combined[:, index] = _web_raw(trace.data)
            print(f"{archive.name} day {day_id}: assembled {index + 1}/{len(station_order)} channels", flush=True)
        combined.flush()
        del combined

    manifest["raw_path"] = "../source/" + output_path.name
    manifest["raw_archive"] = str(archive)
    manifest["raw_workbook"] = str(workbook)
    manifest["raw_semantics"] = "per-station median-centered and 99.5-percentile normalized SAC waveform, time-major [sample, station], workbook-ordered subset matching Gauss/Pre"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {manifest_path} raw_shape=({sample_count}, {len(station_order)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workbook", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert(archive=args.archive, manifest_path=args.manifest, workbook=args.workbook, force=args.force)


if __name__ == "__main__":
    main()
