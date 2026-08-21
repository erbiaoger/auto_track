"""Convert per-station Gauss ZIP data into the replay web input layout.

The replay service consumes one time-major ``[sample, station]`` NPY array and
an ordered station mapping.  The field ZIPs contain one one-dimensional NPY
file per station, so this converter orders the available stations by the
deployment workbook and keeps the first 50 stations supported by the current
trackers.  Raw and Pre are intentionally left absent: the generated manifest
describes a Gauss-only dataset and the web service disables methods that need
the missing planes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
DEFAULT_WORKBOOK = Path("shared_data/legacy_datasets/peaks_20260723/415现场布设详表20260312.xlsx")


def _shared_strings(zf: ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(text.text or "" for text in si.iter(f"{{{MAIN_NS}}}t")) for si in root.findall(f"{{{MAIN_NS}}}si")]


def _cell_value(cell: ET.Element | None, shared: list[str]) -> str | None:
    if cell is None:
        return None
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    return shared[int(value.text)] if cell.attrib.get("t") == "s" else value.text


def read_station_order(workbook: Path) -> list[dict[str, str | int | None]]:
    with ZipFile(workbook) as zf:
        shared = _shared_strings(zf)
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: list[dict[str, str | int | None]] = []
        for row in sheet.findall(f".//{{{MAIN_NS}}}row"):
            number = int(row.attrib["r"])
            if number < 3:
                continue
            cells = {cell.attrib.get("r", ""): cell for cell in row.findall(f"{{{MAIN_NS}}}c")}
            sequence = _cell_value(cells.get(f"B{number}"), shared)
            station_id = _cell_value(cells.get(f"G{number}"), shared)
            location = _cell_value(cells.get(f"H{number}"), shared)
            if sequence is not None and station_id:
                rows.append({"sequence": int(float(sequence)), "station_id": station_id.strip().upper(), "location": location})
        if not rows:
            raise ValueError(f"no station rows found in {workbook}")
        return rows


def _station_id(member: str) -> str | None:
    parts = Path(member).stem.split("_")
    return parts[1].strip().upper() if len(parts) >= 2 else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--channel-count", type=int, default=50)
    parser.add_argument("--sample-rate-hz", type=float, default=1000.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def convert(*, archive: Path, output_dir: Path, workbook: Path, channel_count: int, sample_rate_hz: float, force: bool) -> None:
    if channel_count <= 0 or sample_rate_hz <= 0:
        raise ValueError("channel-count and sample-rate-hz must be positive")
    archive = archive.expanduser().resolve()
    workbook = workbook.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    source_dir = output_dir / "source"
    mapping_dir = output_dir / "mapping"
    cache_dir = output_dir / "cache"
    day_match = archive.stem.upper().replace("GAUSS_OUTPUT_", "")
    if not day_match:
        raise ValueError(f"cannot infer dataset id from {archive.name}")
    gauss_path = source_dir / f"gauss_{day_match}.npy"
    mapping_path = mapping_dir / f"raw_{day_match}.mapping.json"
    manifest_path = cache_dir / "full_day_manifest.json"
    outputs = (gauss_path, mapping_path, manifest_path)
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError("output exists; use --force to replace the generated dataset")

    workbook_rows = read_station_order(workbook)
    with ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".npy")]
        by_station: dict[str, str] = {}
        for member in members:
            station = _station_id(member)
            if station:
                by_station[station] = member
        available = [row for row in workbook_rows if str(row["station_id"]).upper() in by_station]
        if len(available) < channel_count:
            raise ValueError(f"{archive.name}: only {len(available)} workbook stations are present; need {channel_count}")
        selected = available[:channel_count]
        selected_members = [by_station[str(row["station_id"])] for row in selected]
        first = np.load(zf.open(selected_members[0]), mmap_mode=None)
        if first.ndim != 1:
            raise ValueError(f"expected 1-D station array, got {first.shape}")
        sample_count = int(first.shape[0])
        source_dir.mkdir(parents=True, exist_ok=True)
        combined = np.lib.format.open_memmap(gauss_path, mode="w+", dtype=np.float32, shape=(sample_count, channel_count))
        combined[:, 0] = np.asarray(first, dtype=np.float32)
        for index, member in enumerate(selected_members[1:], start=1):
            values = np.load(zf.open(member), mmap_mode=None)
            if values.ndim != 1 or int(values.shape[0]) != sample_count:
                raise ValueError(f"inconsistent array shape for {member}: {values.shape}; expected ({sample_count},)")
            combined[:, index] = np.asarray(values, dtype=np.float32)
            print(f"{archive.name}: assembled {index + 1}/{channel_count} channels", flush=True)
        combined.flush()
        del combined

    mapping = {
        "dataset_id": day_match,
        "archive": str(archive),
        "workbook": str(workbook),
        "output": str(gauss_path),
        "source_array_shape": [sample_count],
        "combined_shape_time_channel": [sample_count, channel_count],
        "channel_count": channel_count,
        "selected_channels": [
            {"channel_index": index, **row, "archive_member": member}
            for index, (row, member) in enumerate(zip(selected, selected_members))
        ],
        "excluded_available_channels": available[channel_count:],
        "missing_workbook_stations": [row for row in workbook_rows if row not in available],
        "archive_npy_count": len(members),
    }
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_id": day_match,
        "dataset_label": f"{day_match} · Gauss-only",
        "source_archive": str(archive),
        "mapping": "../mapping/" + mapping_path.name,
        "station_count": channel_count,
        "sample_count": sample_count,
        "duration_s": sample_count / sample_rate_hz,
        "sample_rate_hz": sample_rate_hz,
        "raw_path": None,
        "pre_path": None,
        "gauss_path": "../source/" + gauss_path.name,
        "input_semantics": "Gauss-only station arrays assembled in workbook spatial order",
        "station_ids_in_channel_order": [row["station_id"] for row in selected],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {manifest_path} shape=({sample_count}, {channel_count})")


if __name__ == "__main__":
    args = _parse_args()
    convert(
        archive=args.archive,
        output_dir=args.output_dir,
        workbook=args.workbook,
        channel_count=args.channel_count,
        sample_rate_hz=args.sample_rate_hz,
        force=args.force,
    )
