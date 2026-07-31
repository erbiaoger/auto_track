"""Assemble per-station NumPy files from a DAS zip archive in Excel order.

The PeakSlot checkpoints in this repository consume a fixed 50-channel
``[time, channel]`` array.  The field data arrives as one one-dimensional
``.npy`` file per station, so this utility uses the station IDs in the
workbook's current-ID column (column G), orders the files spatially, and
writes one combined array plus an auditable mapping JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import numpy as np


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path, help="Input zip containing one .npy per station.")
    parser.add_argument("--workbook", required=True, type=Path, help="Station layout workbook.")
    parser.add_argument("--out", required=True, type=Path, help="Output combined .npy path.")
    parser.add_argument("--mapping-out", type=Path, help="Mapping JSON path; defaults to <out>.mapping.json.")
    parser.add_argument("--channel-count", type=int, default=50, help="Number of ordered stations to keep.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files.")
    return parser.parse_args()


def _shared_strings(zf: ZipFile) -> list[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in si.iter(f"{{{MAIN_NS}}}t"))
        for si in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _cell_value(cell: ET.Element | None, shared: list[str]) -> str | None:
    if cell is None:
        return None
    value = cell.find(f"{{{MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    raw = value.text
    if cell.attrib.get("t") == "s":
        return shared[int(raw)]
    return raw


def read_station_order(workbook: Path) -> list[dict[str, str | int | None]]:
    """Read sequence, current station ID (G), and location (H) from the sheet."""
    with ZipFile(workbook) as zf:
        shared = _shared_strings(zf)
        workbook_root = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = workbook_root.find(f".//{{{MAIN_NS}}}sheet")
        if sheet is None:
            raise ValueError(f"No worksheet found in {workbook}")
        sheet_path = "xl/worksheets/sheet1.xml"
        sheet_root = ET.fromstring(zf.read(sheet_path))
        result: list[dict[str, str | int | None]] = []
        for row in sheet_root.findall(f".//{{{MAIN_NS}}}row"):
            row_number = int(row.attrib["r"])
            if row_number < 3:
                continue
            cells = {cell.attrib.get("r", ""): cell for cell in row.findall(f"{{{MAIN_NS}}}c")}
            sequence = _cell_value(cells.get(f"B{row_number}"), shared)
            station = _cell_value(cells.get(f"G{row_number}"), shared)
            location = _cell_value(cells.get(f"H{row_number}"), shared)
            if sequence is None or station is None:
                continue
            result.append({"sequence": int(float(sequence)), "station_id": station.strip(), "location": location})
        if not result:
            raise ValueError(f"No station rows found in {workbook}")
        return result


def _station_id(name: str) -> str | None:
    stem = Path(name).stem
    parts = stem.split("_")
    return parts[1].strip().upper() if len(parts) >= 2 else None


def assemble(*, archive: Path, workbook: Path, out: Path, mapping_out: Path, channel_count: int, overwrite: bool) -> None:
    if channel_count <= 0:
        raise ValueError("--channel-count must be positive")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out}; use --overwrite")
    if mapping_out.exists() and not overwrite:
        raise FileExistsError(f"Mapping exists: {mapping_out}; use --overwrite")

    workbook_rows = read_station_order(workbook)
    with ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".npy")]
        by_station: dict[str, str] = {}
        duplicate_ids: list[str] = []
        for member in members:
            station = _station_id(member)
            if station is None:
                continue
            if station in by_station:
                duplicate_ids.append(station)
            by_station[station] = member

        ordered_rows = [row for row in workbook_rows if str(row["station_id"]).upper() in by_station]
        missing_rows = [row for row in workbook_rows if str(row["station_id"]).upper() not in by_station]
        if len(ordered_rows) < channel_count:
            raise ValueError(
                f"{archive.name}: only {len(ordered_rows)} workbook stations have files; "
                f"cannot assemble {channel_count} channels"
            )
        selected_rows = ordered_rows[:channel_count]
        selected_members = [by_station[str(row["station_id"]).upper()] for row in selected_rows]
        excluded_rows = ordered_rows[channel_count:]

        first_member = selected_members[0]
        with zf.open(first_member) as fp:
            first = np.asarray(np.load(fp), dtype=np.float32)
        if first.ndim != 1:
            raise ValueError(f"Expected 1-D station array, got {first_member}: {first.shape}")
        sample_count = int(first.shape[0])
        out.parent.mkdir(parents=True, exist_ok=True)
        combined = np.lib.format.open_memmap(out, mode="w+", dtype=np.float32, shape=(sample_count, channel_count))
        combined[:, 0] = first
        for channel, member in enumerate(selected_members[1:], start=1):
            with zf.open(member) as fp:
                values = np.asarray(np.load(fp), dtype=np.float32)
            if values.ndim != 1 or int(values.shape[0]) != sample_count:
                raise ValueError(f"Inconsistent array shape for {member}: {values.shape}; expected ({sample_count},)")
            combined[:, channel] = values
            print(f"{archive.name}: assembled {channel + 1}/{channel_count} channels", flush=True)
        combined.flush()
        del combined

    mapping = {
        "archive": str(archive.resolve()),
        "workbook": str(workbook.resolve()),
        "output": str(out.resolve()),
        "source_array_shape": [sample_count],
        "combined_shape_time_channel": [sample_count, channel_count],
        "channel_count": channel_count,
        "workbook_station_rows": workbook_rows,
        "selected_channels": [
            {"channel_index": index, **row, "archive_member": member}
            for index, (row, member) in enumerate(zip(selected_rows, selected_members))
        ],
        "excluded_available_channels": excluded_rows,
        "missing_workbook_stations": missing_rows,
        "archive_npy_count": len(members),
        "duplicate_station_ids": sorted(set(duplicate_ids)),
    }
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done: {out} shape=({sample_count}, {channel_count})", flush=True)


def main() -> int:
    args = parse_args()
    mapping_out = args.mapping_out or args.out.with_suffix(args.out.suffix + ".mapping.json")
    assemble(
        archive=args.archive.expanduser(),
        workbook=args.workbook.expanduser(),
        out=args.out.expanduser(),
        mapping_out=mapping_out.expanduser(),
        channel_count=int(args.channel_count),
        overwrite=bool(args.overwrite),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
