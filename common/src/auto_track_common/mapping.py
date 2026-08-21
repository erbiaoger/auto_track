from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import Station, StationGeometry

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _position_m(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value) * 1000.0
    match = _NUMBER.search(str(value))
    if match is None:
        raise ValueError(f"cannot parse station position from {value!r}")
    return float(match.group(0)) * 1000.0


def load_station_geometry(path: str | Path) -> StationGeometry:
    mapping_path = Path(path).expanduser()
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = payload.get("selected_channels")
    if not rows:
        raise ValueError(f"{mapping_path} has no selected_channels")
    stations = tuple(
        Station(
            channel_index=int(row["channel_index"]),
            station_id=str(row.get("station_id", row["channel_index"])).upper(),
            position_m=_position_m(row.get("location", row.get("position_m"))),
            sequence=int(row.get("sequence", int(row["channel_index"]) + 1)),
            location=str(row.get("location", "")),
        )
        for row in sorted(rows, key=lambda item: int(item["channel_index"]))
    )
    return StationGeometry(stations)
