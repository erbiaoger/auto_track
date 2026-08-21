from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from hybrid_vehicle_tracker.types import Station, StationGeometry


_FLOAT_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _position_m(location: Any) -> float:
    if isinstance(location, (float, int)):
        return float(location) * 1000.0
    match = _FLOAT_PATTERN.search(str(location))
    if match is None:
        raise ValueError(f"cannot parse station position from {location!r}")
    return float(match.group(0)) * 1000.0


def load_station_geometry(path: str | Path) -> StationGeometry:
    """Load physical station locations from a conversion mapping JSON."""
    mapping_path = Path(path)
    with mapping_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    selected = payload.get("selected_channels")
    if not selected:
        raise ValueError(
            f"{mapping_path} has no selected_channels with physical locations; "
            "use a raw/pre/gauss array mapping JSON"
        )

    stations = []
    for row in sorted(selected, key=lambda item: int(item["channel_index"])):
        station_id = str(row["station_id"]).upper()
        stations.append(
            Station(
                channel_index=int(row["channel_index"]),
                sequence=int(row.get("sequence", int(row["channel_index"]) + 1)),
                station_id=station_id,
                position_m=_position_m(row["location"]),
                location=str(row.get("location", "")),
            )
        )
    return StationGeometry(tuple(stations))

