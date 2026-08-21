from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vehicle_replay_web.methods import MethodManager


class _Source:
    sample_rate_hz = 20.0
    duration_s = 150.0
    station_count = 24

    def __init__(self, root: Path) -> None:
        positions = np.asarray([i * 100.0 for i in range(self.station_count)], dtype=float)
        times = np.arange(int(self.duration_s * self.sample_rate_hz)) / self.sample_rate_hz
        data = np.zeros((len(times), self.station_count), dtype=np.float32)
        for channel, position in enumerate(positions):
            peak_time = 4.0 + (positions[-1] - position) / (75.0 / 3.6)
            data[:, channel] = 2.0 * np.exp(-0.5 * ((times - peak_time) / 0.12) ** 2)
        paths = {}
        for name in ("raw", "pre", "gauss"):
            path = root / f"{name}.npy"
            np.save(path, data)
            paths[f"{name}_path"] = path
        self.raw_path = paths["raw_path"]
        self.pre_path = paths["pre_path"]
        self.gauss_path = paths["gauss_path"]


def _mapping(path: Path, count: int = 24) -> None:
    path.write_text(
        json.dumps(
            {
                "selected_channels": [
                    {"channel_index": i, "station_id": f"S{i}", "location": f"{i / 10:.1f}"}
                    for i in range(count)
                ]
            }
        ),
        encoding="utf-8",
    )


def test_cpu_classic_workers_speak_method_protocol_and_switch(tmp_path: Path) -> None:
    source = _Source(tmp_path)
    mapping = tmp_path / "mapping.json"
    _mapping(mapping)
    manager = MethodManager(source=source, mapping_path=mapping, device="cuda")
    try:
        descriptors = {item["method_id"]: item for item in manager.descriptors()}
        assert descriptors["hungarian_assignment"]["available"] is True
        assert descriptors["hungarian_assignment"]["device"] == "cpu"
        assert descriptors["kalman_seed"]["device"] == "cpu"

        for method_id in ("hungarian_assignment", "kalman_seed"):
            manager.ensure(method_id)
            response = manager.predict(start_s=0.0, duration_s=120.0)
            assert response["protocol"] == "method-worker/v1"
            assert response["method_id"] == method_id
            assert response["diagnostics"]["direction"] == "reverse"
            assert response["tracks"]
            assert all(
                0.0 <= float(point["time_s"]) <= 120.0
                for track in response["tracks"]
                for point in track["points"]
            )
    finally:
        manager.close()
