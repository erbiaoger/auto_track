from __future__ import annotations

import json
import asyncio
from pathlib import Path

import numpy as np

from hybrid_vehicle_tracker.types import TrackBatch, TrackPoint, VehicleTrack
from hybrid_vehicle_tracker.web.runtime import encode_frame
from hybrid_vehicle_tracker.web.source import NpyReplaySource
from hybrid_vehicle_tracker.web.stitch import TrackStitcher
from hybrid_vehicle_tracker.web.runtime import ReplayController


def test_npy_replay_source_downsamples_peak_without_losing_it(tmp_path: Path) -> None:
    arrays = {}
    for name in ("raw", "pre", "gauss"):
        array = np.zeros((2000, 2), dtype=np.float32)
        array[777, 1] = 1.0
        path = tmp_path / f"{name}.npy"
        np.save(path, array)
        arrays[name] = path
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sample_rate_hz": 1000, "duration_s": 2, **{f"{name}_path": str(path) for name, path in arrays.items()}}))
    source = NpyReplaySource(manifest)
    frame = source.frame(0, 1)
    assert len(frame["times_s"]) == 20
    assert max(frame["gauss"]) == 1.0
    assert frame["gauss"][15 * 2 + 1] == 1.0
    assert frame["waveform_rate_hz"] == 50.0
    assert len(frame["raw_wave"]) == 50 * 2
    source.set_waveform_downsample(10)
    assert len(source.frame(0, 1)["raw_wave"]) == 100 * 2


def test_binary_frame_has_magic_and_json_header() -> None:
    packet = encode_frame({"event": "frame", "start_s": 0.0, "duration_s": 1.0, "times_s": [0.0], "station_count": 1, "gauss": [1.0], "raw_min": [0.0], "raw_max": [1.0]})
    assert packet[:4] == b"HVT1"
    header_size = int.from_bytes(packet[4:8], "little")
    header = json.loads(packet[8 : 8 + header_size])
    assert header["event"] == "frame"


def _batch(start: float, track_id: str, shift: float = 0.0) -> TrackBatch:
    points = [
        TrackPoint(i, f"S{i}", i * 200.0, 10.0 + i * 2.0 + shift, True)
        for i in range(5)
    ]
    track = VehicleTrack(track_id, points, 72.0, [72.0], 0.9, 1.0, 5, 800.0, 0.1, 200.0, False, False, False)
    return TrackBatch([track], [], start, 120.0)


def test_track_stitcher_keeps_id_across_overlapping_windows() -> None:
    stitcher = TrackStitcher()
    stitcher.update(_batch(0.0, "a"))
    stitcher.update(_batch(60.0, "b", shift=-60.0))
    result = stitcher.finalize().tracks
    assert result[0].global_vehicle_id == "V0001"
    assert stitcher.cumulative_unique_count == 1


def test_replay_controller_reaches_end_without_blocking() -> None:
    class Source:
        sample_rate_hz = 1000.0
        duration_s = 3.0
        station_count = 1
        raw = pre = gauss = np.zeros((3000, 1), dtype=np.float32)

        def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
            return {"event": "frame", "start_s": start_s, "duration_s": duration_s, "times_s": [start_s], "station_count": 1, "gauss": [0.0], "raw_min": [0.0], "raw_max": [0.0]}

    class Tracker:
        device = "cuda:0"
        has_checkpoint = True

        def predict(self, *_args: object, **_kwargs: object) -> TrackBatch:
            return TrackBatch([], [], 0.0, 120.0, {})

    async def run() -> None:
        controller = ReplayController(Source(), Tracker(), mapping=object(), window_s=2.0)
        await controller.start(0.0, speed="fastest")
        await controller._replay_task
        assert controller.state.cursor_s == 3.0
        assert controller.state.playing is False
        await controller.close()

    asyncio.run(run())


def test_seek_uses_new_window_origin() -> None:
    class Source:
        sample_rate_hz = 1000.0
        duration_s = 131.0
        station_count = 1
        raw = pre = gauss = np.zeros((131000, 1), dtype=np.float32)

        def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
            return {"event": "frame", "start_s": start_s, "duration_s": duration_s, "times_s": [start_s], "station_count": 1, "gauss": [0.0], "raw_min": [0.0], "raw_max": [0.0]}

    class Tracker:
        device = "cuda:0"
        has_checkpoint = True

        def predict(self, *_args: object, **_kwargs: object) -> TrackBatch:
            return TrackBatch([], [], 0.0, 120.0, {})

    async def run() -> None:
        controller = ReplayController(Source(), Tracker(), mapping=object())
        await controller.start(10.0, speed="fastest")
        await controller._replay_task
        await controller._queue.join()
        assert controller.state.last_prediction_s == 10.0
        await controller.close()

    asyncio.run(run())


def test_replay_window_and_stride_are_runtime_parameters() -> None:
    class Source:
        sample_rate_hz = 1000.0
        duration_s = 10.0
        station_count = 1
        raw = pre = gauss = np.zeros((10000, 1), dtype=np.float32)

        def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
            return {"event": "frame", "start_s": start_s, "duration_s": duration_s, "times_s": [start_s], "station_count": 1, "gauss": [0.0], "raw_min": [0.0], "raw_max": [0.0]}

    class Tracker:
        device = "cuda:0"
        has_checkpoint = True

    async def run() -> None:
        controller = ReplayController(Source(), Tracker(), mapping=object())
        await controller.start(0.0, speed="fastest", window_s=4.0, stride_s=2.0, waveform_downsample=10)
        await controller.control({"action": "pause"})
        assert controller.state.window_s == 4.0
        assert controller.state.stride_s == 2.0
        assert controller.state.waveform_downsample == 10
        await controller.close()

    asyncio.run(run())


def test_stride_schedules_predictions_at_configured_interval() -> None:
    calls: list[float] = []

    class Source:
        sample_rate_hz = 1000.0
        duration_s = 101.0
        station_count = 1
        raw = pre = gauss = np.zeros((101000, 1), dtype=np.float32)

        def frame(self, start_s: float, duration_s: float) -> dict[str, object]:
            return {"event": "frame", "start_s": start_s, "duration_s": duration_s, "times_s": [start_s], "station_count": 1, "gauss": [0.0], "raw_min": [0.0], "raw_max": [0.0]}

    class Tracker:
        device = "cuda:0"
        has_checkpoint = True

        def predict(self, *_args: object, **kwargs: object) -> TrackBatch:
            calls.append(float(kwargs["start_s"]))
            return TrackBatch([], [], 0.0, 40.0, {})

    async def run() -> None:
        controller = ReplayController(Source(), Tracker(), mapping=object(), window_s=40.0, stride_s=20.0)
        await controller.start(0.0, speed="fastest")
        await controller._replay_task
        await controller._queue.join()
        await controller.close()

    asyncio.run(run())
    assert calls == [0.0, 20.0, 40.0, 60.0]
