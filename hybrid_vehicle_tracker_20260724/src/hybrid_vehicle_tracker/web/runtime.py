from __future__ import annotations

import asyncio
import csv
import json
import struct
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker
from hybrid_vehicle_tracker.types import TrackBatch

from .methods import MethodManager, WorkerError, response_to_batch
from .source import DataSource
from .stitch import StreamingTrack, TrackStitcher


FRAME_MAGIC = b"HVT1"


def encode_frame(frame: dict[str, object]) -> bytes:
    """Encode display planes and optional 100 Hz raw waveform as one binary frame."""
    array_names = ["gauss", "raw_min", "raw_max"]
    if "raw_wave" in frame:
        array_names.append("raw_wave")
    arrays = [np.asarray(frame.pop(name), dtype="<f4") for name in array_names]
    frame["array_lengths"] = [int(item.size) for item in arrays]
    header = json.dumps(frame, separators=(",", ":")).encode("utf-8")
    payload = b"".join(item.tobytes(order="C") for item in arrays)
    return FRAME_MAGIC + struct.pack("<I", len(header)) + header + payload


@dataclass
class ReplayState:
    session_id: str = ""
    cursor_s: float = 0.0
    total_duration_s: float = 0.0
    window_s: float = 120.0
    stride_s: float = 60.0
    waveform_downsample: int = 20
    playing: bool = False
    speed: float | str = 1.0
    buffer_start_s: float = 0.0
    buffer_end_s: float = 0.0
    buffered_s: float = 0.0
    prediction_queue: int = 0
    last_prediction_s: float | None = None
    last_inference_s: float | None = None
    error: str | None = None
    method_id: str = "hybrid"
    preset_id: str = "hybrid_v9_morphology_16384"

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "cursor_s": self.cursor_s,
            "total_duration_s": self.total_duration_s,
            "window_s": self.window_s,
            "stride_s": self.stride_s,
            "waveform_downsample": self.waveform_downsample,
            "playing": self.playing,
            "speed": self.speed,
            "buffer_start_s": self.buffer_start_s,
            "buffer_end_s": self.buffer_end_s,
            "buffered_s": self.buffered_s,
            "prediction_queue": self.prediction_queue,
            "last_prediction_s": self.last_prediction_s,
            "last_inference_s": self.last_inference_s,
            "error": self.error,
            "method_id": self.method_id,
            "preset_id": self.preset_id,
        }


@dataclass
class ReplayController:
    source: DataSource
    tracker: HybridVehicleTracker | None
    mapping: object
    stride_s: float = 60.0
    window_s: float = 120.0
    waveform_downsample: int = 20
    output_root: Path = Path("runs/web_sessions")
    state: ReplayState = field(default_factory=ReplayState)
    method_manager: MethodManager | None = None

    def __post_init__(self) -> None:
        if self.window_s <= 0 or self.stride_s <= 0:
            raise ValueError("window_s and stride_s must be positive")
        if self.waveform_downsample < 1:
            raise ValueError("waveform_downsample must be >= 1")
        self._set_waveform_downsample(self.waveform_downsample)
        self.state.total_duration_s = float(self.source.duration_s)
        self.state.window_s = float(self.window_s)
        self.state.stride_s = float(self.stride_s)
        self.state.waveform_downsample = int(self.waveform_downsample)
        if self.method_manager is not None:
            self.state.method_id = self.method_manager.default_method
            spec = self.method_manager.specs.get(self.state.method_id, {})
            self.state.preset_id = str(spec.get("preset_id", self.state.method_id))
        self._subscribers: set[Any] = set()
        self._replay_task: asyncio.Task[None] | None = None
        self._worker_task: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[tuple[str, float] | None] = asyncio.Queue(maxsize=8)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hvt-gpu")
        self._stitcher = TrackStitcher(stride_s=self.stride_s, window_s=self.window_s)
        self._session_tracks: dict[str, StreamingTrack] = {}
        self._last_counters: dict[str, Any] = self._stitcher.counters(window_candidates=0)
        self._next_prediction_start_s = 0.0
        self._lock = asyncio.Lock()

    async def ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._prediction_worker())

    async def close(self) -> None:
        if self._replay_task is not None:
            self._replay_task.cancel()
            await asyncio.gather(self._replay_task, return_exceptions=True)
        if self._worker_task is not None:
            await self._queue.put(None)
            await asyncio.gather(self._worker_task, return_exceptions=True)
        if self.method_manager is not None:
            self.method_manager.close()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def subscribe(self, websocket: Any) -> None:
        self._subscribers.add(websocket)

    def unsubscribe(self, websocket: Any) -> None:
        self._subscribers.discard(websocket)

    async def _broadcast_json(self, payload: dict[str, object]) -> None:
        dead: list[Any] = []
        for websocket in tuple(self._subscribers):
            try:
                await websocket.send_json(payload)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.unsubscribe(websocket)

    async def _broadcast_frame(self, payload: dict[str, object]) -> None:
        dead: list[Any] = []
        for websocket in tuple(self._subscribers):
            try:
                # encode_frame consumes plane lists to avoid a second copy.
                await websocket.send_bytes(encode_frame(payload.copy()))
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.unsubscribe(websocket)

    def snapshot(self) -> dict[str, object]:
        counters = self._last_counters
        positions = getattr(self.mapping, "positions_m", np.asarray([], dtype=np.float64))
        return {
            "state": self.state.to_dict(),
            "counters": counters,
            "active_method_id": self.state.method_id,
            "active_preset_id": self.state.preset_id,
            "methods": self.method_manager.descriptors() if self.method_manager is not None else [],
            "station_positions_m": np.asarray(positions, dtype=np.float64).tolist(),
            "prediction_schedule": {
                "next_window_start_s": float(self._next_prediction_start_s),
                "interval_s": float(self.stride_s),
                "window_length_s": float(self.window_s),
                "waveform_downsample": int(self.waveform_downsample),
            },
        }

    def _set_waveform_downsample(self, value: int) -> None:
        value = int(value)
        if value < 1:
            raise ValueError("waveform_downsample must be >= 1")
        setter = getattr(self.source, "set_waveform_downsample", None)
        if setter is not None:
            setter(value)
        elif hasattr(self.source, "waveform_downsample"):
            setattr(self.source, "waveform_downsample", value)
        self.waveform_downsample = value

    async def start(
        self,
        start_s: float = 0.0,
        *,
        speed: float | str | None = None,
        window_s: float | None = None,
        stride_s: float | None = None,
        waveform_downsample: int | None = None,
        method_id: str | None = None,
    ) -> None:
        if window_s is not None:
            if float(window_s) <= 0:
                raise ValueError("window_s must be positive")
            self.window_s = float(window_s)
        if stride_s is not None:
            if float(stride_s) <= 0:
                raise ValueError("stride_s must be positive")
            self.stride_s = float(stride_s)
        if waveform_downsample is not None:
            self._set_waveform_downsample(int(waveform_downsample))
        if self.window_s > float(self.source.duration_s):
            raise ValueError(f"window_s={self.window_s} exceeds source duration {self.source.duration_s}")
        self._stitcher = TrackStitcher(stride_s=self.stride_s, window_s=self.window_s)
        start_s = max(0.0, min(float(start_s), max(0.0, self.source.duration_s)))
        if self._replay_task is not None:
            self._replay_task.cancel()
            await asyncio.gather(self._replay_task, return_exceptions=True)
        if self.method_manager is not None:
            loop = asyncio.get_running_loop()
            try:
                worker = await loop.run_in_executor(self._executor, self.method_manager.ensure, method_id)
            except Exception as exc:
                self.state.error = str(exc)
                await self._broadcast_json({"event": "error", "message": str(exc), **self.snapshot()})
                raise
            self.state.method_id = self.method_manager.active_method_id
            self.state.preset_id = str(worker.spec.get("preset_id", self.state.method_id))
        # The prediction consumer is shared by all backends.  Start it after a
        # method worker has been selected as well as for the legacy in-process
        # Hybrid path.
        await self.ensure_worker()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._stitcher.reset()
        self._session_tracks.clear()
        self._last_counters = self._stitcher.counters(window_candidates=0)
        self._next_prediction_start_s = start_s
        self.state = ReplayState(
            session_id=uuid.uuid4().hex[:12],
            cursor_s=start_s,
            total_duration_s=float(self.source.duration_s),
            window_s=self.window_s,
            stride_s=self.stride_s,
            waveform_downsample=self.waveform_downsample,
            playing=True,
            speed=self.state.speed if speed is None else speed,
            method_id=self.state.method_id,
            preset_id=self.state.preset_id,
        )
        self._replay_task = asyncio.create_task(self._replay_loop())
        await self._broadcast_json({"event": "status", **self.snapshot()})

    async def control(self, payload: dict[str, object]) -> None:
        action = str(payload.get("action", ""))
        if action == "pause":
            self.state.playing = False
        elif action == "resume" or action == "play":
            self.state.playing = True
            await self.ensure_worker()
            if self._replay_task is None or self._replay_task.done():
                self._replay_task = asyncio.create_task(self._replay_loop())
        elif action == "speed":
            value = payload.get("value", 1.0)
            self.state.speed = "fastest" if value == "fastest" else max(0.01, float(value))
        elif action == "seek":
            await self.start(float(payload.get("value", 0.0)))
            return
        elif action == "select_method":
            await self.start(0.0, method_id=str(payload.get("method_id", "")))
            return
        elif action == "configure":
            await self.start(
                float(payload.get("start_s", 0.0)),
                speed=payload.get("speed"),
                window_s=float(payload["window_s"]) if payload.get("window_s") is not None else None,
                stride_s=float(payload["stride_s"]) if payload.get("stride_s") is not None else None,
                waveform_downsample=int(payload["waveform_downsample"]) if payload.get("waveform_downsample") is not None else None,
                method_id=str(payload["method_id"]) if payload.get("method_id") is not None else None,
            )
            return
        elif action == "restart":
            await self.start(0.0, method_id=str(payload["method_id"]) if payload.get("method_id") is not None else None)
            return
        else:
            raise ValueError(f"unknown replay action: {action!r}")
        await self._broadcast_json({"event": "status", **self.snapshot()})

    async def _replay_loop(self) -> None:
        try:
            while self.state.cursor_s < self.source.duration_s - 1e-6:
                if not self.state.playing:
                    await asyncio.sleep(0.05)
                    continue
                start_s = float(self.state.cursor_s)
                duration_s = min(1.0, self.source.duration_s - start_s)
                await self._broadcast_frame(self.source.frame(start_s, duration_s))
                self.state.cursor_s = min(self.source.duration_s, start_s + duration_s)
                self.state.buffer_start_s = max(0.0, self.state.cursor_s - self.window_s)
                self.state.buffer_end_s = self.state.cursor_s
                self.state.buffered_s = min(self.window_s, self.state.cursor_s)
                while self.state.cursor_s >= self._next_prediction_start_s + self.window_s - 1e-6:
                    prediction_start = float(self._next_prediction_start_s)
                    self._next_prediction_start_s += self.stride_s
                    if prediction_start + self.window_s <= self.source.duration_s + 1e-6:
                        await self._queue.put((self.state.session_id, prediction_start))
                        self.state.prediction_queue = self._queue.qsize()
                await self._broadcast_json({"event": "status", **self.snapshot()})
                if self.state.cursor_s >= self.source.duration_s - 1e-6:
                    self.state.playing = False
                    await self._broadcast_json({"event": "end", **self.snapshot()})
                    break
                speed = self.state.speed
                delay = 0.0 if speed == "fastest" else 1.0 / float(speed)
                if delay:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.error = str(exc)
            self.state.playing = False
            await self._broadcast_json({"event": "error", "message": str(exc), **self.snapshot()})

    async def _prediction_worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            session_id, start_s = item
            try:
                if session_id != self.state.session_id:
                    continue
                started = time.perf_counter()
                batch = await loop.run_in_executor(self._executor, self._predict, start_s)
                elapsed = time.perf_counter() - started
                if session_id != self.state.session_id:
                    continue
                tracks = self._stitcher.update(batch)
                for item in tracks:
                    old = self._session_tracks.get(item.global_vehicle_id)
                    if old is None:
                        self._session_tracks[item.global_vehicle_id] = item
                    else:
                        point_map = {
                            (int(point.get("channel_index", -1)), round(float(point.get("time_s", 0.0)), 3)): point
                            for point in old.points
                        }
                        point_map.update({
                            (int(point.get("channel_index", -1)), round(float(point.get("time_s", 0.0)), 3)): point
                            for point in item.points
                        })
                        item.points = sorted(point_map.values(), key=lambda point: (float(point.get("time_s", 0.0)), int(point.get("channel_index", -1))))
                        item.observed_count = sum(1 for point in item.points if point.get("observed", False))
                        item.span_m = float(max((point.get("position_m", 0.0) for point in item.points), default=0.0) - min((point.get("position_m", 0.0) for point in item.points), default=0.0))
                        self._session_tracks[item.global_vehicle_id] = item
                counters = self._stitcher.counters(window_candidates=len(batch.tracks))
                self._last_counters = counters
                self.state.last_prediction_s = start_s
                self.state.last_inference_s = elapsed
                self.state.prediction_queue = self._queue.qsize()
                await self._broadcast_json(
                    {
                        "event": "prediction",
                        "session_id": session_id,
                        "window_start_s": start_s,
                        "window_end_s": start_s + self.window_s,
                        "elapsed_s": elapsed,
                        "method_id": self.state.method_id,
                        "preset_id": self.state.preset_id,
                        "checkpoint": (self.method_manager.specs.get(self.state.method_id, {}).get("checkpoint") if self.method_manager is not None else None),
                        "worker_version": "method-worker/v1",
                        "tracks": [item.to_dict() for item in tracks],
                        "counters": counters,
                        "diagnostics": batch.diagnostics,
                    }
                )
                await self._broadcast_json({"event": "vehicle_update", "session_id": session_id, "method_id": self.state.method_id, "preset_id": self.state.preset_id, "worker_version": "method-worker/v1", "tracks": [item.to_dict() for item in tracks], "counters": counters})
            except Exception as exc:
                self.state.error = str(exc)
                await self._broadcast_json({"event": "error", "message": str(exc), "window_start_s": start_s})
            finally:
                self._queue.task_done()

    def _predict(self, start_s: float) -> TrackBatch:
        if self.method_manager is not None:
            return response_to_batch(self.method_manager.predict(start_s=start_s, duration_s=self.window_s), station_count=getattr(self.source, "station_count", None))  # type: ignore[return-value]
        if self.tracker is None:
            raise WorkerError("no tracker or method manager configured")
        raw = getattr(self.source, "raw", None)
        pre = getattr(self.source, "pre", None)
        gauss = getattr(self.source, "gauss", None)
        if raw is None or pre is None or gauss is None:
            chunk = self.source.read(start_s, self.window_s)
            raw, pre, gauss = chunk.raw, chunk.pre, chunk.gauss
        return self.tracker.predict(
            raw,
            pre,
            gauss,
            self.mapping,
            start_s=start_s,
            duration_s=self.window_s,
        )

    async def export(self) -> dict[str, object]:
        output_root = self.output_root
        if self.method_manager is not None:
            configured = self.method_manager.specs.get(self.state.method_id, {}).get("output_root")
            if configured:
                output_root = Path(configured)
                if not output_root.is_absolute():
                    output_root = self.method_manager.root / output_root
        session_dir = output_root / self.state.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = session_dir / "tracks.jsonl"
        csv_path = session_dir / "track_points.csv"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for item in self._session_tracks.values():
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n")
        fields = ["global_vehicle_id", "source_window_start_s", "source_track_id", "channel_index", "station_id", "position_m", "time_s", "observed", "observation_id", "residual_s", "ambiguous"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in self._session_tracks.values():
                for point in item.points:
                    writer.writerow({"global_vehicle_id": item.global_vehicle_id, "source_window_start_s": item.source_window_start_s, "source_track_id": item.source_track_id, **{key: point.get(key) for key in fields[3:]}})
        manifest_path = session_dir / "session.json"
        method_spec = self.method_manager.specs.get(self.state.method_id, {}) if self.method_manager is not None else {}
        manifest_path.write_text(
            json.dumps(
                {
                    "session_id": self.state.session_id,
                    "method_id": self.state.method_id,
                    "preset_id": self.state.preset_id,
                    "checkpoint": method_spec.get("checkpoint"),
                    "protocol": "method-worker/v1",
                    "input_catalog": f"methods/{self.state.method_id}/data/catalog.yaml",
                    "cache_manifest": str(getattr(self.source, "manifest_path", "")),
                    "window_s": self.window_s,
                    "stride_s": self.stride_s,
                    "track_count": len(self._session_tracks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"session_id": self.state.session_id, "method_id": self.state.method_id, "preset_id": self.state.preset_id, "tracks_jsonl": str(jsonl_path), "track_points_csv": str(csv_path), "manifest": str(manifest_path), "track_count": len(self._session_tracks)}
