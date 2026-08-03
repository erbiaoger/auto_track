"""Method registry and subprocess bridge used by the generic replay server."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
import numpy as np


PROTOCOL = "method-worker/v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


class WorkerError(RuntimeError):
    pass


@dataclass
class WorkerPoint:
    channel_index: int
    station_id: str
    position_m: float
    time_s: float
    observed: bool
    score: float | None = None
    observation_id: int | None = None
    residual_s: float | None = None
    ambiguous: bool = False


@dataclass
class WorkerTrack:
    track_id: str
    direction: str
    points: list[WorkerPoint]
    median_speed_kmh: float
    confidence: float | None
    score: float | None
    observed_count: int
    span_m: float
    max_gap_m: float
    enters_window: bool
    exits_window: bool
    ambiguous_crossing: bool


@dataclass
class WorkerBatch:
    tracks: list[WorkerTrack]
    observations: list[Any]
    start_s: float
    duration_s: float
    diagnostics: dict[str, Any]


def response_to_batch(payload: dict[str, Any], *, station_count: int | None = None) -> WorkerBatch:
    tracks: list[WorkerTrack] = []
    duration_s = float(payload.get("duration_s", 0.0))
    for raw_track in payload.get("tracks", []):
        points = [
            WorkerPoint(
                channel_index=int(item.get("channel_index", -1)),
                station_id=str(item.get("station_id", "")),
                position_m=float(item.get("position_m", 0.0)),
                time_s=float(item.get("time_s", 0.0)),
                observed=bool(item.get("observed", True)),
                score=float(item["score"]) if item.get("score") is not None else None,
            )
            for item in raw_track.get("points", [])
        ]
        if any(point.channel_index < 0 or (station_count is not None and point.channel_index >= station_count) or not np.isfinite(point.time_s) or point.time_s < -1e-6 or point.time_s > duration_s + 1e-6 for point in points):
            raise WorkerError("worker returned an invalid channel or out-of-window point")
        times = [p.time_s for p in points]
        positions = [p.position_m for p in points]
        speed = raw_track.get("median_speed_kmh")
        tracks.append(
            WorkerTrack(
                track_id=str(raw_track.get("track_id", len(tracks))),
                direction=str(raw_track.get("direction", "unknown")),
                points=points,
                median_speed_kmh=float(speed) if speed is not None else float("nan"),
                confidence=float(raw_track["confidence"]) if raw_track.get("confidence") is not None else None,
                score=float(raw_track["score"]) if raw_track.get("score") is not None else None,
                observed_count=sum(1 for p in points if p.observed),
                span_m=float(max(positions, default=0.0) - min(positions, default=0.0)),
                max_gap_m=float(max((b - a for a, b in zip(sorted(positions), sorted(positions)[1:])), default=0.0)),
                enters_window=bool(times and min(times) <= 0.01),
                exits_window=bool(times and max(times) >= duration_s - 0.01),
                ambiguous_crossing=any(p.ambiguous for p in points),
            )
        )
    return WorkerBatch(
        tracks=tracks,
        observations=[],
        start_s=float(payload.get("start_s", 0.0)),
        duration_s=float(payload.get("duration_s", 0.0)),
        diagnostics=dict(payload.get("diagnostics", {})),
    )


class MethodWorker:
    def __init__(self, spec: dict[str, Any], *, root: Path, source: Any, mapping_path: str | Path, device: str) -> None:
        self.spec = spec
        self.root = root
        self.method_id = str(spec["id"])
        # All method workers receive the same accelerator selection.  The
        # classic graph implementation has no checkpoint, but keeping the
        # selected CUDA device in its worker contract makes the service
        # consistent and leaves room for GPU graph kernels.
        self.device = str(device)
        self._lock = threading.Lock()
        # Do not call Path.resolve() for virtualenv executables: resolving the
        # symlink jumps to the base interpreter and drops that environment's
        # site-packages.
        interpreter_value = Path(str(spec.get("interpreter", "python"))).expanduser()
        interpreter = interpreter_value if interpreter_value.is_absolute() else root / interpreter_value
        if not interpreter.exists():
            raise WorkerError(f"interpreter not found: {interpreter}")
        workdir = _resolve(root, str(spec.get("workdir", ".")))
        env = os.environ.copy()
        pythonpath = [_resolve(root, item) for item in spec.get("pythonpath", [])]
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([str(item) for item in pythonpath] + ([existing] if existing else []))
        command = [str(interpreter), "-m", str(spec.get("worker_module", "hybrid_vehicle_tracker.web.worker")), "--method", self.method_id]
        self.process = subprocess.Popen(
            command,
            cwd=str(workdir),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
        )
        init = {
            "op": "init",
            "protocol": PROTOCOL,
            "method_id": self.method_id,
            "preset_id": spec.get("preset_id", self.method_id),
            "device": self.device,
            "checkpoint": str(_resolve(root, spec["checkpoint"])) if spec.get("checkpoint") else "",
            "config": str(_resolve(root, spec["config"])) if spec.get("config") else "",
            "mapping_path": str(Path(mapping_path).expanduser().resolve()),
            "source": {
                "raw_path": str(Path(source.raw_path).resolve()),
                "pre_path": str(Path(source.pre_path).resolve()),
                "gauss_path": str(Path(source.gauss_path).resolve()),
                "sample_rate_hz": float(source.sample_rate_hz),
            },
        }
        response = self._call(init, timeout=180.0)
        if response.get("event") != "ready":
            self.close()
            raise WorkerError(str(response.get("error", "worker did not become ready")))
        self.ready = response

    def _call(self, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        if self.process.poll() is not None:
            raise WorkerError(f"worker exited with code {self.process.returncode}")
        if self.process.stdin is None or self.process.stdout is None:
            raise WorkerError("worker pipes are unavailable")
        with self._lock:
            self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self.process.stdin.flush()
            deadline = time.monotonic() + float(timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WorkerError(f"worker timeout after {timeout:.1f}s")
                ready, _, _ = select.select([self.process.stdout], [], [], remaining)
                if not ready:
                    raise WorkerError(f"worker timeout after {timeout:.1f}s")
                line = self.process.stdout.readline()
                if not line:
                    raise WorkerError(f"worker exited with code {self.process.poll()}")
                response = json.loads(line)
                if response.get("protocol") != PROTOCOL:
                    raise WorkerError("worker protocol mismatch")
                if response.get("event") == "error":
                    raise WorkerError(str(response.get("error", "worker error")))
                return response

    def predict(self, *, start_s: float, duration_s: float) -> dict[str, Any]:
        return self._call(
            {
                "op": "predict",
                "protocol": PROTOCOL,
                "request_id": uuid.uuid4().hex,
                "start_s": float(start_s),
                "duration_s": float(duration_s),
            },
            timeout=max(300.0, float(duration_s) * 5.0),
        )

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            if self.process.stdin is not None:
                self.process.stdin.write(json.dumps({"op": "close", "protocol": PROTOCOL}) + "\n")
                self.process.stdin.flush()
            self.process.wait(timeout=5.0)
        except Exception:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()


class MethodManager:
    def __init__(self, *, source: Any, mapping_path: str | Path, device: str, registry_path: str | Path | None = None) -> None:
        self.root = _repo_root()
        self.source = source
        self.mapping_path = Path(mapping_path).expanduser().resolve()
        self.device = str(device)
        path = Path(registry_path) if registry_path else self.root / "web/backend/methods.yaml"
        if not path.is_absolute():
            path = (self.root / path).resolve()
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.specs: dict[str, dict[str, Any]] = {}
        for raw in payload.get("methods", []):
            spec = dict(raw)
            spec["worker_module"] = spec.get("worker_module", payload.get("worker_module", "hybrid_vehicle_tracker.web.worker"))
            self.specs[str(spec["id"])] = spec
        self.default_method = str(payload.get("default_method", "hybrid"))
        self.active_method_id = ""
        self.active: MethodWorker | None = None
        self.last_errors: dict[str, str] = {}

    def descriptors(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for method_id, spec in self.specs.items():
            available, reason = self._availability(spec)
            if method_id in self.last_errors:
                available, reason = False, self.last_errors[method_id]
            rows.append({
                "id": method_id,
                "method_id": method_id,
                "label": spec.get("label", method_id),
                "kind": spec.get("kind", "unknown"),
                "preset_id": spec.get("preset_id", method_id),
                "checkpoint": spec.get("checkpoint"),
                "version": f"{PROTOCOL}:{spec.get('preset_id', method_id)}",
                "available": bool(available),
                "reason": reason,
                "input": spec.get("input"),
                "active": method_id == self.active_method_id,
            })
        return rows

    def _availability(self, spec: dict[str, Any]) -> tuple[bool, str | None]:
        interpreter = _resolve(self.root, str(spec.get("interpreter", "python")))
        if not interpreter.exists():
            return False, f"interpreter missing: {interpreter}"
        for key in ("checkpoint", "config"):
            if spec.get(key) and not _resolve(self.root, str(spec[key])).exists():
                return False, f"{key} missing: {spec[key]}"
        if not self.mapping_path.exists():
            return False, f"mapping missing: {self.mapping_path}"
        return True, None

    def ensure(self, method_id: str | None = None) -> MethodWorker:
        selected = str(method_id or self.active_method_id or self.default_method)
        if selected not in self.specs:
            raise WorkerError(f"unknown method: {selected}")
        if self.active is not None and self.active_method_id == selected:
            return self.active
        spec = self.specs[selected]
        available, reason = self._availability(spec)
        if not available:
            raise WorkerError(reason or f"method unavailable: {selected}")
        old = self.active
        try:
            worker = MethodWorker(spec, root=self.root, source=self.source, mapping_path=self.mapping_path, device=self.device)
        except Exception as exc:
            self.last_errors[selected] = str(exc)
            raise
        self.active = worker
        self.active_method_id = selected
        self.last_errors.pop(selected, None)
        if old is not None:
            old.close()
        return worker

    def predict(self, *, start_s: float, duration_s: float) -> dict[str, Any]:
        worker = self.ensure()
        try:
            return worker.predict(start_s=start_s, duration_s=duration_s)
        except Exception as exc:
            self.last_errors[self.active_method_id] = str(exc)
            worker.close()
            self.active = None
            raise

    def close(self) -> None:
        if self.active is not None:
            self.active.close()
            self.active = None
        self.active_method_id = ""
