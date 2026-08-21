"""JSON-lines worker for the CPU classic trackers and the GPU graph tracker."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from auto_track_common.mapping import load_station_geometry
from auto_track_common.protocol import PROTOCOL, json_ready
from vehicle_replay_web.separation import normalize_separation_config, separate_large_vehicle_signal


def _window(array: np.ndarray, start_s: float, duration_s: float, fs: float) -> np.ndarray:
    begin = int(round(float(start_s) * fs))
    count = int(round(float(duration_s) * fs))
    end = begin + count
    if begin < 0 or end > int(array.shape[0]):
        raise ValueError(f"window [{start_s}, {start_s + duration_s}) is outside input array")
    return np.asarray(array[begin:end], dtype=np.float32)


def _track_dict(track: Any, geometry: Any) -> dict[str, Any]:
    points = []
    for point in list(getattr(track, "points", [])):
        channel = int(getattr(point, "channel_index", getattr(point, "ch_idx", 0)))
        if channel < 0 or channel >= len(geometry):
            raise ValueError(f"invalid worker point channel: {channel}")
        points.append(
            {
                "channel_index": channel,
                "station_id": str(geometry.stations[channel].station_id),
                "position_m": float(getattr(point, "position_m", getattr(point, "offset_m", geometry.positions_m[channel]))),
                "time_s": float(getattr(point, "time_s", 0.0)),
                "observed": bool(getattr(point, "observed", True)),
                "score": getattr(point, "score", None),
                "observation_id": getattr(point, "observation_id", None),
                "residual_s": getattr(point, "residual_s", None),
                "ambiguous": bool(getattr(point, "ambiguous", False)),
            }
        )
    speed = getattr(track, "median_speed_kmh", getattr(track, "mean_speed_kmh", None))
    return {
        "track_id": str(getattr(track, "track_id", "0")),
        "direction": str(getattr(track, "direction", "reverse")),
        "median_speed_kmh": float(speed) if speed is not None else None,
        "confidence": getattr(track, "confidence", None),
        "score": getattr(track, "score", getattr(track, "total_score", None)),
        "points": points,
    }


class ClassicMethodRunner:
    def __init__(self, init: dict[str, Any]) -> None:
        self.method_id = str(init["method_id"])
        self.preset_id = str(init.get("preset_id", self.method_id))
        self.device = str(init.get("device", "cpu"))
        source = dict(init["source"])
        self.fs = float(source.get("sample_rate_hz", 1000.0))
        self.raw = np.load(str(source["raw_path"]), mmap_mode="r") if source.get("raw_path") else None
        self.pre = np.load(str(source["pre_path"]), mmap_mode="r") if source.get("pre_path") else None
        self.prediction = np.load(str(source["prediction_path"]), mmap_mode="r") if source.get("prediction_path") else None
        self.gauss = np.load(str(source["gauss_path"]), mmap_mode="r")
        self.separation_config = normalize_separation_config(dict(init.get("separation", {})))
        self.geometry = load_station_geometry(init["mapping_path"])
        self.config_path = str(init.get("config", ""))
        self.config = self._load_config()
        self.tracker: Any = None
        with contextlib.redirect_stdout(sys.stderr):
            self._load_method()

    def _load_config(self) -> dict[str, Any]:
        if not self.config_path:
            return {}
        path = Path(self.config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"classic method config not found: {path}")
        return dict(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    def _load_method(self) -> None:
        if self.method_id == "hungarian_assignment":
            if self.device != "cpu":
                raise RuntimeError("hungarian_assignment requires device=cpu")
            from hungarian_assignment_tracker import HungarianAssignmentTracker

            self.tracker = HungarianAssignmentTracker(self.config)
            return
        if self.method_id == "kalman_seed":
            if self.device != "cpu":
                raise RuntimeError("kalman_seed requires device=cpu")
            from kalman_seed_tracker import KalmanVehicleTracker

            self.tracker = KalmanVehicleTracker(self.config)
            return
        if self.method_id == "graph_search":
            if self.device != "cuda":
                raise RuntimeError("graph_search requires device=cuda")
            import cupy

            if int(cupy.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("graph_search requires a CUDA device (CuPy)")
            return
        raise ValueError(f"unknown classic method: {self.method_id}")

    def _predict_graph(self, gauss: np.ndarray, duration_s: float, request_id: str) -> dict[str, Any]:
        from graph_search_tracker.auto_track_gpu import extract_all_gpu
        from graph_search_tracker.track_extractor_graph import ExtractorConfig, Track, TrackPoint

        positions = self.geometry.positions_m
        cfg = ExtractorConfig(
            prominence=float(self.config.get("prominence", 0.4)),
            min_peak_distance=int(round(float(self.config.get("min_distance_s", 0.5)) * self.fs)),
            min_track_channels=int(self.config.get("min_observed_points", 12)),
            edge_min_track_channels=int(self.config.get("edge_min_observed_points", 4)),
            max_skip_channels=8,
            max_tracks=256,
        )
        graph_axis = positions[-1] - positions[::-1]
        with contextlib.redirect_stdout(sys.stderr):
            decoded = extract_all_gpu(
                gauss.T[::-1],
                fs=self.fs,
                dx_m=float(np.median(np.diff(positions))),
                direction="forward",
                vmin_kmh=float(self.config.get("speed_min_kmh", 60.0)),
                vmax_kmh=float(self.config.get("speed_max_kmh", 90.0)),
                config=cfg,
                x_axis_m=graph_axis,
            )
        tracks = []
        for item in decoded:
            points = []
            for point in item.points:
                channel = len(positions) - 1 - int(point.ch_idx)
                points.append(
                    TrackPoint(
                        ch_idx=channel,
                        t_idx=int(point.t_idx),
                        time_s=float(point.time_s),
                        offset_m=float(positions[channel]),
                        amp=float(point.amp),
                        score=float(point.score),
                    )
                )
            tracks.append(
                _track_dict(
                    Track(
                        track_id=item.track_id,
                        direction="reverse",
                        points=points,
                        total_score=item.total_score,
                        mean_speed_kmh=item.mean_speed_kmh,
                    ),
                    self.geometry,
                )
            )
        diagnostics = {
            "extractor": "classic_peak_graph_gpu",
            "engine": "cupy_gpu",
            "direction": "reverse",
            "track_count": len(tracks),
        }
        return {
            "protocol": PROTOCOL,
            "event": "prediction",
            "method_id": self.method_id,
            "preset_id": self.preset_id,
            "worker_version": PROTOCOL,
            "request_id": request_id,
            "start_s": 0.0,
            "duration_s": duration_s,
            "tracks": tracks,
            "diagnostics": diagnostics,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        start_s = float(payload.get("start_s", 0.0))
        duration_s = float(payload.get("duration_s", 120.0))
        request_id = str(payload.get("request_id", ""))
        raw = _window(self.raw, start_s, duration_s, self.fs) if self.raw is not None else None
        pre = _window(self.pre, start_s, duration_s, self.fs) if self.pre is not None else None
        prediction = _window(self.prediction, start_s, duration_s, self.fs) if self.prediction is not None else pre
        gauss = _window(self.gauss, start_s, duration_s, self.fs)
        gauss, separation_diagnostics = separate_large_vehicle_signal(
            raw, prediction, gauss, self.fs, self.separation_config,
        )
        if self.method_id == "graph_search":
            result = self._predict_graph(gauss, duration_s, request_id)
            result["start_s"] = start_s
            result.setdefault("diagnostics", {})["large_vehicle_separation"] = separation_diagnostics
            return result
        batch = self.tracker.predict_window(
            gauss,
            sample_rate_hz=self.fs,
            geometry=self.geometry,
            start_s=0.0,
            duration_s=duration_s,
            request_id=request_id,
        )
        return {
            "protocol": PROTOCOL,
            "event": "prediction",
            "method_id": self.method_id,
            "preset_id": self.preset_id,
            "worker_version": PROTOCOL,
            "request_id": request_id,
            "start_s": start_s,
            "duration_s": duration_s,
            "tracks": [_track_dict(track, self.geometry) for track in batch.tracks],
            "diagnostics": {**dict(batch.diagnostics or {}), "large_vehicle_separation": separation_diagnostics},
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.parse_args()
    runner: ClassicMethodRunner | None = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            operation = payload.get("op")
            if operation == "init":
                runner = ClassicMethodRunner(payload)
                output = {"protocol": PROTOCOL, "event": "ready", "method_id": runner.method_id, "preset_id": runner.preset_id, "device": runner.device}
            elif operation == "predict":
                if runner is None:
                    raise RuntimeError("worker is not initialized")
                output = runner.predict(payload)
            elif operation == "close":
                output = {"protocol": PROTOCOL, "event": "closed"}
                print(json.dumps(output, separators=(",", ":")), flush=True)
                return 0
            else:
                raise ValueError(f"unknown worker operation: {operation!r}")
            print(json.dumps(json_ready(output), ensure_ascii=False, separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps({"protocol": PROTOCOL, "event": "error", "error": str(exc), "traceback": traceback.format_exc(limit=6)}, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
