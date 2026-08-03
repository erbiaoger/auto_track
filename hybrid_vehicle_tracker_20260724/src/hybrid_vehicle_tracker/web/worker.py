"""Isolated method worker for the multi-method replay service.

The worker speaks a deliberately small JSON-lines protocol.  It is launched
with the Python environment belonging to the selected method, so the web
coordinator never imports competing Torch stacks into one process.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


PROTOCOL = "method-worker/v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_mapping(path: str | Path) -> tuple[Any, np.ndarray, list[str]]:
    from hybrid_vehicle_tracker.data.mapping import load_station_geometry

    geometry = load_station_geometry(path)
    return geometry, np.asarray(geometry.positions_m, dtype=np.float64), [s.station_id for s in geometry.stations]


def _window(array: np.ndarray, start_s: float, duration_s: float, fs: float) -> np.ndarray:
    begin = int(round(float(start_s) * float(fs)))
    count = int(round(float(duration_s) * float(fs)))
    end = begin + count
    if begin < 0 or end > int(array.shape[0]):
        raise ValueError(f"window [{start_s}, {start_s + duration_s}) is outside input array")
    return np.asarray(array[begin:end], dtype=np.float32)


def _point_dict(point: Any, positions: np.ndarray, station_ids: list[str], *, observed: bool = True) -> dict[str, Any]:
    channel = int(getattr(point, "channel_index", getattr(point, "ch_idx", 0)))
    time_s = float(getattr(point, "time_s", 0.0))
    if channel < 0 or channel >= len(positions) or not np.isfinite(time_s):
        raise ValueError(f"invalid worker point channel/time: {channel}, {time_s}")
    position = float(getattr(point, "position_m", getattr(point, "offset_m", positions[channel])))
    if not np.isfinite(position):
        raise ValueError(f"invalid worker point position: {position}")
    return {
        "channel_index": channel,
        "station_id": station_ids[channel] if 0 <= channel < len(station_ids) else str(channel),
        "position_m": position,
        "time_s": time_s,
        "observed": bool(observed),
        "score": _json_ready(getattr(point, "score", None)),
    }


def _track_dict(track: Any, positions: np.ndarray, station_ids: list[str], *, observed: Any = True, confidence: Any = None) -> dict[str, Any]:
    points = list(getattr(track, "points", []))
    if isinstance(observed, np.ndarray):
        flags = observed.astype(bool).tolist()
    elif isinstance(observed, (list, tuple)):
        flags = [bool(v) for v in observed]
    else:
        flags = [bool(observed)] * len(points)
    point_rows = [_point_dict(point, positions, station_ids, observed=flags[i] if i < len(flags) else True) for i, point in enumerate(points)]
    speed = getattr(track, "median_speed_kmh", getattr(track, "mean_speed_kmh", None))
    return {
        "track_id": str(getattr(track, "track_id", len(point_rows))),
        "direction": str(getattr(track, "direction", "unknown")),
        "median_speed_kmh": _json_ready(float(speed)) if speed is not None else None,
        "confidence": _json_ready(confidence),
        "score": _json_ready(getattr(track, "total_score", getattr(track, "score", None))),
        "points": point_rows,
    }


def _diagnostic_summary(value: Any) -> dict[str, Any]:
    """Keep protocol diagnostics bounded; never stream model-sized arrays."""
    if not isinstance(value, dict):
        return {}
    allowed = {
        "model_family", "decoder", "extractor", "engine", "track_count", "decoded_tracks",
        "quality_profile", "candidate_source", "raw_candidate_count",
        "prior_candidate_count", "merged_candidate_count", "final_candidate_count",
        "checkpoint", "device", "direction", "motion_direction", "station_count",
    }
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key not in allowed:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            result[key] = item
        elif isinstance(item, (list, tuple)) and len(item) <= 32:
            result[key] = _json_ready(item)
    return result


class MethodRunner:
    def __init__(self, init: dict[str, Any]) -> None:
        self.method_id = str(init["method_id"])
        self.preset_id = str(init.get("preset_id", self.method_id))
        self.device = str(init.get("device", "cpu"))
        source = dict(init["source"])
        self.fs = float(source.get("sample_rate_hz", 1000.0))
        self.raw = np.load(str(source["raw_path"]), mmap_mode="r")
        self.pre = np.load(str(source["pre_path"]), mmap_mode="r")
        self.gauss = np.load(str(source["gauss_path"]), mmap_mode="r")
        self.mapping, self.positions, self.station_ids = _read_mapping(init["mapping_path"])
        self.mapping_path = str(Path(init["mapping_path"]).expanduser().resolve())
        self.checkpoint = str(init.get("checkpoint", ""))
        self.config_path = str(init.get("config", ""))
        self.model: Any = None
        self.config: Any = None
        with contextlib.redirect_stdout(sys.stderr):
            self._load_method()

    def _load_method(self) -> None:
        if self.method_id == "hybrid":
            from hybrid_vehicle_tracker.config import load_tracker_config
            from hybrid_vehicle_tracker.tracker import HybridVehicleTracker

            self.config = load_tracker_config(self.config_path)
            self.config.runtime.device = self.device
            self.config.data.mapping_path = self.mapping_path
            self.config.data.sample_rate_hz = self.fs
            self.config.data.duration_s = 120.0
            if self.checkpoint:
                self.config.model.checkpoint = self.checkpoint
            self.model = HybridVehicleTracker(self.config)
            return
        if self.method_id == "peak_slot":
            from autotrack.dl.peak_slot_model import load_checkpoint_model

            self.model, _ = load_checkpoint_model(self.checkpoint, device=self.device)
            return
        if self.method_id == "vehicle_peak_set":
            from autotrack.dl.vehicle_peak_set_transformer import load_checkpoint_model

            self.model, _ = load_checkpoint_model(self.checkpoint, device=self.device)
            return
        if self.method_id == "graph_search":
            return
        raise ValueError(f"unknown method: {self.method_id}")

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        start_s = float(payload.get("start_s", 0.0))
        duration_s = float(payload.get("duration_s", 120.0))
        raw = _window(self.raw, start_s, duration_s, self.fs)
        pre = _window(self.pre, start_s, duration_s, self.fs)
        gauss = _window(self.gauss, start_s, duration_s, self.fs)
        if self.method_id == "hybrid":
            batch = self.model.predict(raw, pre, gauss, self.mapping, start_s=0.0, duration_s=duration_s)
            tracks = [_track_dict(item, self.positions, self.station_ids, observed=True, confidence=getattr(item, "confidence", None)) for item in batch.tracks]
            diagnostics = _diagnostic_summary(batch.diagnostics)
        elif self.method_id == "peak_slot":
            from autotrack.dl.peak_slot_model import InferenceConfig, predict_tracks_from_window

            cfg = InferenceConfig(
                time_downsample=10,
                objectness_threshold=0.45,
                peak_threshold=0.4,
                min_visible_channels=4,
                max_tracks=96,
                extra_candidate_slots=8,
                candidate_objectness_floor=0.20,
                viterbi_speed_min_kmh=60.0,
                viterbi_speed_max_kmh=90.0,
                decoder_mode="beam_global",
            )
            with contextlib.redirect_stdout(sys.stderr):
                result = predict_tracks_from_window(
                    self.model,
                    gauss.T,
                    fs=self.fs,
                    x_axis_m=self.positions,
                    config=cfg,
                    device=self.device,
                    return_diagnostics=True,
                )
            model_tracks, diagnostics = result
            diagnostics = _diagnostic_summary(diagnostics)
            tracks = [_track_dict(item, self.positions, self.station_ids, observed=True) for item in model_tracks if str(getattr(item, "direction", "")) == "reverse"]
        elif self.method_id == "vehicle_peak_set":
            from autotrack.dl.vehicle_peak_set_transformer import (
                PeakSetInferenceConfig,
                decode_peak_guided_vehicle_tracks,
                decode_vehicle_peak_tracks,
            )

            cfg = PeakSetInferenceConfig(
                objectness_threshold=0.25,
                complete_valid_threshold=0.35,
                anchor_threshold=0.45,
                min_visible_channels=5,
                max_tracks=24,
                dedup_tolerance_samples=30,
                speed_min_kmh=60.0,
                speed_max_kmh=90.0,
                graph_refine=False,
            )
            with contextlib.redirect_stdout(sys.stderr):
                if type(self.model).__name__ == "PeakGuidedVehicleSetTransformer":
                    decoded = decode_peak_guided_vehicle_tracks(self.model, gauss.T, self.fs, self.positions, config=cfg, device=self.device)
                else:
                    decoded = decode_vehicle_peak_tracks(self.model, gauss.T, self.fs, self.positions, config=cfg, device=self.device)
            tracks = []
            for item in decoded:
                if str(getattr(item.track, "direction", "")) != "reverse":
                    continue
                tracks.append(_track_dict(item.track, self.positions, self.station_ids, observed=getattr(item, "observed_valid", True), confidence=getattr(item, "objectness", None)))
            diagnostics = {"decoder": type(self.model).__name__, "decoded_tracks": len(decoded), "quality_profile": "recall"}
        else:
            from autotrack.core.track_extractor_graph import ExtractorConfig, extract_all

            cfg = ExtractorConfig(
                prominence=0.4,
                min_peak_distance=500,
                min_track_channels=12,
                edge_min_track_channels=4,
                max_tracks=256,
            )
            with contextlib.redirect_stdout(sys.stderr):
                model_tracks = extract_all(
                    gauss.T,
                    fs=self.fs,
                    dx_m=float(np.median(np.diff(self.positions))),
                    direction="reverse",
                    vmin_kmh=60.0,
                    vmax_kmh=90.0,
                    config=cfg,
                    x_axis_m=self.positions,
                )
            tracks = [_track_dict(item, self.positions, self.station_ids, observed=True) for item in model_tracks]
            diagnostics = {"extractor": "classic_peak_graph", "track_count": len(tracks)}
        return {
            "protocol": PROTOCOL,
            "event": "prediction",
            "method_id": self.method_id,
            "preset_id": self.preset_id,
            "checkpoint": self.checkpoint or None,
            "worker_version": PROTOCOL,
            "request_id": str(payload.get("request_id", "")),
            "start_s": start_s,
            "duration_s": duration_s,
            "tracks": _json_ready(tracks),
            "diagnostics": _json_ready(diagnostics),
        }


def _write(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(_json_ready(payload), ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    runner: MethodRunner | None = None
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            op = payload.get("op")
            if op == "init":
                runner = MethodRunner(payload)
                _write({"protocol": PROTOCOL, "event": "ready", "method_id": runner.method_id, "preset_id": runner.preset_id, "device": runner.device})
            elif op == "predict":
                if runner is None:
                    raise RuntimeError("worker is not initialized")
                _write(runner.predict(payload))
            elif op == "close":
                _write({"protocol": PROTOCOL, "event": "closed"})
                return 0
            else:
                raise ValueError(f"unknown worker operation: {op!r}")
        except Exception as exc:
            _write({"protocol": PROTOCOL, "event": "error", "error": str(exc), "traceback": traceback.format_exc(limit=6)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
