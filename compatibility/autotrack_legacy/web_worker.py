"""Minimal root-environment worker for PeakSlot, Peak-Set and graph search.

It deliberately avoids importing ``hybrid_vehicle_tracker`` so the root Torch
environment can remain independent from the Hybrid environment's dependencies.
"""
from __future__ import annotations

import argparse, contextlib, json, re, sys, traceback
from pathlib import Path
from typing import Any
import numpy as np

PROTOCOL = "method-worker/v1"
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _ready(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_ready(v) for v in value]
    if isinstance(value, np.ndarray): return [_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value

def _mapping(path: str | Path) -> tuple[np.ndarray, list[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = sorted(payload.get("selected_channels", []), key=lambda x: int(x["channel_index"]))
    positions, station_ids = [], []
    for row in rows:
        match = _NUMBER.search(str(row.get("location", "")))
        if match is None: raise ValueError(f"mapping location missing: {row!r}")
        positions.append(float(match.group(0)) * 1000.0)
        station_ids.append(str(row.get("station_id", row["channel_index"])).upper())
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 1 or len(positions) == 0 or np.any(np.diff(positions) <= 0): raise ValueError("mapping positions must increase")
    return positions, station_ids

def _window(array: np.ndarray, start_s: float, duration_s: float, fs: float) -> np.ndarray:
    begin, count = int(round(start_s * fs)), int(round(duration_s * fs))
    end = begin + count
    if begin < 0 or end > array.shape[0]: raise ValueError("window outside input array")
    return np.asarray(array[begin:end], dtype=np.float32)

def _track(track: Any, positions: np.ndarray, station_ids: list[str], observed: Any = True, confidence: Any = None) -> dict[str, Any]:
    points = list(getattr(track, "points", []))
    flags = observed.astype(bool).tolist() if isinstance(observed, np.ndarray) else ([bool(v) for v in observed] if isinstance(observed, (list, tuple)) else [bool(observed)] * len(points))
    rows = []
    for i, point in enumerate(points):
        ch = int(getattr(point, "channel_index", getattr(point, "ch_idx", 0)))
        if ch < 0 or ch >= len(positions): raise ValueError(f"invalid worker point channel: {ch}")
        point_time = float(getattr(point, "time_s", 0.0))
        if not np.isfinite(point_time): raise ValueError(f"invalid worker point time: {point_time}")
        rows.append({"channel_index": ch, "station_id": station_ids[ch] if 0 <= ch < len(station_ids) else str(ch), "position_m": float(getattr(point, "position_m", getattr(point, "offset_m", positions[ch]))), "time_s": float(getattr(point, "time_s", 0.0)), "observed": flags[i] if i < len(flags) else True, "score": _ready(getattr(point, "score", None))})
    speed = getattr(track, "median_speed_kmh", getattr(track, "mean_speed_kmh", None))
    return {"track_id": str(getattr(track, "track_id", len(rows))), "direction": str(getattr(track, "direction", "unknown")), "median_speed_kmh": _ready(float(speed)) if speed is not None else None, "confidence": _ready(confidence), "score": _ready(getattr(track, "total_score", getattr(track, "score", None))), "points": rows}

def _diag(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict): return {}
    allowed = {"model_family", "decoder", "extractor", "engine", "track_count", "decoded_tracks", "quality_profile", "candidate_source", "raw_candidate_count", "prior_candidate_count", "merged_candidate_count", "final_candidate_count"}
    return {k: (_ready(v) if isinstance(v, (str, int, float, bool, type(None), list, tuple)) and (not isinstance(v, (list, tuple)) or len(v) <= 32) else None) for k, v in value.items() if k in allowed}

class MethodRunner:
    def __init__(self, payload: dict[str, Any]):
        self.method_id, self.preset_id = str(payload["method_id"]), str(payload.get("preset_id", payload["method_id"]))
        src = payload["source"]; self.fs = float(src.get("sample_rate_hz", 1000.0))
        self.raw = np.load(str(src["raw_path"]), mmap_mode="r") if src.get("raw_path") else None
        self.pre = np.load(str(src["pre_path"]), mmap_mode="r") if src.get("pre_path") else None
        self.gauss = np.load(str(src["gauss_path"]), mmap_mode="r")
        self.positions, self.station_ids = _mapping(payload["mapping_path"]); self.device = str(payload.get("device", "cuda")); self.model = None
        if self.device != "cuda":
            raise RuntimeError(f"{self.method_id} requires CUDA; refusing device={self.device!r}")
        if self.method_id == "graph_search":
            import cupy
            if int(cupy.cuda.runtime.getDeviceCount()) < 1:
                raise RuntimeError("graph_search requires a CUDA device (CuPy)")
        else:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f"{self.method_id} requires a CUDA-capable Torch runtime")
        self.checkpoint = str(payload.get("checkpoint", ""))
        with contextlib.redirect_stdout(sys.stderr): self._load()
    def _load(self):
        if self.method_id == "peak_slot":
            from peak_slot_tracker.peak_slot_model import load_checkpoint_model
            self.model, _ = load_checkpoint_model(self.checkpoint, device=self.device)
        elif self.method_id == "vehicle_peak_set":
            from vehicle_peak_set_tracker.vehicle_peak_set_transformer import load_checkpoint_model
            self.model, _ = load_checkpoint_model(self.checkpoint, device=self.device)
        elif self.method_id != "graph_search": raise ValueError(self.method_id)
    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        start, duration = float(payload.get("start_s", 0.0)), float(payload.get("duration_s", 120.0)); gauss = _window(self.gauss, start, duration, self.fs)
        if self.method_id == "peak_slot":
            from peak_slot_tracker.peak_slot_model import InferenceConfig, predict_tracks_from_window
            cfg = InferenceConfig(time_downsample=10, objectness_threshold=.45, peak_threshold=.4, min_visible_channels=4, max_tracks=96, extra_candidate_slots=8, candidate_objectness_floor=.2, viterbi_speed_min_kmh=60., viterbi_speed_max_kmh=90., decoder_mode="beam_global")
            with contextlib.redirect_stdout(sys.stderr): result = predict_tracks_from_window(self.model, gauss.T, fs=self.fs, x_axis_m=self.positions, config=cfg, device=self.device, return_diagnostics=True)
            decoded, diagnostics = result; diagnostics = _diag(diagnostics); tracks = [_track(item, self.positions, self.station_ids) for item in decoded if str(getattr(item, "direction", "")) == "reverse"]
        elif self.method_id == "vehicle_peak_set":
            from vehicle_peak_set_tracker.vehicle_peak_set_transformer import PeakSetInferenceConfig, decode_peak_guided_vehicle_tracks, decode_vehicle_peak_tracks
            cfg = PeakSetInferenceConfig(objectness_threshold=.25, complete_valid_threshold=.35, anchor_threshold=.45, min_visible_channels=5, max_tracks=24, dedup_tolerance_samples=30, speed_min_kmh=60., speed_max_kmh=90., graph_refine=False)
            with contextlib.redirect_stdout(sys.stderr): decoded = decode_peak_guided_vehicle_tracks(self.model, gauss.T, self.fs, self.positions, config=cfg, device=self.device) if type(self.model).__name__ == "PeakGuidedVehicleSetTransformer" else decode_vehicle_peak_tracks(self.model, gauss.T, self.fs, self.positions, config=cfg, device=self.device)
            tracks = [_track(item.track, self.positions, self.station_ids, getattr(item, "observed_valid", True), getattr(item, "objectness", None)) for item in decoded if str(getattr(item.track, "direction", "")) == "reverse"]; diagnostics = {"decoder": type(self.model).__name__, "decoded_tracks": len(decoded)}
        else:
            from graph_search_tracker.auto_track_gpu import extract_all_gpu
            from graph_search_tracker.track_extractor_graph import ExtractorConfig, Track, TrackPoint
            cfg = ExtractorConfig(prominence=.4, min_peak_distance=500, min_track_channels=12, edge_min_track_channels=4, max_tracks=256)
            # The original GPU/DP implementation advances through increasing
            # channel indices.  DAY11 vehicles move toward decreasing physical
            # mapping positions, so reverse only the algorithm boundary and
            # map every returned point back to its original channel.  The
            # transformed axis preserves real (possibly nonuniform) spacing.
            graph_axis = self.positions[-1] - self.positions[::-1]
            with contextlib.redirect_stdout(sys.stderr): decoded = extract_all_gpu(
                gauss.T[::-1],
                fs=self.fs,
                dx_m=float(np.median(np.diff(self.positions))),
                direction="forward",
                vmin_kmh=60.,
                vmax_kmh=90.,
                config=cfg,
                x_axis_m=graph_axis,
            )
            mapped = []
            for item in decoded:
                points = []
                for point in item.points:
                    original_channel = len(self.positions) - 1 - int(point.ch_idx)
                    points.append(TrackPoint(
                        ch_idx=original_channel,
                        t_idx=int(point.t_idx),
                        time_s=float(point.time_s),
                        offset_m=float(self.positions[original_channel]),
                        amp=float(point.amp),
                        score=float(point.score),
                    ))
                mapped.append(Track(track_id=item.track_id, direction="reverse", points=points, total_score=item.total_score, mean_speed_kmh=item.mean_speed_kmh))
            tracks = [_track(item, self.positions, self.station_ids) for item in mapped]
            diagnostics = {"extractor": "classic_peak_graph_gpu", "track_count": len(tracks), "engine": "cupy_gpu", "direction": "reverse", "adapter": "reverse_channel_order_physical_axis"}
        return {"protocol": PROTOCOL, "event": "prediction", "method_id": self.method_id, "preset_id": self.preset_id, "checkpoint": self.checkpoint or None, "worker_version": PROTOCOL, "request_id": str(payload.get("request_id", "")), "start_s": start, "duration_s": duration, "tracks": _ready(tracks), "diagnostics": _ready(diagnostics)}

def main() -> int:
    runner = None
    for line in sys.stdin:
        try:
            payload = json.loads(line); op = payload.get("op")
            if op == "init": runner = MethodRunner(payload); out = {"protocol": PROTOCOL, "event": "ready", "method_id": runner.method_id, "preset_id": runner.preset_id, "device": runner.device}
            elif op == "predict": out = runner.predict(payload) if runner is not None else (_ for _ in ()).throw(RuntimeError("worker is not initialized"))
            elif op == "close": out = {"protocol": PROTOCOL, "event": "closed"}; print(json.dumps(out), flush=True); return 0
            else: raise ValueError(f"unknown operation: {op}")
            print(json.dumps(_ready(out), separators=(",", ":")), flush=True)
        except Exception as exc:
            print(json.dumps({"protocol": PROTOCOL, "event": "error", "error": str(exc), "traceback": traceback.format_exc(limit=6)}, separators=(",", ":")), flush=True)
    return 0

if __name__ == "__main__": raise SystemExit(main())
