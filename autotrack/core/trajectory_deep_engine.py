"""Deep-learning trajectory extraction adapter for AutoTrackBackend.

This file exposes an `extract_all_deep_learning(...)` function with the same
call shape as the legacy peak/DP extractors, so the GUI/backend can select it
as another extraction engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from autotrack.core.track_extractor_graph import Track
from autotrack.dl.trajectory_set_model import auto_torch_device


_MODEL_CACHE: dict[tuple[str, str, str], tuple[object, dict, str]] = {}


def _resolve_model_family(model_path: str, requested_family: Optional[str]) -> str:
    family = str(requested_family or "").strip().lower()
    if family in {"query_points", "query_masks", "track_slot", "peak_slot"}:
        return family
    checkpoint = torch.load(str(Path(model_path).expanduser()), map_location="cpu", weights_only=False)
    ckpt_family = str(checkpoint.get("model_family", "")).strip().lower()
    if ckpt_family in {"query_points", "query_masks", "track_slot", "peak_slot"}:
        return ckpt_family
    model_config = dict(checkpoint.get("model_config", {}))
    if "peak_candidates" in model_config:
        return "peak_slot"
    if "match_time_bins" in model_config:
        return "query_masks"
    if "max_tracks" in model_config:
        return "track_slot"
    return "query_points"


def _load_cached_model(model_path: str, device: Optional[str], model_family: str):
    path = str(Path(model_path).expanduser().resolve())
    raw_device = str(device or "").strip()
    resolved_device = auto_torch_device() if raw_device in {"", "auto", "None"} else raw_device
    key = (path, resolved_device, model_family)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    if model_family == "peak_slot":
        from autotrack.dl import peak_slot_model as pk

        model, checkpoint = pk.load_checkpoint_model(path, device=resolved_device)
    elif model_family == "track_slot":
        from autotrack.dl import track_slot_model as tm

        model, checkpoint = tm.load_checkpoint_model(path, device=resolved_device)
    elif model_family == "query_masks":
        from autotrack.dl import query_mask_instance_model as mm

        model, checkpoint = mm.load_checkpoint_model(path, device=resolved_device)
    else:
        from autotrack.dl import trajectory_set_model as pm

        model, checkpoint = pm.load_checkpoint_model(path, device=resolved_device)
    _MODEL_CACHE[key] = (model, checkpoint, model_family)
    return model, checkpoint, model_family


def _resolve_device(device: Optional[str]) -> str:
    raw_device = str(device or "").strip()
    return auto_torch_device() if raw_device in {"", "auto", "None"} else raw_device


def extract_all_deep_learning(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[dict] = None,
) -> list[Track]:
    requested_direction = str(direction).strip().lower()
    if requested_direction not in {"forward", "reverse"}:
        raise ValueError("direction must be either forward or reverse")
    cfg = dict(config or {})
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Deep-learning engine requires model_path in config")
    diagnostics_sink = cfg.get("_diagnostics_sink")

    resolved_device = _resolve_device(cfg.get("device"))
    model_family = _resolve_model_family(model_path, cfg.get("model_family"))
    model, checkpoint, model_family = _load_cached_model(model_path, resolved_device, model_family)
    dataset_cfg = dict(checkpoint.get("dataset_config", {}))
    trained_window_seconds = float(dict(checkpoint.get("dataset_meta", {})).get("window_seconds", 0.0) or 0.0)
    infer_window_seconds = float(np.asarray(data, dtype=np.float32).shape[1]) / float(fs)
    if model_family == "peak_slot":
        from autotrack.dl import peak_slot_model as pk

        peak_detection = dict(checkpoint.get("dataset_meta", {}).get("peak_detection", {}))
        inference_cfg = pk.InferenceConfig(
            time_downsample=int(cfg.get("time_downsample", dataset_cfg.get("time_downsample", 10))),
            objectness_threshold=float(cfg.get("objectness_threshold", 0.35)),
            peak_threshold=float(cfg.get("peak_threshold", 0.4)),
            min_visible_channels=int(cfg.get("min_visible_channels", 2)),
            max_tracks=int(cfg.get("max_tracks", 96)),
            dedup_tolerance_samples=int(cfg.get("dedup_tolerance_samples", 180)),
            speed_norm_kmh=float(dataset_cfg.get("speed_norm_kmh", 150.0)),
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
            peak_candidates=int(dict(checkpoint.get("model_config", {})).get("peak_candidates", 64)),
            peak_min_distance_s=float(cfg.get("peak_min_distance_s", 0.15)),
            peak_min_height=float(peak_detection.get("min_height", 0.02)),
            peak_prominence=float(peak_detection.get("prominence", 0.02)),
            use_viterbi_decoder=bool(cfg.get("use_viterbi_decoder", True)),
            decoder_mode=str(cfg.get("decoder_mode", "beam_global")),
            viterbi_beam_size=int(cfg.get("viterbi_beam_size", 8)),
            time_prior_weight=float(cfg.get("time_prior_weight", 2.0)),
            global_conflict_penalty=float(cfg.get("global_conflict_penalty", 2.0)),
            conflict_mode=str(cfg.get("conflict_mode", "soft")),
            duplicate_overlap_ratio=float(cfg.get("duplicate_overlap_ratio", 0.75)),
            min_unique_support_channels=int(cfg.get("min_unique_support_channels", 3)),
            extra_candidate_slots=int(cfg.get("extra_candidate_slots", 32)),
            candidate_objectness_floor=float(cfg.get("candidate_objectness_floor", 0.02)),
            viterbi_topk=int(cfg.get("viterbi_topk", 16)),
            viterbi_candidate_threshold=float(cfg.get("viterbi_candidate_threshold", 0.01)),
            viterbi_speed_min_kmh=float(cfg.get("viterbi_speed_min_kmh", vmin_kmh)),
            viterbi_speed_max_kmh=float(cfg.get("viterbi_speed_max_kmh", vmax_kmh)),
            viterbi_max_skip_channels=int(cfg.get("viterbi_max_skip_channels", 4)),
            viterbi_point_bonus=float(cfg.get("viterbi_point_bonus", 3.0)),
            viterbi_skip_penalty=float(cfg.get("viterbi_skip_penalty", 2.0)),
            viterbi_speed_penalty=float(cfg.get("viterbi_speed_penalty", 1.0)),
            viterbi_smoothness_penalty=float(cfg.get("viterbi_smoothness_penalty", 0.6)),
            viterbi_inertia_penalty=float(cfg.get("viterbi_inertia_penalty", 2.5)),
            viterbi_slope_memory=float(cfg.get("viterbi_slope_memory", 0.75)),
            viterbi_fallback_speed_kmh=float(cfg.get("viterbi_fallback_speed_kmh", 80.0)),
        )
        predict_fn = pk.predict_tracks_from_window
    elif model_family == "track_slot":
        from autotrack.dl import track_slot_model as tm

        inference_cfg = tm.InferenceConfig(
            time_downsample=int(cfg.get("time_downsample", dataset_cfg.get("time_downsample", 10))),
            objectness_threshold=float(cfg.get("objectness_threshold", 0.5)),
            visibility_threshold=float(cfg.get("visibility_threshold", 0.5)),
            min_visible_channels=int(cfg.get("min_visible_channels", 3)),
            refine_radius_samples=int(cfg.get("refine_radius_samples", 120)),
            max_tracks=int(cfg.get("max_tracks", 96)),
            dedup_tolerance_samples=int(cfg.get("dedup_tolerance_samples", 180)),
            speed_norm_kmh=float(dataset_cfg.get("speed_norm_kmh", 150.0)),
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
        )
        predict_fn = tm.predict_tracks_from_window
    elif model_family == "query_masks":
        from autotrack.dl import query_mask_instance_model as mm

        inference_cfg = mm.InferenceConfig(
            time_downsample=int(cfg.get("time_downsample", dataset_cfg.get("time_downsample", 10))),
            objectness_threshold=float(cfg.get("objectness_threshold", 0.5)),
            visibility_threshold=float(cfg.get("visibility_threshold", 0.5)),
            min_visible_channels=int(cfg.get("min_visible_channels", 3)),
            refine_radius_samples=int(cfg.get("refine_radius_samples", 120)),
            max_tracks=int(cfg.get("max_tracks", 128)),
            dedup_tolerance_samples=int(cfg.get("dedup_tolerance_samples", 180)),
            speed_norm_kmh=float(dataset_cfg.get("speed_norm_kmh", 150.0)),
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
        )
        predict_fn = mm.predict_tracks_from_window
    else:
        from autotrack.dl import trajectory_set_model as pm

        inference_cfg = pm.InferenceConfig(
            time_downsample=int(cfg.get("time_downsample", dataset_cfg.get("time_downsample", 10))),
            objectness_threshold=float(cfg.get("objectness_threshold", 0.5)),
            visibility_threshold=float(cfg.get("visibility_threshold", 0.5)),
            min_visible_channels=int(cfg.get("min_visible_channels", 3)),
            refine_radius_samples=int(cfg.get("refine_radius_samples", 120)),
            max_tracks=int(cfg.get("max_tracks", 128)),
            dedup_tolerance_samples=int(cfg.get("dedup_tolerance_samples", 180)),
            speed_norm_kmh=float(dataset_cfg.get("speed_norm_kmh", 150.0)),
            clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
        )
        predict_fn = pm.predict_tracks_from_window
    arr = np.asarray(data, dtype=np.float32)
    x_axis_m = np.arange(arr.shape[0], dtype=np.float64) * float(dx_m)
    if model_family == "peak_slot" and isinstance(diagnostics_sink, dict):
        tracks, diagnostics = predict_fn(
            model=model,
            data_window=arr,
            fs=float(fs),
            x_axis_m=x_axis_m,
            config=inference_cfg,
            device=resolved_device,
            return_diagnostics=True,
        )
        diagnostics_sink.clear()
        diagnostics_sink.update(diagnostics)
        if trained_window_seconds > 0.0:
            window_ratio = infer_window_seconds / max(1e-6, trained_window_seconds)
            diagnostics_sink["trained_window_seconds"] = float(trained_window_seconds)
            diagnostics_sink["infer_window_seconds"] = float(infer_window_seconds)
            diagnostics_sink["window_seconds_ratio"] = float(window_ratio)
            if abs(window_ratio - 1.0) > 0.1:
                diagnostics_sink["warning_window_mismatch"] = (
                    f"Checkpoint trained for {trained_window_seconds:.1f}s windows, "
                    f"but inference used {infer_window_seconds:.1f}s "
                    f"(ratio={window_ratio:.2f}). Peak normalization, candidate detection, "
                    "and vehicle-count prior may all be miscalibrated."
                )
    else:
        tracks = predict_fn(
            model=model,
            data_window=arr,
            fs=float(fs),
            x_axis_m=x_axis_m,
            config=inference_cfg,
            device=resolved_device,
        )
        if isinstance(diagnostics_sink, dict):
            diagnostics_sink.clear()
    if model_family == "peak_slot" and str(cfg.get("fusion_mode", "off")).strip().lower() == "graph_extend":
        from autotrack.core.track_fusion import extend_peakslot_tracks_with_graph

        fusion_diagnostics: dict[str, object] = {}
        target_tracks = [
            track
            for track in tracks
            if str(getattr(track, "direction", "")).strip().lower() == requested_direction
        ]
        other_tracks = [
            track
            for track in tracks
            if str(getattr(track, "direction", "")).strip().lower() != requested_direction
        ]
        tracks = extend_peakslot_tracks_with_graph(
            data=arr,
            fs=float(fs),
            dx_m=float(dx_m),
            tracks=list(target_tracks),
            direction=requested_direction,
            vmin_kmh=float(vmin_kmh),
            vmax_kmh=float(vmax_kmh),
            config=cfg,
            diagnostics=fusion_diagnostics,
        )
        tracks = list(tracks) + list(other_tracks)
        if isinstance(diagnostics_sink, dict):
            diagnostics_sink.update(fusion_diagnostics)
    elif model_family == "peak_slot" and isinstance(diagnostics_sink, dict):
        diagnostics_sink.update(
            {
                "fusion_enabled": False,
                "fusion_input_track_count": int(len(tracks)),
                "fusion_output_track_count": int(len(tracks)),
                "fusion_added_point_count": 0,
                "fusion_tracks": [],
            }
        )
    if model_family == "peak_slot":
        from autotrack.core.boundary_completion import complete_tracks_to_boundaries

        boundary_diagnostics: dict[str, object] = {}
        target_tracks = [
            track
            for track in tracks
            if str(getattr(track, "direction", "")).strip().lower() == requested_direction
        ]
        other_tracks = [
            track
            for track in tracks
            if str(getattr(track, "direction", "")).strip().lower() != requested_direction
        ]
        tracks = complete_tracks_to_boundaries(
            data=arr,
            fs=float(fs),
            dx_m=float(dx_m),
            tracks=list(target_tracks),
            direction=requested_direction,
            vmin_kmh=float(vmin_kmh),
            vmax_kmh=float(vmax_kmh),
            config=cfg,
            diagnostics=boundary_diagnostics,
        )
        tracks = list(tracks) + list(other_tracks)
        if isinstance(diagnostics_sink, dict):
            diagnostics_sink.update(boundary_diagnostics)
    return [
        track
        for track in tracks
        if str(getattr(track, "direction", "")).strip().lower() == requested_direction
    ]
