"""Deep-learning trajectory extraction adapter for AutoTrackBackend.

This file exposes an `extract_all_deep_learning(...)` function with the same
call shape as the legacy peak/DP extractors, so the GUI/backend can select it
as another extraction engine.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from autotrack.core.track_extractor_graph import Track
from autotrack.dl.trajectory_set_model import (
    InferenceConfig,
    auto_torch_device,
    load_checkpoint_model,
    predict_tracks_from_window,
)


_MODEL_CACHE: dict[tuple[str, str], tuple[object, dict]] = {}


def _load_cached_model(model_path: str, device: Optional[str]):
    path = str(Path(model_path).expanduser().resolve())
    raw_device = str(device or "").strip()
    resolved_device = auto_torch_device() if raw_device in {"", "auto", "None"} else raw_device
    key = (path, resolved_device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model, checkpoint = load_checkpoint_model(path, device=resolved_device)
    _MODEL_CACHE[key] = (model, checkpoint)
    return model, checkpoint


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
    del direction, vmin_kmh, vmax_kmh
    cfg = dict(config or {})
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Deep-learning engine requires model_path in config")

    resolved_device = _resolve_device(cfg.get("device"))
    model, checkpoint = _load_cached_model(model_path, resolved_device)
    dataset_cfg = dict(checkpoint.get("dataset_config", {}))
    inference_cfg = InferenceConfig(
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
    arr = np.asarray(data, dtype=np.float32)
    x_axis_m = np.arange(arr.shape[0], dtype=np.float64) * float(dx_m)
    return predict_tracks_from_window(
        model=model,
        data_window=arr,
        fs=float(fs),
        x_axis_m=x_axis_m,
        config=inference_cfg,
        device=resolved_device,
    )
