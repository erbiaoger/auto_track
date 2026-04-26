from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from track_extractor_graph import (
    ExtractorConfig,
    Track,
    _as_config,
    _extract_best_track,
    _suppress_nodes,
)


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("No usable PyTorch detected. Please install torch first.") from exc
    return torch, F


def _enhance_with_gaussian_templates_torch(
    abs_data: np.ndarray,
    fs: float,
    sigma_seconds: tuple[float, ...],
) -> np.ndarray:
    torch, F = _import_torch()
    if not torch.backends.mps.is_built():
        raise RuntimeError("Current PyTorch build does not include MPS backend support.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available on this system (requires macOS 12.3+ and supported hardware).")

    device = torch.device("mps")
    n_ch = int(abs_data.shape[0])
    x = torch.as_tensor(abs_data, dtype=torch.float32, device=device).unsqueeze(0)
    enhanced = torch.zeros_like(x)

    for sigma_s in sigma_seconds:
        sigma_samples = max(1.0, float(sigma_s) * float(fs))
        radius = int(max(1, round(4.0 * sigma_samples)))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (coords / float(sigma_samples)) ** 2)
        kernel = kernel / torch.clamp(kernel.sum(), min=1e-12)
        weight = kernel.view(1, 1, -1).repeat(n_ch, 1, 1)
        response = F.conv1d(x, weight, padding=radius, groups=n_ch)
        enhanced = torch.maximum(enhanced, response)

    out = enhanced.squeeze(0).to("cpu").numpy().astype(np.float32, copy=False)
    try:
        torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return out


def _build_nodes_torch_mps(
    data: np.ndarray,
    fs: float,
    config: ExtractorConfig,
) -> list[dict[str, np.ndarray]]:
    abs_data = np.abs(data).astype(np.float32, copy=False)
    decim = int(max(1, config.enhance_decimate))
    if decim > 1:
        abs_work = abs_data[:, ::decim]
        fs_work = float(fs) / float(decim)
        peak_distance = int(max(1, round(float(config.min_peak_distance) / float(decim))))
    else:
        abs_work = abs_data
        fs_work = float(fs)
        peak_distance = int(max(1, config.min_peak_distance))

    if bool(config.use_template_enhancement) and len(config.sigma_seconds) > 0:
        enhanced = _enhance_with_gaussian_templates_torch(abs_work, fs_work, config.sigma_seconds)
    else:
        enhanced = abs_work

    channel_nodes: list[dict[str, np.ndarray]] = []
    for ch in range(data.shape[0]):
        peaks, props = find_peaks(
            enhanced[ch],
            prominence=config.prominence,
            distance=peak_distance,
        )
        if peaks.size == 0:
            channel_nodes.append(
                {
                    "t": np.empty((0,), dtype=np.int32),
                    "amp": np.empty((0,), dtype=np.float32),
                    "score": np.empty((0,), dtype=np.float32),
                }
            )
            continue

        if decim > 1:
            peaks = np.minimum(
                peaks.astype(np.int64, copy=False) * int(decim),
                int(data.shape[1] - 1),
            ).astype(np.int32, copy=False)

        prominences = props.get("prominences", np.zeros(peaks.shape[0], dtype=np.float32)).astype(np.float32)
        amps = abs_data[ch, peaks].astype(np.float32, copy=False)
        amp_ref = float(np.median(amps) + 1e-6)
        amp_norm = amps / amp_ref
        node_score = prominences + 0.2 * amp_norm

        if peaks.size > config.max_peaks_per_channel:
            idx_top = np.argsort(node_score)[-config.max_peaks_per_channel :]
            peaks = peaks[idx_top]
            amps = amps[idx_top]
            node_score = node_score[idx_top]

        order = np.argsort(peaks)
        channel_nodes.append(
            {
                "t": peaks[order].astype(np.int32, copy=False),
                "amp": amps[order].astype(np.float32, copy=False),
                "score": node_score[order].astype(np.float32, copy=False),
            }
        )
    return channel_nodes


def extract_all_torch_mps(
    data: np.ndarray,
    fs: float,
    dx_m: float,
    direction: str,
    vmin_kmh: float,
    vmax_kmh: float,
    config: Optional[ExtractorConfig | dict] = None,
) -> list[Track]:
    cfg = _as_config(config)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("data must be a 2D array")
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if dx_m <= 0:
        raise ValueError("dx_m must be > 0")
    if vmin_kmh <= 0 or vmax_kmh <= 0:
        raise ValueError("speed range must be > 0")
    if vmin_kmh > vmax_kmh:
        raise ValueError("vmin_kmh cannot be greater than vmax_kmh")
    if direction not in {"forward", "reverse"}:
        raise ValueError("direction must be either forward or reverse")

    vmin_mps = float(vmin_kmh) / 3.6
    vmax_mps = float(vmax_kmh) / 3.6
    nodes = _build_nodes_torch_mps(arr, fs, cfg)

    tracks: list[Track] = []
    for tid in range(int(max(1, cfg.max_tracks))):
        best = _extract_best_track(
            nodes=nodes,
            fs=fs,
            dx_m=dx_m,
            direction=direction,
            vmin_mps=vmin_mps,
            vmax_mps=vmax_mps,
            n_samples=arr.shape[1],
            config=cfg,
            track_id=tid,
        )
        if best is None:
            break
        tracks.append(best)
        _suppress_nodes(nodes, best, cfg)

    return tracks
