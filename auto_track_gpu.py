from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import find_peaks

from auto_track_backend import AutoTrackBackend, DEFAULT_DATA_FOLDER
from track_extractor_graph import (
    ExtractorConfig,
    Track,
    _as_config,
    _extract_best_track,
    _suppress_nodes,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU trajectory extraction (CuPy enhancement + CPU dynamic programming)")
    parser.add_argument("--data-folder", type=str, default=DEFAULT_DATA_FOLDER, help="SAC folder")
    parser.add_argument("--out-csv", type=str, default="", help="Output CSV path; default is <data-folder>/auto_tracks_gpu.csv")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0)
    parser.add_argument("--speed-max-kmh", type=float, default=120.0)
    parser.add_argument("--prominence", type=float, default=0.4)
    parser.add_argument("--min-peak-distance", type=int, default=500)
    parser.add_argument("--min-track-channels", type=int, default=12)
    parser.add_argument("--tile-seconds", type=float, default=120.0)
    parser.add_argument("--overlap-seconds", type=float, default=20.0)
    parser.add_argument("--nms-time-radius", type=float, default=0.18, help="seconds")
    parser.add_argument("--window-seconds", type=float, default=240.0, help="Only used in current-window mode")
    parser.add_argument("--window-start-s", type=float, default=0.0, help="Only used in current-window mode")
    parser.add_argument("--current-window-only", action="store_true", help="Extract current window only")
    parser.add_argument(
        "--enhance-decimate",
        type=int,
        default=0,
        help="Enhancement-stage decimation factor; <=0 means auto (recommended)",
    )
    parser.add_argument("--device-id", type=int, default=0, help="CUDA device ID")
    return parser.parse_args()


def _import_cupy():
    try:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter1d as cp_gaussian_filter1d
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No usable CuPy detected. Install a CuPy build matching your CUDA version, e.g. cupy-cuda12x."
        ) from exc
    return cp, cp_gaussian_filter1d


def _enhance_with_gaussian_templates_gpu(
    abs_data: np.ndarray,
    fs: float,
    sigma_seconds: tuple[float, ...],
) -> np.ndarray:
    cp, cp_gaussian_filter1d = _import_cupy()
    abs_cp = cp.asarray(abs_data, dtype=cp.float32)
    enhanced_cp = cp.zeros_like(abs_cp)
    for sigma_s in sigma_seconds:
        sigma_samples = max(1.0, float(sigma_s) * float(fs))
        response = cp_gaussian_filter1d(abs_cp, sigma=sigma_samples, axis=1, mode="nearest")
        cp.maximum(enhanced_cp, response, out=enhanced_cp)
    enhanced = cp.asnumpy(enhanced_cp)
    del abs_cp, enhanced_cp
    cp.get_default_memory_pool().free_all_blocks()
    return enhanced.astype(np.float32, copy=False)


def _build_nodes_gpu(
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
        enhanced = _enhance_with_gaussian_templates_gpu(abs_work, fs_work, config.sigma_seconds)
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


def extract_all_gpu(
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

    nodes = _build_nodes_gpu(arr, fs, cfg)

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


def _reindex_tracks(tracks: list[Track]) -> list[Track]:
    out = []
    for i, tr in enumerate(tracks):
        out.append(
            Track(
                track_id=i,
                direction=tr.direction,
                points=tr.points,
                total_score=tr.total_score,
                mean_speed_kmh=tr.mean_speed_kmh,
            )
        )
    return out


def _run_gpu_extract(args: argparse.Namespace) -> int:
    cp, _ = _import_cupy()
    n_dev = int(cp.cuda.runtime.getDeviceCount())
    if n_dev <= 0:
        raise RuntimeError("No CUDA devices detected")
    if args.device_id < 0 or args.device_id >= n_dev:
        raise ValueError(f"--device-id out of range; available device count: {n_dev}")
    cp.cuda.Device(int(args.device_id)).use()

    backend = AutoTrackBackend(data_folder=args.data_folder)
    if backend.init_error:
        raise RuntimeError(f"Data loading failed: {backend.init_error}")

    if float(args.speed_min_kmh) <= 0 or float(args.speed_max_kmh) <= 0:
        raise ValueError("speed_min/speed_max must be > 0")
    if float(args.speed_min_kmh) > float(args.speed_max_kmh):
        raise ValueError("speed_min_kmh cannot be greater than speed_max_kmh")

    t0 = time.perf_counter()
    nms_samples = int(max(1, round(float(args.nms_time_radius) * backend.fs)))

    if args.current_window_only and backend.data_view.size > 0:
        scope_samples = int(backend.data_view.shape[1])
    else:
        scope_samples = int(max(round(float(args.tile_seconds) * backend.fs), 10 * backend.fs))
    if int(args.enhance_decimate) > 0:
        decim = int(args.enhance_decimate)
    else:
        decim = int(backend._choose_enhance_decimate(scope_samples))

    cfg = ExtractorConfig(
        enhance_decimate=decim,
        prominence=float(args.prominence),
        min_peak_distance=int(max(1, args.min_peak_distance)),
        min_track_channels=int(max(2, args.min_track_channels)),
        nms_time_radius=nms_samples,
    )

    print(f"[gpu] using cuda:{args.device_id}, enhance_decimate={cfg.enhance_decimate}")

    if args.current_window_only:
        backend.window_size = int(round(float(args.window_seconds) * backend.fs))
        backend.current_start = int(round(float(args.window_start_s) * backend.fs))
        backend.update_view_window()

        source = backend.data_view
        start_sample = int(backend.current_start)
        local_tracks = extract_all_gpu(
            data=source,
            fs=backend.fs,
            dx_m=backend.dx_m,
            direction=args.direction,
            vmin_kmh=float(args.speed_min_kmh),
            vmax_kmh=float(args.speed_max_kmh),
            config=cfg,
        )
        global_tracks = [backend._to_global_track(tr, start_sample) for tr in local_tracks]
        dedup = backend._deduplicate_tracks(global_tracks, tol_samples=nms_samples)
        dedup = backend._stitch_track_fragments(
            dedup,
            direction=args.direction,
            speed_min_kmh=float(args.speed_min_kmh),
            speed_max_kmh=float(args.speed_max_kmh),
            tol_samples=nms_samples,
        )
        dedup = backend._deduplicate_tracks(dedup, tol_samples=nms_samples)
        dedup = sorted(dedup, key=lambda tr: tr.total_score, reverse=True)
    else:
        tile_samples = int(max(round(float(args.tile_seconds) * backend.fs), 10 * backend.fs))
        overlap_samples = int(max(0, round(float(args.overlap_seconds) * backend.fs)))
        overlap_samples = min(overlap_samples, tile_samples - 1)
        step = max(1, tile_samples - overlap_samples)

        n_total = int(backend.data_all.shape[1])
        starts = list(range(0, n_total, step))
        all_tracks: list[Track] = []
        print(
            f"[gpu] full-data mode, tiles={len(starts)}, tile={tile_samples/backend.fs:.1f}s, "
            f"overlap={overlap_samples/backend.fs:.1f}s"
        )

        for i, start in enumerate(starts, 1):
            end = min(n_total, start + tile_samples)
            tile = backend.data_all[:, start:end]
            tile_tracks = extract_all_gpu(
                data=tile,
                fs=backend.fs,
                dx_m=backend.dx_m,
                direction=args.direction,
                vmin_kmh=float(args.speed_min_kmh),
                vmax_kmh=float(args.speed_max_kmh),
                config=cfg,
            )
            all_tracks.extend(backend._to_global_track(tr, start) for tr in tile_tracks)
            if i % max(1, len(starts) // 10) == 0 or i == len(starts):
                print(f"[gpu] progress {i}/{len(starts)}")

        merged = backend._merge_tracks(all_tracks, tol_samples=nms_samples, min_overlap=3)
        dedup = backend._deduplicate_tracks(merged, tol_samples=nms_samples)
        dedup = backend._stitch_track_fragments(
            dedup,
            direction=args.direction,
            speed_min_kmh=float(args.speed_min_kmh),
            speed_max_kmh=float(args.speed_max_kmh),
            tol_samples=nms_samples,
        )
        dedup = backend._deduplicate_tracks(dedup, tol_samples=nms_samples)
        dedup = sorted(dedup, key=lambda tr: tr.total_score, reverse=True)

    backend.tracks = _reindex_tracks(dedup)

    elapsed = float(time.perf_counter() - t0)
    total_points = int(sum(len(tr.points) for tr in backend.tracks))
    backend.last_summary = {
        "track_count": int(len(backend.tracks)),
        "total_points": total_points,
        "elapsed_seconds": elapsed,
        "mode": "gpu_current_window" if args.current_window_only else "gpu_full_data",
        "device_id": int(args.device_id),
    }

    out_csv = args.out_csv.strip() if isinstance(args.out_csv, str) else ""
    if not out_csv:
        out_csv = str(Path(args.data_folder).expanduser() / "auto_tracks_gpu.csv")
    csv_path, summary_path = backend.export_csv(out_csv)

    print(f"[gpu] done: tracks={len(backend.tracks)}, points={total_points}, elapsed={elapsed:.3f}s")
    print(f"[gpu] csv={csv_path}")
    print(f"[gpu] summary={summary_path}")
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _run_gpu_extract(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu] failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
