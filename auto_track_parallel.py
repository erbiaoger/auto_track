from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from auto_track_backend import AutoTrackBackend, DEFAULT_DATA_FOLDER
from track_extractor_graph import ExtractorConfig, Track, extract_all


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU parallel trajectory extraction (Peak Map + Dynamic Programming)")
    parser.add_argument("--data-folder", type=str, default=DEFAULT_DATA_FOLDER, help="SAC folder")
    parser.add_argument("--out-csv", type=str, default="", help="Output CSV path; default is <data-folder>/auto_tracks_parallel.csv")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0)
    parser.add_argument("--speed-max-kmh", type=float, default=120.0)
    parser.add_argument("--prominence", type=float, default=0.4)
    parser.add_argument("--min-peak-distance", type=int, default=500)
    parser.add_argument("--min-track-channels", type=int, default=12)
    parser.add_argument("--tile-seconds", type=float, default=120.0)
    parser.add_argument("--overlap-seconds", type=float, default=20.0)
    parser.add_argument("--nms-time-radius", type=float, default=0.18, help="seconds")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)))
    parser.add_argument("--window-seconds", type=float, default=240.0, help="Only used in current-window mode")
    parser.add_argument("--window-start-s", type=float, default=0.0, help="Only used in current-window mode")
    parser.add_argument("--current-window-only", action="store_true", help="Extract current window only")
    parser.add_argument(
        "--enhance-decimate",
        type=int,
        default=0,
        help="Enhancement-stage decimation factor; <=0 means auto (recommended)",
    )
    return parser.parse_args()


def _build_cfg(
    backend: AutoTrackBackend,
    args: argparse.Namespace,
    scope_samples: int,
) -> tuple[ExtractorConfig, int]:
    nms_samples = int(max(1, round(float(args.nms_time_radius) * backend.fs)))
    if int(args.enhance_decimate) > 0:
        decim = int(args.enhance_decimate)
    else:
        decim = int(backend._choose_enhance_decimate(int(scope_samples)))
    cfg = ExtractorConfig(
        enhance_decimate=decim,
        prominence=float(args.prominence),
        min_peak_distance=int(max(1, args.min_peak_distance)),
        min_track_channels=int(max(2, args.min_track_channels)),
        nms_time_radius=nms_samples,
    )
    return cfg, nms_samples


def _extract_tile_worker(
    tile_data,
    start_sample: int,
    fs: float,
    dx_m: float,
    direction: str,
    speed_min_kmh: float,
    speed_max_kmh: float,
    cfg: ExtractorConfig,
) -> tuple[int, list[Track]]:
    tracks = extract_all(
        data=tile_data,
        fs=fs,
        dx_m=dx_m,
        direction=direction,
        vmin_kmh=float(speed_min_kmh),
        vmax_kmh=float(speed_max_kmh),
        config=cfg,
    )
    return int(start_sample), tracks


def _reindex_tracks(tracks: Iterable[Track]) -> list[Track]:
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


def _run_parallel_extract(args: argparse.Namespace) -> int:
    backend = AutoTrackBackend(data_folder=args.data_folder)
    if backend.init_error:
        raise RuntimeError(f"Data loading failed: {backend.init_error}")

    if float(args.speed_min_kmh) <= 0 or float(args.speed_max_kmh) <= 0:
        raise ValueError("speed_min/speed_max must be > 0")
    if float(args.speed_min_kmh) > float(args.speed_max_kmh):
        raise ValueError("speed_min_kmh cannot be greater than speed_max_kmh")

    t0 = time.perf_counter()

    if args.current_window_only:
        backend.window_size = int(round(float(args.window_seconds) * backend.fs))
        backend.current_start = int(round(float(args.window_start_s) * backend.fs))
        backend.update_view_window()

        source = backend.data_view
        start_sample = int(backend.current_start)
        cfg, nms_samples = _build_cfg(backend, args, source.shape[1])
        print(
            f"[parallel] current-window mode, samples={source.shape[1]}, "
            f"enhance_decimate={cfg.enhance_decimate}"
        )
        local_tracks = extract_all(
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
        cfg, nms_samples = _build_cfg(backend, args, tile_samples)
        workers = max(1, int(args.workers))
        print(
            f"[parallel] full-data mode, tiles={len(starts)}, workers={workers}, "
            f"tile={tile_samples/backend.fs:.1f}s, overlap={overlap_samples/backend.fs:.1f}s, "
            f"enhance_decimate={cfg.enhance_decimate}"
        )

        all_tracks: list[Track] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {}
            for start in starts:
                end = min(n_total, start + tile_samples)
                tile = backend.data_all[:, start:end]
                fut = ex.submit(
                    _extract_tile_worker,
                    tile,
                    start,
                    backend.fs,
                    backend.dx_m,
                    args.direction,
                    float(args.speed_min_kmh),
                    float(args.speed_max_kmh),
                    cfg,
                )
                future_map[fut] = (start, end)

            done = 0
            for fut in as_completed(future_map):
                start, _end = future_map[fut]
                tile_start, tile_tracks = fut.result()
                if tile_start != start:
                    raise RuntimeError("Parallel task returned an inconsistent tile start")
                all_tracks.extend(backend._to_global_track(tr, tile_start) for tr in tile_tracks)
                done += 1
                if done % max(1, len(starts) // 10) == 0 or done == len(starts):
                    print(f"[parallel] progress {done}/{len(starts)}")

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
        "mode": "parallel_current_window" if args.current_window_only else "parallel_full_data",
        "workers": int(max(1, int(args.workers))),
    }

    out_csv = args.out_csv.strip() if isinstance(args.out_csv, str) else ""
    if not out_csv:
        out_csv = str(Path(args.data_folder).expanduser() / "auto_tracks_parallel.csv")
    csv_path, summary_path = backend.export_csv(out_csv)

    print(
        f"[parallel] done: tracks={len(backend.tracks)}, points={total_points}, "
        f"elapsed={elapsed:.3f}s"
    )
    print(f"[parallel] csv={csv_path}")
    print(f"[parallel] summary={summary_path}")
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return _run_parallel_extract(args)
    except Exception as exc:  # noqa: BLE001
        print(f"[parallel] failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
