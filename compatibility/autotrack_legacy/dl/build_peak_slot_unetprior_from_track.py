"""Build two-channel PeakSlotNet shards directly from TrackSlotNet shards.

The TrackSlotNet Gaussian-window heatmap is the source of truth. For every
sample this converter:

1. detects PeakSlotNet candidate tables from the track heatmap;
2. maps TrackSlotNet GT trajectories to candidate indices;
3. renders the same heatmap for waveform-line U-Net inference;
4. writes PeakSlotNet shards with x[:, 0] = original track heatmap and
   x[:, 1] = sampled waveform-line U-Net probability prior.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from autotrack.dl.convert_track_slot_to_peak_slot import _convert_one_sample
from autotrack.dl.peak_slot_model import PeakDetectionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PeakSlotNet+U-Net-prior shards from track_slot shards.")
    parser.add_argument("--track-dir", required=True, type=Path, help="Input track_slot dataset directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak_slot dataset directory with in_channels=2.")
    parser.add_argument("--unet-checkpoint", required=True, type=Path, help="waveform_line_task U-Net checkpoint.")
    parser.add_argument(
        "--waveform-task-dir",
        type=Path,
        default=Path("/csim2/zhangzhiyu/MyProjects/waveform_line_task"),
        help="Path containing waveform_line_task model/ and render.py modules.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device for U-Net inference.")
    parser.add_argument("--batch-size", type=int, default=16, help="U-Net inference batch size.")
    parser.add_argument("--image-size", type=int, default=512, help="Rendered image size for U-Net input.")
    parser.add_argument("--waveform-line-width", type=int, default=1, help="Rendered waveform trace width.")
    parser.add_argument("--wiggle-fraction", type=float, default=0.28, help="Rendered trace wiggle fraction.")
    parser.add_argument("--robust-percentile", type=float, default=99.5, help="Rendered trace robust scaling percentile.")
    parser.add_argument("--prior-threshold", type=float, default=0.0, help="Set values below this probability to 0.")
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Multiplier applied to prior channel.")
    parser.add_argument("--x-dtype", choices=["preserve", "float16", "float32"], default="preserve", help="Output x dtype.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates K per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum distance between candidates.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Minimum normalized absolute heatmap height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum normalized peak prominence.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="Max GT-to-candidate match distance.")
    parser.add_argument("--candidate-source", default="raw", choices=["raw", "prior", "raw_prior_union"], help="Peak candidate source used for the output candidate table.")
    parser.add_argument("--prior-peak-min-height", type=float, default=0.08, help="Prior-channel peak height used by prior/union candidate modes.")
    parser.add_argument("--prior-peak-prominence", type=float, default=0.02, help="Prior-channel peak prominence used by prior/union candidate modes.")
    parser.add_argument("--candidate-merge-tolerance-s", type=float, default=0.20, help="Merge raw/prior candidates on the same channel within this many seconds.")
    parser.add_argument("--prior-score-scale", type=float, default=0.85, help="Score multiplier for prior candidates in raw_prior_union mode.")
    parser.add_argument("--workers", type=int, default=1, help="CPU workers for per-sample peak-table conversion.")
    parser.add_argument("--max-shards", type=int, default=0, help="Smoke-test limit. 0 converts all shards.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--resume", action="store_true", help="Keep existing output shards and skip them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    track_dir = Path(args.track_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    waveform_task_dir = Path(args.waveform_task_dir).expanduser().resolve()
    _install_waveform_task_imports(waveform_task_dir)

    from model.network import UNetConfig, WaveformLineUNet  # type: ignore
    from render import RenderConfig, channel_x_positions, render_waveform_image  # type: ignore

    meta = _load_meta(track_dir)
    if str(meta.get("format", "")) != "track_slot_shards_v1":
        raise ValueError(f"Input dataset is not track_slot_shards_v1: {track_dir}")
    if int(meta.get("in_channels", 1)) != 1:
        raise ValueError(f"Expected track in_channels=1, got {meta.get('in_channels')!r}")
    shards = [str(item) for item in meta.get("shards", [])]
    if int(args.max_shards) > 0:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise ValueError(f"No shards listed in {track_dir / 'meta.json'}")

    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite), resume=bool(args.resume))
    device = _resolve_device(str(args.device))
    render_config = RenderConfig(
        image_size=int(args.image_size),
        waveform_line_width=int(args.waveform_line_width),
        robust_percentile=float(args.robust_percentile),
        wiggle_fraction=float(args.wiggle_fraction),
    )
    unet = _load_unet(
        Path(args.unet_checkpoint).expanduser(),
        device=device,
        UNetConfig=UNetConfig,
        WaveformLineUNet=WaveformLineUNet,
    )
    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(args.peak_candidates_per_channel),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=float(args.peak_match_tolerance_s),
        candidate_source=str(args.candidate_source),
        prior_min_height=float(args.prior_peak_min_height),
        prior_prominence=float(args.prior_peak_prominence),
        merge_tolerance_s=float(args.candidate_merge_tolerance_s),
        prior_score_scale=float(args.prior_score_scale),
        return_candidate_stats=True,
    )

    t0 = time.perf_counter()
    out_shards: list[str] = []
    converted_samples = 0
    matched_total = 0
    injected_total = 0
    candidate_stats_total: dict[str, int] = {
        "raw_candidate_count": 0,
        "prior_candidate_count": 0,
        "merged_candidate_count": 0,
    }
    prior_stats: list[dict[str, float | int | str]] = []
    fs = float(meta["fs"])
    time_downsample = int(meta["time_downsample"])
    window_samples = int(meta["window_samples"])
    workers = int(max(1, args.workers))
    if workers > 1 and str(args.candidate_source) != "raw":
        print("candidate_source requires prior-aware conversion; using workers=1 for peak table rebuild.", flush=True)
        workers = 1
    if workers > 1:
        conversion_result = _convert_shards_parallel(
            track_dir=track_dir,
            out_dir=out_dir,
            shards=shards,
            args=args,
            meta=meta,
            peak_cfg=peak_cfg,
            unet=unet,
            device=device,
            render_config=render_config,
            channel_x_positions=channel_x_positions,
            render_waveform_image=render_waveform_image,
            t0=t0,
            workers=workers,
        )
        out_shards = conversion_result["out_shards"]
        converted_samples = int(conversion_result["converted_samples"])
        matched_total = int(conversion_result["matched_total"])
        injected_total = int(conversion_result["injected_total"])
        candidate_stats_total = dict(conversion_result["candidate_stats_total"])
        skipped_shards = int(conversion_result["skipped_shards"])
        prior_stats = conversion_result["prior_stats"]
    else:
        skipped_shards = 0
        for shard_idx, shard_name in enumerate(shards):
            out_name = f"shard_{shard_idx:06d}.pt"
            out_path = out_dir / out_name
            if bool(args.resume) and out_path.is_file():
                existing = torch.load(str(out_path), map_location="cpu", weights_only=False)
                sample_count = int(existing["x"].shape[0])
                out_shards.append(out_name)
                converted_samples += sample_count
                skipped_shards += 1
                prior = existing["x"][:, 1].to(torch.float32) if int(existing["x"].shape[1]) > 1 else torch.zeros(())
                prior_stats.append(
                    {
                        "shard": out_name,
                        "samples": sample_count,
                        "prior_mean": float(prior.mean().item()) if prior.numel() else 0.0,
                        "prior_max": float(prior.max().item()) if prior.numel() else 0.0,
                        "prior_positive_ratio_0p5": float((prior >= 0.5).to(torch.float32).mean().item()) if prior.numel() else 0.0,
                        "skipped_existing": 1,
                    }
                )
                print(f"skipped {out_name}: samples={converted_samples}, elapsed={time.perf_counter() - t0:.1f}s", flush=True)
                continue

            payload = torch.load(str(track_dir / shard_name), map_location="cpu", weights_only=False)
            x = payload["x"]
            if int(x.ndim) != 4 or int(x.shape[1]) != 1:
                raise ValueError(f"{shard_name} x must have shape [N,1,C,T], got {tuple(x.shape)}")
            time_labels = payload["time"].to(torch.float32)
            visibility = payload["visibility"].to(torch.float32)
            gt_valid = payload["gt_valid"].to(torch.bool)

            prior = _build_prior_channel(
                x[:, 0].to(torch.float32),
                unet=unet,
                device=device,
                render_config=render_config,
                channel_x_positions=channel_x_positions,
                render_waveform_image=render_waveform_image,
                batch_size=int(args.batch_size),
                prior_threshold=float(args.prior_threshold),
                prior_scale=float(args.prior_scale),
            )
            out_x = torch.cat([x.to(torch.float32), prior[:, None, :, :].to(torch.float32)], dim=1)
            out_x = _convert_x_dtype(out_x, original=x, mode=str(args.x_dtype))
            peak_input = out_x if str(args.candidate_source) != "raw" else x
            peak_result = _convert_peak_samples(
                x=peak_input,
                time_labels=time_labels,
                visibility=visibility,
                gt_valid=gt_valid,
                fs=fs,
                time_downsample=time_downsample,
                window_samples=window_samples,
                peak_cfg=peak_cfg,
            )
            torch.save(
                {
                    "x": out_x.contiguous(),
                    "peak_time": peak_result["peak_time"].contiguous(),
                    "peak_amp": peak_result["peak_amp"].contiguous(),
                    "peak_valid": peak_result["peak_valid"].contiguous(),
                    "peak_index": peak_result["peak_index"].contiguous(),
                    "gt_peak_index": peak_result["gt_peak_index"].contiguous(),
                    "visibility": visibility.contiguous(),
                    "direction": payload["direction"].to(torch.long).contiguous(),
                    "speed": payload["speed"].to(torch.float32).contiguous(),
                    "gt_valid": gt_valid.contiguous(),
                },
                str(out_path),
            )
            shard_matched = int(peak_result["matched"])
            shard_injected = int(peak_result["injected"])
            out_shards.append(out_name)
            converted_samples += int(out_x.shape[0])
            matched_total += int(shard_matched)
            injected_total += int(shard_injected)
            _merge_candidate_stats(candidate_stats_total, peak_result["candidate_stats"])
            prior_stats.append(
                {
                    "shard": out_name,
                    "samples": int(out_x.shape[0]),
                    "prior_mean": float(prior.mean().item()),
                    "prior_max": float(prior.max().item()),
                    "prior_positive_ratio_0p5": float((prior >= 0.5).to(torch.float32).mean().item()),
                }
            )
            print(
                f"wrote {out_name}: samples={converted_samples}, matched={shard_matched}, "
                f"injected={shard_injected}, elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    out_meta = dict(meta)
    out_meta.update(
        {
            "format": "peak_slot_shards_v1",
            "mode": "peak_slot_from_track_with_waveform_line_unet_prior",
            "source_format": str(meta.get("format", "track_slot_shards_v1")),
            "source_dir": str(track_dir),
            "track_source_data_dir": str(track_dir),
            "created_at_unix": time.time(),
            "num_samples": int(converted_samples),
            "shards": out_shards,
            "in_channels": 2,
            "peak_candidates_per_channel": int(args.peak_candidates_per_channel),
            "peak_none_index": int(args.peak_candidates_per_channel),
            "peak_detection": {
                "min_distance_s": float(args.peak_min_distance_s),
                "min_height": float(args.peak_min_height),
                "prominence": float(args.peak_prominence),
                "match_tolerance_s": float(args.peak_match_tolerance_s),
                "candidate_source": str(args.candidate_source),
                "prior_min_height": float(args.prior_peak_min_height),
                "prior_prominence": float(args.prior_peak_prominence),
                "merge_tolerance_s": float(args.candidate_merge_tolerance_s),
                "prior_score_scale": float(args.prior_score_scale),
            },
            "peak_conversion_stats": {
                "samples": int(converted_samples),
                "matched_gt_points": int(matched_total),
                "injected_gt_points": int(injected_total),
                **candidate_stats_total,
                "workers": int(workers),
                "skipped_existing_shards": int(skipped_shards),
            },
            "prior_channel": {
                "index": 1,
                "kind": "waveform_line_unet_probability",
                "source_x": "track_slot_gaussian_window_x",
                "unet_checkpoint": str(Path(args.unet_checkpoint).expanduser()),
                "waveform_task_dir": str(waveform_task_dir),
                "image_size": int(args.image_size),
                "waveform_line_width": int(args.waveform_line_width),
                "wiggle_fraction": float(args.wiggle_fraction),
                "robust_percentile": float(args.robust_percentile),
                "prior_threshold": float(args.prior_threshold),
                "prior_scale": float(args.prior_scale),
                "stats": prior_stats,
            },
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(_json_ready(out_meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: samples={converted_samples}, shards={len(out_shards)}, matched={matched_total}, "
        f"injected={injected_total}, out_dir={out_dir}",
        flush=True,
    )
    return 0


def _install_waveform_task_imports(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"waveform_task_dir does not exist: {path}")
    sys.path.insert(0, str(path))


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _convert_shards_parallel(
    *,
    track_dir: Path,
    out_dir: Path,
    shards: list[str],
    args: argparse.Namespace,
    meta: dict[str, Any],
    peak_cfg: PeakDetectionConfig,
    unet: torch.nn.Module,
    device: torch.device,
    render_config: Any,
    channel_x_positions: Any,
    render_waveform_image: Any,
    t0: float,
    workers: int,
) -> dict[str, Any]:
    cache_dir = out_dir / ".peak_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fs = float(meta["fs"])
    time_downsample = int(meta["time_downsample"])
    window_samples = int(meta["window_samples"])
    out_entries: list[tuple[int, str]] = []
    converted_samples = 0
    matched_total = 0
    injected_total = 0
    candidate_stats_total: dict[str, int] = {
        "raw_candidate_count": 0,
        "prior_candidate_count": 0,
        "merged_candidate_count": 0,
    }
    skipped_shards = 0
    prior_stats: list[dict[str, float | int | str]] = []
    futures = {}
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for shard_idx, shard_name in enumerate(shards):
            out_name = f"shard_{shard_idx:06d}.pt"
            out_path = out_dir / out_name
            if bool(args.resume) and out_path.is_file():
                existing = torch.load(str(out_path), map_location="cpu", weights_only=False)
                sample_count = int(existing["x"].shape[0])
                out_entries.append((int(shard_idx), out_name))
                converted_samples += sample_count
                skipped_shards += 1
                prior = existing["x"][:, 1].to(torch.float32) if int(existing["x"].shape[1]) > 1 else torch.zeros(())
                prior_stats.append(
                    {
                        "shard": out_name,
                        "samples": sample_count,
                        "prior_mean": float(prior.mean().item()) if prior.numel() else 0.0,
                        "prior_max": float(prior.max().item()) if prior.numel() else 0.0,
                        "prior_positive_ratio_0p5": float((prior >= 0.5).to(torch.float32).mean().item()) if prior.numel() else 0.0,
                        "skipped_existing": 1,
                    }
                )
                print(f"skipped {out_name}: samples={converted_samples}, elapsed={time.perf_counter() - t0:.1f}s", flush=True)
                continue
            future = executor.submit(
                _convert_peak_shard_task,
                int(shard_idx),
                str(shard_name),
                str(track_dir),
                str(cache_dir),
                fs,
                time_downsample,
                window_samples,
                peak_cfg,
            )
            futures[future] = (int(shard_idx), str(shard_name), out_name)

        for future in as_completed(futures):
            shard_idx, shard_name, out_name = futures[future]
            peak_info = future.result()
            payload = torch.load(str(track_dir / shard_name), map_location="cpu", weights_only=False)
            x = payload["x"]
            if int(x.ndim) != 4 or int(x.shape[1]) != 1:
                raise ValueError(f"{shard_name} x must have shape [N,1,C,T], got {tuple(x.shape)}")
            peak_payload = torch.load(str(peak_info["cache_path"]), map_location="cpu", weights_only=False)
            prior = _build_prior_channel(
                x[:, 0].to(torch.float32),
                unet=unet,
                device=device,
                render_config=render_config,
                channel_x_positions=channel_x_positions,
                render_waveform_image=render_waveform_image,
                batch_size=int(args.batch_size),
                prior_threshold=float(args.prior_threshold),
                prior_scale=float(args.prior_scale),
            )
            out_x = torch.cat([x.to(torch.float32), prior[:, None, :, :].to(torch.float32)], dim=1)
            out_x = _convert_x_dtype(out_x, original=x, mode=str(args.x_dtype))
            torch.save(
                {
                    "x": out_x.contiguous(),
                    "peak_time": peak_payload["peak_time"].contiguous(),
                    "peak_amp": peak_payload["peak_amp"].contiguous(),
                    "peak_valid": peak_payload["peak_valid"].contiguous(),
                    "peak_index": peak_payload["peak_index"].contiguous(),
                    "gt_peak_index": peak_payload["gt_peak_index"].contiguous(),
                    "visibility": payload["visibility"].to(torch.float32).contiguous(),
                    "direction": payload["direction"].to(torch.long).contiguous(),
                    "speed": payload["speed"].to(torch.float32).contiguous(),
                    "gt_valid": payload["gt_valid"].to(torch.bool).contiguous(),
                },
                str(out_dir / out_name),
            )
            try:
                Path(str(peak_info["cache_path"])).unlink()
            except FileNotFoundError:
                pass
            sample_count = int(out_x.shape[0])
            converted_samples += sample_count
            matched_total += int(peak_info["matched"])
            injected_total += int(peak_info["injected"])
            _merge_candidate_stats(candidate_stats_total, dict(peak_info.get("candidate_stats", {})))
            out_entries.append((int(shard_idx), out_name))
            prior_stats.append(
                {
                    "shard": out_name,
                    "samples": sample_count,
                    "prior_mean": float(prior.mean().item()),
                    "prior_max": float(prior.max().item()),
                    "prior_positive_ratio_0p5": float((prior >= 0.5).to(torch.float32).mean().item()),
                }
            )
            print(
                f"wrote {out_name}: samples={converted_samples}, matched={int(peak_info['matched'])}, "
                f"injected={int(peak_info['injected'])}, elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    return {
        "out_shards": [name for _, name in sorted(out_entries, key=lambda item: item[0])],
        "converted_samples": int(converted_samples),
        "matched_total": int(matched_total),
        "injected_total": int(injected_total),
        "candidate_stats_total": dict(candidate_stats_total),
        "skipped_shards": int(skipped_shards),
        "prior_stats": sorted(prior_stats, key=lambda item: str(item["shard"])),
    }


def _convert_peak_shard_task(
    shard_idx: int,
    shard_name: str,
    track_dir: str,
    cache_dir: str,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
) -> dict[str, Any]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    payload = torch.load(str(Path(track_dir) / shard_name), map_location="cpu", weights_only=False)
    result = _convert_peak_samples(
        x=payload["x"],
        time_labels=payload["time"].to(torch.float32),
        visibility=payload["visibility"].to(torch.float32),
        gt_valid=payload["gt_valid"].to(torch.bool),
        fs=float(fs),
        time_downsample=int(time_downsample),
        window_samples=int(window_samples),
        peak_cfg=peak_cfg,
    )
    cache_path = Path(cache_dir) / f"peak_{int(shard_idx):06d}.pt"
    torch.save(
        {
            "peak_time": result["peak_time"].contiguous(),
            "peak_amp": result["peak_amp"].contiguous(),
            "peak_valid": result["peak_valid"].contiguous(),
            "peak_index": result["peak_index"].contiguous(),
            "gt_peak_index": result["gt_peak_index"].contiguous(),
        },
        str(cache_path),
    )
    return {
        "shard_idx": int(shard_idx),
        "shard_name": str(shard_name),
        "cache_path": str(cache_path),
        "samples": int(payload["x"].shape[0]),
        "matched": int(result["matched"]),
        "injected": int(result["injected"]),
        "candidate_stats": dict(result["candidate_stats"]),
    }


class _NullExecutor:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None


def _submit_peak_conversion(
    executor: ProcessPoolExecutor | None,
    *,
    x: torch.Tensor,
    time_labels: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
    workers: int,
) -> Any:
    if workers <= 1 or executor is None:
        return _convert_peak_samples(
            x=x,
            time_labels=time_labels,
            visibility=visibility,
            gt_valid=gt_valid,
            fs=fs,
            time_downsample=time_downsample,
            window_samples=window_samples,
            peak_cfg=peak_cfg,
        )
    futures = [
        executor.submit(
            _convert_peak_sample_task,
            i,
            x[i],
            time_labels[i],
            visibility[i],
            gt_valid[i],
            fs,
            time_downsample,
            window_samples,
            peak_cfg,
        )
        for i in range(int(x.shape[0]))
    ]
    return futures


def _collect_peak_conversion(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    results: list[Any] = [None] * len(value)
    for future in as_completed(value):
        item = future.result()
        results[int(item[0])] = item
    peak_times = []
    peak_amps = []
    peak_valids = []
    peak_indices = []
    gt_peak_indices = []
    matched_total = 0
    injected_total = 0
    for item in results:
        _, peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, matched, injected = item
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        gt_peak_indices.append(gt_peak_index)
        matched_total += int(matched)
        injected_total += int(injected)
    return {
        "peak_time": torch.stack(peak_times, dim=0),
        "peak_amp": torch.stack(peak_amps, dim=0),
        "peak_valid": torch.stack(peak_valids, dim=0),
        "peak_index": torch.stack(peak_indices, dim=0),
        "gt_peak_index": torch.stack(gt_peak_indices, dim=0),
        "matched": int(matched_total),
        "injected": int(injected_total),
    }


def _convert_peak_samples(
    *,
    x: torch.Tensor,
    time_labels: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
) -> dict[str, Any]:
    peak_times = []
    peak_amps = []
    peak_valids = []
    peak_indices = []
    gt_peak_indices = []
    matched_total = 0
    injected_total = 0
    candidate_stats: dict[str, int] = {
        "raw_candidate_count": 0,
        "prior_candidate_count": 0,
        "merged_candidate_count": 0,
    }
    for i in range(int(x.shape[0])):
        item = _convert_peak_sample_task(
            i,
            x[i],
            time_labels[i],
            visibility[i],
            gt_valid[i],
            fs,
            time_downsample,
            window_samples,
            peak_cfg,
        )
        _, peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, matched, injected, stats = item
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        gt_peak_indices.append(gt_peak_index)
        matched_total += int(matched)
        injected_total += int(injected)
        _merge_candidate_stats(candidate_stats, dict(stats))
    return {
        "peak_time": torch.stack(peak_times, dim=0),
        "peak_amp": torch.stack(peak_amps, dim=0),
        "peak_valid": torch.stack(peak_valids, dim=0),
        "peak_index": torch.stack(peak_indices, dim=0),
        "gt_peak_index": torch.stack(gt_peak_indices, dim=0),
        "matched": int(matched_total),
        "injected": int(injected_total),
        "candidate_stats": dict(candidate_stats),
    }


def _convert_peak_sample_task(
    i: int,
    x_i: torch.Tensor,
    time_label_i: torch.Tensor,
    visibility_i: torch.Tensor,
    gt_valid_i: torch.Tensor,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, dict[str, int | float | str]]:
    converted = _convert_one_sample(
        x_i,
        time_label_i,
        visibility_i,
        gt_valid_i,
        fs=float(fs),
        time_downsample=int(time_downsample),
        window_samples=int(window_samples),
        peak_cfg=peak_cfg,
    )
    peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, matched, injected, stats = converted
    return int(i), peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, int(matched), int(injected), dict(stats)


def _merge_candidate_stats(target: dict[str, int], stats: dict[str, object]) -> None:
    for key in ("raw_candidate_count", "prior_candidate_count", "merged_candidate_count"):
        target[key] = int(target.get(key, 0)) + int(stats.get(key, 0) or 0)


def _prepare_out_dir(out_dir: Path, *, overwrite: bool, resume: bool) -> None:
    import shutil

    if out_dir.exists() and any(out_dir.iterdir()):
        if resume:
            return
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _resolve_device(requested: str) -> torch.device:
    name = str(requested or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is unavailable")
    return torch.device(name)


def _load_unet(checkpoint_path: Path, *, device: torch.device, UNetConfig: Any, WaveformLineUNet: Any) -> torch.nn.Module:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = UNetConfig(**dict(checkpoint.get("model_config", {})))
    model = WaveformLineUNet(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def _build_prior_channel(
    x: torch.Tensor,
    *,
    unet: torch.nn.Module,
    device: torch.device,
    render_config: Any,
    channel_x_positions: Any,
    render_waveform_image: Any,
    batch_size: int,
    prior_threshold: float,
    prior_scale: float,
) -> torch.Tensor:
    import numpy as np
    import torch.nn.functional as F

    n, n_ch, t_down = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    image_size = int(render_config.image_size)
    grid = _build_sampling_grid(n_ch, t_down, image_size, channel_x_positions)
    priors: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, max(1, int(batch_size))):
            batch = x[start : start + max(1, int(batch_size))]
            images = [
                torch.from_numpy(render_waveform_image(sample.detach().cpu().numpy(), render_config).astype(np.float32) / 255.0)
                for sample in batch
            ]
            image_tensor = torch.stack(images, dim=0)[:, None, :, :].to(device=device, dtype=torch.float32)
            logits = unet(image_tensor)
            prob = torch.sigmoid(logits).to(torch.float32)
            sample_grid = grid.to(device=device).expand(int(prob.shape[0]), -1, -1, -1)
            sampled = F.grid_sample(prob, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            sampled = sampled[:, 0].detach().cpu()
            if float(prior_threshold) > 0.0:
                sampled = torch.where(sampled >= float(prior_threshold), sampled, torch.zeros_like(sampled))
            if float(prior_scale) != 1.0:
                sampled = sampled * float(prior_scale)
            priors.append(sampled.clamp(0.0, 1.0))
    return torch.cat(priors, dim=0).contiguous()


def _build_sampling_grid(n_ch: int, t_down: int, image_size: int, channel_x_positions: Any) -> torch.Tensor:
    xs = torch.as_tensor(channel_x_positions(int(n_ch), int(image_size)), dtype=torch.float32)
    ys = torch.linspace(0.0, float(image_size - 1), int(t_down), dtype=torch.float32)
    x_norm = xs / float(max(1, image_size - 1)) * 2.0 - 1.0
    y_norm = ys / float(max(1, image_size - 1)) * 2.0 - 1.0
    yy = y_norm.view(1, int(t_down)).expand(int(n_ch), int(t_down))
    xx = x_norm.view(int(n_ch), 1).expand(int(n_ch), int(t_down))
    return torch.stack([xx, yy], dim=-1).unsqueeze(0)


def _convert_x_dtype(out_x: torch.Tensor, *, original: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "float32":
        return out_x.to(torch.float32)
    if mode == "float16":
        return out_x.to(torch.float16)
    return out_x.to(dtype=original.dtype)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
