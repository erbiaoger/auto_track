"""Convert TrackSlotNet tensor shards into PeakSlotNet tensor shards.

Purpose:
    Reuse existing `track_slot` generated datasets for the `peak_slot` model.
    The source heatmap and vehicle labels are kept, but each channel is first
    converted into a fixed number of peak candidates. GT vehicle times are then
    mapped to candidate indices, with the final class K reserved for `none`.

Example:
    uv run python -m autotrack.dl.convert_track_slot_to_peak_slot \
        --in-dir datasets/track_slot/train \
        --out-dir datasets/peak_slot/train \
        --peak-candidates-per-channel 64 \
        --workers 8 \
        --peak-min-distance-s 0.15 \
        --peak-match-tolerance-s 0.25 \
        --overwrite

Arguments:
    --in-dir must contain a `track_slot` `meta.json` and `shard_*.pt`.
    --out-dir receives `peak_slot` `meta.json` and converted shards.
    --peak-candidates-per-channel is K; gt_peak_index == K means none.
    --workers parallelizes conversion at shard granularity.

Outputs:
    x, peak_time, peak_amp, peak_valid, gt_peak_index, visibility, direction,
    speed, gt_valid in each shard.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TrackSlotNet shards to PeakSlotNet shards.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Source track_slot dataset directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak_slot dataset directory.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates K per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum distance between candidates on one channel.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Minimum normalized absolute heatmap height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum normalized peak prominence.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="Max GT-to-candidate match distance before GT injection.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel shard workers. 1 runs sequentially.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output directory.")
    return parser.parse_args()


def _load_meta(in_dir: Path) -> dict[str, Any]:
    meta_path = in_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def _insert_candidate(
    peak_time: torch.Tensor,
    peak_amp: torch.Tensor,
    peak_valid: torch.Tensor,
    peak_index: torch.Tensor,
    *,
    ch: int,
    t_norm: float,
    down_idx: int,
    amp: float,
) -> int:
    valid = torch.where(peak_valid[ch])[0]
    if valid.numel() < int(peak_valid.shape[1]):
        slot = int(valid.numel())
    else:
        weakest = torch.argmin(torch.abs(peak_amp[ch]))
        slot = int(weakest.item())
    peak_time[ch, slot] = float(t_norm)
    peak_amp[ch, slot] = float(amp)
    peak_valid[ch, slot] = True
    peak_index[ch, slot] = int(down_idx)
    order = torch.argsort(peak_time[ch])
    peak_time[ch] = peak_time[ch, order]
    peak_amp[ch] = peak_amp[ch, order]
    peak_valid[ch] = peak_valid[ch, order]
    peak_index[ch] = peak_index[ch, order]
    matches = torch.where((peak_index[ch] == int(down_idx)) & peak_valid[ch])[0]
    return int(matches[0].item())


def _convert_one_sample(
    x: torch.Tensor,
    time_label: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    heatmap = x[0].to(torch.float32)
    peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates_from_tensor(
        heatmap,
        fs=float(fs),
        time_downsample=int(time_downsample),
        window_samples=int(window_samples),
        config=peak_cfg,
    )
    k_count = int(peak_valid.shape[1])
    gt_peak_index = torch.full_like(time_label, fill_value=k_count, dtype=torch.long)
    injected = 0
    matched = 0
    tolerance_norm = float(peak_cfg.match_tolerance_s) * float(fs) / float(max(1, window_samples - 1))

    # First make sure every visible GT point has a candidate in the per-channel
    # candidate table. Candidate insertion sorts the table by time, so GT indices
    # cannot be finalized during this pass without becoming stale after later
    # insertions on the same channel.
    for g in torch.where(gt_valid)[0].tolist():
        for ch in torch.where(visibility[g] > 0.5)[0].tolist():
            t_norm = float(time_label[g, ch].item())
            valid = torch.where(peak_valid[ch])[0]
            if valid.numel() > 0:
                diffs = torch.abs(peak_time[ch, valid] - float(t_norm))
                min_pos = int(torch.argmin(diffs).item())
                if float(diffs[min_pos].item()) <= tolerance_norm:
                    matched += 1
                    continue
            down_idx = int(round(float(t_norm) * float(max(1, window_samples - 1)) / float(max(1, time_downsample))))
            down_idx = max(0, min(int(heatmap.shape[1]) - 1, down_idx))
            _insert_candidate(
                peak_time,
                peak_amp,
                peak_valid,
                peak_index,
                ch=int(ch),
                t_norm=float(t_norm),
                down_idx=int(down_idx),
                amp=float(heatmap[int(ch), int(down_idx)].item()),
            )
            injected += 1

    # Now that all per-channel candidate arrays have their final sorted order,
    # map each GT point to the nearest candidate. This avoids stale indices when
    # a later injected point shifted previously assigned candidate slots.
    for g in torch.where(gt_valid)[0].tolist():
        for ch in torch.where(visibility[g] > 0.5)[0].tolist():
            valid = torch.where(peak_valid[ch])[0]
            if valid.numel() <= 0:
                continue
            t_norm = float(time_label[g, ch].item())
            diffs = torch.abs(peak_time[ch, valid] - float(t_norm))
            min_pos = int(torch.argmin(diffs).item())
            if float(diffs[min_pos].item()) <= max(tolerance_norm, 1.0 / float(max(1, window_samples - 1))):
                gt_peak_index[g, ch] = int(valid[min_pos].item())
    return peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, int(matched), int(injected)


def _convert_shard(
    in_path: Path,
    out_path: Path,
    *,
    meta: dict[str, Any],
    peak_cfg: PeakDetectionConfig,
) -> dict[str, int]:
    payload = torch.load(str(in_path), map_location="cpu", weights_only=False)
    xs = payload["x"]
    time_labels = payload["time"].to(torch.float32)
    visibility = payload["visibility"].to(torch.float32)
    gt_valid = payload["gt_valid"].to(torch.bool)
    fs = float(meta["fs"])
    time_downsample = int(meta["time_downsample"])
    window_samples = int(meta["window_samples"])
    peak_times = []
    peak_amps = []
    peak_valids = []
    peak_indices = []
    gt_peak_indices = []
    matched_total = 0
    injected_total = 0
    for i in range(int(xs.shape[0])):
        peak_time, peak_amp, peak_valid, peak_index, gt_peak_index, matched, injected = _convert_one_sample(
            xs[i],
            time_labels[i],
            visibility[i],
            gt_valid[i],
            fs=fs,
            time_downsample=time_downsample,
            window_samples=window_samples,
            peak_cfg=peak_cfg,
        )
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        gt_peak_indices.append(gt_peak_index)
        matched_total += matched
        injected_total += injected
    out_payload = {
        "x": xs.contiguous(),
        "peak_time": torch.stack(peak_times, dim=0).contiguous(),
        "peak_amp": torch.stack(peak_amps, dim=0).contiguous(),
        "peak_valid": torch.stack(peak_valids, dim=0).contiguous(),
        "peak_index": torch.stack(peak_indices, dim=0).contiguous(),
        "gt_peak_index": torch.stack(gt_peak_indices, dim=0).contiguous(),
        "visibility": visibility.contiguous(),
        "direction": payload["direction"].to(torch.long).contiguous(),
        "speed": payload["speed"].to(torch.float32).contiguous(),
        "gt_valid": gt_valid.contiguous(),
    }
    torch.save(out_payload, str(out_path))
    return {
        "samples": int(xs.shape[0]),
        "matched_gt_points": int(matched_total),
        "injected_gt_points": int(injected_total),
    }


def _convert_shard_task(
    shard_idx: int,
    shard_name: str,
    in_dir: Path,
    out_dir: Path,
    meta: dict[str, Any],
    peak_cfg: PeakDetectionConfig,
) -> tuple[int, str, dict[str, int]]:
    stats = _convert_shard(in_dir / shard_name, out_dir / shard_name, meta=meta, peak_cfg=peak_cfg)
    return int(shard_idx), str(shard_name), stats


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    meta = _load_meta(in_dir)
    shards = [str(item) for item in meta.get("shards", [])]
    if not shards:
        raise ValueError(f"No shards listed in {in_dir / 'meta.json'}")
    _prepare_out_dir(out_dir, bool(args.overwrite))
    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(args.peak_candidates_per_channel),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=float(args.peak_match_tolerance_s),
    )
    t0 = time.perf_counter()
    converted_shards = []
    shard_results: list[tuple[int, str, dict[str, int]]] = []
    total_samples = 0
    total_matched = 0
    total_injected = 0
    workers = int(max(1, args.workers))
    if workers <= 1:
        for idx, shard in enumerate(shards):
            shard_results.append(_convert_shard_task(idx, shard, in_dir, out_dir, meta, peak_cfg))
            _, done_shard, stats = shard_results[-1]
            print(
                f"converted {done_shard}: samples={stats['samples']}, matched={stats['matched_gt_points']}, "
                f"injected={stats['injected_gt_points']}, elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(shards))) as executor:
            futures = [
                executor.submit(_convert_shard_task, idx, shard, in_dir, out_dir, meta, peak_cfg)
                for idx, shard in enumerate(shards)
            ]
            for future in as_completed(futures):
                result = future.result()
                shard_results.append(result)
                _, done_shard, stats = result
                print(
                    f"converted {done_shard}: samples={stats['samples']}, matched={stats['matched_gt_points']}, "
                    f"injected={stats['injected_gt_points']}, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
    for _, shard, stats in sorted(shard_results, key=lambda item: item[0]):
        converted_shards.append(shard)
        total_samples += int(stats["samples"])
        total_matched += int(stats["matched_gt_points"])
        total_injected += int(stats["injected_gt_points"])
    out_meta = dict(meta)
    out_meta.update(
        {
            "format": "peak_slot_shards_v1",
            "source_format": str(meta.get("format", "track_slot_shards_v1")),
            "source_dir": str(in_dir),
            "created_at_unix": time.time(),
            "shards": converted_shards,
            "peak_candidates_per_channel": int(args.peak_candidates_per_channel),
            "peak_none_index": int(args.peak_candidates_per_channel),
            "peak_detection": {
                "min_distance_s": float(args.peak_min_distance_s),
                "min_height": float(args.peak_min_height),
                "prominence": float(args.peak_prominence),
                "match_tolerance_s": float(args.peak_match_tolerance_s),
            },
            "peak_conversion_stats": {
                "samples": int(total_samples),
                "matched_gt_points": int(total_matched),
                "injected_gt_points": int(total_injected),
                "workers": int(workers),
            },
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"done: samples={total_samples}, shards={len(converted_shards)}, "
        f"matched={total_matched}, injected={total_injected}, out_dir={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
