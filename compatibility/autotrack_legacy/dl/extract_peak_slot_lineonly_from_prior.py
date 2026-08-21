"""Extract line-only PeakSlot shards from an existing raw+prior PeakSlot dataset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from autotrack.dl.convert_track_slot_to_peak_slot import _insert_candidate
from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract x[:,prior_channel] into one-channel line-only PeakSlot shards.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Input peak_slot dataset with raw+prior x.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output one-channel line-only peak_slot dataset.")
    parser.add_argument("--prior-channel", type=int, default=-1, help="Input x channel to extract. -1 uses meta prior_channel.index.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=96, help="Peak candidates K per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum distance between candidates.")
    parser.add_argument("--peak-min-height", type=float, default=0.08, help="Minimum line-probability peak height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum line-probability peak prominence.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="Max GT-to-candidate match distance.")
    parser.add_argument("--workers", type=int, default=1, help="Shard-level worker count.")
    parser.add_argument("--max-shards", type=int, default=0, help="Smoke-test limit. 0 converts all shards.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    meta = _load_meta(in_dir)
    if str(meta.get("format", "")) != "peak_slot_shards_v1":
        raise ValueError(f"Input is not peak_slot_shards_v1: {in_dir}")
    prior_channel = int(args.prior_channel)
    if prior_channel < 0:
        prior_channel = int(dict(meta.get("prior_channel", {})).get("index", 1))
    shards = [str(item) for item in meta.get("shards", [])]
    if int(args.max_shards) > 0:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise ValueError(f"No shards listed in {in_dir / 'meta.json'}")
    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))

    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(args.peak_candidates_per_channel),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=float(args.peak_match_tolerance_s),
        candidate_source="raw",
    )
    fs = float(meta["fs"])
    time_downsample = int(meta["time_downsample"])
    window_samples = int(meta["window_samples"])
    t0 = time.perf_counter()
    converted: list[tuple[int, str, dict[str, int]]] = []
    workers = int(max(1, args.workers))
    if workers <= 1:
        for idx, shard in enumerate(shards):
            stats = _convert_shard(
                in_dir / shard,
                out_dir / f"shard_{idx:06d}.pt",
                prior_channel=prior_channel,
                peak_cfg=peak_cfg,
                fs=fs,
                time_downsample=time_downsample,
                window_samples=window_samples,
            )
            converted.append((idx, f"shard_{idx:06d}.pt", stats))
            print(
                f"wrote shard_{idx:06d}.pt: samples={stats['samples']}, matched={stats['matched_gt_points']}, "
                f"injected={stats['injected_gt_points']}, elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(shards))) as executor:
            futures = [
                executor.submit(
                    _convert_shard_task,
                    idx,
                    str(in_dir / shard),
                    str(out_dir / f"shard_{idx:06d}.pt"),
                    prior_channel,
                    peak_cfg,
                    fs,
                    time_downsample,
                    window_samples,
                )
                for idx, shard in enumerate(shards)
            ]
            for future in as_completed(futures):
                idx, out_name, stats = future.result()
                converted.append((idx, out_name, stats))
                print(
                    f"wrote {out_name}: samples={stats['samples']}, matched={stats['matched_gt_points']}, "
                    f"injected={stats['injected_gt_points']}, elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

    out_shards = []
    total_samples = 0
    matched_total = 0
    injected_total = 0
    for _, out_name, stats in sorted(converted, key=lambda item: item[0]):
        out_shards.append(out_name)
        total_samples += int(stats["samples"])
        matched_total += int(stats["matched_gt_points"])
        injected_total += int(stats["injected_gt_points"])

    out_meta = dict(meta)
    out_meta.update(
        {
            "format": "peak_slot_shards_v1",
            "mode": "peak_slot_lineonly_from_existing_prior",
            "source_data_dir": str(in_dir),
            "created_at_unix": time.time(),
            "num_samples": int(total_samples),
            "shards": out_shards,
            "in_channels": 1,
            "peak_candidates_per_channel": int(args.peak_candidates_per_channel),
            "peak_none_index": int(args.peak_candidates_per_channel),
            "peak_detection": {
                "candidate_source": "line_prior_channel",
                "source_prior_channel": int(prior_channel),
                "min_distance_s": float(args.peak_min_distance_s),
                "min_height": float(args.peak_min_height),
                "prominence": float(args.peak_prominence),
                "match_tolerance_s": float(args.peak_match_tolerance_s),
            },
            "peak_conversion_stats": {
                "samples": int(total_samples),
                "matched_gt_points": int(matched_total),
                "injected_gt_points": int(injected_total),
                "workers": int(workers),
            },
            "elapsed_seconds": float(time.perf_counter() - t0),
        }
    )
    (out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={total_samples}, shards={len(out_shards)}, out_dir={out_dir}", flush=True)
    return 0


def _convert_shard_task(
    idx: int,
    in_path: str,
    out_path: str,
    prior_channel: int,
    peak_cfg: PeakDetectionConfig,
    fs: float,
    time_downsample: int,
    window_samples: int,
) -> tuple[int, str, dict[str, int]]:
    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass
    out_name = Path(out_path).name
    stats = _convert_shard(
        Path(in_path),
        Path(out_path),
        prior_channel=int(prior_channel),
        peak_cfg=peak_cfg,
        fs=float(fs),
        time_downsample=int(time_downsample),
        window_samples=int(window_samples),
    )
    return int(idx), out_name, stats


def _convert_shard(
    in_path: Path,
    out_path: Path,
    *,
    prior_channel: int,
    peak_cfg: PeakDetectionConfig,
    fs: float,
    time_downsample: int,
    window_samples: int,
) -> dict[str, int]:
    payload = torch.load(str(in_path), map_location="cpu", weights_only=False)
    x = payload["x"]
    if not torch.is_tensor(x) or x.ndim != 4:
        raise ValueError(f"{in_path} x must have shape [N,in_channels,C,T], got {tuple(x.shape)}")
    if prior_channel < 0 or prior_channel >= int(x.shape[1]):
        raise ValueError(f"prior_channel={prior_channel} outside x shape {tuple(x.shape)}")
    out_x = x[:, int(prior_channel) : int(prior_channel) + 1].contiguous()
    time_labels = _resolve_time_labels(payload)
    visibility = payload["visibility"].to(torch.float32)
    gt_valid = payload["gt_valid"].to(torch.bool)
    peak_times = []
    peak_amps = []
    peak_valids = []
    peak_indices = []
    gt_peak_indices = []
    matched_total = 0
    injected_total = 0
    for i in range(int(out_x.shape[0])):
        peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates_from_input(
            out_x[i].to(torch.float32),
            fs=float(fs),
            time_downsample=int(time_downsample),
            window_samples=int(window_samples),
            config=peak_cfg,
        )
        gt_peak_index = torch.full_like(time_labels[i], fill_value=int(peak_cfg.candidates_per_channel), dtype=torch.long)
        matched, injected = _map_gt_to_candidates(
            peak_time,
            peak_amp,
            peak_valid,
            peak_index,
            time_labels[i],
            visibility[i],
            gt_valid[i],
            heatmap=out_x[i, 0].to(torch.float32),
            peak_cfg=peak_cfg,
            fs=float(fs),
            window_samples=int(window_samples),
            time_downsample=int(time_downsample),
        )
        for g, ch, idx in matched:
            gt_peak_index[int(g), int(ch)] = int(idx)
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        gt_peak_indices.append(gt_peak_index)
        matched_total += len(matched)
        injected_total += int(injected)
    out_payload = dict(payload)
    out_payload["x"] = out_x.contiguous()
    out_payload["peak_time"] = torch.stack(peak_times, dim=0).contiguous()
    out_payload["peak_amp"] = torch.stack(peak_amps, dim=0).contiguous()
    out_payload["peak_valid"] = torch.stack(peak_valids, dim=0).contiguous()
    out_payload["peak_index"] = torch.stack(peak_indices, dim=0).contiguous()
    out_payload["gt_peak_index"] = torch.stack(gt_peak_indices, dim=0).contiguous()
    torch.save(out_payload, str(out_path))
    return {
        "samples": int(out_x.shape[0]),
        "matched_gt_points": int(matched_total),
        "injected_gt_points": int(injected_total),
    }


def _resolve_time_labels(payload: dict[str, Any]) -> torch.Tensor:
    if "time" in payload and torch.is_tensor(payload["time"]):
        return payload["time"].to(torch.float32)
    peak_time = payload["peak_time"].to(torch.float32)
    gt_peak_index = payload["gt_peak_index"].to(torch.long)
    visibility = payload["visibility"].to(torch.float32)
    gt_valid = payload["gt_valid"].to(torch.bool)
    out = torch.zeros_like(visibility, dtype=torch.float32)
    none_index = int(peak_time.shape[2])
    for sample in range(int(gt_peak_index.shape[0])):
        for g in torch.where(gt_valid[sample])[0].tolist():
            for ch in torch.where(visibility[sample, g] > 0.5)[0].tolist():
                idx = int(gt_peak_index[sample, g, ch].item())
                if 0 <= idx < none_index:
                    out[sample, g, ch] = float(peak_time[sample, ch, idx].item())
    return out


def _map_gt_to_candidates(
    peak_time: torch.Tensor,
    peak_amp: torch.Tensor,
    peak_valid: torch.Tensor,
    peak_index: torch.Tensor,
    time_label: torch.Tensor,
    visibility: torch.Tensor,
    gt_valid: torch.Tensor,
    *,
    heatmap: torch.Tensor,
    peak_cfg: PeakDetectionConfig,
    fs: float,
    window_samples: int,
    time_downsample: int,
) -> tuple[list[tuple[int, int, int]], int]:
    tolerance_norm = float(peak_cfg.match_tolerance_s) * float(fs) / float(max(1, window_samples - 1))
    injected = 0
    matches: list[tuple[int, int, int]] = []
    for g in torch.where(gt_valid)[0].tolist():
        for ch in torch.where(visibility[g] > 0.5)[0].tolist():
            t_norm = float(time_label[g, ch].item())
            valid = torch.where(peak_valid[ch])[0]
            if valid.numel() > 0:
                diffs = torch.abs(peak_time[ch, valid] - float(t_norm))
                min_pos = int(torch.argmin(diffs).item())
                if float(diffs[min_pos].item()) <= tolerance_norm:
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
    for g in torch.where(gt_valid)[0].tolist():
        for ch in torch.where(visibility[g] > 0.5)[0].tolist():
            valid = torch.where(peak_valid[ch])[0]
            if valid.numel() <= 0:
                continue
            t_norm = float(time_label[g, ch].item())
            diffs = torch.abs(peak_time[ch, valid] - float(t_norm))
            min_pos = int(torch.argmin(diffs).item())
            if float(diffs[min_pos].item()) <= max(tolerance_norm, 1.0 / float(max(1, window_samples - 1))):
                matches.append((int(g), int(ch), int(valid[min_pos].item())))
    return matches, int(injected)


def _load_meta(data_dir: Path) -> dict[str, Any]:
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _prepare_out_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {out_dir}. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
