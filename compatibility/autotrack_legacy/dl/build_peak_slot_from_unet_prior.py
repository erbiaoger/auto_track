"""Build 1-channel PeakSlotNet shards directly from waveform-line U-Net output.

This experimental converter replaces the original peak-slot input heatmap with
the waveform-line U-Net probability map itself, then re-detects peak
candidates on that line-probability signal. It is intended for cascade tests:

    DAS/gauss -> waveform-line U-Net -> line probability -> PeakSlotNet
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from autotrack.dl.add_unet_prior_to_peak_slot import (
    _build_prior_channel,
    _convert_x_dtype,
    _install_waveform_task_imports,
    _json_ready,
    _load_meta,
    _load_unet,
    _prepare_out_dir,
    _resolve_device,
)
from autotrack.dl.convert_track_slot_to_peak_slot import _insert_candidate
from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace peak_slot input with waveform-line U-Net output and rebuild peaks.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Input peak_slot dataset directory.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak_slot dataset directory with line-only x.")
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
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Multiplier applied to the prior channel.")
    parser.add_argument("--x-dtype", choices=["preserve", "float16", "float32"], default="preserve", help="Output x dtype.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=64, help="Peak candidates K per channel.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Minimum distance between candidates.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Minimum normalized peak height.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Minimum normalized peak prominence.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="Max GT-to-candidate match distance.")
    parser.add_argument("--max-shards", type=int, default=0, help="Smoke-test limit. 0 converts all shards.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    waveform_task_dir = Path(args.waveform_task_dir).expanduser().resolve()
    _install_waveform_task_imports(waveform_task_dir)

    from model.network import UNetConfig, WaveformLineUNet  # type: ignore
    from render import RenderConfig, channel_x_positions, render_waveform_image  # type: ignore

    meta = _load_meta(in_dir)
    if str(meta.get("format", "")) != "peak_slot_shards_v1":
        raise ValueError(f"Input dataset is not peak_slot_shards_v1: {in_dir}")
    if int(meta.get("in_channels", 1)) != 1:
        raise ValueError(f"Expected input in_channels=1, got {meta.get('in_channels')!r}")
    shards = [str(item) for item in meta.get("shards", [])]
    if int(args.max_shards) > 0:
        shards = shards[: int(args.max_shards)]
    if not shards:
        raise ValueError(f"No shards to convert in {in_dir}")

    _prepare_out_dir(out_dir, overwrite=bool(args.overwrite))
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
    )

    fs = float(meta["fs"])
    time_downsample = int(meta["time_downsample"])
    window_samples = int(meta["window_samples"])
    t0 = time.perf_counter()
    out_shards: list[str] = []
    converted_samples = 0
    prior_stats: list[dict[str, float | int | str]] = []
    remap_stats = {"matched_gt_points": 0, "injected_gt_points": 0}
    for shard_idx, shard_name in enumerate(shards):
        payload = torch.load(str(in_dir / shard_name), map_location="cpu", weights_only=False)
        x = payload["x"]
        if int(x.ndim) != 4 or int(x.shape[1]) != 1:
            raise ValueError(f"{shard_name} x must have shape [N,1,C,T], got {tuple(x.shape)}")
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
        rebuilt = _rebuild_peak_slot_targets(
            prior=prior,
            old_peak_time=payload["peak_time"].to(torch.float32),
            old_gt_peak_index=payload["gt_peak_index"].to(torch.long),
            visibility=payload["visibility"].to(torch.float32),
            gt_valid=payload["gt_valid"].to(torch.bool),
            fs=fs,
            time_downsample=time_downsample,
            window_samples=window_samples,
            peak_cfg=peak_cfg,
        )
        remap_stats["matched_gt_points"] += int(rebuilt["matched_gt_points"])
        remap_stats["injected_gt_points"] += int(rebuilt["injected_gt_points"])
        out_x = _convert_x_dtype(prior[:, None, :, :].to(torch.float32), original=x, mode=str(args.x_dtype))
        out_payload = dict(payload)
        out_payload["x"] = out_x.contiguous()
        out_payload["peak_time"] = rebuilt["peak_time"].contiguous()
        out_payload["peak_amp"] = rebuilt["peak_amp"].contiguous()
        out_payload["peak_valid"] = rebuilt["peak_valid"].contiguous()
        out_payload["peak_index"] = rebuilt["peak_index"].contiguous()
        out_payload["gt_peak_index"] = rebuilt["gt_peak_index"].contiguous()
        out_name = f"shard_{shard_idx:06d}.pt"
        torch.save(out_payload, str(out_dir / out_name))
        out_shards.append(out_name)
        converted_samples += int(out_x.shape[0])
        prior_stats.append(
            {
                "shard": out_name,
                "samples": int(out_x.shape[0]),
                "prior_mean": float(prior.mean().item()),
                "prior_max": float(prior.max().item()),
                "prior_positive_ratio_0p5": float((prior >= 0.5).to(torch.float32).mean().item()),
            }
        )
        print(f"wrote {out_name}: samples={converted_samples}", flush=True)

    out_meta = dict(meta)
    out_meta.update(
        {
            "format": "peak_slot_shards_v1",
            "mode": "peak_slot_from_waveform_line_unet_only",
            "source_data_dir": str(in_dir),
            "num_samples": int(converted_samples),
            "shards": out_shards,
            "in_channels": 1,
            "peak_candidates_per_channel": int(args.peak_candidates_per_channel),
            "peak_none_index": int(args.peak_candidates_per_channel),
            "peak_detection": {
                "min_distance_s": float(args.peak_min_distance_s),
                "min_height": float(args.peak_min_height),
                "prominence": float(args.peak_prominence),
                "match_tolerance_s": float(args.peak_match_tolerance_s),
                **remap_stats,
            },
            "line_source": {
                "kind": "waveform_line_unet_probability",
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
    if int(args.max_shards) > 0:
        out_meta["shard_size"] = int(payload["x"].shape[0])
    (out_dir / "meta.json").write_text(json.dumps(_json_ready(out_meta), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"done: samples={converted_samples}, shards={len(out_shards)}, out_dir={out_dir}", flush=True)
    return 0


def _rebuild_peak_slot_targets(
    *,
    prior: torch.Tensor,
    old_peak_time: torch.Tensor,
    old_gt_peak_index: torch.Tensor,
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
    tolerance_norm = float(peak_cfg.match_tolerance_s) * float(fs) / float(max(1, window_samples - 1))
    for i in range(int(prior.shape[0])):
        heatmap = prior[i].to(torch.float32)
        peak_time, peak_amp, peak_valid, peak_index = detect_peak_candidates_from_tensor(
            heatmap,
            fs=float(fs),
            time_downsample=int(time_downsample),
            window_samples=int(window_samples),
            config=peak_cfg,
        )
        k_count = int(peak_valid.shape[1])
        gt_peak_index = torch.full_like(old_gt_peak_index[i], fill_value=k_count, dtype=torch.long)
        for g in torch.where(gt_valid[i])[0].tolist():
            for ch in torch.where(visibility[i, int(g)] > 0.5)[0].tolist():
                old_idx = int(old_gt_peak_index[i, int(g), int(ch)].item())
                if not (0 <= old_idx < int(old_peak_time.shape[2])):
                    continue
                t_norm = float(old_peak_time[i, int(ch), old_idx].item())
                valid = torch.where(peak_valid[int(ch)])[0]
                if valid.numel() > 0:
                    diffs = torch.abs(peak_time[int(ch), valid] - float(t_norm))
                    min_pos = int(torch.argmin(diffs).item())
                    if float(diffs[min_pos].item()) <= tolerance_norm:
                        matched_total += 1
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
                injected_total += 1
        for g in torch.where(gt_valid[i])[0].tolist():
            for ch in torch.where(visibility[i, int(g)] > 0.5)[0].tolist():
                old_idx = int(old_gt_peak_index[i, int(g), int(ch)].item())
                if not (0 <= old_idx < int(old_peak_time.shape[2])):
                    continue
                valid = torch.where(peak_valid[int(ch)])[0]
                if valid.numel() <= 0:
                    continue
                t_norm = float(old_peak_time[i, int(ch), old_idx].item())
                diffs = torch.abs(peak_time[int(ch), valid] - float(t_norm))
                min_pos = int(torch.argmin(diffs).item())
                if float(diffs[min_pos].item()) <= max(tolerance_norm, 1.0 / float(max(1, window_samples - 1))):
                    gt_peak_index[int(g), int(ch)] = int(valid[min_pos].item())
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        gt_peak_indices.append(gt_peak_index)
    return {
        "peak_time": torch.stack(peak_times, dim=0),
        "peak_amp": torch.stack(peak_amps, dim=0),
        "peak_valid": torch.stack(peak_valids, dim=0),
        "peak_index": torch.stack(peak_indices, dim=0),
        "gt_peak_index": torch.stack(gt_peak_indices, dim=0),
        "matched_gt_points": int(matched_total),
        "injected_gt_points": int(injected_total),
    }


if __name__ == "__main__":
    raise SystemExit(main())
