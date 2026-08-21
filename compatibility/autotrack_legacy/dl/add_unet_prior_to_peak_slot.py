"""Add waveform-line U-Net prior channels to PeakSlotNet shards.

This keeps the original PeakSlotNet peak-candidate tables intact, then appends
one extra input channel sampled from a trained waveform-line U-Net probability
image. When --track-dir is provided, the U-Net image input is rendered from the
matching TrackSlotNet Gaussian-window heatmap while labels and candidate tables
come from the PeakSlotNet shard. The output remains a `peak_slot_shards_v1`
dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from autotrack.dl.peak_slot_model import PeakDetectionConfig, detect_peak_candidates_from_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a waveform-line U-Net prior channel to peak_slot shards.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Input peak_slot dataset directory.")
    parser.add_argument(
        "--track-dir",
        type=Path,
        default=None,
        help=(
            "Optional matching track_slot dataset directory. If set, its Gaussian-window x tensor is rendered "
            "for waveform-line U-Net inference; peak candidates and labels are still read from --in-dir."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output peak_slot dataset directory with x in_channels=2.")
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
    parser.add_argument("--prior-threshold", type=float, default=0.0, help="Set values below this probability to 0; 0 keeps continuous probabilities.")
    parser.add_argument("--prior-scale", type=float, default=1.0, help="Multiplier applied to the prior channel after sampling.")
    parser.add_argument("--x-dtype", choices=["preserve", "float16", "float32"], default="preserve", help="Output x dtype.")
    parser.add_argument("--candidate-source", default="raw", choices=["raw", "prior", "raw_prior_union"], help="Optionally rebuild peak tables from raw/prior output channels.")
    parser.add_argument("--peak-candidates-per-channel", type=int, default=0, help="K per channel when rebuilding peak tables. 0 keeps input meta/default.")
    parser.add_argument("--peak-min-distance-s", type=float, default=0.15, help="Raw/prior peak minimum distance when rebuilding peak tables.")
    parser.add_argument("--peak-min-height", type=float, default=0.02, help="Raw peak minimum height when rebuilding peak tables.")
    parser.add_argument("--peak-prominence", type=float, default=0.02, help="Raw peak prominence when rebuilding peak tables.")
    parser.add_argument("--peak-match-tolerance-s", type=float, default=0.25, help="GT-to-candidate tolerance recorded in rebuilt metadata.")
    parser.add_argument("--prior-peak-min-height", type=float, default=0.08, help="Prior peak minimum height when rebuilding peak tables.")
    parser.add_argument("--prior-peak-prominence", type=float, default=0.02, help="Prior peak prominence when rebuilding peak tables.")
    parser.add_argument("--candidate-merge-tolerance-s", type=float, default=0.20, help="Merge raw/prior candidates on the same channel within this many seconds.")
    parser.add_argument("--prior-score-scale", type=float, default=0.85, help="Score multiplier for prior candidates in raw_prior_union mode.")
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
    track_dir = Path(args.track_dir).expanduser() if args.track_dir is not None else None
    track_meta: dict[str, Any] | None = None
    if track_dir is not None:
        track_meta = _load_meta(track_dir)
        if str(track_meta.get("format", "")) != "track_slot_shards_v1":
            raise ValueError(f"--track-dir is not track_slot_shards_v1: {track_dir}")
        if int(track_meta.get("in_channels", 1)) != 1:
            raise ValueError(f"Expected --track-dir in_channels=1, got {track_meta.get('in_channels')!r}")
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

    t0 = time.perf_counter()
    out_shards: list[str] = []
    converted_samples = 0
    prior_stats: list[dict[str, float | int | str]] = []
    candidate_stats: dict[str, int] = {"raw_candidate_count": 0, "prior_candidate_count": 0, "merged_candidate_count": 0}
    peak_k = int(args.peak_candidates_per_channel) if int(args.peak_candidates_per_channel) > 0 else int(meta.get("peak_candidates_per_channel", 64))
    peak_cfg = PeakDetectionConfig(
        candidates_per_channel=int(peak_k),
        min_distance_s=float(args.peak_min_distance_s),
        min_height=float(args.peak_min_height),
        prominence=float(args.peak_prominence),
        match_tolerance_s=float(args.peak_match_tolerance_s),
        candidate_source=str(args.candidate_source),
        prior_min_height=float(args.prior_peak_min_height),
        prior_prominence=float(args.prior_peak_prominence),
        merge_tolerance_s=float(args.candidate_merge_tolerance_s),
        prior_score_scale=float(args.prior_score_scale),
    )
    fs = float(meta.get("fs", 1000.0))
    time_downsample = int(meta.get("time_downsample", 10))
    window_samples = int(meta.get("window_samples", 0))
    for shard_idx, shard_name in enumerate(shards):
        payload = torch.load(str(in_dir / shard_name), map_location="cpu", weights_only=False)
        x = payload["x"]
        if int(x.ndim) != 4 or int(x.shape[1]) != 1:
            raise ValueError(f"{shard_name} x must have shape [N,1,C,T], got {tuple(x.shape)}")
        prior_source_x = x
        if track_dir is not None:
            track_payload = torch.load(str(track_dir / shard_name), map_location="cpu", weights_only=False)
            prior_source_x = track_payload["x"]
            if tuple(prior_source_x.shape) != tuple(x.shape):
                raise ValueError(
                    f"{shard_name} track x shape {tuple(prior_source_x.shape)} does not match peak x shape {tuple(x.shape)}"
                )
            if int(prior_source_x.ndim) != 4 or int(prior_source_x.shape[1]) != 1:
                raise ValueError(f"{shard_name} track x must have shape [N,1,C,T], got {tuple(prior_source_x.shape)}")
        prior = _build_prior_channel(
            prior_source_x[:, 0].to(torch.float32),
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
        out_payload = dict(payload)
        out_payload["x"] = out_x.contiguous()
        if str(args.candidate_source) != "raw":
            rebuilt = _rebuild_peak_tables(
                out_x.to(torch.float32),
                fs=float(fs),
                time_downsample=int(time_downsample),
                window_samples=int(window_samples),
                peak_cfg=peak_cfg,
            )
            out_payload["peak_time"] = rebuilt["peak_time"].contiguous()
            out_payload["peak_amp"] = rebuilt["peak_amp"].contiguous()
            out_payload["peak_valid"] = rebuilt["peak_valid"].contiguous()
            out_payload["peak_index"] = rebuilt["peak_index"].contiguous()
            _merge_candidate_stats(candidate_stats, rebuilt["candidate_stats"])
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
            "mode": "peak_slot_with_waveform_line_unet_prior",
            "source_data_dir": str(in_dir),
            "track_source_data_dir": None if track_dir is None else str(track_dir),
            "num_samples": int(converted_samples),
            "shards": out_shards,
            "in_channels": 2,
            "peak_candidates_per_channel": int(peak_k),
            "peak_none_index": int(peak_k),
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
                **candidate_stats,
            },
            "prior_channel": {
                "index": 1,
                "kind": "waveform_line_unet_probability",
                "source_x": "track_slot_gaussian_window_x" if track_dir is not None else "peak_slot_x",
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


def _install_waveform_task_imports(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"waveform_task_dir does not exist: {path}")
    sys.path.insert(0, str(path))


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


def _rebuild_peak_tables(
    x: torch.Tensor,
    *,
    fs: float,
    time_downsample: int,
    window_samples: int,
    peak_cfg: PeakDetectionConfig,
) -> dict[str, Any]:
    peak_times = []
    peak_amps = []
    peak_valids = []
    peak_indices = []
    candidate_stats: dict[str, int] = {"raw_candidate_count": 0, "prior_candidate_count": 0, "merged_candidate_count": 0}
    for i in range(int(x.shape[0])):
        peak_time, peak_amp, peak_valid, peak_index, stats = detect_peak_candidates_from_input(
            x[i].to(torch.float32),
            fs=float(fs),
            time_downsample=int(time_downsample),
            window_samples=int(window_samples),
            config=peak_cfg,
            return_stats=True,
        )
        peak_times.append(peak_time)
        peak_amps.append(peak_amp)
        peak_valids.append(peak_valid)
        peak_indices.append(peak_index)
        _merge_candidate_stats(candidate_stats, dict(stats))
    return {
        "peak_time": torch.stack(peak_times, dim=0),
        "peak_amp": torch.stack(peak_amps, dim=0),
        "peak_valid": torch.stack(peak_valids, dim=0),
        "peak_index": torch.stack(peak_indices, dim=0),
        "candidate_stats": dict(candidate_stats),
    }


def _merge_candidate_stats(target: dict[str, int], stats: dict[str, object]) -> None:
    for key in ("raw_candidate_count", "prior_candidate_count", "merged_candidate_count"):
        target[key] = int(target.get(key, 0)) + int(stats.get(key, 0) or 0)


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
