from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autotrack.dl.multi_vehicle_pipeline import MultiVehiclePipelineConfig, _proposal_peak_boxes, extract_multi_vehicle_tracks
from autotrack.dl.vehicle_proposal_net import load_checkpoint_model, prepare_window_input


def main() -> int:
    root = ROOT
    bench = torch.load(str(root / "data" / "multi_vehicle_set_rich_bench.pt"), map_location="cpu", weights_only=False)
    meta = dict(bench.get("meta", {}))
    sample = bench["samples"][0]
    raw = sample["target"]["raw_window"].to(torch.float32).cpu().numpy()
    fs = float(meta.get("fs", 1000.0))
    dx_m = float(meta.get("dx_m", 100.0))
    time_downsample = int(meta.get("time_downsample", 10))
    clip_ratio = float(meta.get("clip_ratio", 1.35))
    window_seconds = raw.shape[1] / fs

    proposal_model, _ = load_checkpoint_model(root / "models" / "vehicle_proposal_rich_v4_checkpoint_best.pt", device="cpu")
    proposal_model.eval()
    x = prepare_window_input(raw, time_downsample=time_downsample, clip_ratio=clip_ratio).unsqueeze(0)
    with torch.inference_mode():
        outputs = proposal_model(x)
    heat = torch.sigmoid(outputs["heatmap_logits"][0]).cpu().numpy()
    boxes = _proposal_peak_boxes(heat, min_score=0.5, pad_channels=3, pad_time=20, max_peaks=12)

    tracks = extract_multi_vehicle_tracks(
        raw,
        fs=fs,
        dx_m=dx_m,
        direction="forward",
        vmin_kmh=70.0,
        vmax_kmh=90.0,
        config=MultiVehiclePipelineConfig(
            candidate_limit=2,
            candidate_min_score=2.0,
            dedup_tolerance_samples=180,
            dedup_min_overlap_channels=3,
            dedup_min_overlap_ratio=0.45,
            crop_channel_margin=4,
            crop_time_margin_s=4.0,
            refine_with_model=True,
            min_model_confidence=0.1,
            proposal_model_path=None,
            iterative_extraction=False,
            max_iterations=1,
        ),
        model_path=str(root / "models" / "vehicle_trace_exact_train_checkpoint_best.pt"),
        device="cuda",
    )

    raw_ds = raw[:, ::time_downsample]
    fig, axes = plt.subplots(1, 2, figsize=(18, 5.6), dpi=180, constrained_layout=True)
    colors = ["cyan", "lime", "yellow", "orange", "red", "white"]

    ax = axes[0]
    ax.imshow(raw_ds, aspect="auto", origin="lower", cmap="magma", extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5))
    ax.imshow(
        heat,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        alpha=0.55,
        extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5),
        vmin=0.0,
        vmax=1.0,
    )
    for score, (ch0, ch1, t0, t1) in boxes:
        x0 = float(t0) * window_seconds / float(heat.shape[1])
        x1 = float(t1) * window_seconds / float(heat.shape[1])
        rect = Rectangle((x0, ch0 - 0.5), x1 - x0, ch1 - ch0, fill=False, edgecolor="white", linewidth=1.4, alpha=0.95)
        ax.add_patch(rect)
    ax.set_title(f"DL coarse recognition | sample 0 | boxes={len(boxes)}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("channel")

    ax = axes[1]
    ax.imshow(raw_ds, aspect="auto", origin="lower", cmap="magma", extent=(0.0, window_seconds, -0.5, raw.shape[0] - 0.5))
    for j, tr in enumerate(tracks):
        pts = sorted(tr.points, key=lambda p: float(p.time_s))
        ax.plot([float(p.time_s) for p in pts], [int(p.ch_idx) for p in pts], color=colors[j % len(colors)], linewidth=2.3)
    ax.set_title(f"Postprocessed result | tracks={len(tracks)}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("channel")

    out = root / "results" / "final_dl_vs_postprocess_overlay.png"
    fig.savefig(out)
    print(out)
    print({"dl_boxes": len(boxes), "post_tracks": len(tracks)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
