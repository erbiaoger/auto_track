from __future__ import annotations

from pathlib import Path

import torch

from autotrack.dl.build_reference_style_peakslot_dataset import main as build_reference_style_peakslot_dataset


def test_build_reference_style_peakslot_dataset(tmp_path: Path) -> None:
    out_dir = tmp_path / "peakslot_ref"
    rc = build_reference_style_peakslot_dataset(
        [
            "--out-dir",
            str(out_dir),
            "--unet-checkpoint",
            "/csim2/zhangzhiyu/MyProjects/waveform_line_task/models/unet_waveform_profile_only_cuda/checkpoint_best.pt",
            "--num-samples",
            "2",
            "--shard-size",
            "2",
            "--plot-peaks",
            "--overwrite",
        ]
    )
    assert rc == 0

    meta = torch.load(str(out_dir / "shard_000000.pt"), map_location="cpu", weights_only=False)
    assert tuple(meta["x"].shape)[1] == 2
    assert "peak_time" in meta
    assert "gt_peak_index" in meta
    assert tuple(meta["peak_time"].shape)[0] == 2
    assert tuple(meta["x"].shape)[2] == 50

    plot_dir = out_dir.with_name(f"{out_dir.name}_plots")
    assert (plot_dir / "sample_000000_input_target.png").is_file()
    assert (plot_dir / "sample_000001_input_target.png").is_file()
