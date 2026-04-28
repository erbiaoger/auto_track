"""Run a trained trajectory-query model on SAC data and export tracks.

用途：
    加载 `train_trajectory_model.py` 训练得到的 PyTorch checkpoint，对一个 SAC
    数据目录执行深度学习轨迹提取，并导出与现有 GUI 兼容的 `auto_tracks.csv`。

用例：
    uv run python KF/auto_track/infer_trajectory_model.py \
        --data-folder KF/auto_track/data \
        --model KF/auto_track/models/trajectory_query_demo/checkpoint_best.pt \
        --window-start-s 0 \
        --window-seconds 120 \
        --out-csv KF/auto_track/data/auto_tracks_deep.csv

参数说明：
    默认只推理一个当前窗口；加 `--full-hour` 后按 tile 分块处理整段数据。
    --objectness-threshold 控制保留多少条车辆 query。
    --visibility-threshold 控制每条轨迹上哪些通道点被保留。
    --refine-radius-samples 会在预测点附近搜索局部振幅峰值，提高坐标精度。

输出：
    - auto_tracks_deep.csv：轨迹点表。
    - summary.json：推理参数、轨迹数量和耗时。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotrack.core.auto_track_backend import AutoTrackBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer vehicle trajectories with a trained deep-learning model.")
    parser.add_argument("--data-folder", required=True, help="SAC data folder.")
    parser.add_argument("--model", required=True, help="Trajectory model checkpoint path (.pt/.pth).")
    parser.add_argument("--out-csv", default="", help="Output CSV path; default is <data-folder>/auto_tracks_deep.csv.")
    parser.add_argument("--device", default="", help="Torch device: cuda, mps, cpu, auto, or empty for auto.")
    parser.add_argument("--window-start-s", type=float, default=0.0, help="Current-window start time in seconds.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Current-window length in seconds.")
    parser.add_argument("--full-hour", action="store_true", help="Run tiled inference on the whole data span.")
    parser.add_argument("--tile-seconds", type=float, default=120.0, help="Tile length for full-hour mode.")
    parser.add_argument("--overlap-seconds", type=float, default=20.0, help="Tile overlap for full-hour mode.")
    parser.add_argument("--objectness-threshold", type=float, default=0.5, help="Query objectness threshold.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5, help="Per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum visible points per trajectory.")
    parser.add_argument("--refine-radius-samples", type=int, default=120, help="Local peak refinement radius in samples.")
    parser.add_argument("--nms-time-radius", type=float, default=0.18, help="Track deduplication tolerance in seconds.")
    parser.add_argument("--speed-min-kmh", type=float, default=60.0, help="Legacy-compatible minimum speed metadata.")
    parser.add_argument("--speed-max-kmh", type=float, default=120.0, help="Legacy-compatible maximum speed metadata.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    backend = AutoTrackBackend(data_folder=args.data_folder)
    if backend.init_error:
        raise RuntimeError(backend.init_error)

    if not args.full_hour:
        backend.window_size = int(round(float(args.window_seconds) * backend.fs))
        backend.current_start = int(round(float(args.window_start_s) * backend.fs))
        backend.update_view_window()

    summary = backend.run_auto_extract(
        direction="forward",
        speed_min_kmh=float(args.speed_min_kmh),
        speed_max_kmh=float(args.speed_max_kmh),
        prominence=0.4,
        min_peak_distance=500,
        min_track_channels=12,
        edge_min_track_channels=4,
        edge_time_margin_seconds=15.0,
        edge_min_score_scale=0.2,
        tile_seconds=float(args.tile_seconds),
        overlap_seconds=float(args.overlap_seconds),
        nms_time_radius=float(args.nms_time_radius),
        current_window_only=not bool(args.full_hour),
        engine="deep_learning",
        dl_model_path=str(args.model),
        dl_device=str(args.device),
        dl_objectness_threshold=float(args.objectness_threshold),
        dl_visibility_threshold=float(args.visibility_threshold),
        dl_min_visible_channels=int(args.min_visible_channels),
        dl_refine_radius_samples=int(args.refine_radius_samples),
    )
    out_csv = str(args.out_csv).strip() or str(Path(args.data_folder).expanduser() / "auto_tracks_deep.csv")
    if not backend.tracks:
        out_path = Path(out_csv).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path = out_path.parent / "summary.json"
        summary = dict(backend.last_summary) if backend.last_summary else dict(summary)
        summary["csv_path"] = str(out_path)
        summary["summary_path"] = str(summary_path)
        summary["note"] = (
            "No tracks predicted. Try lowering --objectness-threshold / "
            "--visibility-threshold or check that the checkpoint has trained long enough."
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            "done: tracks=0, points=0. "
            f"No CSV written because no tracks were predicted. summary={summary_path}"
        )
        return 0
    csv_path, summary_path = backend.export_csv(out_csv)
    print(
        f"done: tracks={summary.get('track_count', 0)}, "
        f"points={summary.get('total_points', 0)}, csv={csv_path}, summary={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
