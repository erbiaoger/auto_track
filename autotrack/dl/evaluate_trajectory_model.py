"""Evaluate a trained trajectory-query model against simulated `tracks.json`.

用途：
    在模拟数据上抽取多个时间窗，运行深度学习轨迹模型，并与 `tracks.json`
    真值轨迹做车辆级匹配，计算车辆数量误差、Precision/Recall/F1 和轨迹点
    时间坐标误差。

用例：
    uv run python KF/auto_track/evaluate_trajectory_model.py \
        --data-folder KF/auto_track/data \
        --model KF/auto_track/models/trajectory_query_demo/checkpoint_best.pt \
        --window-count 50 \
        --window-seconds 120 \
        --out-json KF/auto_track/data/eval_deep.json

参数说明：
    --match-tolerance-s 表示同一通道预测点和真值点允许的最大时间误差。
    --min-overlap-channels 表示预测轨迹和真值轨迹至少要有多少个共同通道点。

输出：
    - 控制台摘要。
    - 可选 --out-json 指定的详细评估 JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from autotrack.dl.trajectory_set_model import (
    InferenceConfig,
    Track,
    TrackPoint,
    load_checkpoint_model,
    load_sac_matrix,
    load_tracks_json,
    predict_tracks_from_window,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trajectory-query model on simulated SAC data.")
    parser.add_argument("--data-folder", required=True, help="SAC data folder containing tracks.json.")
    parser.add_argument("--model", required=True, help="Trajectory model checkpoint path.")
    parser.add_argument("--out-json", default="", help="Optional evaluation JSON output path.")
    parser.add_argument("--device", default="", help="Torch device: cuda, mps, cpu, or empty for auto.")
    parser.add_argument("--window-count", type=int, default=50, help="Number of sampled evaluation windows.")
    parser.add_argument("--window-seconds", type=float, default=120.0, help="Evaluation window length.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for window sampling.")
    parser.add_argument("--objectness-threshold", type=float, default=0.5, help="Query objectness threshold.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5, help="Per-channel visibility threshold.")
    parser.add_argument("--min-visible-channels", type=int, default=3, help="Minimum predicted points per trajectory.")
    parser.add_argument("--refine-radius-samples", type=int, default=120, help="Local peak refinement radius in samples.")
    parser.add_argument("--match-tolerance-s", type=float, default=0.25, help="Maximum median matched-point time error.")
    parser.add_argument("--min-overlap-channels", type=int, default=3, help="Minimum common channels for a match.")
    return parser.parse_args()


def _gt_tracks_for_window(payload: dict, start: int, end: int, fs: float, min_points: int) -> list[Track]:
    tracks: list[Track] = []
    for raw in payload.get("tracks", []):
        points: list[TrackPoint] = []
        for point in raw.get("points", []):
            t_idx = int(point.get("t_idx", -1))
            if not (start <= t_idx < end):
                continue
            ch = int(point.get("ch_idx", -1))
            if ch < 0:
                continue
            local_t = int(t_idx - start)
            points.append(
                TrackPoint(
                    ch_idx=ch,
                    t_idx=local_t,
                    time_s=float(local_t) / float(fs),
                    offset_m=float(point.get("offset_m", ch)),
                    amp=float(point.get("amp", 1.0)),
                    score=1.0,
                )
            )
        if len(points) >= int(min_points):
            tracks.append(
                Track(
                    track_id=int(raw.get("track_id", len(tracks))),
                    direction=str(raw.get("direction", "forward")),
                    points=sorted(points, key=lambda p: p.ch_idx),
                    total_score=float(len(points)),
                    mean_speed_kmh=float(raw.get("speed_kmh", np.nan)),
                )
            )
    return tracks


def _pair_cost(pred: Track, gt: Track, tol_samples: int, min_overlap: int) -> float:
    pred_map = {int(p.ch_idx): int(p.t_idx) for p in pred.points}
    gt_map = {int(p.ch_idx): int(p.t_idx) for p in gt.points}
    common = sorted(set(pred_map) & set(gt_map))
    if len(common) < int(min_overlap):
        return 1e9
    diffs = np.array([abs(pred_map[ch] - gt_map[ch]) for ch in common], dtype=np.float64)
    med = float(np.median(diffs))
    if med > float(tol_samples):
        return 1e9
    return med + 0.05 * abs(len(pred.points) - len(gt.points))


def _match_tracks(pred: list[Track], gt: list[Track], tol_samples: int, min_overlap: int) -> tuple[list[tuple[int, int, float]], int, int]:
    if not pred or not gt:
        return [], len(pred), len(gt)
    cost = np.full((len(pred), len(gt)), 1e9, dtype=np.float64)
    for i, pred_track in enumerate(pred):
        for j, gt_track in enumerate(gt):
            cost[i, j] = _pair_cost(pred_track, gt_track, tol_samples, min_overlap)
    rows, cols = linear_sum_assignment(cost)
    matches = [(int(i), int(j), float(cost[i, j])) for i, j in zip(rows, cols) if cost[i, j] < 1e9]
    return matches, len(pred) - len(matches), len(gt) - len(matches)


def main() -> int:
    args = parse_args()
    data, fs, x_axis_m, _ = load_sac_matrix(args.data_folder)
    payload = load_tracks_json(args.data_folder)
    model, checkpoint = load_checkpoint_model(args.model, device=str(args.device).strip() or None)
    dataset_cfg = dict(checkpoint.get("dataset_config", {}))
    infer_cfg = InferenceConfig(
        time_downsample=int(dataset_cfg.get("time_downsample", 10)),
        objectness_threshold=float(args.objectness_threshold),
        visibility_threshold=float(args.visibility_threshold),
        min_visible_channels=int(args.min_visible_channels),
        refine_radius_samples=int(args.refine_radius_samples),
        speed_norm_kmh=float(dataset_cfg.get("speed_norm_kmh", 150.0)),
        clip_ratio=float(dataset_cfg.get("clip_ratio", 1.35)),
    )

    rng = np.random.default_rng(int(args.seed))
    window_samples = int(round(float(args.window_seconds) * float(fs)))
    window_samples = max(1, min(window_samples, int(data.shape[1])))
    max_start = max(0, int(data.shape[1]) - window_samples)
    tol_samples = int(round(float(args.match_tolerance_s) * float(fs)))

    details = []
    tp = fp = fn = 0
    all_costs: list[float] = []
    count_errors: list[int] = []
    for _ in range(int(args.window_count)):
        start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        end = start + window_samples
        gt_tracks = _gt_tracks_for_window(payload, start, end, fs, int(args.min_visible_channels))
        pred_tracks = predict_tracks_from_window(
            model=model,
            data_window=data[:, start:end],
            fs=float(fs),
            x_axis_m=x_axis_m,
            config=infer_cfg,
            device=str(args.device).strip() or None,
        )
        matches, fp_i, fn_i = _match_tracks(pred_tracks, gt_tracks, tol_samples, int(args.min_overlap_channels))
        tp += len(matches)
        fp += fp_i
        fn += fn_i
        all_costs.extend([m[2] for m in matches])
        count_errors.append(len(pred_tracks) - len(gt_tracks))
        details.append(
            {
                "start_sample": int(start),
                "end_sample": int(end),
                "gt_count": int(len(gt_tracks)),
                "pred_count": int(len(pred_tracks)),
                "matches": int(len(matches)),
                "fp": int(fp_i),
                "fn": int(fn_i),
            }
        )

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    summary = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "time_mae_samples": float(np.mean(all_costs)) if all_costs else None,
        "time_mae_seconds": float(np.mean(all_costs) / fs) if all_costs else None,
        "mean_count_error": float(np.mean(count_errors)) if count_errors else 0.0,
        "details": details,
    }
    print(
        f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}, "
        f"time_mae_s={summary['time_mae_seconds']}"
    )
    if str(args.out_json).strip():
        out_path = Path(args.out_json).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
