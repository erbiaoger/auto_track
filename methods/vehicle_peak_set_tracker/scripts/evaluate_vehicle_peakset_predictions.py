from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate vehicle peak-set predictions against exported shard labels.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Labeled exported shard dataset.")
    parser.add_argument("--predictions", required=True, type=Path, help="Prediction JSONL file.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional output JSON.")
    parser.add_argument("--match-mae-s", type=float, default=3.0, help="Track match MAE threshold in seconds.")
    parser.add_argument("--anchor-hit-s", type=float, default=1.2, help="Observed anchor hit tolerance in seconds.")
    parser.add_argument("--jump-second-diff-s", type=float, default=4.0, help="Second-difference jump threshold in seconds.")
    parser.add_argument("--switch-margin-s", type=float, default=0.20, help="Margin in seconds for counting a point as switched to another GT track.")
    parser.add_argument("--switch-track-fraction", type=float, default=0.20, help="Fraction of comparable points that must switch to mark a track as switched.")
    parser.add_argument("--switch-track-min-points", type=int, default=2, help="Minimum switched points needed to mark a track as switched.")
    parser.add_argument("--fixed-dead-channels", default="5,6,15,16,22,36,38,45", help="Comma-separated fixed dead channels.")
    return parser.parse_args(argv)


def _parse_int_csv(text: str) -> set[int]:
    result: set[int] = set()
    for part in str(text).split(","):
        part = part.strip()
        if part:
            result.add(int(part))
    return result


def _load_predictions(path: Path) -> dict[int, dict[str, Any]]:
    items: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            items[int(item["sample_index"])] = item
    return items


class ShardLabels:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir).expanduser()
        meta = json.loads((self.dataset_dir / "meta.json").read_text(encoding="utf-8"))
        self.meta = meta
        self.shards = [self.dataset_dir / str(name) for name in meta.get("shards", [])]
        self.shard_sizes = [int(size) for size in meta.get("shard_sizes", [])]
        self.total = int(sum(self.shard_sizes))
        self._cache: dict[int, dict[str, Any]] = {}

    def _resolve(self, sample_index: int) -> tuple[int, int]:
        idx = int(sample_index)
        acc = 0
        for shard_idx, size in enumerate(self.shard_sizes):
            nxt = acc + int(size)
            if idx < nxt:
                return shard_idx, idx - acc
            acc = nxt
        raise IndexError(sample_index)

    def _load(self, shard_idx: int) -> dict[str, Any]:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(str(self.shards[shard_idx]), map_location="cpu", weights_only=False)
        return self._cache[shard_idx]

    def get(self, sample_index: int) -> dict[str, torch.Tensor]:
        shard_idx, local_idx = self._resolve(int(sample_index))
        payload = self._load(shard_idx)
        targets = payload["targets"]
        return {key: value[local_idx].clone() for key, value in targets.items() if torch.is_tensor(value)}


def _gt_tracks(target: dict[str, torch.Tensor], window_seconds: float) -> list[dict[str, Any]]:
    full_time = target["full_time"].to(torch.float32).cpu().numpy() * float(window_seconds)
    full_valid = target["full_valid"].to(torch.bool).cpu().numpy()
    gt_valid = target.get("gt_valid", torch.ones((full_time.shape[0],), dtype=torch.bool)).to(torch.bool).cpu().numpy()
    observed = target.get("observed_visibility", target["full_valid"]).to(torch.bool).cpu().numpy()
    missing = target.get("missing_channel_mask", torch.zeros_like(target["full_valid"])).to(torch.bool).cpu().numpy()
    tracks: list[dict[str, Any]] = []
    for g in range(int(full_time.shape[0])):
        if not bool(gt_valid[g]):
            continue
        valid = full_valid[g].astype(bool, copy=False)
        if not valid.any():
            continue
        tracks.append(
            {
                "gt_index": g,
                "time_s": full_time[g],
                "valid": valid,
                "observed": observed[g].astype(bool, copy=False),
                "missing": missing[g].astype(bool, copy=False),
            }
        )
    return tracks


def _pred_tracks(item: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for track in item.get("tracks", []):
        points = track.get("points", [])
        by_ch: dict[int, dict[str, Any]] = {}
        for point in points:
            by_ch[int(point["channel"])] = point
        result.append(
            {
                "track_id": int(track.get("track_id", len(result))),
                "objectness": float(track.get("objectness", 0.0)),
                "observed_ratio": float(track.get("observed_ratio", 0.0)),
                "total_score": float(track.get("total_score", 0.0)),
                "points_by_channel": by_ch,
            }
        )
    return result


def _pair_mae(pred: dict[str, Any], gt: dict[str, Any]) -> tuple[float, int]:
    diffs: list[float] = []
    by_ch = pred["points_by_channel"]
    for ch, valid in enumerate(gt["valid"]):
        if not bool(valid) or ch not in by_ch:
            continue
        diffs.append(abs(float(by_ch[ch]["time_s"]) - float(gt["time_s"][ch])))
    if not diffs:
        return float("inf"), 0
    return float(np.mean(diffs)), int(len(diffs))


def _greedy_match(preds: list[dict[str, Any]], gts: list[dict[str, Any]], threshold_s: float) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            mae, overlap = _pair_mae(pred, gt)
            if overlap > 0 and math.isfinite(mae):
                candidates.append((mae, -overlap, pi, gi))
    matches: list[tuple[int, int, float]] = []
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    for mae, _neg_overlap, pi, gi in sorted(candidates):
        if float(mae) > float(threshold_s):
            continue
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, float(mae)))
    return matches


def _switch_stats_for_match(
    pred: dict[str, Any],
    gt_match: dict[str, Any],
    gts: list[dict[str, Any]],
    *,
    matched_gt_index: int,
    switch_margin_s: float,
) -> tuple[int, int]:
    by_ch = pred["points_by_channel"]
    matched_valid = np.asarray(gt_match["valid"], dtype=bool)
    if not by_ch or not matched_valid.any():
        return 0, 0

    compared = 0
    switched = 0
    for ch, is_valid in enumerate(matched_valid):
        if not bool(is_valid) or ch not in by_ch:
            continue
        pred_time = float(by_ch[ch]["time_s"])
        matched_time = float(gt_match["time_s"][ch])
        matched_err = abs(pred_time - matched_time)
        best_other_err = float("inf")
        for gi, other in enumerate(gts):
            if gi == int(matched_gt_index):
                continue
            other_valid = np.asarray(other["valid"], dtype=bool)
            if ch >= other_valid.size or not bool(other_valid[ch]):
                continue
            other_err = abs(pred_time - float(other["time_s"][ch]))
            if other_err < best_other_err:
                best_other_err = other_err
        if not math.isfinite(best_other_err):
            continue
        compared += 1
        if best_other_err + float(switch_margin_s) < matched_err:
            switched += 1
    return switched, compared


def _jump_rate(preds: list[dict[str, Any]], threshold_s: float) -> tuple[int, int]:
    jump = 0
    total = 0
    for pred in preds:
        pts = pred["points_by_channel"]
        channels = sorted(pts)
        times = np.asarray([float(pts[ch]["time_s"]) for ch in channels], dtype=np.float32)
        if times.size < 3:
            continue
        second = np.diff(times, n=2)
        total += int(second.size)
        jump += int((np.abs(second) > float(threshold_s)).sum())
    return jump, total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = ShardLabels(Path(args.dataset_dir))
    predictions = _load_predictions(Path(args.predictions))
    window_seconds = float(labels.meta.get("dataset_config", {}).get("window_seconds", labels.meta.get("window_seconds", 120.0)) or 120.0)
    fixed_dead = _parse_int_csv(str(args.fixed_dead_channels))

    total_gt = 0
    total_pred = 0
    matched = 0
    false_positive = 0
    track_counts: list[int] = []
    match_maes: list[float] = []
    missing_errors: list[float] = []
    fixed_dead_errors: list[float] = []
    observed_hits = 0
    observed_total = 0
    jumps = 0
    jump_total = 0
    switch_points = 0
    switch_compared = 0
    switch_tracks = 0

    sample_metrics: list[dict[str, Any]] = []
    for sample_index in sorted(predictions):
        target = labels.get(sample_index)
        gts = _gt_tracks(target, window_seconds)
        preds = _pred_tracks(predictions[sample_index])
        matches = _greedy_match(preds, gts, float(args.match_mae_s))
        total_gt += len(gts)
        total_pred += len(preds)
        matched += len(matches)
        false_positive += max(0, len(preds) - len(matches))
        track_counts.append(len(preds))
        sj, st = _jump_rate(preds, float(args.jump_second_diff_s))
        jumps += sj
        jump_total += st

        for pi, gi, mae in matches:
            pred = preds[pi]
            gt = gts[gi]
            match_maes.append(float(mae))
            switched, compared = _switch_stats_for_match(
                pred,
                gt,
                gts,
                matched_gt_index=int(gi),
                switch_margin_s=float(args.switch_margin_s),
            )
            switch_points += int(switched)
            switch_compared += int(compared)
            if int(compared) >= int(args.switch_track_min_points):
                threshold = max(int(args.switch_track_min_points), int(math.ceil(float(compared) * float(args.switch_track_fraction))))
                if int(switched) >= int(threshold):
                    switch_tracks += 1
            by_ch = pred["points_by_channel"]
            for ch, valid in enumerate(gt["valid"]):
                if not bool(valid) or ch not in by_ch:
                    continue
                err = abs(float(by_ch[ch]["time_s"]) - float(gt["time_s"][ch]))
                if bool(gt["observed"][ch]):
                    observed_total += 1
                    if err <= float(args.anchor_hit_s):
                        observed_hits += 1
                if bool(gt["missing"][ch]):
                    missing_errors.append(err)
                if ch in fixed_dead:
                    fixed_dead_errors.append(err)

        sample_metrics.append(
            {
                "sample_index": int(sample_index),
                "gt_count": int(len(gts)),
                "pred_count": int(len(preds)),
                "matched_count": int(len(matches)),
            }
        )

    precision = matched / max(1, total_pred)
    recall = matched / max(1, total_gt)
    metrics = {
        "dataset_dir": str(Path(args.dataset_dir)),
        "predictions": str(Path(args.predictions)),
        "sample_count": int(len(predictions)),
        "gt_count": int(total_gt),
        "pred_count": int(total_pred),
        "matched_count": int(matched),
        "recall": float(recall),
        "precision": float(precision),
        "false_positive_count": int(false_positive),
        "false_positive_rate_per_sample": float(false_positive / max(1, len(predictions))),
        "avg_track_count": float(np.mean(track_counts)) if track_counts else 0.0,
        "zero_track_samples": int(sum(1 for value in track_counts if value == 0)),
        "track_match_mae_s": float(np.mean(match_maes)) if match_maes else None,
        "observed_anchor_hit_rate": float(observed_hits / max(1, observed_total)),
        "observed_anchor_count": int(observed_total),
        "missing_completion_mae_s": float(np.mean(missing_errors)) if missing_errors else None,
        "missing_completion_count": int(len(missing_errors)),
        "fixed_dead_completion_mae_s": float(np.mean(fixed_dead_errors)) if fixed_dead_errors else None,
        "fixed_dead_completion_count": int(len(fixed_dead_errors)),
        "jump_rate": float(jumps / max(1, jump_total)),
        "jump_count": int(jumps),
        "jump_total": int(jump_total),
        "switch_point_rate": float(switch_points / max(1, switch_compared)),
        "switch_point_count": int(switch_points),
        "switch_point_total": int(switch_compared),
        "switch_track_rate": float(switch_tracks / max(1, matched)),
        "switch_track_count": int(switch_tracks),
        "samples": sample_metrics,
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out_json is not None:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
