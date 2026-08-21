#!/usr/bin/env python3
"""Convert one full measured day to the model's mapped Raw/Pre/Gauss arrays.

The reference threshold picker is loaded read-only from the dataset directory.
Unlike the sliding-window renderer, this command loads and thresholds each
station exactly once for the complete day.  Later 120 s windows can mmap and
slice the three output arrays without repeating conversion or file I/O.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from convert_threshold_and_predict import _selected_mapping, _station_id


def _load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("hvt_full_day_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, default=Path(
        "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/0002pre2guass.py"
    ))
    parser.add_argument("--sig-dir", type=Path, default=Path(
        "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_raw/11"
    ))
    parser.add_argument("--pred-dir", type=Path, default=Path(
        "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_pre/new11"
    ))
    parser.add_argument("--mapping", type=Path, default=Path(
        "/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json"
    ))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "runs/day11_threshold_090_full_day_cache"
    ))
    parser.add_argument("--station-thresholds-json", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "full_day_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        print(f"cache already exists: {manifest_path}; use --overwrite to rebuild")
        return

    module = _load_converter(args.converter)
    module.SIG_DIR = str(args.sig_dir)
    module.PRED_DIR = str(args.pred_dir)
    module.PRED_THRESH = float(args.threshold)
    module.STATION_THRESHOLDS_JSON = str(args.station_thresholds_json or "")

    pairs = module.scan_pairs(str(args.sig_dir), str(args.pred_dir))
    pair_by_station = {_station_id(Path(sig)): (fname, sig, pred) for fname, sig, pred in pairs}
    mapping_rows = _selected_mapping(args.mapping)
    station_ids = [str(row["station_id"]).upper() for row in mapping_rows]
    missing = [station for station in station_ids if station not in pair_by_station]
    if missing:
        raise FileNotFoundError(f"missing mapped stations: {missing}")

    first_fname, first_sig, _ = pair_by_station[station_ids[0]]
    first = np.load(first_sig, mmap_mode="r")
    sample_count = int(np.asarray(first).size)
    if sample_count < 2:
        raise ValueError("input waveform is too short")
    shape = (sample_count, len(station_ids))
    raw_path = args.output_dir / "raw_threshold_full_DAY11.npy"
    pre_path = args.output_dir / "pre_probability_full_DAY11.npy"
    gauss_path = args.output_dir / "gauss_threshold_full_DAY11.npy"
    raw_out = np.lib.format.open_memmap(raw_path, mode="w+", dtype=np.float32, shape=shape)
    pre_out = np.lib.format.open_memmap(pre_path, mode="w+", dtype=np.float32, shape=shape)
    gauss_out = np.lib.format.open_memmap(gauss_path, mode="w+", dtype=np.float32, shape=shape)

    station_stats: list[dict[str, object]] = []
    full_range = None  # load_pair interprets None as the complete source array.
    for col, station_id in enumerate(station_ids):
        fname, sig_path, pred_path = pair_by_station[station_id]
        sig, pred_raw, time_s = module.load_pair(
            sig_path,
            pred_path,
            module.DT,
            full_range,
            module.AGC_APPLY,
            module.AGC_WIN_S,
        )
        n = min(sample_count, len(sig), len(pred_raw), len(time_s))
        if n != sample_count:
            raise ValueError(f"{fname}: length {n}, expected {sample_count}")
        pred = module.postprocess_prob(
            pred_raw,
            module.DT,
            smooth_s=module.SMOOTH_S,
            power=module.SHARPEN_POWER,
            morph_open_s=module.MORPH_OPEN_S,
            normalize=module.NORMALIZE,
        )
        event_score = 1.0 - pred if module.FLIP_PROB else pred
        threshold = module.get_station_threshold(fname)
        pick_times, _ = module.pick_peaks_1d(
            event_score,
            time_s,
            threshold,
            module.MIN_GAP_S,
            module.DT,
        )
        gauss = module.build_gaussian_curve(
            time_s,
            sig,
            pick_times,
            module.GAUSS_WIDTH_S,
            module.GAUSS_AMP_SCALE,
        )
        raw_out[:, col] = sig
        pre_out[:, col] = pred_raw
        gauss_out[:, col] = gauss
        station_stats.append({
            "channel_index": col,
            "station_id": station_id,
            "source_file": fname,
            "threshold": float(threshold),
            "peak_count": int(len(pick_times)),
        })
        print(f"[{col + 1:02d}/{len(station_ids)}] {station_id}: {len(pick_times)} peaks")

    raw_out.flush()
    pre_out.flush()
    gauss_out.flush()
    del raw_out, pre_out, gauss_out
    manifest = {
        "source_converter": str(args.converter),
        "sig_dir": str(args.sig_dir),
        "pred_dir": str(args.pred_dir),
        "mapping": str(args.mapping),
        "threshold": float(args.threshold),
        "station_count": len(station_ids),
        "sample_count": sample_count,
        "duration_s": float(sample_count * module.DT),
        "sample_rate_hz": float(1.0 / module.DT),
        "raw_path": str(raw_path),
        "pre_path": str(pre_path),
        "gauss_path": str(gauss_path),
        "pre_semantics": "unflipped probability after reference loader; event score is 1-pre",
        "gauss_semantics": "reference threshold picker, Gaussian width 0.5 s, amp scale 0.8",
        "station_ids_in_channel_order": station_ids,
        "station_stats": station_stats,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "shape": shape, "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
