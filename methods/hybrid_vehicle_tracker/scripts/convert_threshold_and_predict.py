#!/usr/bin/env python3
"""Convert the supplied Raw+probability traces with 0002pre2guass.py and track.

The conversion module is loaded read-only from the dataset directory.  Its
threshold picker (including FLIP_PROB, 2 s minimum gap and 0.5 s Gaussian
window) is used verbatim; this file only assembles the resulting per-station
curves into the 50-channel physical mapping used by the tracker.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import write_prediction_outputs
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


STATION_RE = re.compile(r"_([^_]+)_EHZ_", re.IGNORECASE)


def _load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("hvt_threshold_converter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load conversion script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _station_id(path: Path) -> str:
    match = STATION_RE.search(path.name)
    if match is None:
        raise ValueError(f"cannot parse station id from {path.name}")
    return match.group(1).upper()


def _selected_mapping(mapping_path: Path) -> list[dict]:
    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    rows = payload.get("selected_channels") or []
    if len(rows) != 50:
        raise ValueError(f"expected 50 selected channels, got {len(rows)}")
    return sorted(rows, key=lambda row: int(row["channel_index"]))


def convert_and_assemble(
    *,
    converter_path: Path,
    sig_dir: Path,
    pred_dir: Path,
    gauss_dir: Path,
    mapping_path: Path,
    cache_dir: Path,
    threshold: float,
    duration_s: float,
    start_s: float = 0.0,
    station_thresholds_json: str = "",
    save_mode: str = "single",
) -> tuple[Path, Path, Path, dict]:
    """Run the reference converter and export aligned [time, channel] arrays."""
    module = _load_converter(converter_path)
    module.SIG_DIR = str(sig_dir)
    module.PRED_DIR = str(pred_dir)
    module.PRED_THRESH = float(threshold)
    module.STATION_THRESHOLDS_JSON = station_thresholds_json
    module.OUTPUT_DIR = str(cache_dir / "visual")
    module.GAUSS_SAVE_DIR = str(gauss_dir)
    module.T_RANGE = (float(start_s), float(start_s + duration_s))
    module.PLOT_T_RANGE = (float(start_s), float(start_s + duration_s))
    module.SAVE_MODE = save_mode
    mapping_rows = _selected_mapping(mapping_path)
    expected_station_ids = {str(row["station_id"]).upper() for row in mapping_rows}
    existing_gauss_ids = {
        _station_id(path) for path in gauss_dir.glob("*.npy")
        if STATION_RE.search(path.name)
    }
    if not expected_station_ids.issubset(existing_gauss_ids):
        module.main()
    else:
        print(f"reusing {len(existing_gauss_ids)} existing threshold Gaussian curves in {gauss_dir}")

    raw_files = {_station_id(path): path for path in sig_dir.glob("*.npy")}
    pred_files = {_station_id(path): path for path in pred_dir.glob("*.npy")}
    gauss_files = {_station_id(path): path for path in gauss_dir.glob("*.npy")}
    station_ids = [str(row["station_id"]).upper() for row in mapping_rows]
    missing = {
        "raw": sorted(set(station_ids) - raw_files.keys()),
        "pred": sorted(set(station_ids) - pred_files.keys()),
        "gauss": sorted(set(station_ids) - gauss_files.keys()),
    }
    if any(missing.values()):
        raise FileNotFoundError(f"missing mapped stations after conversion: {missing}")

    sample_count = int(round(duration_s / module.DT))
    raw = np.empty((sample_count, len(station_ids)), dtype=np.float32)
    pre = np.empty_like(raw)
    gauss = np.empty_like(raw)
    for col, station in enumerate(station_ids):
        raw_col, pred_col, _ = module.load_pair(
            str(raw_files[station]),
            str(pred_files[station]),
            module.DT,
            (float(start_s), float(start_s + duration_s)),
            module.AGC_APPLY,
            module.AGC_WIN_S,
        )
        gauss_col = np.load(gauss_files[station], mmap_mode="r")[:sample_count]
        if len(raw_col) < sample_count or len(pred_col) < sample_count or len(gauss_col) < sample_count:
            raise ValueError(
                f"station {station}: expected {sample_count} samples, "
                f"got raw={len(raw_col)} gauss={len(gauss_col)}"
            )
        # The reference script picks on FLIP_PROB=(1-p), but the tracker uses
        # Pre as a negative valley and computes 1-pre_score as event evidence.
        # Keeping the unflipped sigmoid probability therefore preserves the
        # exact selected centres without inverting the tracker's Pre semantics.
        raw[:, col] = raw_col[:sample_count]
        pre[:, col] = pred_col[:sample_count]
        gauss[:, col] = np.asarray(gauss_col, dtype=np.float32)

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": cache_dir / "raw_threshold.npy",
        "pre": cache_dir / "pre_probability.npy",
        "gauss": cache_dir / "gauss_threshold.npy",
    }
    np.save(paths["raw"], raw)
    np.save(paths["pre"], pre)
    np.save(paths["gauss"], gauss)
    manifest = {
        "source_converter": str(converter_path),
        "sig_dir": str(sig_dir),
        "pred_dir": str(pred_dir),
        "gauss_dir": str(gauss_dir),
        "mapping": str(mapping_path),
        "threshold": float(threshold),
        "start_s": float(start_s),
        "duration_s": float(duration_s),
        "sample_rate_hz": 1000.0,
        "station_count": len(station_ids),
        "station_ids_in_channel_order": station_ids,
        "pre_semantics": "unflipped sigmoid probability; event evidence is 1-pre_score",
        "gauss_semantics": "0002pre2guass.py FLIP_PROB threshold picks, sigma=0.5 s",
    }
    (cache_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return paths["raw"], paths["pre"], paths["gauss"], manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converter", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/0002pre2guass.py"))
    parser.add_argument("--sig-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_raw/11"))
    parser.add_argument("--pred-dir", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/threshold_test/day11_pre/new11"))
    parser.add_argument("--mapping", type=Path, default=Path("/csim2/zhangzhiyu/MyProjects/auto_track/datasets/peaks_20260723/converted_50ch/arrays/raw_DAY11.mapping.json"))
    parser.add_argument("--gauss-dir", type=Path, default=Path("runs/day11_threshold_090_real/gauss"))
    parser.add_argument("--cache-dir", type=Path, default=Path("runs/day11_threshold_090_real/modal"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/day11_threshold_090_real/tracks"))
    parser.add_argument("--config", type=Path, default=Path("configs/day11_120s.yaml"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--station-thresholds-json", default="")
    parser.add_argument("--skip-conversion", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.gauss_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_conversion:
        raw_path, pre_path, gauss_path, manifest = convert_and_assemble(
            converter_path=args.converter,
            sig_dir=args.sig_dir,
            pred_dir=args.pred_dir,
            gauss_dir=args.gauss_dir,
            mapping_path=args.mapping,
            cache_dir=args.cache_dir,
            threshold=args.threshold,
            start_s=args.start_s,
            duration_s=args.duration_s,
            station_thresholds_json=args.station_thresholds_json,
        )
    else:
        raw_path = args.cache_dir / "raw_threshold.npy"
        pre_path = args.cache_dir / "pre_probability.npy"
        gauss_path = args.cache_dir / "gauss_threshold.npy"
        manifest = json.loads((args.cache_dir / "conversion_manifest.json").read_text())

    config = load_tracker_config(args.config)
    config.data.raw_path = str(raw_path)
    config.data.pre_path = str(pre_path)
    config.data.gauss_path = str(gauss_path)
    config.data.mapping_path = str(args.mapping)
    config.data.start_s = 0.0
    config.data.duration_s = float(args.duration_s)
    config.runtime.output_dir = str(args.output_dir)
    config.runtime.device = "cuda"
    tracker = HybridVehicleTracker(config)
    batch = tracker.predict_from_paths()
    if tracker.last_artifacts is None:
        raise RuntimeError("inference completed without diagnostic artifacts")
    destination = write_prediction_outputs(
        batch,
        tracker.last_artifacts,
        args.output_dir,
        config=config,
    )
    summary = {
        "conversion": manifest,
        "prediction_dir": str(destination),
        "track_count": len(batch.tracks),
        "diagnostics": batch.diagnostics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "threshold_conversion_tracking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"wrote {len(batch.tracks)} tracks to {destination}")


if __name__ == "__main__":
    main()
