from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.data.mapping import load_station_geometry
from hybrid_vehicle_tracker.evaluation.null_control import run_null_controls
from hybrid_vehicle_tracker.evaluation.synthetic_benchmark import run_synthetic_benchmark
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate physical consistency and null controls")
    parser.add_argument("--config", default="configs/day11_120s.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, help="Explicit evaluation device override")
    parser.add_argument("--null-samples", type=int, default=200)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_tracker_config(args.config)
    if args.checkpoint:
        config.model.checkpoint = args.checkpoint
    if args.device:
        config.runtime.device = args.device
    tracker = HybridVehicleTracker(config)
    if args.synthetic_samples > 0:
        geometry = load_station_geometry(config.data.mapping_path)
        payload = run_synthetic_benchmark(
            tracker.model,
            geometry,
            config,
            samples=args.synthetic_samples,
            seed=config.runtime.seed + 999983,
            device=tracker.device,
        )
        destination = Path(args.output or "reports/synthetic_benchmark.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    batch = tracker.predict_from_paths()
    if tracker.last_artifacts is None:
        raise RuntimeError("prediction did not retain inference artifacts")
    geometry = load_station_geometry(config.data.mapping_path)
    null_result = run_null_controls(
        batch.observations,
        geometry,
        config.association,
        config.model,
        duration_s=config.data.duration_s,
        dense_evidence=tracker.last_artifacts.network_probability,
        feature_tensor=tracker.last_artifacts.features.tensor,
        feature_rate_hz=tracker.last_artifacts.features.feature_rate_hz,
        samples=args.null_samples,
        seed=config.runtime.seed,
        edge_gnn=tracker.model.edge_gnn if tracker.has_checkpoint else None,
        model=tracker.model if tracker.has_checkpoint else None,
        device=tracker.device,
    )
    payload = {
        "real_track_count": len(batch.tracks),
        "real_tracks_above_calibrated_score": sum(
            track.score > null_result.calibrated_min_score for track in batch.tracks
        ),
        "null_control": null_result.to_dict(),
        "meets_target": null_result.mean_tracks_per_window < 0.1,
        "note": "No manual DAY11 ground truth; this is a false-association control, not precision/recall.",
    }
    destination = Path(args.output or "reports/day11_null_control.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
