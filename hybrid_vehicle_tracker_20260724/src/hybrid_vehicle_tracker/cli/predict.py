from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_vehicle_tracker.config import load_tracker_config
from hybrid_vehicle_tracker.evaluation.report import write_prediction_outputs
from hybrid_vehicle_tracker.tracker import HybridVehicleTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hybrid DAS vehicle tracking")
    parser.add_argument("--config", default="configs/day11_120s.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, help="Explicit inference device override")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_tracker_config(args.config)
    if args.checkpoint is not None:
        config.model.checkpoint = args.checkpoint
    if args.device is not None:
        config.runtime.device = args.device
    if args.output_dir is not None:
        config.runtime.output_dir = args.output_dir
    if args.start_s is not None:
        config.data.start_s = args.start_s
    if args.duration_s is not None:
        config.data.duration_s = args.duration_s
    tracker = HybridVehicleTracker(config)
    batch = tracker.predict_from_paths()
    if tracker.last_artifacts is None:
        raise RuntimeError("inference completed without diagnostic artifacts")
    destination = write_prediction_outputs(
        batch,
        tracker.last_artifacts,
        Path(config.runtime.output_dir),
        config=config,
    )
    print(f"wrote {len(batch.tracks)} tracks to {destination}")


if __name__ == "__main__":
    main()
