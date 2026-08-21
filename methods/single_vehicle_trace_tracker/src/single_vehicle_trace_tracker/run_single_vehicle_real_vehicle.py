"""End-to-end real-vehicle workflow: train the mixed model, then run raw real-data smoke."""

from __future__ import annotations

import argparse
from pathlib import Path

from autotrack.dl.predict_single_vehicle_real_vehicle import main as predict_real_vehicle_main
from autotrack.dl.train_single_vehicle_real_vehicle import main as train_real_vehicle_main


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and validate the real-vehicle single-track workflow.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for training and smoke results.")
    parser.add_argument("--realbg-npy", type=Path, default=Path("datasets/03gauss_large.npy"), help="Real background .npy.")
    parser.add_argument(
        "--trackslot-dir",
        type=Path,
        default=Path("datasets/track_slot_v3_120s_realistic/train"),
        help="TrackSlot shard directory used during training.",
    )
    parser.add_argument("--device", default="auto", help="Torch device for training and smoke inference.")
    parser.add_argument("--clean-samples", type=int, default=128, help="Synthetic clean samples used for training.")
    parser.add_argument("--realbg-samples", type=int, default=16, help="Real-background samples used for training.")
    parser.add_argument("--trackslot-samples", type=int, default=256, help="TrackSlot-derived samples used for training.")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=48, help="Single-vehicle model hidden size.")
    parser.add_argument("--smoke-windows", type=int, default=24, help="Number of windows for the raw smoke check.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller training mix and one-epoch smoke-friendly settings.")
    parser.add_argument("--keep-benchmarks", action="store_true", help="Keep generated benchmark files from training.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.quick):
        args.clean_samples = min(int(args.clean_samples), 8)
        args.realbg_samples = min(int(args.realbg_samples), 4)
        args.trackslot_samples = min(int(args.trackslot_samples), 8)
        args.epochs = 1
        args.batch_size = 1
        args.hidden_dim = min(int(args.hidden_dim), 32)
        args.smoke_windows = min(int(args.smoke_windows), 6)

    train_args = [
        "--out-dir",
        str(out_dir / "train"),
        "--realbg-npy",
        str(Path(args.realbg_npy).expanduser()),
        "--trackslot-dir",
        str(Path(args.trackslot_dir).expanduser()),
        "--clean-samples",
        str(int(args.clean_samples)),
        "--realbg-samples",
        str(int(args.realbg_samples)),
        "--trackslot-samples",
        str(int(args.trackslot_samples)),
        "--device",
        str(args.device),
        "--epochs",
        str(int(args.epochs)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
    ]
    if bool(args.keep_benchmarks):
        train_args.append("--keep-benchmarks")
    if train_real_vehicle_main(train_args) != 0:
        return 1

    smoke_args = [
        "--preset",
        "real_vehicle",
        "--model",
        str(out_dir / "train" / "checkpoint_best.pt"),
        "--input",
        str(Path(args.realbg_npy).expanduser()),
        "--out-dir",
        str(out_dir / "smoke"),
        "--device",
        str(args.device),
        "--max-windows",
        str(int(args.smoke_windows)),
        "--plot-samples",
        "2",
    ]
    return int(predict_real_vehicle_main(smoke_args))


if __name__ == "__main__":
    raise SystemExit(main())
