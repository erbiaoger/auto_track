"""Convenience wrapper for the validated real-vehicle single-track inference preset."""

from __future__ import annotations

import sys

from autotrack.dl.predict_single_vehicle_real_npy import main as predict_main


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--preset" not in args:
        args = ["--preset", "real_vehicle", *args]
    return int(predict_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
