from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn

from auto_track_common.mapping import load_station_geometry

from .app import create_app
from .methods import MethodManager
from .runtime import ReplayController
from .source import NpyReplaySource


DEFAULT_MAPPING = Path(
    "shared_data/day11/mapping/raw_DAY11.mapping.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the DAY11 rolling vehicle recognition UI")
    parser.add_argument("--cache-dir", type=Path, default=Path("shared_data/day11/cache"))
    parser.add_argument("--config", type=Path, default=Path("methods/hybrid_vehicle_tracker/configs/day11_120s_v9.yaml"))
    parser.add_argument("--mapping", type=Path, default=None, help="Station mapping; defaults to the manifest mapping when present.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--frontend-dist", type=Path, default=Path("vehicle_replay_web/frontend/dist"))
    parser.add_argument("--video-dir", type=Path, default=Path("shared_data/2025-08-09"), help="Directory containing synchronized replay videos")
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--method", default="hybrid", help="Initial method id from web/backend/methods.yaml")
    parser.add_argument("--window-s", type=float, default=120.0, help="recognition window length in seconds")
    parser.add_argument("--stride-s", type=float, default=60.0, help="window movement/recognition interval in seconds")
    parser.add_argument("--waveform-downsample", type=int, default=20, help="raw waveform downsample factor (default: 20x)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.window_s <= 0 or args.stride_s <= 0 or args.waveform_downsample < 1:
        raise ValueError("--window-s and --stride-s must be positive; --waveform-downsample must be >= 1")
    manifest_path = args.cache_dir / "full_day_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"full-day cache manifest not found: {manifest_path}")
    source = NpyReplaySource(manifest_path, waveform_downsample=args.waveform_downsample)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.mapping is not None:
        mapping_path = args.mapping
    elif manifest.get("mapping"):
        mapping_path = Path(manifest_path.parent / str(manifest["mapping"]))
    else:
        mapping_path = DEFAULT_MAPPING
    geometry = load_station_geometry(mapping_path)
    manager = MethodManager(source=source, mapping_path=mapping_path, device=args.device)
    initial_method = str(args.method)
    if initial_method in manager.specs:
        manager.default_method = initial_method
    if initial_method not in manager.specs or not manager._availability(manager.specs[initial_method])[0]:
        preferred = ["graph_search"] if str(args.device) == "cuda" else ["hungarian_assignment", "kalman_seed"]
        candidates = preferred + [method_id for method_id in manager.specs if method_id not in preferred]
        fallback = next(
            (
                method_id
                for method_id in candidates
                if str(manager.specs[method_id].get("input", "")) == "gauss"
                and (not manager.specs[method_id].get("device") or str(manager.specs[method_id].get("device")) == str(args.device))
                and manager._availability(manager.specs[method_id])[0]
            ),
            None,
        )
        if fallback is not None:
            manager.default_method = fallback
    if args.checkpoint is not None and initial_method == "hybrid":
        # The registry is the source of truth for normal operation.  Keep the
        # existing --checkpoint flag as a compatibility override for Hybrid.
        manager.specs[initial_method]["checkpoint"] = str(args.checkpoint)
    output_root = args.output_root
    if output_root is None:
        output_root = Path(__file__).resolve().parents[3] / "methods/hybrid_vehicle_tracker/results/web_sessions"
    controller = ReplayController(
        source=source,
        tracker=None,
        mapping=geometry,
        window_s=args.window_s,
        stride_s=args.stride_s,
        waveform_downsample=args.waveform_downsample,
        output_root=output_root,
        method_manager=manager,
    )
    app = create_app(controller, frontend_dist=args.frontend_dist, video_dir=args.video_dir, access_token=args.access_token)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
