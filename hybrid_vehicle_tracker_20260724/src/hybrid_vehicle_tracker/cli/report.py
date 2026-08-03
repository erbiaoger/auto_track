from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate a compact report from prediction files")
    parser.add_argument("run_dir")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_dir = Path(args.run_dir)
    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    tracks = [
        json.loads(line)
        for line in (run_dir / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lines = [
        "# Hybrid Vehicle Tracker summary",
        "",
        f"- Tracks: {len(tracks)}",
        f"- Observations: {diagnostics.get('observation_count', 0)}",
        f"- Candidate graph edges: {diagnostics.get('edge_count', 0)}",
        f"- Device: {diagnostics.get('device', 'unknown')}",
        "",
        "| track | speed km/h | observed | span m | confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for track in tracks:
        lines.append(
            f"| {track['track_id']} | {track['median_speed_kmh']:.2f} | "
            f"{track['observed_count']} | {track['span_m']:.0f} | {track['confidence']:.3f} |"
        )
    destination = Path(args.output) if args.output else run_dir / "summary.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
