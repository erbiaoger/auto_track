#!/usr/bin/env python3
"""Create small ownership catalogs for the newly created method projects."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "hybrid_vehicle_tracker": "hybrid",
    "peak_slot_tracker": "peak_slot",
    "vehicle_peak_set_tracker": "vehicle_peak_set",
    "graph_search_tracker": "graph_search",
    "track_slot_tracker": "track_slot",
    "compact_slot_tracker": "compact_slot",
    "trajectory_query_tracker": "trajectory_query",
    "query_mask_tracker": "query_mask",
    "trajectory_energy_tracker": "trajectory_energy",
    "single_vehicle_trace_tracker": "single_vehicle_trace",
    "single_vehicle_focus_tracker": "single_vehicle_focus",
    "vehicle_proposal_trace_pipeline": "vehicle_proposal_trace",
    "vehicle_set_tracker": "vehicle_set",
    "kalman_seed_tracker": "kalman_seed",
}


def main() -> int:
    for directory, method_id in METHODS.items():
        project = ROOT / "methods" / directory
        (project / "README.md").write_text(
            f"# {method_id}\n\nThis is the isolated `{method_id}` algorithm project.\n\n"
            "The shared workspace uses the top-level `.venv`; archived routes are "
            "kept under `archive/` and are not exposed by the web selector.\n",
            encoding="utf-8",
        )
        for section in ("data", "checkpoints", "results", "archive"):
            path = project / section / "catalog.yaml"
            path.write_text(
                f"method: {method_id}\nstatus: pending_migration\nentries: []\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
