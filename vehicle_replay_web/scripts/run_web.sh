#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/common/src:$ROOT_DIR/vehicle_replay_web/backend:$ROOT_DIR/methods/hybrid_vehicle_tracker/src:$ROOT_DIR/methods/peak_slot_tracker/src:$ROOT_DIR/methods/trajectory_query_tracker/src:$ROOT_DIR/methods/vehicle_peak_set_tracker/src:$ROOT_DIR/methods/graph_search_tracker/src:$ROOT_DIR/methods/hungarian_assignment_tracker/src:$ROOT_DIR/methods/kalman_seed_tracker/src:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$ROOT_DIR/.venv/bin/python" -m vehicle_replay_web.cli "$@"
