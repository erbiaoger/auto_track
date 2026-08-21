#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

uv sync

cat <<'EOF'

Auto-track uv environment is ready.

Run examples:
  uv run python -m autotrack.gui.auto_track_gui
  uv run python -m autotrack.gui.train_data_label_viewer_gui --data-folder datasets/train/sim_0001
  uv run python -m autotrack.dl.train_trajectory_model --help
  uv run python -m autotrack.dl.infer_trajectory_model --help

NVIDIA/CUDA note:
  If this machine installs a CPU-only torch wheel, install the CUDA torch build
  according to your CUDA version, then rerun this script or train with --device cuda.
EOF
