#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec .venv/bin/hvt-evaluate --config configs/day11_120s_v9.yaml --null-samples 200 "$@"
