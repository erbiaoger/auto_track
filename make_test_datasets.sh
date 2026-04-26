#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)

COUNT=${COUNT:-2}
START_SEED=${START_SEED:-1001}

for i in $(seq "$START_SEED" $((START_SEED + COUNT - 1))); do
  (cd "$PROJECT_ROOT" && uv run python "$SCRIPT_DIR/simulate_vehicle_sac.py" \
    --out-dir "$SCRIPT_DIR/datasets/test/sim_$(printf "%04d" "$i")" \
    --seed "$i" \
    --primary-count 300 \
    --secondary-count 60 \
    --noise-std 0.3 \
    --fixed-amp 6.0 \
    --duration-s 3600 \
    --fs 1000 \
    --n-ch 50 \
    --dx-m 100 \
    --speed-range-kmh 60 90 \
    --speed-jitter-kmh-range -1 1 \
    --speed-jitter-channel-count 3 \
    --accel-count 20 \
    --decel-count 20 \
    --stop-go-count 10 \
    --accel-mps2 0.9 \
    --decel-mps2 1.1 \
    --stop-brake-mps2 1.5 \
    --restart-accel-mps2 0.8 \
    --no-noise)
done
