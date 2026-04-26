#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

DEVICE=${DEVICE:-auto}
COUNT=${COUNT:-50}
START_SEED=${START_SEED:-1}

for i in $(seq "$START_SEED" $((START_SEED + COUNT - 1))); do
  uv run python simulate_vehicle_sac_torch.py \
    --out-dir "$SCRIPT_DIR/datasets/train/sim_$(printf "%04d" "$i")" \
    --seed "$i" \
    --primary-count 300 \
    --secondary-count 60 \
    --noise-std 0.3 \
    --fixed-amp 6.0 \
    --duration-s 3600 \
    --fs 1000 \
    --n-ch 50 \
    --dx-m 100 \
    --speed-range-kmh 60 110 \
    --speed-jitter-kmh-range -3 3 \
    --accel-count 20 \
    --decel-count 20 \
    --stop-go-count 10 \
    --device "$DEVICE"
done
