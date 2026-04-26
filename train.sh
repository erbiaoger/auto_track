uv run python train_trajectory_model.py \
  $(for d in datasets/train/sim_*; do echo --data-folder "$d"; done) \
  --out-dir models/trajectory_query_v1 \
  --epochs 20 \
  --batch-size 1 \
  --window-seconds 45 \
  --time-downsample 20 \
  --max-queries 96 \
  --hidden-dim 96 \
  --decoder-layers 1 \
  --num-heads 4 \
  --pooled-channels 8 \
  --pooled-time 128 \
  --trajectory-points 32 \
  --device mps \
  --no-object-weight 0.05 \
  --log-every 5
