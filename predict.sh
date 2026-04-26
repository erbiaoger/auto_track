uv run python infer_trajectory_model.py \
  --data-folder datasets/test/sim_1001 \
  --model models/trajectory_query_online_v1_cuda/checkpoint_best.pt \
  --device auto \
  --window-start-s 0 \
  --window-seconds 120 \
  --out-csv datasets/test/sim_1001/auto_tracks_deep.csv \
  --objectness-threshold 0.1 \
  --visibility-threshold 0.2 \
  --min-visible-channels 2
