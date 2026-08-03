cd /csim2/zhangzhiyu/MyProjects/auto_track/hybrid_vehicle_tracker_20260724

PYTHONPATH=src .venv/bin/python -m hybrid_vehicle_tracker.web.cli \
  --method hybrid \
  --device cuda \
  --port 8000