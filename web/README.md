# Local unified replay service

Run the existing compatible CLI from the Hybrid package:

```bash
hybrid_vehicle_tracker_20260724/.venv/bin/python -m hybrid_vehicle_tracker.web.cli \
  --method hybrid --device cuda
```

The registry in `web/backend/methods.yaml` starts one isolated worker for the
selected method. All four methods use the CUDA worker setting; if CUDA is not
available, the affected method is reported as unavailable instead of silently
falling back to CPU. The service remains local and uses the existing DAY11
cache.
