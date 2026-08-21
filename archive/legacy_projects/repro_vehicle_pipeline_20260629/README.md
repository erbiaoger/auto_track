# Vehicle Pipeline Repro Bundle

This directory collects the current vehicle-recognition work into one place.

Contents:
- `src/` current source files used by the proposal / trace / multi-vehicle pipeline
- `tests/` regression tests for the current route
- `models/` checkpoints used for the current results
- `data/` benchmark pointer and metadata
- `results/` current overlays
- `scripts/` reproduction entrypoints

Reproduce the current artifacts from this bundle:

```bash
cd /csim2/zhangzhiyu/MyProjects/auto_track/repro_vehicle_pipeline_20260629
PYTHONPATH="$PWD/src" python scripts/reproduce_current_results.py
```

The benchmark file is linked at:
- `data/multi_vehicle_set_rich_bench.pt`

Notes:
- The benchmark is large, so it is linked to the original file in `/tmp` on this machine.
- The included checkpoints and overlays are the ones used to generate the current example results.
