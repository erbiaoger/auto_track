# autotrack.core

Core trajectory extraction and backend integration code.

- `track_extractor_graph.py`: shared `Track` / `TrackPoint` dataclasses and the
  classic peak-to-graph dynamic-programming extractor.
- `trajectory_deep_engine.py`: adapter that loads deep-learning checkpoints and
  exposes them through the backend extraction API. It supports `query_points`,
  `query_masks`, and `track_slot`.
- `track_fusion.py`: optional PeakSlotNet post-processing helper that uses
  graph search to extend detected track fragments and bridge small channel gaps.
- `single_vehicle_tracker.py`: one-vehicle candidate decoder, Hungarian gap
  matching, and Kalman smoothing for the redesigned single-track pipeline.
- `auto_track_backend.py`: data loading, tiling, deduplication, stitching, CSV
  export, and GUI-facing orchestration. It accepts SAC folders and direct real
  DAS `.npy` imports for the GUI.
- `auto_track_torch_mps.py`: PyTorch/MPS helper path for classic extraction.

The deep-learning adapter returns the same `Track` objects as the classic
extractors, so GUI and CSV export code can remain shared.

Enable PeakSlotNet graph fusion through `dl_extra_config` / advanced DL params:

```json
{
  "fusion_mode": "graph_extend",
  "fusion_graph_prominence": 0.18,
  "fusion_graph_min_peak_distance": 120,
  "fusion_graph_max_skip_channels": 8,
  "fusion_min_seed_channels": 4
}
```
