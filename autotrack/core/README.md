# autotrack.core

Core trajectory extraction and backend integration code.

- `track_extractor_graph.py`: shared `Track` / `TrackPoint` dataclasses and the
  classic peak-to-graph dynamic-programming extractor.
- `trajectory_deep_engine.py`: adapter that loads deep-learning checkpoints and
  exposes them through the backend extraction API. It supports `query_points`,
  `query_masks`, and `track_slot`.
- `auto_track_backend.py`: data loading, tiling, deduplication, stitching, CSV
  export, and GUI-facing orchestration.
- `auto_track_torch_mps.py`: PyTorch/MPS helper path for classic extraction.

The deep-learning adapter returns the same `Track` objects as the classic
extractors, so GUI and CSV export code can remain shared.
