# Classic graph-search source boundary

The web entry point is the existing GPU implementation
`autotrack/cli/auto_track_gpu.py:extract_all_gpu`. It reuses
`autotrack/core/track_extractor_graph.py` for dynamic programming. The web
adapter reverses channel order for the existing GPU kernel and maps resulting
points back to physical `x_axis_m` positions.
