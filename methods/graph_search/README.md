# Classic graph search

The classic peak-to-graph dynamic-programming extractor is exposed as the
`graph_search` web method through the existing
`autotrack.cli.auto_track_gpu.extract_all_gpu` entry point. That implementation
uses CuPy for Gaussian enhancement and the existing graph dynamic-programming
code; it has no checkpoint. Its active configuration is stored in
`configs/day11.yaml` and uses the real station positions when the web adapter
runs.

For the existing DAY11 cache, the adapter reverses the channel order before
calling the original GPU CLI kernel and maps points back to the true station
positions. The public result is therefore the physical `reverse` direction
without changing the original graph-search implementation.
