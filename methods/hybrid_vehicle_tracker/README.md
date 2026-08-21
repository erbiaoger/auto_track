# Hybrid Vehicle Tracker

This is the isolated `hybrid` algorithm project.

This project owns the ResUNet + physical Hough + GNN/MILP implementation,
the v9 checkpoint and Hybrid-specific reports. DAY11 arrays and mapping are
memory-mapped from `../shared_data`.

Run the unified web service from the workspace root with
`./vehicle_replay_web/scripts/run_web.sh`.
The active worker is CUDA-only (`device=cuda`).
