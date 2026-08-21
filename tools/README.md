# Workspace tools

Only neutral data preparation, labeling, simulation, catalog validation and
migration utilities belong here. Model inference and decoding stay in method
projects. The workspace installer lives at `tools/install_uv.sh`.

Algorithm shell entrypoints live beside their implementation:

- `methods/track_slot_tracker/scripts/` — TrackSlot/profile/data generation
- `methods/peak_slot_tracker/scripts/` — PeakSlot training, conversion and prediction
- `methods/vehicle_peak_set_tracker/scripts/` — Peak-Set training/evaluation pipelines
- `methods/hybrid_vehicle_tracker/scripts/` — Hybrid training and DAY11 replay
- `vehicle_replay_web/scripts/run_web.sh` — shared web launcher

旧的根目录 `.sh` 名称不再放在规范目录中；需要兼容旧调用时，请从
`compatibility/` 查找对应包装入口。
