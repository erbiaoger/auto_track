# docs

Design and algorithm notes for the project.

- `auto_track_algorithm.md`: classic extraction algorithm notes.
- `current_peak_slot_model_training.md`: current deep-learning mainline method,
  including the active PeakSlotNet model, loss terms, training setup, and the
  physical/mathematical principles behind them.
- `trajectory_query_network.md`: legacy query model design.
- `track_slot_network.md`: TrackSlotNet design, training target, matching, and
  variable vehicle-count behavior.
- `track_slot_method_complete.md`: complete TrackSlotNet method note covering
  physical model, math formulation, network structure, training, inference, and
  visualization.
- `compact_slot_model.md`: lightweight replacement mainline for the 50-channel
  task, centered on a channel-first encoder and slot attention. Use the new
  compact slot CLI entrypoints for this branch.
- `single_vehicle_tracking.md`: new one-vehicle tracking direction with a
  dense heatmap scorer, Hungarian gap bridging, and Kalman smoothing.
- `compact_slot_model.md`: compact slot mainline for the 50-channel task.
- `compact_slot_model.md` and `build_single_vehicle_benchmark_realistic.py`
  together define the new narrow single-vehicle training path.
- `realbg_profile_workflow.md`: profile-driven real-background generation,
  calibration, and PeakSlotNet data-production workflow.
- `real_data_label_gui.md`: real-data auto-label + manual calibration GUI
  workflow and label-file format.
- `project_structure.md`: high-level package layout.
- `network_structure.png`: generated network diagram asset.
