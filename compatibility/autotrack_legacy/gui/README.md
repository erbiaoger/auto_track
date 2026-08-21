# autotrack.gui

PyQt user interfaces.

- `auto_track_gui.py`: main interactive DAS trajectory extraction GUI. It can
  import either a SAC folder, a single real DAS `.npy` file, or one tensor
  shard `.pt/.pth` file. For tensor shards it currently loads the first sample,
  and it remembers the last Deep Learning checkpoint path across launches.
- `kalman_track_gui.py`: full Kalman-filter interactive GUI adapted from the
  original `KF03.py`, but reading real `.npy` DAS arrays directly.
- `real_data_label_gui.py`: real-data labeling GUI. It reuses the classic graph
  search backend to auto-label the current window, then supports manual
  point-level calibration and saves `manual_labels.json` / `manual_labels.csv`.
- `train_data_label_viewer_gui.py`: viewer for simulated training data and
  `tracks.json` labels.

Launch from the project root with `uv run python -m autotrack.gui.<module>`.
