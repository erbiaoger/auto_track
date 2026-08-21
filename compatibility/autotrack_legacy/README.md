# autotrack Package

This package contains the implementation code.

- `dl/`: PyTorch dataset generation, models, training, inference, and evaluation.
- `core/`: shared trajectory data types, extraction engines, and backend adapter.
- `gui/`: PyQt user interfaces.
- `kalman/`: Kalman-filter based tracking utilities adapted from standalone KF scripts.
- `labeling/`: editable label-project storage used by the real-data labeling GUI.
- `cli/`: command-line classic extraction tools.
- `simulation/`: synthetic DAS/SAC generation helpers.

Run package modules from the project root with `uv run python -m autotrack...`.
