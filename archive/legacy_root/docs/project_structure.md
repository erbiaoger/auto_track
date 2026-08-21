# Project Structure

Implementation code lives in the `autotrack` package. Shell scripts in the
project root call package modules with `python -m autotrack...`.

## Root Entry Points

- `simulate_vehicle_sac.py`: original SAC simulator, intentionally kept at root.
- `*.sh`: shell shortcuts that keep the existing command workflow.

## Implementation Package

- `autotrack/core/`: classic extraction backend, graph extractor, MPS helper, and
  deep-learning extraction engine used by the GUI/backend.
- `autotrack/dl/`: trajectory-query model, offline/online training, inference,
  evaluation, checkpointing, and training metrics.
- `autotrack/gui/`: PyQt GUI applications.
- `autotrack/cli/`: command-line extraction variants.
- `autotrack/simulation/`: accelerated simulation helpers. The original
  `simulate_vehicle_sac.py` remains at root unchanged.
- `docs/`: design notes and generated diagrams.
- `notebooks/`: exploratory notebooks.
