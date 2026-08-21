"""Deprecated compatibility wrapper; implementation lives in TrajectoryEnergy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/trajectory_energy_tracker/src"))
from trajectory_energy_tracker.trajectory_energy_model import *  # noqa: F401,F403
from trajectory_energy_tracker.trajectory_energy_model import _smooth_track_with_kalman  # noqa: F401
from trajectory_energy_tracker.trajectory_energy_model import _select_decoder_profile  # noqa: F401
