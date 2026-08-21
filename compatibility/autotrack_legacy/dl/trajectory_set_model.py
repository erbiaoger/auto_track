"""Deprecated compatibility wrapper; implementation lives in TrajectoryQuery."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/trajectory_query_tracker/src"))
from trajectory_query_tracker.trajectory_set_model import *  # noqa: F401,F403
from trajectory_query_tracker.trajectory_set_model import _refine_t_idx, _robust_scale  # noqa: F401
