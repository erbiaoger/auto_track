"""Deprecated compatibility wrapper; implementation lives in VehicleSet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/vehicle_set_tracker/src"))
from vehicle_set_tracker.vehicle_set_net import *  # noqa: F401,F403
