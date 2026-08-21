"""Deprecated compatibility wrapper; implementation lives in SingleVehicleFocus."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "methods/single_vehicle_focus_tracker/src"))
from single_vehicle_focus_tracker.single_vehicle_focus_net import *  # noqa: F401,F403
